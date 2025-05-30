import torch
import re
import numpy as np
import os
from pathlib import Path
from itertools import chain
import networkx as nx
from tqdm import tqdm
import math
import os, tempfile
from copy import deepcopy
import mdtraj as md
from mdtraj import Trajectory
import warnings
from enum import Enum
from typing import Optional, Any, Callable, Sequence, Union
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
from ase.data import atomic_numbers

from utils import DummyClass

from evaluate.msm_utils import discretize_trajectory
from evaluate.committor_utils import get_gt_committor_probs

# MDGen imports
from mdgen.mdgen.rigid_utils import Rigid
from mdgen.mdgen.residue_constants import restype_order
import mdgen.mdgen.residue_constants as rc
import numpy as np
import pandas as pd
from mdgen.mdgen.geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames


class AtomSelection(Enum):
    PROTEIN = "protein"
    A_CARBON = "c-alpha"
    ALL = "all"


class Molecules(Enum):
    CHIGNOLIN = "CLN025"
    TRP_CAGE = "2JOF"
    BBA = "1FME"
    VILLIN = "2F4K"
    WW_DOMAIN = "GTT"
    NTL9 = "NTL9"
    BBL = "2WAV"
    PROTEIN_B = "PRB"
    HOMEODOMAIN = "UVF"
    PROTEIN_G = "NuG2"
    ALPHA3D = "A3D"
    LAMBDA_REPRESSOR = "lambda"


class AtlasProteins(Enum):
    DELTA = "5w82_E"
    MEMBRANE = "1l2w_I"
    GPROTEIN = "2pbi_A"
    CAPSID = "3qc7_A"
    ADHESIN = "3wp8_A"


ATLAS_PDB_ID_TO_NAME = {
    "5w82_E": "delta",
    "1l2w_I": "membrane",
    "2pbi_A": "gprotein",
    "3qc7_A": "capsid",
    "3wp8_A": "adhesin",
}

PDB_ID_TO_NAME = {
    "cln025": "chignolin",
    "2jof": "trp_cage",
    "1fme": "bba",
    "2f4k": "villin",
    "1mi0": "protein_g",
}


all_molecules = ["alanine_dipeptide"] + [mol.name.lower() for mol in Molecules]

norm_stds = {
    Molecules.CHIGNOLIN: 3.113133430480957,
    Molecules.TRP_CAGE: 5.08211088180542,
    Molecules.BBA: 6.294918537139893,
    Molecules.VILLIN: 6.082900047302246,
    Molecules.PROTEIN_G: 6.354289531707764,
    "alanine_fold1": 0.9449278712272644,
    "alanine_fold2": 0.944965124130249,
    "alanine_fold3": 0.9452606439590454,
    "alanine_fold4": 0.9454087018966675,
}

# default cluster endpoints for testing interpolation
# (obtained by visual inspection of what are hard transition paths to capture)
CLUSTER_ENDPOINTS = {
    "chignolin": [11, 13],
    "trp_cage": [2, 13],
    "bba": [9, 17],
    "villin": [0, 17],
    "protein_g": [11, 14],
}


def mae_to_pdb_atom_mapping(protein_name, forward=True):
    """
    In the case of all-atom proteins, we need to correct for the fact that the pdb and mae/dcd files have different atom orderings.
    """

    pdb_topology = md.load_topology(
        f"./datasets/folded_pdbs/{Molecules[protein_name.upper()].value}-0-protein.pdb"
    )
    pdb_bonds = torch.tensor(
        [(bond[0].index, bond[1].index) for bond in pdb_topology.bonds]
    )
    mae_bonds = extract_bonds_from_mae(
        f"./datasets/Reference_MD_Sims/{Molecules[protein_name.upper()].value}/simulation_0/protein/{Molecules[protein_name.upper()].value}-0-protein/{Molecules[protein_name.upper()].value}-0-protein.mae"
    )
    if forward:
        return recover_permutation(mae_bonds, pdb_bonds)
    return recover_permutation(pdb_bonds, mae_bonds)


def extract_bonds_from_mae(file_path):
    """
    Extract bond indices from a .mae file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    bond_section = False
    bonds = []

    for line in lines:
        line = line.strip()

        # Detect the start of the m_bond block
        if line.startswith("m_bond"):
            bond_section = True
            continue

        # Detect the end of the block
        if bond_section and line.startswith("}"):
            break

        # Skip the header inside the block (first few lines)
        if bond_section and ":::" in line:
            continue

        # Extract bond data
        if bond_section:
            parts = re.split(r"\s+", line)  # Split by whitespace
            if len(parts) >= 4:  # Ensure valid data row
                i_m_from, i_m_to = int(parts[1]), int(parts[2])
                bonds.append([i_m_from, i_m_to])

    # Convert to torch.Tensor
    bond_tensor = torch.tensor(bonds, dtype=torch.int64) - 1

    return bond_tensor


def recover_permutation(bonds1, bonds2):
    """
    Recovers the permutation mapping node indices in the permuted graph (bonds1)
    to those in the original one (bonds2) using vf2pp_isomorphism from networkx.

    Args:
        bonds1 (torch.Tensor): Permuted graph edges of shape [N, 2]
        bonds2 (torch.Tensor): Original graph edges of shape [N, 2]

    Returns:
        dict or None: A dictionary mapping original node indices to permuted ones if an isomorphism exists, None otherwise.
    """
    # Create NetworkX graphs
    G1 = nx.Graph()
    G2 = nx.Graph()

    G1.add_edges_from(bonds1.tolist())
    G2.add_edges_from(bonds2.tolist())

    # Compute isomorphism
    iso_mapping = nx.vf2pp_isomorphism(G1, G2)

    if iso_mapping is None:
        return None

    # Convert mapping to tensor
    max_node = max(max(G1.nodes), max(G2.nodes)) + 1
    perm_tensor = torch.full((max_node,), -1, dtype=torch.long)

    for perm, orig in iso_mapping.items():
        perm_tensor[orig] = perm

    return perm_tensor


def get_dataset(
    mol,
    mean0,
    data_folder=None,
    fold=None,
    atom_selection=AtomSelection.A_CARBON,
    traindata_subset=None,
    shuffle_before_splitting=False,
    pdb_folder=None,
    tic_evaluator=None,
    committor_remove_range=[],  # range of committor values to remove
    remove_freq=0,  # frequency of removing clusters from data
    tetra_atom_selection="all-atom",
):
    """
    Get dataset for a specific molecule.

    Args:
        mol (str): molecule name
        mean0 (bool): whether or not to center at zero
        data_folder: path to folder containing data, run with empty dataset if None
        fold (int in [1,2,3,4]): fold number, only for alanine dipeptide
        traindata_subset (int): subset for training data
        shuffle_before_splitting (bool): whether or not to shuffle the data
        pdb_folder (str): path to folded pdb files, use "datasets/folded_pdbs/" if None (default)

    NB: the relevant file (ala2_cg_2fs_Hmass_2_HBonds.npz) for the alanine dipeptide dataset
    can be downloaded freely from https://ftp.imp.fu-berlin.de/pub/cmb-data/
    """

    if pdb_folder is None:
        pdb_folder = os.path.join("./datasets/folded_pdbs/")
    if mol.lower() == "alanine_dipeptide_fuberlin":
        assert fold is not None and fold in [
            1,
            2,
            3,
            4,
        ], "Please supply a fold in [1,2,3,4]"

        dataset = FUBerlinAlanine2pDataset(data_folder, fold, pdb_folder, mean0=mean0)

        if data_folder is not None:
            idx_range = torch.arange(len(dataset))
            assert (
                not shuffle_before_splitting
            ), f"Shuffling data before split not supported for dataset {mol}."
            allfolds = idx_range.chunk(4)
            testrange = allfolds[fold - 1]
            trainvalrange = torch.cat(allfolds[: fold - 1] + allfolds[fold:])
            # shuffle trainval data
            trainvalrange = trainvalrange[torch.randperm(len(trainvalrange))[:]]
            trainrange = trainvalrange[0:500000]
            valrange = trainvalrange[500000:]
            assert len(trainrange) + len(valrange) == len(trainvalrange)

            if traindata_subset is not None:
                print("should not go here")
                assert (
                    type(traindata_subset) == int
                    and traindata_subset > 0
                    and len(trainrange) >= traindata_subset
                ), "Provide valid number of points for subset"
                trainrange = trainrange[:traindata_subset]

            trainset = dataset.get_subset(trainrange, dataset.topology, train=True)
            valset = dataset.get_subset(valrange, train=False)
            testset = dataset.get_subset(testrange, train=False)
        else:
            trainset = dataset
            valset = dataset
            testset = dataset

    elif mol.upper() in AtlasProteins.__members__:
        mol = AtlasProteins[mol.upper()].value
        dataset = AtlasDataset(data_folder, mol)

        trainset = dataset
        valset = dataset
        testset = dataset

    elif mol == "tetrapeptides":
        trainset = MDGenDataset(
            data_folder,
            suffix="_i100",
            split="./mdgen/splits/4AA_train.csv",
            atom_selection=tetra_atom_selection,
            train=True,
        )
        valset = MDGenDataset(
            data_folder,
            suffix="_i100",
            split="./mdgen/splits/4AA_val.csv",
            atom_selection=tetra_atom_selection,
            train=False,
        )
        testset = MDGenDataset(
            data_folder,
            suffix="_i100",
            split="./mdgen/splits/4AA_test.csv",
            atom_selection=tetra_atom_selection,
            train=False,
        )

    elif "alanine_dipeptide" not in mol.lower():
        # D.E. Shaw fast folding proteins data
        if fold is not None:
            warnings.warn("Fold not implemented for this dataset")
        if traindata_subset is not None:
            warnings.warn(
                "Traindata subset is not implemented for this molecule. Ignoring this argument"
            )

        molecule = Molecules[mol.upper()]
        print(molecule)
        full_simulation_id = "-".join([molecule.value, str(0), "c-alpha"])
        pdb_file = os.path.join(pdb_folder, full_simulation_id + ".pdb")
        topology = md.load_topology(pdb_file)

        if data_folder is None:
            dataset = None
        else:
            dataset = DEShawDataset(
                data_root=data_folder,
                molecule=molecule,
                simulation_id=0,
                atom_selection=atom_selection,
                return_bond_graph=False,
                transform=to_angstrom,
                align=False,
                tic_evaluator=tic_evaluator,
                committor_remove_range=committor_remove_range,  # range of committor values to remove
                remove_freq=remove_freq,  # frequency of removing clusters from data
            )

        dataset = CGDataset(
            to_angstrom(dataset.traj.xyz) if dataset is not None else dataset,
            topology,
            molecule,
            mean0=mean0,
            shuffle=shuffle_before_splitting,
        )
        if dataset.dataset is not None:
            print(f"Creating dataset splits")
            valratio, testratio = 0.1, 0.001
            num_val = math.floor(valratio * dataset.__len__())
            num_test = math.floor(testratio * dataset.__len__())
            num_train = dataset.__len__() - num_val - num_test
            idx_range = torch.arange(len(dataset))
            train_idx = idx_range[:num_train]
            val_idx = idx_range[num_train : num_train + num_val]
            test_idx = idx_range[num_train + num_val :]
            trainset = dataset.get_subset(train_idx, topology, train=True)
            valset = dataset.get_subset(val_idx, topology, train=False)
            testset = dataset.get_subset(test_idx, topology, train=False)
        else:
            trainset = dataset
            valset = dataset
            testset = dataset
    else:
        raise Exception(
            f"Wrong dataset mol/dataset name {mol}. Provide valid molecule from {all_molecules}"
        )

    return trainset, valset, testset


def to_angstrom(x):
    """
    Convert from nanometer to angstrom.
    """
    return x * 10.0


class DummyClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CGDataset(torch.utils.data.TensorDataset):
    """
    Dataset class specific for CG experiments
    Args:
        dataset: atom coordinates
        topology: topology of the molecule
        molecule (str): molecule name
        mean0 (bool): center molecules at zero
        atom_selection (list of ints): list containing atoms to keep
        shuffle: whether or not to shuffle data
    """

    def __init__(
        self,
        dataset,
        topology,
        molecule,
        mean0=True,
        atom_selection=None,
        shuffle=False,
    ):
        if dataset is not None:
            if not isinstance(dataset, torch.Tensor):
                dataset = torch.tensor(dataset)
        self.dataset = dataset
        self.mean0 = mean0
        self.atom_selection = atom_selection
        if dataset is not None:
            self.dataset = self.prepare_dataset(dataset, mean0, atom_selection, shuffle)
        self.topology = topology
        self.molecule = molecule
        self.std = norm_stds[molecule]
        if hasattr(molecule, "name"):
            assert "alanine" not in molecule.name.lower()
            self.num_beads = topology.n_residues
        elif "alanine" in molecule.lower():
            self.num_beads = 5
        else:
            raise NotImplementedError("Invalid molecule name")
        self.bead_onehot = torch.eye(self.num_beads)
        if dataset is None:
            dataset = torch.zeros(1)
        super().__init__(dataset)

    def prepare_dataset(self, dataset, mean0, atom_selection, shuffle):
        """
        Prepare dataset
        """
        data = dataset[:]
        if atom_selection is not None:
            data = data[:, atom_selection, :]
        if mean0:
            data -= data.mean(dim=1, keepdim=True)
        if shuffle:
            data = data.numpy()
            np.random.seed(2342361)
            np.random.shuffle(data)
            data = torch.Tensor(data)
        return data

    def add_attributes(self, topology):
        """
        Add extra attributes to dataset.
        """
        self.topology = topology
        if self.atom_selection is not None:
            self.topology = topology.subset(self.atom_selection)
        self.num_beads = self[:][0].shape[1]
        self.bead_onehot = torch.eye(self.num_beads)
        self.std = norm_stds[self.molecule]

    def get_subset(self, ind_range, topology=None, train=True, forces=False):
        """
        Get subset of entire dataset
        """
        subset = torch.utils.data.Subset(self.dataset, ind_range)
        subset = CGDataset(
            subset.dataset[subset.indices],
            topology,
            self.molecule,
            self.mean0,
            self.atom_selection,
        )
        if train:
            assert topology is not None, "Provide topology for train set"
            subset.add_attributes(topology)
        return subset


class FUBerlinAlanine2pDataset(CGDataset):
    """
    Dataset for FU Berlin alanine-dipeptide data. Inherits from CGDataset.
    Args:
        data_root (string): root directory for alanine dipeptide data
        mean0 (bool): whether or not to center at zero

    NB: the relevant file (ala2_cg_2fs_Hmass_2_HBonds.npz) for the alanine dipeptide dataset
    can be downloaded freely from https://ftp.imp.fu-berlin.de/pub/cmb-data/
    """

    def __init__(self, data_root, fold, pdb_folder, mean0=True):
        if data_root is None:
            data_coords = None
        else:
            npz_file = "ala2_cg_2fs_Hmass_2_HBonds.npz"
            local_npz_file = os.path.join(data_root, npz_file)
            data_coords = torch.from_numpy(np.load(local_npz_file)["coords"])

        self.topology = md.load(os.path.join(pdb_folder, "ala2_cg.pdb")).topology

        super().__init__(data_coords, self.topology, f"alanine_fold{fold}", mean0=mean0)


class TemporalSequence:
    def __init__(self, timestep: float):
        """A sequence with a timestep attribute indicating the time between consecutive elements.

        Args:
            timestep (float): The time resolution of the sequence in picoseconds.
        """
        self.timestep = timestep


class MDTrajectory(TemporalSequence, Dataset):
    def __init__(
        self,
        traj: Trajectory,
        extra_features: Sequence = None,
        transform: Optional[Callable[[Any], Any]] = None,
        return_bond_graph: bool = False,
        timestep: Optional[float] = None,
        align: bool = False,
    ):
        """Dataset object for a Molecular Dinamics Trajectory object from the mdtraj module

        Args:
            traj (Trajectory): the trajectory to be transformed into a dataset
            extra_features (Sequence): A sequence of (labeled) features to return (other than positions).
            transform (Optional[Callable[[Any] ,Any]], optional): A function to be applied to the atom coordinates (after indexing). Defaults to None.
            return_bond_graph (bool, optional): flag to specify if the data-items are graphs or coordinates only. Defaults to False.
            timestep (float, optional): the time in between consetutive simulation frames in picoseconds. Defaults to None.
            align (bool): flat to align the Trajectories. Defaults to False.
        """

        # Align the trajectory if required
        if align:
            traj = traj.superpose(traj, 0)

        # Save the original trajectory
        self.traj = traj
        self.return_bond_graph = return_bond_graph
        if not (extra_features is None):
            assert len(extra_features) == len(
                traj.xyz
            ), "The extra features must have the same lenght as the trajectory"
        self.extra_features = extra_features
        self.transform = transform

        # Make a lookup dictionary for the atoms in the molecule
        self.atomsdict = {atom: i for i, atom in enumerate(traj.topology.atoms)}

        # Compute an edge index based on the bonds if required
        if return_bond_graph:
            self.edge_index = torch.LongTensor(
                [
                    [self.atomsdict[edge[0]], self.atomsdict[edge[1]]]
                    for edge in traj.topology.bonds
                ]
            ).T

        if timestep is None:
            timestep = traj.timestep

        super(MDTrajectory, self).__init__(timestep=timestep)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Data]:
        x = torch.FloatTensor(self.traj.xyz[idx])
        atom_ids = [a.element.atomic_number - 1 for a in self.traj.topology.atoms]

        if self.return_bond_graph:
            # wrap into a torch_geometric graph
            x = Data(
                pos=x, atom_labels=torch.tensor(atom_ids), edge_index=self.edge_index
            )

        if not (self.transform is None):
            x = self.transform(x)

        # Add the extra features
        if not (self.extra_features is None):
            extra_features = self.extra_features[idx]

            # if the extra features are a dictionary, make sure the is no key 'x'
            if isinstance(extra_features, dict):
                assert not (
                    x in extra_features
                ), "The extra features can't specify a key named 'x'"

            # In case a graph is returned
            if self.return_bond_graph:
                if not isinstance(extra_features, dict):
                    extra_features = {"y": extra_features}
                # Add the attributes to the data object
                for k, v in extra_features.items():
                    setattr(x, k, v)
            else:
                if isinstance(extra_features, dict):
                    # add x to the features
                    extra_features["x"] = x
                    x = extra_features
                else:
                    x = (x, extra_features)
        return x

    def __len__(self):
        return len(self.traj.xyz)


class DEShawDataset(MDTrajectory):
    def __init__(
        self,
        data_root: str,
        molecule: Molecules,
        simulation_id: int,
        atom_selection: AtomSelection = AtomSelection.PROTEIN,
        transform: Optional[Callable[[Any], Any]] = None,
        return_bond_graph: bool = False,
        align: bool = False,
        tic_evaluator=None,
        committor_remove_range=[],  # range of committor values to remove
        remove_freq=0,  # frequency of removing from data
    ):
        self.data_root = data_root
        self.simulation_id = simulation_id
        self.atom_selection = atom_selection
        self.tic_evaluator = tic_evaluator
        full_simulation_id = "-".join(
            [molecule.value, str(simulation_id), atom_selection.value]
        )

        simulation_path = os.path.join(
            molecule.value,
            f"simulation_{simulation_id}",
            atom_selection.value,
            full_simulation_id,
        )
        full_simulation_id = simulation_path.split("/")[-1]

        time_file = os.path.join(simulation_path, full_simulation_id + "_times.csv")
        pdb_file = os.path.join(simulation_path, full_simulation_id + ".pdb")

        time_data = pd.read_csv(
            os.path.join(self.data_root, time_file), names=["time", "file"]
        )

        local_trajectory_files = [
            os.path.join(self.data_root, simulation_path, trajectory_part_file)
            for trajectory_part_file in time_data["file"].values
        ]
        local_pdb_file = os.path.join(self.data_root, pdb_file)

        # Load the trajectory using mdtraj
        traj = md.load(local_trajectory_files, top=local_pdb_file)

        super(DEShawDataset, self).__init__(
            traj=traj,
            transform=transform,
            return_bond_graph=return_bond_graph,
            timestep=time_data["time"].values[0],
            align=align,
        )

        if committor_remove_range:
            # remove clusters from the data

            cluster_centers_path = Path(
                os.path.join(
                    "evaluate",
                    "saved_references",
                    f"saved_cluster_centers_{PDB_ID_TO_NAME[molecule.value.lower()].upper()}.npy",
                )
            )

            cluster_coords = np.load(cluster_centers_path)
            cluster_assignments, transformed_samples = discretize_trajectory(
                10 * torch.tensor(self.traj.xyz), self.tic_evaluator, cluster_coords
            )
            protein_name = PDB_ID_TO_NAME[molecule.value.lower()]
            start_cluster, end_cluster = CLUSTER_ENDPOINTS[protein_name]
            committor_probs = get_gt_committor_probs(
                protein_name,
                transformed_samples,
                start_cluster,
                end_cluster,
                cluster_assignments,
                self.tic_evaluator,
            )

            old_len = len(self.traj.xyz)

            remove_idxs = np.where(
                np.logical_and(
                    committor_probs > committor_remove_range[0],
                    committor_probs < committor_remove_range[1],
                )
            )[0]
            remove_idxs = np.random.choice(
                remove_idxs, size=int(len(remove_idxs) * remove_freq), replace=False
            )
            self.traj.xyz = np.delete(self.traj.xyz, remove_idxs, axis=0)
            new_len = len(self.traj.xyz)
            print(
                f"Removed {old_len - new_len} ({(old_len - new_len) / old_len * 100} %) out of {old_len} frames from the dataset"
            )


class AtlasDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        name,
    ):
        self.data_root = data_root
        self.name = name

        # assumes data is preprocessed and stored in npz files (by AlphaFlow repo)
        path = os.path.join(data_root, f"{name}.npz")
        xyz = dict(np.load(path, allow_pickle=True))["all_atom_positions"]
        # extract only Alpha Carbons
        xyz = np.expand_dims(xyz[:, :, 1], 0)
        self.traj = DummyClass(
            xyz=torch.tensor(xyz)
        )  # shape is (1, n_frames, n_residues, 3)

    def __len__(self):
        return len(self.traj.xyz)

    def __getitem__(self, idx):
        x = self.traj.xyz[idx]
        return x


class MDGenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        suffix,
        split,
        overfit=False,
        overfit_peptide=None,
        atlas=False,
        repeat=1,
        atom_selection="all-atom",
        train=True,
    ):
        super().__init__()

        self.df = pd.read_csv(split, index_col="name")
        self.repeat = repeat
        if atom_selection == "c-alpha":
            self.num_beads = 4
        elif atom_selection == "backbone":
            self.num_beads = 12
        else:
            self.num_beads = 56  # maximum number of atoms in tetrapeptide with 14-atom representation ( 14 *4 = 56)
        self.bead_onehot = torch.eye(self.num_beads)
        self.data_dir = data_dir
        self.suffix = suffix
        self.overfit = overfit
        self.overfit_peptide = overfit_peptide
        self.atom_selection = atom_selection
        self.train = train

        # remove proteins for which we don't have any data (not sure why we couldn't download these)
        new_index = deepcopy(self.df.index)
        for protein in self.df.index:
            if not any([protein in f for f in os.listdir(data_dir)]):
                new_index = new_index.drop(protein)
        self.df = self.df.loc[new_index]
        self.atlas = atlas

    def __len__(self):
        if self.overfit_peptide:
            return 1000
        return 10000 * len(self.df) if self.train else 1000 * len(self.df)

    def __getitem__(self, idx):
        if self.train:
            protein = idx // 10000
            t_idx = idx % 10000
        else:
            protein = idx // 1000
            t_idx = (idx % 1000) * 10

        if self.overfit:
            idx = 0

        if self.overfit_peptide is None:
            name = self.df.index[protein]
            seqres = self.df.seqres[name]
        else:
            name = self.overfit_peptide
            seqres = name

        if self.atlas:
            i = np.random.randint(1, 4)
            full_name = f"{name}_R{i}"
        else:
            full_name = name

        arr = np.lib.format.open_memmap(
            f"{self.data_dir}/{full_name}{self.suffix}.npy", "r"
        )

        # arr should be in ANGSTROMS
        frame = torch.tensor(arr[t_idx], dtype=torch.float32)

        if self.atom_selection == "c-alpha":
            frame = frame[:, 1]
            seqres = np.array([restype_order[c] for c in seqres])
            atom_types = torch.from_numpy(seqres)  # Amino acid types

        elif self.atom_selection == "backbone":
            frame = frame[:, 0:3]
            seqres = np.array([restype_order[c] for c in seqres])
            atom_types = 2 * torch.from_numpy(seqres).repeat_interleave(
                3
            )  # Amino acid types
            atom_types[::3] += 1  # distinguish between N and C backbone atoms

        else:  # all-atom
            atom_names = [
                rc.restype_name_to_atom14_names[rc.aa_one_to_three_letter[c]]
                for c in seqres
            ]
            # find num atoms per residue that are not ''
            num_atoms_per_residue = [
                len([a for a in atom if a != ""]) for atom in atom_names
            ]
            atom_names = list(chain.from_iterable(atom_names))

            # remove '' elements from list
            atom_names = [x[0] if x else "" for x in atom_names]
            atom_types = torch.tensor(
                [atomic_numbers[a] if a else 0 for a in atom_names]
            ).long()

        frame = frame.reshape(-1, 3)
        return frame, atom_types
