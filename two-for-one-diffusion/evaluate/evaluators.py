import mdtraj as md
import numpy as np
import torch
import os
from datasets.dataset_utils_empty import AtomSelection
from evaluate.evaluators_CGflowmatching import (
    mse as mse_CGFM,
    get_prob,
    get_torsions,
    plot_free_E_2d,
    kl_div,
    K_BT_IN_KCAL_PER_MOL,
)
from utils import center_zero
import matplotlib.pyplot as plt

from deeptime.decomposition import TICA
import pyemma.coordinates as coor

from matplotlib.colors import LogNorm, Normalize
import matplotlib.path as mpath
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
import types
import json
import math
from datasets.dataset_utils_empty import (
    get_dataset,
    Molecules,
    AtlasProteins,
    mae_to_pdb_atom_mapping,
)
import pickle

# OM Paper plotting stuff
import scienceplots
import matplotlib.pylab as pylab

params = {
    "figure.dpi": 600,
    "axes.labelsize": "small",
    "legend.fontsize": "x-small",
    "axes.titlesize": "medium",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "font.family": "DejaVu Sans",
}
from matplotlib import rc

pylab.rcParams.update(params)


CLUSTER_CENTERS = {
    "chignolin": np.array(
        [[0.69400153, -0.34598462], [-0.48732213, 0.00642035], [1.87483537, 0.06285344]]
    ),
    "trp_cage": np.array(
        [[-2.15921372, 0.0062795], [0.47752285, -0.38050238], [0.40182245, 2.0690773]]
    ),
    "bba": np.array(
        [
            [-0.5756589, -0.60663654],
            [1.7861676, -0.87717611],
            [0.91295128, 1.07518898],
            [-0.49210152, 0.40313689],
        ]
    ),
    "villin": np.array(
        [
            [1.08971813, -0.98522752],
            [-2.49001353, -2.31375028],
            [-0.12929561, 0.53703407],
        ]
    ),
}


class Evaluator:
    """
    Evaluator used in training and in main eval.
    Only considers dihedral free energy and pairwise distances.

    Arguments:
    val_data: validation data
    topology: pdb file with topology of this molecule
    mol_name: name of molecule
    eval_folder: root directory to save results
    folded_pdb_folder: directory where to find the pdb files for folded protein structures
    """

    def __init__(
        self,
        ref_data,
        topology,
        mol_name="alanine",
        eval_folder=None,
        folded_pdb_folder="./datasets/folded_pdbs",
        data_folder="./data",
        evalsetname="",
        atom_selection=AtomSelection.A_CARBON,
    ):
        self.ref_data = ref_data[:][0]
        self.topology = topology
        self.eval_folder = eval_folder
        self.folded_pdb_folder = folded_pdb_folder
        self.mol_name = mol_name
        self.atom_selection = atom_selection

        # Dihedral energies evaluator
        if "alanine" in mol_name:
            # Initialize dihedral energies evaluator (ramachandran plot)
            self.dihedral_evaluator = DihedralEnergiesEvaluator(
                self.ref_data, topology, self.eval_folder
            )
        elif "protein_g" != mol_name.lower():
            # Initialize tic evaluator

            self.tic = TicEvaluator(
                self.ref_data,
                mol_name,
                eval_folder=self.eval_folder,
                data_folder=data_folder,
                folded_pdb_folder=folded_pdb_folder,
                evalset=evalsetname,
                atom_selection=atom_selection,
            )
        if "protein_g" != mol_name.lower():
            # Pairwise distance evaluator
            self.pwd_evaluator = PwdEvaluator(
                self.ref_data, self.eval_folder, mol_name, evalset=evalsetname
            )

    def eval(self, sampled_mol, milestone, save_plots=False):
        """
        Saves and evaluates generated samples. Milestone specifies which model checkpoint was used.
        """
        # Compute metrics
        dict_results = {}

        # Dihedral results
        if "alanine" in self.mol_name:
            print(f"Dihedral analysis {milestone}")
            _, dihedral_js, _, _ = self.dihedral_evaluator.eval(
                sampled_mol, save_plots, milestone
            )
            dict_results["Dihedral JS"] = dihedral_js
        elif "protein_g" != self.mol_name.lower():
            print(f"TIC analysis {milestone}")
            dict_results["TIC JS"] = self.tic.eval(
                sampled_mol, title=f"tic_{milestone}", plot_tic=save_plots
            )[0]

        if "protein_g" != self.mol_name.lower():
            # PWD Results
            if self.atom_selection == AtomSelection.A_CARBON:
                print(f"PWD Analysis {milestone}")
                dict_results["PWD JS"] = self.pwd_evaluator.eval(sampled_mol)

        # Print metrics
        for key in dict_results:
            print(key + f": {dict_results[key]:.4f}")

        with open(self.eval_folder + f"/results-{milestone}.json", "w") as f:
            json.dump(dict_results, f)
        print("Evaluation done \n")
        return dict_results


class DihedralEnergiesEvaluator:
    """
    Evaluator for dihedral free energy.

    Arguments:
    val_data: validation data
    topology: pdb file with topology of this molecule
    plots_folder: folder to save the results
    """

    def __init__(
        self,
        val_data,
        topology,
        plots_folder=None,
        n_bins=61,
        saved_ref="./evaluate/saved_references/saved_dih_probs_ala2_testset.pickle",
    ):
        self.topology = topology
        self.plots_folder = plots_folder
        self.n_bins = n_bins
        if os.path.exists(saved_ref):
            with open(saved_ref, "rb") as f:
                self.gt_probs = pickle.load(f)
        else:
            gt_traj = val_data.numpy()
            t0_dihe = get_torsions(gt_traj, topology)
            self.gt_probs = get_prob(t0_dihe, n_bins=self.n_bins)
            with open(saved_ref, "wb") as f:
                pickle.dump(self.gt_probs, f)

    def eval(
        self,
        all_mol,
        plot_freeE=False,
        milestone=0,
        plot_title="Ramachandran plot",
        save_plot=True,
    ):
        """
        Evaluates samples in terms of dihedral free energy.
        Plotting optional.
        """
        t0_dihe = get_torsions(all_mol.numpy(), self.topology)
        probs = get_prob(t0_dihe, n_bins=self.n_bins)
        dihedral_mse = mse_CGFM(probs, self.gt_probs)
        dihedral_js = js_divergence(probs, self.gt_probs)
        kl_1 = kl_div(probs, self.gt_probs)
        kl_2 = kl_div(self.gt_probs, probs)
        if plot_freeE:
            self._plot_freeE_2d(
                probs,
                file_name=self.plots_folder + f"/ramachandran_sampled_{milestone}.png",
                plot_title=plot_title,
                save_plot=save_plot,
            )
            self._plot_freeE_2d(
                self.gt_probs,
                file_name=self.plots_folder + "/ramachandran_valid.png",
                plot_title=plot_title,
                save_plot=save_plot,
            )
        return dihedral_mse, dihedral_js, kl_1, kl_2

    def _plot_freeE_2d(self, probs, file_name, plot_title="", save_plot=True):
        """
        Plot dihedral free energy and save plot.
        """
        plt.rcParams.update({"font.size": 15})
        _, ax = plt.subplots()
        plot_free_E_2d(
            probs,
            ax,
            unit_conv=K_BT_IN_KCAL_PER_MOL,
            cax=None,
            title=plot_title,
            n_bins=self.n_bins,
        )
        plt.xticks([-math.pi, 0, math.pi], ["-π", "0", "π"])
        plt.yticks([-math.pi, 0, math.pi], ["-π", "0", "π"])
        plt.xlabel("ϕ")
        plt.ylabel("ψ")
        if save_plot:
            plt.savefig(file_name)
        plt.show()
        plt.close()


class PwdEvaluator:
    """
    Evaluator for pairwise distances.

    Arguments:
    val_data: validation data
    plots_folder: where to save histogram plot (only for ala2)
    mol_name: name of molecule
    offset: diagonal offset to calculate pairwise distance Jensen-Shannon
    saved_ref: path to saved reference
    evalset: which set to use for evaluation (testset or valset)
    """

    def __init__(
        self,
        val_data,
        plots_folder="",
        mol_name="",
        offset=0,
        saved_ref="none",
        evalset="testset",
    ):
        self.offset = offset
        self.plots_folder = plots_folder
        self.mol_name = mol_name.lower()
        self.resolution = 0.1

        if saved_ref == "none":
            saved_ref = f"./evaluate/saved_references/saved_pwd_{mol_name.upper()}_{evalset}_offset_{self.offset}.pickle"

        if os.path.exists(saved_ref):
            with open(saved_ref, "rb") as f:
                data = pickle.load(f)
                self.gt_max = data["gt_max"]
                self.gt_hist = data["gt_hist"]
        else:
            self.gt_pwd_triu = get_pwd_triu_batch(val_data, self.offset)
            self.gt_max = self.gt_pwd_triu.max(dim=0)[0]
            self.gt_hist = []
            for pwd, m in zip(self.gt_pwd_triu.t(), self.gt_max):
                nbins = int(torch.div(m, self.resolution, rounding_mode="floor") + 1)
                self.gt_hist.append(
                    torch.histc(
                        pwd, bins=int(nbins), min=0, max=self.resolution * nbins
                    )
                )
            with open(saved_ref, "wb") as f:
                pickle.dump({"gt_max": self.gt_max, "gt_hist": self.gt_hist}, f)

    def js_divergence_pwd(self, hist_gt, pwd_sampled, gt_max, resolution):
        """
        Calculate Jensen-Shannon divergence between pairwise distance histograms,
        one from the ground truth and one calculated from sampled coordinates.
        """

        result_js = np.empty(len(hist_gt))
        for i, (hgt, pwd, gtm) in enumerate(zip(hist_gt, pwd_sampled.t(), gt_max)):
            maxval = max(gtm, pwd.max())
            nbins = int(torch.div(maxval, resolution, rounding_mode="floor") + 1)
            hist_sampled = torch.histc(
                pwd, bins=int(nbins), min=0, max=resolution * nbins
            )

            if nbins > len(hgt):
                hgt = torch.cat((hgt, torch.zeros(nbins - len(hgt))))

            result_js[i] = js_divergence(hgt.numpy(), hist_sampled.numpy())

        return result_js.mean()

    def eval(self, all_mol, plot_pwds=False, milestone=0):
        """
        Calculate Jensen-Shannon divergence for all pairwise distance distributions.
        Plotting optional (only ala2).
        """
        pwd_sampled = get_pwd_triu_batch(all_mol, self.offset)
        pwd_js = self.js_divergence_pwd(
            self.gt_hist, pwd_sampled, self.gt_max, self.resolution
        )
        if plot_pwds:
            self._plot_pwds(
                pwd_sampled,
                file_name=self.plots_folder
                + f"/PWDS_{self.mol_name}_DM_{milestone}.png",
            )
        return pwd_js

    def _plot_pwds(self, pwd_sampled, file_name, save_plot=True):
        """
        Pairwise distance histogram plot (only ala2), ground truth versus sampled.
        """
        assert self.gt_pwd_triu.shape[-1] == pwd_sampled.shape[-1], "Shape mismatch"
        assert self.offset == 1, "Offset needs to be set to 1 for this plot"
        c1 = "tab:green"
        c2 = "tab:orange"
        c1_patch = mpatches.Patch(color=c1, label="Ground truth")
        c2_patch = mpatches.Patch(color=c2, label="Sampled")

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 4))
        axes = axes.flatten()
        for i in range(self.gt_pwd_triu.shape[-1]):
            axes[i].hist(
                self.gt_pwd_triu[:, i],
                bins=20,
                density=True,
                color=c1,
                alpha=0.5,
                edgecolor=c1,
            )
            axes[i].hist(
                pwd_sampled[:, i],
                bins=20,
                density=True,
                color=c2,
                alpha=0.5,
                edgecolor=c2,
            )
            axes[i].set_title(f"{i+1}", fontsize=14)

        ax0 = fig.add_subplot(111, frameon=False)
        ax0.set_xlabel("Pairwise distance (Å)", labelpad=20, fontsize=12)
        ax0.set_ylabel("Density", labelpad=20, fontsize=12)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.legend(
            handles=[c1_patch, c2_patch],
            loc="lower center",
            ncol=2,
            borderaxespad=-6,
            fontsize=12,
        )
        plt.tight_layout()
        if save_plot:
            plt.savefig(file_name)
        plt.show()
        plt.close()


class TicEvaluator:
    """
    Evaluator for time-lagged independent component analysis (TICA).
    We're interested in the two slowest modes.

    Arguments:
        trn_data: train data
        val_data: validation data
        folded_pdb: pdb file of folded structure
        mol_name: molecule name
        eval_folder: folder to save results
    """

    def __init__(
        self,
        val_data,
        mol_name,
        eval_folder,
        data_folder,
        atom_selection=AtomSelection.A_CARBON,
        folded_pdb_folder="../datasets/folded_pdbs",
        bins=101,
        lagtime=100,
        saved_ref="none",
        evalset="testset",
        gt_traj=None,
    ):
        self.mol_name = mol_name
        self.plots_folder = eval_folder
        self.bins = bins
        self.atom_selection = atom_selection
        if mol_name.upper() in Molecules.__members__:
            protid = Molecules[mol_name.upper()].value
        else:
            protid = AtlasProteins[mol_name.upper()].value

        folded_pdb = f"{folded_pdb_folder}/{protid}-0-{atom_selection.value}.pdb"
        if self.atom_selection == AtomSelection.A_CARBON:
            # Only keep alpha carbons
            self.folded = process_pdb(folded_pdb, mol_name)
        elif self.atom_selection == AtomSelection.PROTEIN:
            self.folded = md.load(folded_pdb).remove_solvent()
            self.feat = coor.featurizer(folded_pdb)
            self.feat.add_backbone_torsions(cossin=True)
            self.feat.add_sidechain_torsions(cossin=True)
            self.mae_to_pdb_mapping = mae_to_pdb_atom_mapping(mol_name)

        # Check if the computed objects are already saved
        if saved_ref == "none":
            if atom_selection == AtomSelection.A_CARBON:
                saved_ref = f"./evaluate/saved_references/saved_TICA_{mol_name.upper()}_{evalset}.pickle"
            elif atom_selection == AtomSelection.PROTEIN:
                saved_ref = f"./evaluate/saved_references/saved_TICA_{mol_name.upper()}_all_atom_{evalset}.pickle"
        if os.path.exists(saved_ref):
            # Load the saved objects
            with open(saved_ref, "rb") as f:
                (
                    self.tica,
                    self.gt_prob,
                    self.bin_edges_x,
                    self.bin_edges_y,
                ) = pickle.load(f)
        else:
            if gt_traj is not None:
                sorted_data_xyz = 10 * center_zero(gt_traj)
            else:
                gt_traj_path = os.path.join(
                    data_folder,
                    protid,
                    (
                        "gt_traj.pt"
                        if self.atom_selection == AtomSelection.A_CARBON
                        else "gt_traj_all_atom.pt"
                    ),
                )
                if os.path.exists(gt_traj_path):
                    # load the existing dataset
                    sorted_data_xyz = 10 * center_zero(torch.load(gt_traj_path))
                else:
                    # load the dataset from the raw MD trajectories
                    trainset_sorted, valset_sorted, testset_sorted = get_dataset(
                        mol_name,
                        mean0=True,
                        data_folder=data_folder,
                        pdb_folder=folded_pdb_folder,
                        fold=None,
                        atom_selection=atom_selection,
                        traindata_subset=None,
                        shuffle_before_splitting=False,
                    )

                    sorted_data_xyz = torch.cat(
                        (
                            trainset_sorted[:][0],
                            valset_sorted[:][0],
                            testset_sorted[:][0],
                        ),
                        dim=0,
                    )
                    torch.save(sorted_data_xyz / 10, gt_traj_path)

            print("Computing TIC features")
            tic_features = self.get_tic_features(sorted_data_xyz, self.folded)

            # We compute the TIC eigenvalues on the training and validation partitions both together
            # to be consistent with previous works.
            print("Fitting TICA")
            self.tica = TICA(lagtime=lagtime, dim=2)
            _ = self.tica.fit_transform(tic_features)

            if val_data is not None:
                # We then compute the features only on the val partition that we will use for evaluation.
                tic_features = self.get_tic_features(val_data, self.folded)

            transformed_data = self.tica(tic_features)

            self.gt_prob, self.bin_edges_x, self.bin_edges_y = np.histogram2d(
                transformed_data[:, 0],
                transformed_data[:, 1],
                bins=self.bins,
                density=True,
            )
            # Save the computed TICA objects
            if not os.path.exists(os.path.dirname(saved_ref)):
                os.makedirs(os.path.dirname(saved_ref))
            with open(saved_ref, "wb") as f:
                pickle.dump(
                    (self.tica, self.gt_prob, self.bin_edges_x, self.bin_edges_y), f
                )
            self.bin_mids_x = (self.bin_edges_x[1:] + self.bin_edges_x[:-1]) / 2
            self.bin_mids_y = (self.bin_edges_y[1:] + self.bin_edges_y[:-1]) / 2

        self.bin_mids_x = (self.bin_edges_x[1:] + self.bin_edges_x[:-1]) / 2
        self.bin_mids_y = (self.bin_edges_y[1:] + self.bin_edges_y[:-1]) / 2

        if "protein_g" != mol_name.lower():
            self.folded_transform = self.tica.transform(
                self.get_tic_features(
                    torch.from_numpy(self.folded.xyz) * 10,
                    self.folded,
                    convert_to_pdb_ordering=False,
                )
            )[0]

            self.bin_x_folded = np.argmin(
                abs(self.bin_mids_x - self.folded_transform[0])
            )
            self.bin_y_folded = np.argmin(
                abs(self.bin_mids_y - self.folded_transform[1])
            )

    def get_tic_features(
        self, xyz, folded, separate=False, convert_to_pdb_ordering=True
    ):
        """
        Calculate features for TIC analysis.
        For A_CARBON, we calculate dihedrals and pairwise distances.
        For PROTEIN, we calculate backbone and sidechain torsions.
        """

        traj = md.Trajectory(
            (xyz.numpy() if isinstance(xyz, torch.Tensor) else xyz) / 10,
            topology=folded.topology,
        )

        if self.atom_selection == AtomSelection.A_CARBON:
            # backbone dihedrals and pairwise distances
            ind = np.arange(0, xyz.shape[1] - 3)
            ind = np.stack((ind, ind + 1, ind + 2, ind + 3)).T
            dihedrals = md.compute_dihedrals(traj, ind)
            pwds = get_pwd_triu_batch(xyz).numpy()
            if separate:
                return dihedrals, pwds
            return np.hstack((dihedrals, pwds))
        elif self.atom_selection == AtomSelection.PROTEIN:
            if convert_to_pdb_ordering:
                traj.xyz = traj.xyz[:, self.mae_to_pdb_mapping]
            # backbone and sidechain torsions
            feat = self.feat.transform(traj)
            return feat

    def eval(
        self,
        xyz_samples,
        title,
        plot_tic=True,
        save_object=False,
        path=None,
        cmap="OrRd",
        gradient=True,
        steps=3,
        linewidth=2,
    ):
        """
        Evaluate TIC. Jensen-Shannon divergence with ground truth and optional plot.
        """
        sample_tic_features = self.get_tic_features(xyz_samples, self.folded)
        transformed_samples = self.tica(sample_tic_features)

        prob_samp, _, _ = np.histogram2d(
            transformed_samples[:, 0],
            transformed_samples[:, 1],
            bins=[self.bin_edges_x, self.bin_edges_y],
            density=True,
        )

        tic_js = js_divergence(self.gt_prob.flatten(), prob_samp.flatten())

        if save_object:
            with open("prob_samp.npy", "wb") as f:
                np.save(f, prob_samp)
            with open("bin_mids_x.npy", "wb") as f:
                np.save(f, self.bin_mids_x)
            with open("bin_mids_y.npy", "wb") as f:
                np.save(f, self.bin_mids_y)

        file_name = (
            f"TICA_{self.mol_name}_{title}.png"
            if path is None
            else f"TICA_{self.mol_name}_{title}_path.png"
        )
        file_name = os.path.join(self.plots_folder, file_name)
        if plot_tic:
            fig = self._plot_tic(
                prob_samp,
                title,
                file_name,
                path,
                cmap,
                gradient,
                steps,
                linewidth,
            )

        return tic_js, fig

    def tic_to_xyz(self, query_tic_coords, xyz_samples):
        """
        Convert TIC coordinates back to Cartesian coordinates using the TIC evaluator.
        """
        sample_tic_features = self.get_tic_features(xyz_samples, self.folded)
        transformed_samples = self.tica(sample_tic_features)
        # Find the closest point in the transformed samples
        dists = np.linalg.norm(
            transformed_samples[None] - query_tic_coords[:, None], axis=-1
        )
        closest_idx = np.argmin(dists, axis=-1)
        return xyz_samples[closest_idx]

    def _plot_tic(
        self,
        probs,
        title,
        file_name,
        path=None,
        cmap="OrRd",
        gradient=True,
        steps=3,
        linewidth=2,
        save_plot=True,
        endpoints=None,
        gen_paths=None,
        ref_paths=None,
    ):
        """
        Plot slowest two TIC components versus each other. Mostly based on CG flow matching plotting code.
        """
        fig, (ax1, ax2) = plt.subplots(
            1, 2, dpi=150, gridspec_kw={"width_ratios": [24, 1]}
        )
        ax1.imshow(probs.T, norm=LogNorm(vmax=10, vmin=1e-4), origin="lower", zorder=1)
        ax1.set_xticks(
            range(len(self.bin_mids_x))[5::15],
            [f"{num:.02f}" for num in self.bin_mids_x[5::15]],
        )
        ax1.set_yticks(
            range(len(self.bin_mids_y))[5::15],
            [f"{num:.02f}" for num in self.bin_mids_y[5::15]],
        )
        ax = plt.gca()  # Get the current axis
        ax.set_facecolor("gray")  # Set the background color to gray

        if path is not None:
            edges_x = self.bin_edges_x[0], self.bin_edges_x[-1]
            edges_y = self.bin_edges_y[0], self.bin_edges_y[-1]
            xrange = edges_x[1] - edges_x[0]
            yrange = edges_y[1] - edges_y[0]

            axlimx = ax1.get_xlim()
            axlimy = ax1.get_ylim()
            axxrange = axlimx[1] - axlimx[0]
            axyrange = axlimy[1] - axlimy[0]

            xfactor = axxrange / xrange
            yfactor = axyrange / yrange

            plotx = (path[:, 0] - edges_x[0]) * xfactor
            ploty = (path[:, 1] - edges_y[0]) * yfactor

            ax1.plot(plotx, ploty, color="orange", linewidth=linewidth, zorder=2)

            if gradient:

                class GC(GraphicsContextBase):
                    def __init__(self):
                        super().__init__()
                        self._capstyle = "round"

                def custom_new_gc(self):
                    return GC()

                RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)
                path = mpath.Path(np.column_stack([plotx, ploty]))
                verts = path.interpolated(steps=steps).vertices
                plotx, ploty = verts[:, 0], verts[:, 1]
                segments = np.array(
                    [plotx[:-1], ploty[:-1], plotx[1:], ploty[1:]]
                ).T.reshape(-1, 2, 2)
                norm = plt.Normalize(0, len(plotx))
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(range(len(plotx)))
                lc.set_linewidth(linewidth)
                ax1.get_lines()[0].remove()
                ax1.add_collection(lc)

        # plot folded structure
        # ax1.scatter(
        #     self.bin_x_folded,
        #     self.bin_y_folded,
        #     marker="X",
        #     c="firebrick",
        #     s=200,
        #     linewidth=0,
        #     zorder=3,
        # )

        # plot path endpoints used in interpolation
        if endpoints is not None:
            # reshape from [2, P, N, 3] to [2*P, N, 3]
            if endpoints.shape[-1] == 3:
                endpoints = endpoints.reshape(
                    -1, endpoints.shape[-2], endpoints.shape[-1]
                )
            elif endpoints.shape[-1] == 2:
                endpoints = endpoints.reshape(-1, 2)
            for i, point in enumerate(endpoints):
                if point.shape[-1] == 3:  # in case the point is in xyz space
                    endpoint_transform = self.tica.transform(
                        self.get_tic_features(
                            torch.from_numpy(point[None, :, :]), self.folded
                        )
                    )[0]
                elif point.shape[-1] == 2:  # in case the point is already in tic space
                    endpoint_transform = point

                bin_x_endpoint = np.argmin(abs(self.bin_mids_x - endpoint_transform[0]))
                bin_y_endpoint = np.argmin(abs(self.bin_mids_y - endpoint_transform[1]))
                ax1.scatter(
                    bin_x_endpoint,
                    bin_y_endpoint,
                    marker="X",
                    c="blue" if i % 2 == 0 else "red",
                    s=150,
                    linewidth=0,
                    zorder=3,
                )

        if ref_paths is not None:
            # plot a path connecting each of the points in ref_path
            for p, ref_path in enumerate(ref_paths):
                for i in range(len(ref_path) - 1):
                    start = ref_path[i]
                    end = ref_path[i + 1]
                    start_x = np.argmin(abs(self.bin_mids_x - start[0]))
                    start_y = np.argmin(abs(self.bin_mids_y - start[1]))
                    end_x = np.argmin(abs(self.bin_mids_x - end[0]))
                    end_y = np.argmin(abs(self.bin_mids_y - end[1]))
                    ax1.plot(
                        [start_x, end_x],
                        [start_y, end_y],
                        color="orange",
                        linewidth=2,
                        zorder=2,
                    )

        if gen_paths is not None:
            # plot a path connecting each of the points in gen_path
            for p, gen_path in enumerate(gen_paths):
                for i in range(len(gen_path) - 1):
                    start = gen_path[i]
                    end = gen_path[i + 1]
                    start_x = np.argmin(abs(self.bin_mids_x - start[0]))
                    start_y = np.argmin(abs(self.bin_mids_y - start[1]))
                    end_x = np.argmin(abs(self.bin_mids_x - end[0]))
                    end_y = np.argmin(abs(self.bin_mids_y - end[1]))
                    ax1.plot(
                        [start_x, end_x],
                        [start_y, end_y],
                        color="red",
                        linewidth=1,
                        zorder=2,
                    )

        ax1.set_xlabel("TIC 0", labelpad=10, size=12)
        ax1.set_ylabel("TIC 1", labelpad=10, size=12)
        ax1.set_title(title, fontsize=14, pad=10)
        ax1.axis("off")

        cmap = plt.cm.viridis_r
        norm = Normalize(vmin=0, vmax=10)
        bounds = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        cb1 = ColorbarBase(
            ax2,
            cmap=cmap,
            norm=norm,
            boundaries=[0] + bounds + [15],
            extend="max",
            extendfrac=0.1,
            ticks=bounds,
            spacing="uniform",
            orientation="vertical",
        )
        cb1.set_label("Free energy / $k_BT$", labelpad=-1, fontsize=16)
        plt.tight_layout()
        if save_plot:
            plt.savefig(file_name)

        plt.close()


class RmsdEvaluator:
    """
    Evaluator for RMSD with folded structure.

    Arguments:
    mol_name: molecule name
    folded_pdb: pdb file corresponding to flded structure
    eval_folder: folder to save results
    """

    def __init__(self, mol_name, folded_pdb, eval_folder):
        self.plots_folder = eval_folder
        self.folded = process_pdb(folded_pdb, mol_name)
        self.plot_dict = {}
        self.mol_name = mol_name

        # Parameters from the reference data plot
        self.saved_ref = f"e/saved_references/saved_rmsd_{self.mol_name.upper()}_reference_total.pickle"
        self.cutoff_dict_ref = {
            "chignolin": 10,
            "trp_cage": 12,
            "bba": 14,
            "villin": 14,
            "protein_g": 20,
        }
        self.cutoff_ref = self.cutoff_dict_ref[mol_name.lower()]
        self.nbins_ref = 100

    def eval(self, method, xyz, nbins, cutoff=None, save_dynamics=False):
        """
        Evaluate RMSD to folded structure. Builds a dictionary to make it
        easy to plot into one figure later. RMSD cutoff (optional) determines
        where to terminate the x-axis. Argument save_dynamics can be set to
        true in case dynamics plots need to be made.
        xyz should be a torch tensor in angstrom.
        """

        # Load from pickle
        if method == "Reference" and os.path.exists(self.saved_ref):
            assert (
                nbins == self.nbins_ref and cutoff == self.cutoff_ref == cutoff
            ), f"Reference data only exists for nbins={self.bins_ref} and cutoff={self.cutoff_ref}"
            with open(self.saved_ref, "rb") as f:
                data = pickle.load(f)
                self.plot_dict[method] = data
        else:
            self.plot_dict[method] = {}

            valid_mask = np.all(np.all(np.isfinite(xyz.numpy()), -1), -1)
            traj = md.Trajectory(
                xyz.numpy()[valid_mask] / 10, topology=self.folded.topology
            )

            rmsd = np.full(len(xyz), np.nan)
            rmsd[valid_mask] = md.rmsd(traj, self.folded) * 10

            if save_dynamics:
                self.plot_dict[method]["rmsd"] = rmsd

            if cutoff is None:
                cutoff = rmsd[~np.isnan(rmsd)].max()

            h, bin_edges = np.histogram(
                rmsd, bins=nbins, range=[0, cutoff], density=True
            )
            self.plot_dict[method]["bin_mids"] = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            with np.errstate(divide="ignore"):
                self.plot_dict[method]["energies"] = -np.log(h)
            # Save pickle
            if method == "Reference":
                with open(self.saved_ref, "wb") as f:
                    pickle.dump(self.plot_dict[method], f)

    def _plot_rmsd(
        self,
        save=True,
        colors=None,
        linestyles=None,
        legend_bool=True,
        font_size=10,
        linewidth=None,
    ):
        """
        Plot RMSD between sampled and folded structure from histogram dictionary
        """
        for i, (method, method_dict) in enumerate(self.plot_dict.items()):
            color = None if colors is None else colors[i]
            linestyle = None if linestyles is None else linestyles[i]
            plt.plot(
                method_dict["bin_mids"],
                method_dict["energies"],
                label=method,
                c=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        plt.tick_params(axis="both", labelsize=font_size)
        plt.xlabel(r"$C_{\alpha}$ RMSD to folded (Å)")
        plt.ylabel(r"Free energy / $k_BT$")
        if legend_bool:
            plt.legend(prop={"size": font_size})
        if save:
            plt.savefig(
                os.path.join(self.plots_folder, f"RMSD_{self.mol_name}_free_energy.png")
            )

    def _plot_rmsd_dynamics(
        self, method, start, stop, time_step, save=True, figsize=(1, 5)
    ):
        """
        Plot RMSD over time, start and stop in frames, time_step in ns
        """
        plt.figure(figsize=figsize)
        rmsd = self.plot_dict[method]["rmsd"]
        plt.plot(rmsd[start:stop], (np.arange(len(rmsd)) * time_step)[start:stop])
        plt.xlabel(r"$C_{\alpha}$ RMSD to folded (Å)")
        plt.ylabel("CG simulation time (ns)")
        if save:
            plt.savefig(
                os.path.join(
                    self.plots_folder, f"RMSD_{self.mol_name}_dynamics_{method}.png"
                )
            )

        return rmsd[~np.isnan(rmsd)].mean()


class ContactEvaluator:
    """
    Evaluator for contacts (i.e. places in the protein where the pairwise distance is
    smaller than a certain cutoff of 6-12 Å).

    Arguments:
    mol_name: molecule name
    folded_pdb: pdb file corresponding to flded structure
    eval_folder: folder to save results
    contact_cutoff: threshold for when a pairwise distance is counted as a contact
    """

    def __init__(self, mol_name, folded_pdb, eval_folder, contact_cutoff=10):
        self.mol_name = mol_name
        self.contact_cutoff = contact_cutoff
        self.plots_folder = eval_folder
        self.folded = torch.from_numpy(process_pdb(folded_pdb, mol_name).xyz[0]) * 10
        self.pwd_folded = torch.norm(
            self.folded[:, None, :] - self.folded[None, :, :], dim=-1
        )
        self.contacts_folded = (
            self.pwd_folded < self.contact_cutoff
        )  # Wikipedia: 6-12 Å

    def _plot_pwd_contacts_gt(self, save=True):
        """
        Plot the ground truth pairwise distance matrix and contact map corresponding to
        the chosen cutoff.
        """
        _, (ax1, ax2) = plt.subplots(
            1, 2, dpi=150, gridspec_kw={"width_ratios": [24, 19]}, figsize=(8, 4)
        )

        pl1 = ax1.imshow(self.pwd_folded)
        ax1.set_xticks(np.arange(0, len(self.folded), 5))
        ax1.set_yticks(np.arange(0, len(self.folded), 5))
        ax1.set_title("Pairwise distance matrix folded", fontsize=12, y=1.02)
        cb = plt.colorbar(pl1, ax=ax1, shrink=0.788, format=lambda x, _: f"{x:.1f}")
        cb.set_label("Pairwise distance (Å)", fontsize=12)

        ax2.imshow(self.contacts_folded, cmap="binary", vmin=0, vmax=1)
        ax2.set_xticks(np.arange(0, len(self.folded), 5))
        ax2.set_yticks(np.arange(0, len(self.folded), 5))
        ax2.set_title("Contact map folded", fontsize=12, y=1.02)

        plt.tight_layout()
        if save:
            plt.savefig(self.plots_folder + f"pwd_contacts_{self.mol_name}_gt.png")

    def _get_samp_contacts(self, xyz_sampled):
        """
        Get contacts for sampled molecules
        """
        pwd_sampled = torch.norm(
            xyz_sampled[:, :, None, :] - xyz_sampled[:, None, :, :], dim=-1
        )
        contacts_samp = pwd_sampled < self.contact_cutoff
        return contacts_samp

    def _plot_contact_normcount(
        self, xyz_sampled, method, save=True, take_log=False, vmin_log=None
    ):
        """
        Make a contact map for each sample, and then take a normalized count across samples.
        The log is plotted to make the differences more clear.
        """
        contacts_samp = self._get_samp_contacts(xyz_sampled)

        norm_sum = contacts_samp.sum(dim=0) / len(contacts_samp)

        plt.figure(figsize=(6, 6))
        if take_log:
            norm_sum_log = torch.log(norm_sum)
            plt.imshow(norm_sum_log, cmap="viridis_r", vmin=vmin_log)
        else:
            plt.imshow(norm_sum, cmap="viridis_r", vmin=0, vmax=1)
        plt.xticks(np.arange(0, len(norm_sum), 5))
        plt.yticks(np.arange(0, len(norm_sum), 5))
        cb = plt.colorbar(format=lambda x, _: f"{x:.1f}", shrink=0.788)
        if take_log:
            cb.set_label("Log of normalized contact count", fontsize=12)
        else:
            cb.set_label("Normalized contact count", fontsize=12)
        plt.title(f"{method}", fontsize=12, y=1.02)

        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(
                    self.plots_folder, f"contact_normcount_{self.mol_name}_{method}.png"
                )
            )
        return torch.min(norm_sum_log)

    def _eval_bce_dynamics(
        self, xyz_sampled, method, start, stop, time_step, save=True, figsize=(1, 5)
    ):
        """
        Calculates binary cross entropy between ground truth and sampled contacts,
        plot over time, start and stop in frames, time_step in ns
        """
        contacts_samp = self._get_samp_contacts(xyz_sampled)

        triu_ind = torch.triu_indices(
            self.contacts_folded.shape[-2], self.contacts_folded.shape[-1], offset=3
        )
        contacts_samp_triu = contacts_samp[:, triu_ind[0], triu_ind[1]] * 1.0
        contacts_triu = self.contacts_folded[triu_ind[0], triu_ind[1]] * 1.0

        bce = torch.nn.functional.binary_cross_entropy(
            *torch.broadcast_tensors(contacts_samp_triu, contacts_triu),
            reduction="none",
        ).mean(dim=-1)

        plt.figure(figsize=figsize)
        plt.plot(bce[start:stop], (np.arange(len(bce)) * time_step)[start:stop])
        plt.xlabel("Contact BCE to folded")
        plt.ylabel("CG simulation time (ns)")
        if save:
            plt.savefig(
                self.plots_folder + f"contact_bce_{self.mol_name}_dynamics_{method}.png"
            )

        return bce.mean()


class PhysicalEvaluator:
    def __init__(self, reference_trajectory, tolerances=None):
        """
        Initializes the PhysicalEvaluator with a reference trajectory and tolerances.

        Parameters:
        - reference_trajectory: torch.Tensor of shape [n_frames, n_residues, 3] containing alpha carbons
        - tolerances: dict containing tolerance values for deviations
          {
              "bond_length": float,
              "bond_angle": float,
              "radius_of_gyration": float,
              "min_distance": float
          }
        """
        self.reference_trajectory = reference_trajectory
        if isinstance(self.reference_trajectory, np.ndarray):
            self.reference_trajectory = torch.tensor(self.reference_trajectory)
        self.tolerances = tolerances
        if self.tolerances is None:
            self.tolerances = {
                "bond_length": 0.5,  # (Angstroms) offset relative to reference min/max
                "bond_angle": 0.25,  # (radians) offset relative to reference min/max
                "radius_of_gyration": 5,  # (Angstroms) offset relative to reference min/max
                "min_distance": 0.5,  # (Angstroms) absolute
            }
        self.reference_metrics = self._compute_reference_metrics()

    def _compute_reference_metrics(self):
        """
        Compute reference minimum and maximum values for various metrics.

        Returns:
        - Dictionary of reference metrics.
        """
        metrics = {}

        # Bond lengths
        bond_lengths = torch.norm(
            self.reference_trajectory[:, 1:, :] - self.reference_trajectory[:, :-1, :],
            dim=-1,
        )
        metrics["bond_length_min"] = bond_lengths.min(dim=0)[0]
        metrics["bond_length_max"] = bond_lengths.max(dim=0)[0]

        # Bond angles
        vec1 = (
            self.reference_trajectory[:, 1:-1, :] - self.reference_trajectory[:, :-2, :]
        )
        vec2 = (
            self.reference_trajectory[:, 2:, :] - self.reference_trajectory[:, 1:-1, :]
        )
        cos_theta = torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)
        bond_angles = torch.acos(cos_theta)  # Radians
        metrics["bond_angle_min"] = bond_angles.min(dim=0)[0]
        metrics["bond_angle_max"] = bond_angles.max(dim=0)[0]

        # Radius of gyration
        rg = self._compute_radius_of_gyration(self.reference_trajectory)
        metrics["rg_min"] = rg.min().item()
        metrics["rg_max"] = rg.max().item()

        return metrics

    @staticmethod
    def _compute_radius_of_gyration(trajectory):
        center_of_mass = trajectory.mean(dim=1, keepdim=True)
        deviations = trajectory - center_of_mass
        rg = torch.sqrt((deviations**2).sum(dim=-1).mean(dim=-1))
        return rg

    def validate(self, new_trajectory):
        """
        Validate a new trajectory against the reference metrics and tolerances.

        Parameters:
        - new_trajectory: torch.Tensor of shape [n_frames, n_residues, 3]

        Returns:
        - Dictionary indicating which frames are unphysical for each check.
        """
        results = {}
        assert (
            len(new_trajectory.shape) == 3
            and new_trajectory.shape[1] == self.reference_trajectory.shape[1]
        )

        # Bond lengths

        bond_lengths = torch.norm(
            new_trajectory[:, 1:, :] - new_trajectory[:, :-1, :], dim=-1
        )
        bond_length_min = (
            self.reference_metrics["bond_length_min"] - self.tolerances["bond_length"]
        )
        bond_length_max = (
            self.reference_metrics["bond_length_max"] + self.tolerances["bond_length"]
        )
        results["bond_length_issues"] = (
            ((bond_lengths < bond_length_min) | (bond_lengths > bond_length_max))
            .any()
            .unsqueeze(0)
        )

        # Bond angles
        vec1 = new_trajectory[:, 1:-1, :] - new_trajectory[:, :-2, :]
        vec2 = new_trajectory[:, 2:, :] - new_trajectory[:, 1:-1, :]
        cos_theta = torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)
        bond_angles = torch.acos(cos_theta)  # Radians
        bond_angle_min = (
            self.reference_metrics["bond_angle_min"] - self.tolerances["bond_angle"]
        )
        bond_angle_max = (
            self.reference_metrics["bond_angle_max"] + self.tolerances["bond_angle"]
        )
        results["bond_angle_issues"] = (
            ((bond_angles < bond_angle_min) | (bond_angles > bond_angle_max))
            .any()
            .unsqueeze(0)
        )

        # Pairwise distances
        pwd = get_pwd_triu_batch(new_trajectory)
        results["distance_issues"] = (
            (pwd < self.tolerances["min_distance"]).any().unsqueeze(0)
        )  # Flag frames with clashes

        # Radius of gyration
        rg = self._compute_radius_of_gyration(new_trajectory)
        rg_min = (
            self.reference_metrics["rg_min"] - self.tolerances["radius_of_gyration"]
        )
        rg_max = (
            self.reference_metrics["rg_max"] + self.tolerances["radius_of_gyration"]
        )
        results["rg_issues"] = ((rg < rg_min) | (rg > rg_max)).any().unsqueeze(0)

        # Combine results
        unphysical_frames = (
            results["bond_length_issues"]
            | results["bond_angle_issues"]
            | results["distance_issues"]
            | results["rg_issues"]
        )
        results["unphysical_frames"] = unphysical_frames

        return results


def process_pdb(pdb_path, mol_name):
    """
    Take a fine-grained pdb file and slice out relevant atoms
    """
    folded = md.load(pdb_path).remove_solvent()
    ind_CA = np.array(
        [i for i, m in enumerate(folded.topology.atoms) if "CA" in str(m)]
    )
    # if mol_name.upper() == "PROTEIN_G":
    #     ind_CA = ind_CA[5:61]
    return folded.atom_slice(ind_CA)


def sample_from_model(sampler, num_saved_samples, batch_size, verbose=False, z=None):
    """
    Sample molecules from the model.
    """
    print(f"Generating {num_saved_samples} samples per GPU. This may take some time.")
    batches = num_to_groups(num_saved_samples, batch_size)
    all_mol_list = []
    for i, batch_size in enumerate(batches):
        if z is not None:
            if len(z.shape) == 1:
                z = z.unsqueeze(0)
            z = z.repeat(math.ceil(batch_size / z.shape[0]), 1)[:batch_size]
        all_mol_list.append(sampler(batch_size=batch_size, z=z))
        if verbose:
            print(f"Batch {i+1} from {len(batches)} generated")
    # all_mol_list = list(map(lambda n: model.sample(batch_size=n), batches))
    all_mol = torch.cat(all_mol_list, dim=0).cpu()
    print(f"{len(all_mol)} samples generated")
    return all_mol


def sample_interpolations_from_model(
    interpolator,
    endpoint_1_samples,
    endpoint_2_samples,
    batch_size,
    verbose=False,
    z=None,
):
    """
    Sample interpolations from the model.
    """
    num_paths = endpoint_1_samples.shape[0] // torch.cuda.device_count()
    print(
        f"Generating {num_paths} interpolation paths per GPU. This may take some time."
    )
    all_path_list = []
    path_optimization_list = []
    all_actions_list = []
    all_path_terms_list = []
    all_force_terms_list = []
    endpoint_1_split = endpoint_1_samples.split(batch_size)
    endpoint_2_split = endpoint_2_samples.split(batch_size)
    for i, (x1, x2) in enumerate(zip(endpoint_1_split, endpoint_2_split)):
        output = interpolator(x1, x2, z)
        all_path_list.append(output["final_path"])

        if "all_paths" in output.keys():
            path_optimization_list.append(output["all_paths"])
        if "actions" in output.keys():
            all_actions_list.append(output["actions"])
        if "path_terms" in output.keys():
            all_path_terms_list.append(output["path_terms"])
        if "force_terms" in output.keys():
            all_force_terms_list.append(output["force_terms"])

        if verbose:
            print(f"Batch {i+1} from {len(endpoint_1_split)} generated")
    # all_mol_list = list(map(lambda n: model.sample(batch_size=n), batches))
    all_path = torch.cat(all_path_list, dim=0).cpu()

    output = {"sampled_mol": all_path}

    if len(path_optimization_list) > 0:
        all_paths = torch.cat(path_optimization_list, dim=1).cpu()
        all_actions = torch.cat(all_actions_list, dim=0).cpu()
        all_path_terms = torch.cat(all_path_terms_list, dim=0).cpu()
        all_force_terms = torch.cat(all_force_terms_list, dim=0).cpu()

        output.update(
            {
                "all_paths": all_paths,
                "actions": all_actions,
                "path_terms": all_path_terms,
                "force_terms": all_force_terms,
            }
        )

    print(f"{int(len(all_path) / interpolator.path_length)} paths generated")

    return output


def num_to_groups(num, divisor):
    """
    Converts a number into an array with num // divisor elements of
    size divisor and one element with the remainder num % divisor.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# Jensen–Shannon divergence
def js_divergence(h1: np.ndarray, h2: np.ndarray) -> np.float64:
    """
    Calculate Jensen-Shannon divergence between two histograms
    """
    # Normalize histograms to obtain probability distributions
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10

    M = (p1 + p2) / 2
    js = (kl_divergence(p1, M) + kl_divergence(p2, M)) / 2
    return js


def normalize_histogram(hist: np.ndarray) -> np.ndarray:
    """
    Normalize input histogram
    """
    hist = np.array(hist)
    prob = hist / np.sum(hist)
    return prob


def kl_divergence(p1: np.ndarray, p2: np.ndarray) -> np.float64:
    """
    Calculate KL divergence
    """
    return np.sum(p1 * np.log(p1 / p2))


def get_pwd_triu_batch(x, offset=1):
    """
    Get pairwise distances (PWD) for a batch of structures,
    only for the upper triangle without the diagonal since
    the PWD matrix is symmetric and the diagonal is zero.
    So for structures with dimensions bs x num_beads x 3, this
    function returns bs x (num_beads**2-num_beads)/2 distances
    if offset=1 (default). Offset can also be specified to only
    take into account further off-diagonal distances.
    """
    assert len(x.shape) == 3 and x.shape[-1] == 3, "Shape mismatch"
    pwd = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)
    assert pwd.shape[-2] == pwd.shape[-1], "PWD matrix must be square"
    triu_ind = torch.triu_indices(pwd.shape[-2], pwd.shape[-1], offset=offset)
    return pwd[:, triu_ind[0], triu_ind[1]]


if __name__ == "__main__":
    test_name = "jensen_shannon"

    if test_name == "jensen_shannon":
        # Test Jensen–Shannon divergence between two unnormalized or normalized probability distributions:
        h1 = np.array([0.1, 0.2, 0.5, 0.3, 0])
        h2 = np.array([0, 0.25, 0.5, 0.21, 0])

        value = js_divergence(h1, h2)
        value_equal = js_divergence(h1, h1)

        print(f"This value should be larger than 0: {value:.4f}")
        print(f"This value should be 0 {value_equal:.4f}")
    else:
        raise Exception(f"Wrong test name {test_name:s}")
