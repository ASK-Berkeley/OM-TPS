import math
import scipy
import torch
from inspect import isfunction
import numpy as np
import mdtraj as md
import random
from actions import TruncatedAction
from rmsd import kabsch_rmsd, kabsch_rotate
from scipy.linalg import svd


NUM_RESIDUES_TO_PROTEIN = {
    10: "chignolin",
    20: "trp_cage",
    28: "bba",
    35: "villin",
    56: "protein_g",
}


class DummyClass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def exists(x):
    """
    Check if variable exists.
    """
    return x is not None


def default(val, d):
    """
    Apply function d or replace with value d if val doesn't exist.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    """
    Cycle through data.
    """
    while True:
        for data_i in dl:
            yield data_i


def extract(a, t, x_shape):
    """
    Extract the required elements using gather, and reshape.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    Linear beta schedule.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def compute_batched_forces(x, force_func, force_batch_size):
    forces = []
    original_shape = x.shape
    n_atoms = x.shape[-2]
    x = x.reshape(-1, n_atoms, 3)
    for i in range(0, x.shape[0], force_batch_size):
        forces.append(force_func(x[i : i + force_batch_size]))
    forces = torch.cat(forces)

    return forces.reshape(original_shape)


def center_zero(x):
    """
    Move the molecule center to zero.
    """
    if isinstance(x, tuple):
        x = x[0]
    assert len(x.shape) == 3 and x.shape[-1] == 3, "Dimensionality error"
    return x - x.mean(dim=1, keepdim=True)


def assert_center_zero(x, eps=1e-3):
    """
    Check if molecule center is at zero within tolerance eps.
    """
    assert len(x.shape) == 3 and x.shape[-1] == 3, "Dimensionality error"
    abs_mean = x.mean(dim=1).abs()

    center_max = abs_mean.max().item()

    if center_max >= eps:
        max_ind = (abs_mean == abs_mean.max()).nonzero()[0]
        x_max = x[max_ind[0]]
        max_dist = torch.norm(x_max[:, None, :] - x_max[None, :, :], dim=-1).max()
        raise AssertionError(
            f"Center not at zero: abs max at {center_max} for molecule with max pairwise distance {max_dist}"
        )


def rotation_aligned(x, y, atol=1e-2):
    y = y.cpu()
    x = x.cpu()
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    if not isinstance(y, np.ndarray):
        y = y.numpy()
    return np.allclose(y, kabsch_rotate(y, x), atol=atol)


def random_rotation(x, return_rotation_matrices=False):
    """
    Add a random rotation to input molecule with shape
    batch size x number of nodes x number of dims.
    Only implemented for 3 dimensions.
    """
    x_shape = x.shape
    bs, _, n_dims = x_shape
    device = x.device
    angle_range = np.pi * 2

    if n_dims == 3:
        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = -sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        x = torch.matmul(Ry, x)
        x = torch.matmul(Rz, x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    assert x.shape == x_shape, "Shape changed after rotation"

    if return_rotation_matrices:
        return x.contiguous(), (Rx, Ry, Rz)
    else:
        return x.contiguous()


def reverse_rotation(x, rotation_matrices):
    """
    Do reverse rotation given rotation matrices
    """
    Rx, Ry, Rz = rotation_matrices
    x = x.transpose(1, 2)
    x = torch.matmul(torch.linalg.inv(Rz), x)
    x = torch.matmul(torch.linalg.inv(Ry), x)
    x = torch.matmul(torch.linalg.inv(Rx), x)
    x = x.transpose(1, 2)

    return x.contiguous()


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """
    Custom PyTorch operation to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def check_reflection_equivariance(model_gnn, device, h):
    x_a = torch.randn(256, 5, 3).to(device)
    x_b = x_a.clone()
    x_b[:, :, 0] = x_b[:, :, 0] * (-1)
    t_norm = torch.Tensor([0.5]).to(device)
    t_norm = t_norm.reshape(-1, 1, 1).repeat(x_a.shape[0], 1, 1)

    output_a = model_gnn(x_a, h, t_norm)
    output_b = model_gnn(x_b, h, t_norm)

    print("Checking Invariance")
    print(torch.nn.functional.l1_loss(output_a, output_b))

    output_b[:, :, 0] = output_b[:, :, 0] * (-1)
    print("Checking Equivariance")
    print(torch.nn.functional.l1_loss(output_a, output_b))


def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation between two vectors.
    Source: Andrej Karpathy's gist: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
    Args:
        v0 (torch.Tensor): the first vector
        v1 (torch.Tensor): the second vector
        t (torch.Tensor): scalar from 0 to 1 parameterizing the interpolation
    """
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1  # simple linear interpolation
    else:
        theta_0 = np.arccos(dot)  # angle between latent vectors
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2


def filter_by_rmsd(coords: torch.Tensor, n: int = 2) -> torch.Tensor:
    """
    From a set of coordinates, determine the n most diverse coordinates, where "most diverse" means "most different, in terms of minimum RMSD.
    Note: The Max-Min Diversity Problem (MMDP) is in general NP-hard. This algorithm generates a candidate solution to MMDP for these coords
    by assuming that the random seed point is actually in the MMDP set (which there's no reason a priori to assume). As a result, if we ran
    this function multiple times, we would get different results.

    Args:
        shell: List of Schrodinger structure objects containing solvation shells
        n: number of most diverse shells to return
    Returns:
        List of n Schrodinger structures that are the most diverse in terms of minimum RMSD
    """
    assert_center_zero(coords)
    coords = coords.cpu().numpy()
    # seed_point = random.randint(0, coords.shape[0] - 1)
    seed_point = 0  # fix seed point for reproducibility
    final_idxs = [seed_point]
    min_rmsds = np.array([kabsch_rmsd(coords[seed_point], coord) for coord in coords])
    for _ in range(n - 1):

        best = np.argmax(min_rmsds)
        min_rmsds = np.minimum(
            min_rmsds,
            np.array([kabsch_rmsd(coords[best], coord) for coord in coords]),
        )
        final_idxs.append(best)
    return torch.tensor(coords[final_idxs])


class SamplerWrapper(torch.nn.Module):
    """
    The network becomes a sampler, such that we can sample in parallel GPUs by passing SamplerModule into a
    """

    def __init__(self, model):
        super(SamplerWrapper, self).__init__()
        self.model = model

    def forward(self, **kwargs):
        "The only kwarg should be 'batch_size'"
        return self.model.sample(**kwargs)


class InterpolatorWrapper(torch.nn.Module):
    """
    The network becomes an interpolator, such that we can sample in parallel GPUs by passing SamplerModule into a
    """

    def __init__(
        self, model, path_length, latent_time, interpolation_fn, temperature, log
    ):
        super(InterpolatorWrapper, self).__init__()
        self.model = model
        self.model.log = log
        self.path_length = path_length
        self.latent_time = latent_time
        self.interpolation_fn = interpolation_fn
        self.temperature = temperature

    def forward(self, x1, x2, z=None):
        return self.model.interpolate(
            x1=x1,
            x2=x2,
            z=z,
            path_length=self.path_length,
            latent_time=self.latent_time,
            temperature=self.temperature,
            interpolation_fn=self.interpolation_fn,
        )


class OMInterpolatorWrapper(torch.nn.Module):
    """
    The network becomes an OM interpolator, such that we can sample in parallel GPUs by passing SamplerModule into a
    """

    def __init__(
        self,
        model,
        path_length,
        latent_time,
        encode_and_decode=True,
        action_cls=TruncatedAction,
        initial_guess_fn=torch.lerp,
        initial_guess_level=0,
        om_steps=100,
        optimizer=torch.optim.Adam,
        lr=2e-1,
        dt=0.1,
        gamma=10,
        D=0.01,
        path_batch_size=-1,
        anneal=False,
        cosine_scheduler=False,
        add_noise=False,
        truncated_gradient=False,
        temperature=1.0,
        log=False,
    ):
        super(OMInterpolatorWrapper, self).__init__()
        self.model = model
        self.model.log = log
        self.path_length = path_length
        self.latent_time = latent_time
        self.encode_and_decode = encode_and_decode
        self.action_cls = action_cls
        self.initial_guess_fn = initial_guess_fn
        self.initial_guess_level = initial_guess_level
        self.om_steps = om_steps
        self.optimizer = optimizer
        self.lr = lr
        self.dt = dt
        self.gamma = gamma
        self.D = D
        self.path_batch_size = path_batch_size
        self.anneal = anneal
        self.cosine_scheduler = cosine_scheduler
        self.add_noise = add_noise
        self.truncated_gradient = truncated_gradient
        self.temperature = temperature

    def forward(self, x1, x2, z=None):
        return self.model.om_interpolate(
            x1=x1,
            x2=x2,
            z=z,
            path_length=self.path_length,
            encode_and_decode=self.encode_and_decode,
            latent_time=self.latent_time,
            action_cls=self.action_cls,
            initial_guess_fn=self.initial_guess_fn,
            initial_guess_level=self.initial_guess_level,
            om_steps=self.om_steps,
            optimizer=self.optimizer,
            lr=self.lr,
            dt=self.dt,
            gamma=self.gamma,
            D=self.D,
            path_batch_size=self.path_batch_size,
            anneal=self.anneal,
            cosine_scheduler=self.cosine_scheduler,
            add_noise=self.add_noise,
            truncated_gradient=self.truncated_gradient,
            temperature=self.temperature,
        )


def save_samples(sampled_mol, eval_folder, topology, milestone):
    torch.save(sampled_mol, str(eval_folder + f"/sample-{milestone}.pt"))
    all_mol_traj = md.Trajectory(sampled_mol[0:100].numpy() / 10, topology=topology)
    all_mol_traj.save_pdb(str(eval_folder + f"/sample-{milestone}.pdb"))


class TorchMD_CGProteinPriorForces(torch.nn.Module):
    def __init__(
        self,
        yaml_file,
        topology_file,
        device,
        forceterms=["Bonds", "RepulsionCG", "Dihedrals"],
    ):
        super(TorchMD_CGProteinPriorForces, self).__init__()
        mol = Molecule(topology_file)
        exclusions = "bonds"

        ff = ForceField.create(mol, yaml_file)
        self.device = device
        parameters = Parameters(ff, mol, forceterms, device=self.device)
        parameters.A, parameters.B = parameters.get_AB()
        self.parameters = parameters
        self.bond_params = {
            k: torch.tensor(v).to(self.device)
            for k, v in parameters.bond_params.items()
        }
        self.dihedral_params = {
            k: torch.tensor(v).to(self.device)
            for k, v in parameters.dihedral_params.items()
        }
        self.mapped_atom_types = torch.tensor(parameters.mapped_atom_types).to(
            self.device
        )
        self.A, self.B = torch.tensor(parameters.A).to(self.device), torch.tensor(
            parameters.B
        ).to(self.device)
        self.box = torch.tensor([50.0, 50.0, 50.0]).to(self.device)
        self.explicit_forces = True
        self.natoms = len(self.parameters.masses)
        self.ava_idx = make_indices(
            self.natoms, parameters.get_exclusions("bonds"), parameters.device
        ).to(self.device)

    def forward(self, x):
        # Assume x is in angstroms
        pot = 0
        forces = torch.zeros_like(x)

        # Bond stuff
        pairs = self.bond_params["idx"]
        param_idx = self.bond_params["map"][:, 1]
        bond_dist, bond_unitvec, _ = calculate_distances(x, pairs, self.box)

        bond_params = self.bond_params["params"][param_idx]

        E, force_coeff = evaluate_bonds(bond_dist, bond_params, self.explicit_forces)

        pot += E.sum()
        if self.explicit_forces:
            forcevec = bond_unitvec * force_coeff[:, None]
            forces.index_add_(0, pairs[:, 0], -forcevec)
            forces.index_add_(0, pairs[:, 1], forcevec)

        # Dihedral stuff
        dihed_idx = self.dihedral_params["idx"]
        param_idx = self.dihedral_params["map"][:, 1]
        _, _, r12 = calculate_distances(x, dihed_idx[:, [0, 1]], self.box)
        _, _, r23 = calculate_distances(x, dihed_idx[:, [1, 2]], self.box)
        _, _, r34 = calculate_distances(x, dihed_idx[:, [2, 3]], self.box)
        E, dihedral_forces = evaluate_torsion(
            r12,
            r23,
            r34,
            self.dihedral_params["map"][:, 0],
            self.dihedral_params["params"][param_idx],
            self.explicit_forces,
        )

        pot += E.sum()
        if self.explicit_forces:
            forces.index_add_(0, dihed_idx[:, 0], dihedral_forces[0])
            forces.index_add_(0, dihed_idx[:, 1], dihedral_forces[1])
            forces.index_add_(0, dihed_idx[:, 2], dihedral_forces[2])
            forces.index_add_(0, dihed_idx[:, 3], dihedral_forces[3])

        # Repulsion stuff
        nb_dist, nb_unitvec, _ = calculate_distances(x, self.ava_idx, self.box)
        ava_idx = self.ava_idx

        E, force_coeff = evaluate_repulsion_CG(
            nb_dist,
            ava_idx,
            self.mapped_atom_types,
            self.B,
            self.explicit_forces,
        )

        pot += E.sum()

        if self.explicit_forces:
            forcevec = nb_unitvec * force_coeff[:, None]
            forces.index_add_(0, ava_idx[:, 0], -forcevec)
            forces.index_add_(0, ava_idx[:, 1], forcevec)

        return pot, forces


# Utility Functions
def wrap_dist(dist, box):
    if box is None or torch.all(box == 0):
        wdist = dist
    else:
        wdist = dist - box.unsqueeze(0) * torch.round(dist / box.unsqueeze(0))
    return wdist


def calculate_distances(atom_pos, atom_idx, box):
    direction_vec = wrap_dist(atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]], box)
    dist = torch.norm(direction_vec, dim=1)
    direction_unitvec = direction_vec / dist.unsqueeze(1)
    return dist, direction_unitvec, direction_vec


def evaluate_bonds(dist, bond_params, explicit_forces=True):
    force = None

    k0 = bond_params[:, 0]
    d0 = bond_params[:, 1]
    x = dist - d0
    pot = k0 * (x**2)
    if explicit_forces:
        force = 2 * k0 * x
    return pot, force


def evaluate_torsion(r12, r23, r34, dih_idx, torsion_params, explicit_forces=True):
    # Calculate dihedral angles from vectors
    crossA = torch.cross(r12, r23, dim=1)
    crossB = torch.cross(r23, r34, dim=1)
    crossC = torch.cross(r23, crossA, dim=1)
    normA = torch.norm(crossA, dim=1)
    normB = torch.norm(crossB, dim=1)
    normC = torch.norm(crossC, dim=1)
    normcrossB = crossB / normB.unsqueeze(1)
    cosPhi = torch.sum(crossA * normcrossB, dim=1) / normA
    sinPhi = torch.sum(crossC * normcrossB, dim=1) / normC
    phi = -torch.atan2(sinPhi, cosPhi)

    ntorsions = r12.shape[0]
    pot = torch.zeros(ntorsions, dtype=r12.dtype, layout=r12.layout, device=r12.device)
    if explicit_forces:
        coeff = torch.zeros(
            ntorsions, dtype=r12.dtype, layout=r12.layout, device=r12.device
        )

    k0 = torsion_params[:, 0]
    phi0 = torsion_params[:, 1]
    per = torsion_params[:, 2]

    if torch.all(per > 0):  # AMBER torsions
        angleDiff = per * phi[dih_idx] - phi0
        pot = torch.scatter_add(pot, 0, dih_idx, k0 * (1 + torch.cos(angleDiff)))
        if explicit_forces:
            coeff = torch.scatter_add(
                coeff, 0, dih_idx, -per * k0 * torch.sin(angleDiff)
            )
    else:  # CHARMM torsions
        angleDiff = phi[dih_idx] - phi0
        angleDiff[angleDiff < -pi] = angleDiff[angleDiff < -pi] + 2 * pi
        angleDiff[angleDiff > pi] = angleDiff[angleDiff > pi] - 2 * pi
        pot = torch.scatter_add(pot, 0, dih_idx, k0 * angleDiff**2)
        if explicit_forces:
            coeff = torch.scatter_add(coeff, 0, dih_idx, 2 * k0 * angleDiff)

    # coeff.unsqueeze_(1)

    force0, force1, force2, force3 = None, None, None, None
    if explicit_forces:
        # Taken from OpenMM
        normDelta2 = torch.norm(r23, dim=1)
        norm2Delta2 = normDelta2**2
        forceFactor0 = (-coeff * normDelta2) / (normA**2)
        forceFactor1 = torch.sum(r12 * r23, dim=1) / norm2Delta2
        forceFactor2 = torch.sum(r34 * r23, dim=1) / norm2Delta2
        forceFactor3 = (coeff * normDelta2) / (normB**2)

        force0vec = forceFactor0.unsqueeze(1) * crossA
        force3vec = forceFactor3.unsqueeze(1) * crossB
        s = (
            forceFactor1.unsqueeze(1) * force0vec
            - forceFactor2.unsqueeze(1) * force3vec
        )

        force0 = -force0vec
        force1 = force0vec + s
        force2 = force3vec - s
        force3 = -force3vec

    return pot, (force0, force1, force2, force3)


def make_indices(natoms, excludepairs, device):
    fullmat = torch.full((natoms, natoms), True, dtype=bool)
    if len(excludepairs):
        excludepairs = torch.tensor(excludepairs)
        fullmat[excludepairs[:, 0], excludepairs[:, 1]] = False
        fullmat[excludepairs[:, 1], excludepairs[:, 0]] = False
    fullmat = torch.triu(fullmat, +1)
    allvsall_indices = torch.vstack(torch.where(fullmat)).T
    return allvsall_indices


def evaluate_repulsion_CG(
    dist, pair_indeces, atom_types, B, scale=1, explicit_forces=True
):  # Repulsion like from CGNet
    force = None

    atomtype_indices = atom_types[pair_indeces]
    coef = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1**6

    pot = (coef * rinv6) / scale
    if explicit_forces:
        force = (-6 * coef * rinv6) * rinv1 / scale
    return pot, force
