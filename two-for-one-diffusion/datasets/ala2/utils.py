import gsd.hoomd
from rmsd import kabsch_rmsd, kabsch_rotate
import torch
import numpy as np

import mdtraj as md

def save_ovito_traj(
    positions,
    filename,
    align=True,
    bonds=None,
):
    """
    Save the given positions to a GSD file using Ovito.
    Expects that the positions are in the shape (n_frames, n_residues, 3) (only alpha carbons).
    """

    t = gsd.hoomd.open(name=filename, mode="w")
    cell = 1.5 * torch.eye(3) * positions.cpu().abs().max()

    if align:
        positions = center_zero(positions)

    for i, pos in enumerate(positions):
        if align:
            pos = kabsch_rotate(pos, positions[0])
        t.append(create_frame(i, pos, cell, bonds))

    t.close()

def export_xtc(traj, pdf_file="data/chignolin_folded.pdb"):
    topology = md.load(pdf_file).topology
    traj = md.Trajectory(traj, topology)

    # Save the trajectory to an XTC file
    traj.save_xtc('output.xtc')


def center_zero(x):
    """
    Move the molecule center to zero.
    """
    if isinstance(x, tuple):
        x = x[0]
    assert len(x.shape) == 3 and x.shape[-1] == 3, "Dimensionality error"
    return x - x.mean(dim=1, keepdim=True)

def create_frame(step, position, cell, bonds=None):
    """
    Create an Ovito frame from the given positions.
    """
    # Particle positions, velocities, diameter
    # TODO: add option to add bonds between C and N atoms

    natoms = position.shape[0]
    position = torch.Tensor(position)
    partpos = position.tolist()
    diameter = 0.8 * np.ones((natoms,))
    diameter = diameter.tolist()
    # Now make gsd file
    s = gsd.hoomd.Frame()
    s.configuration.step = step
    s.particles.N = natoms
    s.particles.position = partpos
    s.particles.diameter = diameter
    s.configuration.box = [cell[0][0], cell[1][1], cell[2][2], 0, 0, 0]

    # Bonds for visualization
    if bonds is not None:
        s.bonds.N = bonds.shape[0]
        s.bonds.group = bonds
    return s

def expand_path(path, factor):
    new_path = torch.repeat_interleave(path, factor, dim=0)
    return new_path


#omega1, phi, psi, omega2, xi    
indices_dihedral = torch.tensor([
    [4,6,8,14],      #Phi
    [6,8,14,16],     #Psi
    [1,4,6,8],       #Omega 1
    [8,14,16,18],    #Omega 2
    [4,6,8,10]]).T   #Xi

dihedral_dict = {
    "phi": indices_dihedral[:,0],
    "psi": indices_dihedral[:,1],
    "omega1": indices_dihedral[:,2],
    "omega2": indices_dihedral[:,3],
    "xi": indices_dihedral[:,4]
}

        
#Get plain descriptors
def get_descriptors(R, indices):
    a0 = R[...,indices[0],:]
    a1 = R[...,indices[1],:]
    a2 = R[...,indices[2],:]
    a3 = R[...,indices[3],:]
    r12 = a1-a0
    r23 = a2-a1
    r34 = a3-a2
    crossA = torch.cross(r12, r23, dim=-1)
    crossB = torch.cross(r23, r34, dim=-1)
    crossC = torch.cross(r23, crossA, dim=-1)
    normA = torch.norm(crossA, dim=-1)
    normB = torch.norm(crossB, dim=-1)
    normC = torch.norm(crossC, dim=-1)
    normcrossB = crossB / normB.unsqueeze(-1)
    cosPhi = torch.sum(crossA * normcrossB, dim=-1) / normA
    sinPhi = torch.sum(crossC * normcrossB, dim=-1) / normC
    dihedrals = torch.atan2(sinPhi, cosPhi)
    return dihedrals

def get_psi_angle(R):
    return get_descriptors(R, dihedral_dict["psi"])

def get_phi_angle(R):
    return get_descriptors(R, dihedral_dict["phi"])

