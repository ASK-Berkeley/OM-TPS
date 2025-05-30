import numpy as np
from PIL import Image
import io
import os
import torch
from tqdm import tqdm
import gsd.hoomd
import numpy as np

from IPython.display import Image as IPyImage, display
from utils import center_zero
from rmsd import kabsch_rotate


def save_ovito_traj(
    positions,
    filename,
    align=False,
    create_bonds=True,
    all_backbone=False,
    bonds=None,
):
    """
    Save the given positions to a GSD file using Ovito.
    Expects that the positions are in the shape (n_frames, n_residues, 3) (only alpha carbons).
    """

    t = gsd.hoomd.open(name=filename, mode="w")
    # cell = 1.5 * torch.eye(3) * positions.cpu().abs().max()
    cell = 25 * torch.eye(3)

    if align:
        positions = center_zero(positions)

    for i, pos in enumerate(positions):
        if align:
            try:
                pos = kabsch_rotate(pos, positions[0])
            except:
                pass
        t.append(create_frame(i, pos, cell, create_bonds, all_backbone, bonds))

    t.close()


def create_frame(
    step, position, cell, create_bonds=True, all_backbone=False, bonds=None
):
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
    if create_bonds:
        if bonds is None:
            if all_backbone:
                # construct bonds between CA and N atoms AND between CA and CB atoms
                senders = np.arange(position.shape[0] - 3)[1::3]
                receivers = np.arange(3, position.shape[0])[1::3]
                N_senders = senders
                N_receivers = N_senders - 1
                CB_senders = senders
                CB_receivers = CB_senders + 1
                last_sender = np.array([position.shape[0] - 2, position.shape[0] - 2])
                last_receiver = np.array([position.shape[0] - 3, position.shape[0] - 1])
                senders = np.concatenate([senders, N_senders, CB_senders, last_sender])
                receivers = np.concatenate(
                    [receivers, N_receivers, CB_receivers, last_receiver]
                )
                bonds = np.stack([senders, receivers], axis=1)

            else:
                senders = np.arange(position.shape[0] - 1)
                receivers = np.arange(1, position.shape[0])

                bonds = np.stack([senders, receivers], axis=1)

        s.bonds.N = bonds.shape[0]
        s.bonds.group = bonds
    return s
