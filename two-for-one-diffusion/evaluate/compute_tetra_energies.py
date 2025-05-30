from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer
from io import StringIO
from tqdm import tqdm
import numpy as np
import os
import mdtraj as md
import io
import matplotlib.pyplot as plt
from openmm.app import PDBFile, ForceField, Modeller, Simulation, NoCutoff, HBonds
from openmm import VerletIntegrator, LocalEnergyMinimizer
from openmm.unit import *
import torch
from rmsd import kabsch_rmsd
from logging_utils import save_ovito_traj


def minimize_with_rmsd_limit(
    simulation, max_rmsd=0.5, max_total_steps=100, step_size=10
):
    """Run energy minimization in steps, stopping early if RMSD exceeds threshold."""
    initial_pos = (
        simulation.context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(angstroms)
    )
    current_pos = initial_pos
    rmsd = 0.0

    for step in range(0, max_total_steps, step_size):
        LocalEnergyMinimizer.minimize(simulation.context, maxIterations=step_size)
        current_pos = (
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(angstroms)
        )

        rmsd = kabsch_rmsd(initial_pos, current_pos)

        if rmsd > max_rmsd:
            print(
                f"Stopping early: RMSD {rmsd:.3f} Å exceeds cutoff {max_rmsd:.3f} Å after {step + step_size} steps"
            )
            return current_pos, rmsd
    return current_pos, rmsd  # final pos and rmsd if threshold not exceeded


def fix_pdb_file(pdb_path, freq=5):
    """
    Adds missing heavy atoms and hydrogens to a potentially multi-frame pdb file.
    """
    # first use MDtraj to fix issues with pdb file by loading and resaving
    top = md.load(pdb_path)
    # Save the fixed PDB file
    new_path = os.path.splitext(pdb_path)[0] + "_fixed.pdb"
    top.save_pdb(new_path)

    input_pdb = PDBFile(new_path)
    has_written_header = False
    print("Adding Missing Heavy Atoms and Hydrogens to PDB File")
    with open(new_path, "w") as output_pdb:
        count = 0
        for i in tqdm(range(input_pdb.getNumFrames())):
            # Create an in-memory PDB file containing just the one frame.
            if i % freq == 0 or i == input_pdb.getNumFrames() - 1:
                count += 1
                output = StringIO()
                PDBFile.writeFile(
                    input_pdb.topology, input_pdb.getPositions(frame=i), output
                )
                # Process it with PDBFixer.
                fixer = PDBFixer(pdbfile=StringIO(output.getvalue()))
                fixer.missingResidues = {}
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(pH=7.0)
                # Write the result to the output file.
                if not has_written_header:
                    PDBFile.writeHeader(fixer.topology, output_pdb)
                    has_written_header = True
                PDBFile.writeModel(fixer.topology, fixer.positions, output_pdb, count)
        PDBFile.writeFooter(fixer.topology, output_pdb)
    return new_path


def compute_energies(
    pdb_path, compute_freq=5, max_minimization_steps=200, max_rmsd=0.5
):
    """
    Top level function to compute energies of a multi-frame PDB file.
    Adds missing heavy atoms and hydrogens, and then performs some energy minimization,
    before evaluating energies with Amber FF.

    """
    new_path = fix_pdb_file(pdb_path, freq=compute_freq)  # Add missing atoms

    topology = md.load_topology(new_path)
    bonds = [(bond[0].index, bond[1].index) for bond in topology.bonds]
    bonds = np.array(bonds)

    with open(new_path, "r") as f:
        pdb_text = f.read()

    frames = pdb_text.split("ENDMDL")
    frames = [frame.strip() + "\nENDMDL\n" for frame in frames if "MODEL" in frame]

    forcefield = ForceField("amber14-all.xml")
    energies = []
    positions = []

    print("Computing energies with RMSD thresholding...")

    for i, frame in tqdm(enumerate(frames)):
        pdb = PDBFile(io.StringIO(frame))
        modeller = Modeller(pdb.topology, pdb.positions)

        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
        )

        integrator = VerletIntegrator(1.0 * femtoseconds)
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        state = simulation.context.getState(getEnergy=True)
        energy_before = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        # Minimize with RMSD limit (keep endpoints fixed)
        minimized_pos, rmsd = minimize_with_rmsd_limit(
            simulation,
            max_rmsd=max_rmsd,
            max_total_steps=(
                0 if i == 0 or i == len(frames) - 1 else max_minimization_steps
            ),
            step_size=10,
        )

        state = simulation.context.getState(getEnergy=True)
        energy_after = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        energies.append(energy_after)
        positions.append(minimized_pos)

        print(
            f"Frame {i}: Energy Change = {energy_after - energy_before:.2f} kJ/mol | RMSD = {rmsd:.3f} Å"
        )
    # Save as xtc
    positions = np.array(positions)
    traj = md.Trajectory(positions / 10, topology=topology)
    traj.save(new_path.split(".pdb")[0] + ".xtc")
    return np.array(energies), positions, bonds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute energies of a multi-frame PDB file."
    )
    parser.add_argument("--pdb_dir", type=str, help="Dir of the multi-frame PDB files.")
    parser.add_argument(
        "--gen_mode",
        type=str,
        help="generation mode: iid, interpolate, or om_interpolate.",
    )
    parser.add_argument("--name", type=str, help="Name of the tetrapeptide.")
    parser.add_argument(
        "--num_paths",
        type=int,
        default=16,
        help="Number of paths to compute energies for.",
    )
    parser.add_argument(
        "--max_minimization_steps",
        type=int,
        default=200,
        help="Maximum number of energy minimization steps to perform.",
    )
    parser.add_argument(
        "--dont_plot", action="store_true", help="Whether to plot the energies."
    )

    args = parser.parse_args()

    if args.gen_mode == "iid":
        pdb_files = [
            os.path.join(args.pdb_dir, f"{args.name}_{i}.pdb") for i in range(1)
        ]
    else:
        pdb_files = [
            os.path.join(args.pdb_dir, f"{args.name}_{i}.pdb")
            for i in range(args.num_paths)
        ]

    out = [
        compute_energies(
            pdb_file,
            compute_freq=100 if args.gen_mode == "iid" else 10,
            max_minimization_steps=args.max_minimization_steps,
            max_rmsd=1.0,
        )
        for pdb_file in pdb_files
    ]
    energies = torch.stack([torch.tensor(o[0]) for o in out], dim=0)
    positions = torch.cat([torch.tensor(o[1]) for o in out], dim=0)
    bonds = out[0][2]

    # Save new OVITO trajectory
    save_ovito_traj(
        positions,
        os.path.join(args.pdb_dir, f"sample-{args.gen_mode}_{args.name}_fixed.gsd"),
        align=args.gen_mode == "iid",
        all_backbone=False,
        create_bonds=True,
        bonds=bonds,
    )

    torch.save(energies, os.path.join(args.pdb_dir, f"energies_{args.name}.pt"))
    if not args.dont_plot:
        plt.figure(figsize=(10, 6))
        for energy in energies:
            energy = energy - energy.min() + 1
            plt.plot(range(len(energy)), energy, label="Potential Energy")
        plt.xlabel("Frame")
        plt.ylabel("Potential Energy (kJ/mol)")
        plt.yscale("log")
        plt.title("Energy Profile Across PDB Frames")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.pdb_dir, f"energy_profiles_{args.name}.png"))
