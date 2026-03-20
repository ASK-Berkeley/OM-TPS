import torch
import numpy as np
import argparse
from copy import deepcopy
import mdtraj as md
from argparse import Namespace
from PIL import Image
from datetime import datetime
from pathlib import Path
from os.path import join
from tqdm import tqdm
import json
import os
import sys
import matplotlib.pyplot as plt


sys.path.insert(0, "../")
from datasets.ala2.molecule_reader import load_ala2_data
from .evaluators import DihedralEnergiesEvaluator, js_divergence
from .evaluators_CGflowmatching import (
    get_prob,
    get_torsions,
)
from datasets.dataset_utils_empty import AtomSelection

CLUSTER_ENDPOINTS_ALA = [0, 1]

def evaluate_ala2(
    gen_mode,
    append_exp_name,
    checkpoint_folder,
    reference_folder,
    pdb_folder,
    atom_selection=AtomSelection.PROTEIN,
    fold=None,
    subsample=None,
    n_sims=-1,
    opt_steps=0,
    window_size=3,
    compute_rates=False,
    gif=True,
    model=None,
    num_paths=8,
    traj_len=10,
    endpoints=None,
    log=False,
):

    append_exp_name_str = "_" + append_exp_name if append_exp_name else ""
    if atom_selection == AtomSelection.PROTEIN:
        eval_folder = os.path.join(
            checkpoint_folder,
            f"alanine_dipeptide/main_eval_output_{gen_mode}{append_exp_name_str}",
        )
    else:
        eval_folder = os.path.join(
            checkpoint_folder,
            f"alanine/fold{fold}/main_eval_output_{gen_mode}{append_exp_name_str}",
        )

    # include parent dir for references
    sample_path = Path(eval_folder, f"sample-{gen_mode}.pt")
    pdb_file = os.path.join(pdb_folder, f"folded_pdbs/ala2_cg.pdb")
    # Load sampled molecules
    sampled_mol = torch.load(sample_path, weights_only=True)
    if subsample is not None:
        sampled_mol = sampled_mol[np.random.permutation(subsample)]

    # Load topology from pdb file
    cg_topology = md.load(pdb_file).topology

    if fold is None:
        fold = 1
    dihedral_evaluator_test = DihedralEnergiesEvaluator(
        val_data=None,
        topology=cg_topology,
        plots_folder=eval_folder,
        n_bins=61,
        saved_ref=os.path.join(
            reference_folder, f"saved_dih_probs_ala2_fold_{fold}_testset.pickle"
        ),
    )

    if sampled_mol.shape[1] == 22:
        # For all-atom alanine dipeptide, just slice out the 5 backbone atoms
        sampled_mol = sampled_mol[:, [4, 6, 8, 14, 16]]

    # Testset Reference dihedral distribution
    dihedral_evaluator_test._plot_freeE_2d(
        dihedral_evaluator_test.gt_probs,
        file_name=join(
            dihedral_evaluator_test.plots_folder, "ramachandran_reference.png"
        ),
        plot_title="Reference test",
        save_plot=True,
    )

    loop = [sampled_mol]

    # Load path along optimization if OM interpolation is used
    if gif and gen_mode == "om_interpolate":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        gif_folder = f"temp_gif_images_{timestamp}"
        os.makedirs(gif_folder, exist_ok=True)
        path_history = Path(eval_folder, f"path_history-{gen_mode}.pt")
        # path_history = Path(eval_folder, f"sample-{gen_mode}_refined.pt")
        path_history = torch.load(path_history, weights_only=True)
        args = Namespace(
            path_length=path_history.shape[1],
            precision="float32",
            device=sampled_mol.device,
        )
        _, _, true_force_func, _, system = load_ala2_data(args)
        if path_history.shape[2] == 22:
            original_path_history = deepcopy(path_history)
            path_history = path_history[:, :, [4, 6, 8, 14, 16]]

        loop = path_history

        # Sample at most 100 evenly spaced steps for the GIF
        if len(loop) > 100:
            indices = np.linspace(0, len(loop) - 1, 100, dtype=int)
            loop = loop[indices]
            # Apply same indices to original_path_history to keep them synchronized
            original_path_history = original_path_history[indices]
        ramachandran_plots = []
        energy_plots = []
        path_energies = []

        for i, (path, og_path) in tqdm(enumerate(zip(loop, original_path_history))):
            file_name = os.path.join(
                gif_folder,
                f"ramachandran_opt_{i}.png",
            )
            bin_edges = np.linspace(-np.pi, np.pi, dihedral_evaluator_test.n_bins)

            sampled_dihedrals = get_torsions(path.numpy(), cg_topology)
            sampled_probs = get_prob(
                sampled_dihedrals, n_bins=dihedral_evaluator_test.n_bins
            )
            dihedral_endpoints = sampled_dihedrals.reshape(num_paths, -1, 2)[
                :, [0, -1], :
            ]

            # superimpose the optimized path on the true free energy plot
            dihedral_evaluator_test._plot_freeE_2d(
                dihedral_evaluator_test.gt_probs,
                endpoints=dihedral_endpoints,
                optimized_path=sampled_dihedrals,
                file_name=file_name,
                plot_title=f"Optimization step {i}",
                save_plot=True,
            )
            ramachandran_plots.append(file_name)

            # get energies along the path
            system.set_positions(og_path.permute((1, 2, 0)))  # [N_atoms, 3, nreplicas]
            path_energy, _ = torch.vmap(true_force_func.compute)(system.pos)
            path_energy = path_energy.squeeze().detach().cpu()
            

            plt.figure()
            # plot free energy profile of all paths
            for profile in path_energy.chunk(num_paths):
                profile -= profile.min()
                plt.plot(profile+1e-3)
            plt.ylim(0,100)
            plt.xlabel("Path Step")
            plt.ylabel("Potential Energy (kcal/mol)")
            plt.title(f"Step {i}: Transition energy profile: Alanine Dipeptide")
            file_name = join(gif_folder, f"energy_{i}.png")
            plt.savefig(file_name)
            plt.close()
            energy_plots.append(file_name)

        # Save as GIF
        ram_gif_path = join(
            dihedral_evaluator_test.plots_folder, "ramachandran_samples.gif"
        )
        images = [Image.open(ram_path) for ram_path in ramachandran_plots]

        # Save as GIF
        images[0].save(
            ram_gif_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=100,  # Duration for each frame in milliseconds
            loop=0,  # Loop forever
        )

        # Save as GIF
        energy_gif_path = join(
            dihedral_evaluator_test.plots_folder, "energy_profile.gif"
        )
        images = [Image.open(en_path) for en_path in energy_plots]

        # Save as GIF
        images[0].save(
            energy_gif_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=100,  # Duration for each frame in milliseconds
            loop=0,  # Loop forever
        )

        # delete temp images
        for ram_path in ramachandran_plots:
            os.remove(ram_path)
        for en_path in energy_plots:
            os.remove(en_path)
        os.rmdir(gif_folder)

    else:
        # Get samples dihedral distribution
        sampled_dihedrals = get_torsions(sampled_mol.numpy(), cg_topology)
        sampled_probs = get_prob(
            sampled_dihedrals, n_bins=dihedral_evaluator_test.n_bins
        )
        dihedral_evaluator_test._plot_freeE_2d(
            sampled_probs,
            file_name=join(
                dihedral_evaluator_test.plots_folder,
                f"ramachandran_samples.png",
            ),
            plot_title="Samples",
            save_plot=True,
        )

    dihedral_js = js_divergence(sampled_probs, dihedral_evaluator_test.gt_probs)
    metrics = {}
    metrics["js_divergence"] = dihedral_js
    final_metrics_file = "final_metrics.json"

    with open(join(eval_folder, final_metrics_file), "w") as f:
        json.dump(metrics, f, indent=4)