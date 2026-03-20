from .actions import ZeroTActionMultiPath
from .utils import center_zero, short_minimize, save_ovito_traj, replace_multipathendpoints

from .molecule_reader import load_ala2_data
from .analyze_ramachandran import analyze_ramachandran_plot

import wandb
wandb.require("core")

import argparse
from tqdm import tqdm
from rmsd import kabsch_rotate

import torch
from torch.func import vmap

import numpy as np

import os

from matplotlib import pyplot as plt


def bfgs_multipath_optimization(
    position_tensor_path,
    n_paths=8,
    path_length=300,
    steps=20000,
    om_dt=1.0,
    om_gamma=10.0,
    precision="float32",
    device=None,
    disable_logging=True,
    wandb_project="om-diffusion-experiments",
    output_suffix="_classical_relaxed"
):
    """
    Perform BFGS multipath optimization on ala2 trajectories.
    
    Parameters
    ----------
    position_tensor_path : str
        Path to the input position tensor file
    n_paths : int
        Number of paths to optimize
    path_length : int
        Length of each interpolation path
    steps : int
        Maximum number of OM optimization steps
    om_dt : float
        Time step for OM optimization
    om_gamma : float
        Gamma parameter for OM optimization
    precision : str
        Precision to use ("float32" or "float64")
    device : str or None
        Device to use for computation (None for auto-detect)
    disable_logging : bool
        Whether to disable wandb logging
    wandb_project : str
        Wandb project name
    output_suffix : str
        Suffix to add to output filename
        
    Returns
    -------
    output_path : str
        Path to the saved optimized trajectory
    """
    # Create args namespace
    from argparse import Namespace
    samp_args = Namespace(
        position_tensor_path=position_tensor_path,
        n_paths=n_paths,
        path_length=path_length,
        steps=steps,
        om_dt=om_dt,
        om_gamma=om_gamma,
        precision=precision,
        device=device,
        disable_logging=disable_logging,
        wandb_project=wandb_project,
        data_folder=None
    )
    
    # Auto-detect device if not specified
    if samp_args.device is None:
        if torch.cuda.is_available():
            samp_args.device = f"cuda:{torch.cuda.current_device()}"
        else:
            samp_args.device = "cpu"

    print(f"Using device: {samp_args.device}")

    reactants, products, sample_force, sample_forces_laplace, system = load_ala2_data(samp_args)
    n_atoms = system.natoms
    
    data_dir = os.path.dirname(samp_args.position_tensor_path)

    # Don't create subdirectories - save everything in the same folder as the input
    if not samp_args.disable_logging:
        wandb.login()
        run = wandb.init(
            project=samp_args.wandb_project,
            config=samp_args,
        )

    print(f"Loading position tensor from: {samp_args.position_tensor_path}")
    position_tensor = torch.load(samp_args.position_tensor_path, map_location=samp_args.device)
    position_tensor_reshaped = position_tensor.reshape(samp_args.n_paths, -1, 22, 3)
    samp_args.n_paths = position_tensor_reshaped.shape[0]
    print(f"Loaded {samp_args.n_paths} paths with shape {position_tensor_reshaped.shape}")

    position_tensor = position_tensor_reshaped.reshape(-1, 22, 3)
    position_tensor.requires_grad = True
    system.pos = position_tensor
    
    print("Starting energy minimization (1400 steps)...")
    short_minimize(system, lambda x: vmap(sample_force.compute)(x), n_steps=1400, step_size=0.001, device=samp_args.device)
    print("Energy minimization complete")
    
    position_tensor_reshaped = position_tensor_reshaped.reshape(samp_args.n_paths, -1, 22, 3)
    system.pos = system.pos.reshape(samp_args.n_paths, -1, 22, 3)
    system.pos = system.pos.clone().detach()
    
    # Replace endpoints with non-optimized endpoints
    system.pos[:, 0] = position_tensor_reshaped[:, 0]
    system.pos[:, -1] = position_tensor_reshaped[:, -1]

    system.pos = system.pos.reshape(-1, 22, 3)
    save_ovito_traj(system.pos.detach().cpu(), os.path.join(data_dir, "init_path.gsd"), bonds=reactants.bonds)
    torch.save(position_tensor_reshaped.detach().cpu(), os.path.join(data_dir, "init_path.pt"))

    action = ZeroTActionMultiPath(
        force_func=lambda x: vmap(sample_force.compute)(x),
        gamma=samp_args.om_gamma,
        dt=samp_args.om_dt,
        M=system.M,
        n_paths=samp_args.n_paths
    )

    action_func = lambda path: action(path)

    energies = []
    actions = []
    path_terms = []
    force_terms = []
    laplace_terms = []

    new_path = center_zero(system.pos.detach())
    np_path = new_path.detach().cpu().numpy()
    rotated_list = []
    for i, pos in enumerate(np_path):
        pos = kabsch_rotate(pos, np_path[0])
        rotated_list.append(torch.tensor(pos))
    new_path = torch.stack(rotated_list, axis=0).to(samp_args.device)

    new_path.requires_grad = True
    system.pos = new_path

    precision_dtype = torch.float32 if samp_args.precision == "float32" else torch.float64
    if system.pos.dtype != precision_dtype:
        system.pos = system.pos.to(dtype=precision_dtype)

    print(f"\nStarting OM action optimization with LBFGS ({samp_args.steps} max iterations)")
    print(f"Parameters: dt={samp_args.om_dt}, gamma={samp_args.om_gamma}")

    with tqdm(total=samp_args.steps, desc="Processing") as pbar:

        def bfgs_closure():
            path_term, force_term, laplace_term, total_energy = action_func(system.pos)

            total_action = path_term + force_term

            actions.append(total_action.item())

            (grads,) = torch.autograd.grad(total_action, system.pos)
            
            grads = grads.reshape(samp_args.n_paths, -1, 22, 3)
            grads[:, 0, :], grads[:, -1, :] = torch.zeros(samp_args.n_paths, n_atoms, 3), torch.zeros(samp_args.n_paths, n_atoms, 3)
            grads = grads.reshape(-1, 22, 3)
            ammount_clipped = (grads - torch.clip(grads, -100, 100)).abs().sum()
            system.pos.grad = grads

            with torch.no_grad():
                
                term_sum = path_term.abs().item() + force_term.abs().item() + laplace_term.abs().item()

                path_terms.append(path_term.item())
                force_terms.append(force_term.item())
                laplace_terms.append(laplace_term.item())
                path_contribution = path_term.abs().item() / term_sum * 100
                force_contribution = force_term.abs().item() / term_sum * 100
                laplace_contribution = abs(laplace_term.item()) / term_sum * 100

                if not samp_args.disable_logging:
                    wandb.log(
                        {
                            "OM Action": total_action.item(),
                            "Path term Norm": path_term.item(),
                            "Force term Norm": force_term.item(),
                            "Laplace term norm": laplace_term.item(),
                            "Path Contribution": path_contribution,
                            "Force Contribution": force_contribution,
                            "Laplace Contribution": laplace_contribution,
                            "Ammount clipped": ammount_clipped,
                            "Max energy": total_energy.max()
                        })

            pbar.set_postfix(cost=f"action: {total_action.item():.4f}")   
            pbar.update(1)
            pbar.refresh()                

            return total_action

        optimizer = torch.optim.LBFGS([system.pos],
                                history_size=10, 
                                tolerance_change=1e-12,
                                max_iter=samp_args.steps, 
                                line_search_fn="strong_wolfe")
        optimizer.step(bfgs_closure)

    print("\nOM optimization complete!")
        
    final_traj_reshaped = system.pos.detach().cpu().reshape(samp_args.n_paths, -1, 22, 3)

    # Save optimized trajectory with suffix
    base_name = os.path.basename(position_tensor_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_name = f"{name_without_ext}{output_suffix}.pt"
    output_path = os.path.join(data_dir, output_name)
    
    print(f"Saving optimized trajectory to: {output_path}")
    torch.save(final_traj_reshaped.detach().cpu(), output_path)
    
    # Also save auxiliary files to the same directory
    print(f"Saving results to: {data_dir}")
    save_ovito_traj(system.pos.detach().cpu(), os.path.join(data_dir, "final_traj.gsd"), bonds=reactants.bonds)
    torch.save(final_traj_reshaped.detach().cpu(), os.path.join(data_dir, "final_traj.pt"))

    # Calculate and save energy along the path
    E, f = vmap(sample_force.compute)(system.pos)
    torch.save(E, os.path.join(data_dir, "final_energy.pt"))

    E_reshaped = E.detach().cpu().reshape(samp_args.n_paths, -1)
    print(f"Final energy - Min: {E.min().item():.4f}, Max: {E.max().item():.4f}, Mean: {E.mean().item():.4f} kcal/mol")

    # Plot and save energy profile
    plt.figure(figsize=(10, 6))
    for i in range(E_reshaped.shape[0]):
        plt.plot(E_reshaped[i].detach().cpu().numpy(), label=f'Path {i+1}')
    plt.title("Energy along the path")
    plt.xlabel("Path index")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(os.path.join(data_dir, 'energy_profile.png'))
    plt.close()


    if not samp_args.disable_logging:
        wandb.log({
            "Final_Energy_Profile": wandb.Image(os.path.join(data_dir, 'energy_profile.png')),
            "Final_Max_Energy": E.max().item(),
            "Final_Min_Energy": E.min().item(),
            "Final_Mean_Energy": E.mean().item()
        })
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="coarse-graining-evaluator")

    parser.add_argument("--disable_logging", action="store_true", help="Don't log to wandb")

    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="directory root where data is stored, if None (default) work with empty datasets and saved reference from saved_histograms",
    )


    parser.add_argument(
        "--steps", type=int, help="number of OM optimization steps", default=20000
    )

    parser.add_argument("--om_dt", type=float, help="dt for OM optimization", default=1)

    parser.add_argument(
        "--om_gamma", type=float, help="gamma for OM optimization", default=10
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for computation (default: auto-detect GPU or CPU)",
        default=None
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="Use double or single precision?",
        default="float32"
    )

    parser.add_argument(
        "--n_paths",
        type=int,
        help="number of paths to optimize",
        default=8
    )

    parser.add_argument(
        "--position_tensor_path",
        type=str,
        help="path to the position tensor file to load",
        default="saved_models/alanine_dipeptide/main_eval_output_om_interpolate/sample-om_interpolate.pt"
    )

    parser.add_argument(
        "--path_length",
        type=int,
        help="length of interpolation path",
        default=300
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        help="wandb project name for logging",
        default="om-diffusion-experiments"
    )

    samp_args = parser.parse_args()
    
    bfgs_multipath_optimization(
        position_tensor_path=samp_args.position_tensor_path,
        n_paths=samp_args.n_paths,
        path_length=samp_args.path_length,
        steps=samp_args.steps,
        om_dt=samp_args.om_dt,
        om_gamma=samp_args.om_gamma,
        precision=samp_args.precision,
        device=samp_args.device,
        disable_logging=samp_args.disable_logging,
        wandb_project=samp_args.wandb_project
    ) 