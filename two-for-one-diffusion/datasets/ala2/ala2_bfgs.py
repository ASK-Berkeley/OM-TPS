from actions import FDAction, ZeroTAction, HutchinsonAction
from utils import center_zero
from molecule_reader import load_ala2_data

import wandb
wandb.require("core")

import argparse
from tqdm import tqdm
from rmsd import kabsch_rotate

import torch
from torch.func import vmap

import numpy as np

from utils import save_ovito_traj

import os


parser = argparse.ArgumentParser(description="coarse-graining-evaluator")

parser.add_argument("--disable_logging", action="store_true", help="Don't log to wandb")

parser.add_argument(
    "--data_folder",
    type=str,
    default=None,
    help="directory root where data is stored, if None (default) work with empty datasets and saved reference from saved_histograms",
)

parser.add_argument(
    "--parallel_sim", type=int, default=2, help="Number of parallel simulations UNUSED"
)
parser.add_argument(
    "--save_interval", type=int, default=250, help="save interval (in timesteps)"
)

parser.add_argument("--write_every", type=int, default=100, help="when to write")

parser.add_argument(
    "--initial_guess_method",
    type=str,
    help="method to generate initial interpolation path (options: 'load' or 'linear' or 'expand')",
    default="load",
)

parser.add_argument(
    "--path_length", type=int, help="length of interpolation path", default=200
)

parser.add_argument(
    "--steps", type=int, help="number of OM optimization steps", default=1000
)

parser.add_argument(
    "--optimizer",
    type=str,
    help="Which action to use. Options: adam, sgd",
    default="adam",
)

parser.add_argument(
    "--lr", type=float, help="learning rate for OM optimization", default=1e-6
)
parser.add_argument("--om_dt", type=float, help="dt for OM optimization", default=1)

parser.add_argument(
    "--om_gamma", type=float, help="gamma for OM optimization", default=10
)

parser.add_argument(
    "--om_T", type=float, help="Temperature for OM optimization", default=300
)

parser.add_argument(
    "--laplace_iters", type=int, help="Number of internal iterations in the laplace calculations", default=1
)

parser.add_argument(
    "--action",
    type=str,
    help="Which action to use. Options: hutch, differences",
    default="zeroT",
)


parser.add_argument(
    "--precision",
    type=str,
    help="Use double or single precision?",
    default="float32"
)

parser.add_argument(
    "--path_idx",
    type=int,
    default=0
)

parser.add_argument(
    "--exp_name",
    type=str,
    default="default"
)


samp_args = parser.parse_args()

device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
samp_args.device = device


reactants, products, sample_force, sample_forces_laplace, system = load_ala2_data(samp_args)
n_atoms = system.natoms

data_dir = "result_data"

if not samp_args.disable_logging:
    # validate_git_status()
    wandb.login()
    run = wandb.init(
        entity="om-interpolation",
        project="all-atom-ala2-bfgs",
        config=samp_args,
        name = samp_args.exp_name
    )
    data_dir = os.path.join(data_dir, run.name)
    os.makedirs(data_dir, exist_ok=True)
else:
    data_dir = os.path.join(data_dir, "default")
    os.makedirs(data_dir, exist_ok=True)

if samp_args.initial_guess_method == "load":
    run = "pleasant-darkness-14"
    # position_tensor = torch.load("result_data/" + run + "/unwrapped6.pt", map_location=samp_args.device)
    position_tensor = torch.load("michael_samples.pt")
    position_tensor = position_tensor.to(samp_args.device)[samp_args.path_length * samp_args.path_idx:samp_args.path_length * (samp_args.path_idx + 1)]
    position_tensor.requires_grad=True
    system.pos = position_tensor

if samp_args.action == "hutch":
    action = HutchinsonAction(
                        force_func=lambda x : vmap(sample_force.compute)(x),
                        laplace_forces=lambda x : vmap(sample_forces_laplace.compute)(x),
                        gamma=samp_args.om_gamma,
                        dt=samp_args.om_dt,
                        T=samp_args.om_T,
                        M=system.M,
                        N=samp_args.laplace_iters
                    )
elif samp_args.action == "differences":
    action = FDAction(
                        force_func=lambda x : vmap(sample_force.compute)(x),
                        laplace_forces=lambda x : vmap(sample_forces_laplace.compute)(x),
                        gamma=samp_args.om_gamma,
                        dt=samp_args.om_dt,
                        T=samp_args.om_T,
                        M=system.M,
                        N=samp_args.laplace_iters
                    )
elif samp_args.action == "zeroT":
    action = ZeroTAction(
                        force_func=lambda x : vmap(sample_force.compute)(x),
                        laplace_forces=lambda x : vmap(sample_forces_laplace.compute)(x),
                        gamma=samp_args.om_gamma,
                        dt=samp_args.om_dt,
                        T=samp_args.om_T,
                        M=system.M,
                        N=samp_args.laplace_iters
                    )
else:
    raise ValueError("Unknown action")
action_func = lambda path: action(path)



energies = []
max_energies = []
actions = []
path_terms = []
force_terms = []
laplace_terms = []

path_length = system.pos.shape[0]

new_path = center_zero(system.pos.detach())
np_path = new_path.detach().cpu().numpy()
rotated_list = []
for i, pos in enumerate(np_path):
    pos = kabsch_rotate(pos, np_path[0])
    rotated_list.append(torch.tensor(pos))
new_path = torch.stack(rotated_list, axis=0).to(samp_args.device)

new_path.requires_grad = True
system.pos = new_path

precision = torch.float32 if samp_args.precision=="float32" else torch.float64
if system.pos.dtype != samp_args.precision:
    system.pos = system.pos.to(dtype=precision)
print(system.pos.dtype)


with tqdm(total=samp_args.steps, desc="Processing") as pbar:


    def bfgs_closure():
        global pbar
        path_term, force_term, laplace_term, energy = action_func(system.pos)
        total_energy = energy.sum()
        max_energy = energy.max()

        # It seems likely that they are the same. It probably can be proven
        total_action = path_term + force_term
        #print(total_action.item(), terms)

        actions.append(total_action.item())

        (grads,) = torch.autograd.grad(total_action, system.pos)
        
        grads[0, :], grads[-1, :] = torch.zeros(n_atoms, 3), torch.zeros(n_atoms,3)
        ammount_clipped = (grads - torch.clip(grads, -100, 100)).abs().sum()
        system.pos.grad = grads

        with torch.no_grad():
            
            term_sum = path_term.abs().item() + force_term.abs().item() + laplace_term.abs().item()

            path_terms.append(path_term.item())
            force_terms.append(force_term.item())
            laplace_terms.append(laplace_term.item())
            energies.append(total_energy.item())
            max_energies.append(max_energy.item())
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
                        "Total energy": total_energy,
                        "Max energy": max_energy,
                        "Ammount clipped": ammount_clipped
                        })

        pbar.set_postfix(cost=f"action: {total_action.item():.4f}")   
        pbar.update(1)
        pbar.refresh()                

        return total_action

    optimizer = torch.optim.LBFGS([system.pos],
                            history_size=10, 
                            tolerance_change=1e-6,
                            max_iter=samp_args.steps, 
                            line_search_fn="strong_wolfe")
    optimizer.step(bfgs_closure)
    


save_ovito_traj(system.pos.detach().cpu(), os.path.join(data_dir, f"warmup_unwrapped_{samp_args.path_idx}.gsd"), bonds=reactants.bonds)
torch.save(system.pos.detach().cpu(), os.path.join(data_dir, f"unwrapped_{samp_args.path_idx}.pt"))