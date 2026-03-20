from actions import ZeroTActionMultiPath
from utils import center_zero, short_minimize, save_ovito_traj, replace_multipathendpoints

from molecule_reader import load_ala2_data

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
    default="short_optimize",
)

parser.add_argument(
    "--path_length", type=int, help="length of interpolation path", default=200
)

parser.add_argument(
    "--steps", type=int, help="number of OM optimization steps", default=50000
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
    "--om_gamma", type=float, help="gamma for OM optimization", default=1
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
    "--device",
    type=str,
    help="Use GPU if available",
    default="cuda:8"
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


samp_args = parser.parse_args()


reactants, products, sample_force, sample_forces_laplace, system = load_ala2_data(samp_args)
n_atoms = system.natoms

data_dir = "result_data"

path_filter = [True, True, False, True, True, True, True, True]


if not samp_args.disable_logging:
    # validate_git_status()
    wandb.login()
    run = wandb.init(
        entity="om-interpolation",
        project="all-atom-ala2-bfgs",
        config=samp_args,
    )
    data_dir = os.path.join(data_dir, run.name)
    os.makedirs(data_dir, exist_ok=True)
else:
    data_dir = os.path.join(data_dir, "default")
    os.makedirs(data_dir, exist_ok=True)

if samp_args.initial_guess_method == "load":
    run = "pleasant-darkness-14"
    position_tensor = torch.load("result_data/" + run + "/unwrapped6.pt", map_location=samp_args.device)
    position_tensor.requires_grad=True
    system.pos = position_tensor
elif samp_args.initial_guess_method == "short_optimize":
    position_tensor = torch.load("data/sample-om_interpolate.pt", map_location=samp_args.device)
    position_tensor_reshaped = position_tensor.reshape(samp_args.n_paths, -1, 22, 3)
    position_tensor_reshaped = position_tensor_reshaped[path_filter]
    samp_args.n_paths = position_tensor_reshaped.shape[0]

    print(position_tensor.shape)
    position_tensor = position_tensor_reshaped.reshape(-1, 22, 3)
    position_tensor.requires_grad=True
    system.pos = position_tensor
    #optimize_endpoints(system, lambda x: vmap(sample_force.compute)(x), n_steps=5000)
    #replace_multipathendpoints(system, samp_args.n_paths)
    short_minimize(system, lambda x: vmap(sample_force.compute)(x), n_steps=1400, step_size=0.001, device=samp_args.device)
    replace_multipathendpoints(system, samp_args.n_paths)
    save_ovito_traj(system.pos.detach().cpu(), os.path.join(data_dir, "init_path.gsd"), bonds=reactants.bonds)
    torch.save(position_tensor_reshaped.detach().cpu(), os.path.join(data_dir, "init_path.pt"))

action = ZeroTActionMultiPath(
    force_func=lambda x: vmap(sample_force.compute)(x),
    laplace_forces=lambda x: vmap(sample_forces_laplace.compute)(x),
    gamma=samp_args.om_gamma,
    dt=samp_args.om_dt,
    T=samp_args.om_T,
    M=system.M,
    N=samp_args.laplace_iters,
    n_paths=samp_args.n_paths
)

action_func = lambda path: action(path)

print(system.pos.shape)

energies = []
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
        path_term, force_term, laplace_term, total_energy = action_func(system.pos)

        # It seems likely that they are the same. It probably can be proven
        total_action = path_term + force_term
        #print(total_action.item(), terms)

        actions.append(total_action.item())

        (grads,) = torch.autograd.grad(total_action, system.pos)
        
        grads = grads.reshape(samp_args.n_paths, -1, 22, 3)
        grads[:,0, :], grads[:,-1, :] = torch.zeros(samp_args.n_paths, n_atoms, 3), torch.zeros(samp_args.n_paths, n_atoms,3)
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
    
final_traj_reshaped = system.pos.detach().cpu().reshape(samp_args.n_paths, -1, 22, 3)

save_ovito_traj(system.pos.detach().cpu(), os.path.join(data_dir, "final_traj.gsd"), bonds=reactants.bonds)
torch.save(final_traj_reshaped.detach().cpu(), os.path.join(data_dir, "final_traj.pt"))

# Calculate and save energy along the path
print(system.pos.shape)
E, f = vmap(sample_force.compute)(system.pos)
torch.save(E, os.path.join(data_dir, "final_energy.pt"))

E_reshaped = E.detach().cpu().reshape(samp_args.n_paths, -1)
print(E_reshaped.shape)

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