from actions import ZeroTAction, ExpandingAction
import actions

from utils import center_zero, save_ovito_traj
from molecule_reader import load_ala2_data
from rmsd import kabsch_rotate

import wandb
wandb.require("core")

import argparse
from tqdm import tqdm
import torch
from torch.func import vmap
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import save_ovito_traj, expand_path, center_zero


parser = argparse.ArgumentParser(description="Combined ala2 optimization with warmup")

# Common arguments
parser.add_argument("--disable_logging", action="store_true", help="Don't log to wandb")
parser.add_argument("--data_folder", type=str, default=None, help="directory root where data is stored")
parser.add_argument("--device", type=str, help="Use GPU if available", default="cuda:0")
parser.add_argument("--precision", type=str, help="Use double or single precision?", default="float32")

# Warmup specific arguments
parser.add_argument("--warmup_steps", type=int, default=5000, help="How many steps to get the initial trajectory")
parser.add_argument("--path_length", type=int, default=20, help="Initial path length for warmup")
parser.add_argument("--warmup_om_dt", type=float, default=1, help="dt for warmup OM optimization")
parser.add_argument("--warmup_om_gamma", type=float, default=10, help="gamma for warmup OM optimization")
parser.add_argument("--warmup_om_T", type=float, default=300, help="Temperature for warmup OM optimization")

# Main optimization arguments
parser.add_argument("--steps", type=int, default=5000, help="number of main OM optimization steps")
parser.add_argument("--om_dt", type=float, default=1, help="dt for main OM optimization")
parser.add_argument("--om_gamma", type=float, default=10, help="gamma for main OM optimization")
parser.add_argument("--om_T", type=float, default=300, help="Temperature for main OM optimization")
parser.add_argument("--laplace_iters", type=int, default=1, help="Number of internal iterations in laplace calculations")
parser.add_argument("--action", type=str, default="zeroT", help="Which action to use: hutch, differences, zeroT")

args = parser.parse_args()

# Initialize data directory
data_dir = "result_data"
if not args.disable_logging:
    wandb.login()
    run = wandb.init(
        entity="om-interpolation",
        project="all-atom-ala2-combined",
        config=args,
    )
    data_dir = os.path.join(data_dir, run.name)
    os.makedirs(data_dir, exist_ok=True)
else:
    data_dir = os.path.join(data_dir, "default")
    os.makedirs(data_dir, exist_ok=True)

# Load initial data
reactants, products, sample_force, sample_forces_laplace, system = load_ala2_data(args)
n_atoms = system.natoms

# Set precision
precision = torch.float32 if args.precision == "float32" else torch.float64
system.pos = system.pos.to(dtype=precision)

# Warmup phase
print("Starting warmup phase...")
multiplication_factor = [2, 2, 2, 2, 2, 2, 2]
# How much is initial path shorter?
dt_scale = 1
for f in multiplication_factor:
    dt_scale *= f

zeta = args.warmup_om_gamma / actions.PICOSEC2TIMEU
scaled_dt = args.warmup_om_dt / actions.TIMEFACTOR
k = zeta / scaled_dt

print("Values of k:", k)

for j, factor in enumerate(multiplication_factor):
    with tqdm(total=args.warmup_steps, desc=f"Warmup expansion {j+1}") as pbar:
        def bfgs_closure():
            global pbar
            path_term, force_term, total_energy = expand_func(system.pos, k)
            total_action = path_term + force_term

            (grads,) = torch.autograd.grad(total_action, system.pos)
            grads[0, :], grads[-1, :] = torch.zeros(n_atoms, 3), torch.zeros(n_atoms, 3)
            ammount_clipped = (grads - torch.clip(grads, -100, 100)).abs().sum()
            system.pos.grad = grads

            with torch.no_grad():
                term_sum = path_term.abs().item() + force_term.abs().item()
                path_contribution = path_term.abs().item() / term_sum * 100
                force_contribution = force_term.abs().item() / term_sum * 100

                if not args.disable_logging:
                    wandb.log({
                        "Warmup_OM_Action": total_action.item(),
                        "Warmup_Path_term": path_term.item(),
                        "Warmup_Force_term": force_term.item(),
                        "Warmup_Path_Contribution": path_contribution,
                        "Warmup_Force_Contribution": force_contribution,
                        "Warmup_Total_energy": total_energy,
                        "Warmup_Clipped": ammount_clipped,
                        "Warmup_Step": j
                    })

            pbar.set_postfix(cost=f"action: {total_action.item():.4f}")
            pbar.update(1)
            pbar.refresh()
            return total_action

        new_path = expand_path(system.pos, factor)
        new_path = center_zero(new_path)
        new_path = new_path.detach().clone()
        new_path.requires_grad = True
        system.pos = new_path

        print(f"Expanding by {factor}, expansion {j}, k={k}")
        print("New shape:", system.pos.shape)

        optimizer = torch.optim.LBFGS(
            [system.pos],
            history_size=10,
            tolerance_change=1e-7,
            max_iter=args.warmup_steps,
            line_search_fn="strong_wolfe"
        )

        expand_func = lambda path, k: ExpandingAction(
            force_func=lambda x: vmap(sample_force.compute)(x)
        )(path, k)

        optimizer.step(bfgs_closure)

        save_ovito_traj(system.pos.detach().cpu(), os.path.join(data_dir, f"warmup_unwrapped_{j}"), bonds=reactants.bonds)
        torch.save(system.pos.detach().cpu(), os.path.join(data_dir, f"unwrapped_{j}.pt"))

# Main optimization phase
print("\nStarting main optimization phase...")

action = ZeroTAction(
force_func=lambda x: vmap(sample_force.compute)(x),
laplace_forces=lambda x: vmap(sample_forces_laplace.compute)(x),
gamma=args.om_gamma,
dt=args.om_dt,
T=args.om_T,
M=system.M,
N=args.laplace_iters
)

action_func = lambda path: action(path)

# Prepare initial path for main optimization
new_path = center_zero(system.pos.detach())
np_path = new_path.detach().cpu().numpy()
rotated_list = []
for i, pos in enumerate(np_path):
    pos = kabsch_rotate(pos, np_path[0])
    rotated_list.append(torch.tensor(pos))
new_path = torch.stack(rotated_list, axis=0).to(args.device)
new_path.requires_grad = True
system.pos = new_path

scaling_loss = 100

print(system.pos.shape)

# Main optimization
with tqdm(total=args.steps, desc="Main optimization") as pbar:
    def bfgs_closure():
        global pbar
        path_term, force_term, laplace_term, total_energy = action_func(system.pos)
        total_action = path_term + force_term

        (grads,) = torch.autograd.grad(total_action, system.pos)
        grads[0, :], grads[-1, :] = torch.zeros(n_atoms, 3), torch.zeros(n_atoms, 3)
        ammount_clipped = (grads - torch.clip(grads, -100, 100)).abs().sum()
        system.pos.grad = grads

        with torch.no_grad():
            term_sum = path_term.abs().item() + force_term.abs().item() + laplace_term.abs().item()
            path_contribution = path_term.abs().item() / term_sum * 100
            force_contribution = force_term.abs().item() / term_sum * 100
            laplace_contribution = abs(laplace_term.item()) / term_sum * 100

            if not args.disable_logging:
                wandb.log({
                    "Main_OM_Action": total_action.item(),
                    "Main_Path_term": path_term.item(),
                    "Main_Force_term": force_term.item(),
                    "Main_Laplace_term": laplace_term.item(),
                    "Main_Path_Contribution": path_contribution,
                    "Main_Force_Contribution": force_contribution,
                    "Main_Laplace_Contribution": laplace_contribution,
                    "Main_Total_energy": total_energy,
                    "Main_Clipped": ammount_clipped
                })

        pbar.set_postfix(cost=f"action: {total_action.item():.4f}")
        pbar.update(1)
        pbar.refresh()
        return total_action * scaling_loss

    optimizer = torch.optim.LBFGS(
        [system.pos],
        history_size=10,
        tolerance_change=1e-9,
        max_iter=args.steps,
        line_search_fn="strong_wolfe"
    )
    optimizer.step(bfgs_closure)

# Save final results
save_ovito_traj(system.pos.detach().cpu(), os.path.join(data_dir, "final_unwrapped"), bonds=reactants.bonds)
torch.save(system.pos.detach().cpu(), os.path.join(data_dir, "final_unwrapped.pt"))

# Calculate and save energy along the path
E, f = vmap(sample_force.compute)(system.pos)
torch.save(E, os.path.join(data_dir, "final_energy.pt"))

# Plot and save energy profile
plt.figure(figsize=(10, 6))
plt.plot(E.detach().cpu().numpy())
plt.title("Energy along the path")
plt.xlabel("Path index")
plt.ylabel("Energy")
plt.savefig(os.path.join(data_dir, 'energy_profile.png'))
plt.close()

if not args.disable_logging:
    wandb.log({
        "Final_Energy_Profile": wandb.Image(os.path.join(data_dir, 'energy_profile.png')),
        "Final_Max_Energy": E.max().item(),
        "Final_Min_Energy": E.min().item(),
        "Final_Mean_Energy": E.mean().item()
    }) 