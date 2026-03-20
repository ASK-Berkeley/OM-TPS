import torch
import numpy as np
from tqdm import tqdm
from molecule_reader import load_ala2_data
from utils import center_zero
import argparse
import os
from actions import PICOSEC2TIMEU, TIMEFACTOR, BOLTZMAN
from torch.func import vmap
parser = argparse.ArgumentParser(description="Langevin dynamics simulation")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
parser.add_argument("--precision", type=str, default="float32", help="Precision to use")
parser.add_argument("--dt", type=float, default=1.0, help="Time step in femtoseconds")
parser.add_argument("--gamma", type=float, default=10.0, help="Friction coefficient in 1/picoseconds")
parser.add_argument("--T", type=float, default=300.0, help="Temperature in Kelvin")
parser.add_argument("--n_steps", type=int, default=10000, help="Number of simulation steps")
parser.add_argument("--save_interval", type=int, default=10, help="Save every nth step")
parser.add_argument("--burn_in", type=int, default=1000, help="Number of burn-in steps")
parser.add_argument("--output_dir", type=str, default="langevin_trajectories", help="Output directory")
parser.add_argument("--start_from", type=str, default="reactants", help="Start from reactants or products")
parser.add_argument("--path_length", type=int, default=2000, help="Path length")

args = parser.parse_args()

reactants, products, sample_force, sample_forces_laplace, system = load_ala2_data(args)

# Set up Langevin parameters
dt =torch.tensor(args.dt / TIMEFACTOR, device=args.device, dtype=system.pos.dtype)  # Convert to picoseconds
gamma = torch.tensor(args.gamma / PICOSEC2TIMEU, device=args.device, dtype=system.pos.dtype)
T = args.T
M = system.M
zeta = args.gamma * M
D = torch.tensor(BOLTZMAN * args.T / zeta, device=args.device, dtype=system.pos.dtype)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize velocities
v = torch.randn_like(system.pos) * torch.sqrt(D / dt)

# Initialize storage for trajectories
n_save_steps = (args.n_steps - args.burn_in) // args.save_interval
trajectories = torch.zeros((n_save_steps, args.path_length, system.natoms, 3), 
                         device=args.device, dtype=system.pos.dtype)

print(system.pos.shape)

force_func = lambda x: vmap(sample_force.compute)(x)

# Run Langevin dynamics
save_idx = 0
with torch.no_grad():
    with tqdm(total=args.n_steps, desc="Running Langevin dynamics") as pbar:
        for step in range(args.n_steps):
            # Compute forces
            _, forces = force_func(system.pos)
            
            # Update positions and velocities
            # Deterministic part: forces/M - gamma*v
            # Stochastic part: sqrt(2D*dt) * noise
            v = v + (forces / M - gamma * v) * dt + torch.randn_like(v) * torch.sqrt(2 * D * dt)
            system.pos = system.pos + v * dt
            
            # Save trajectory after burn-in
            if step >= args.burn_in and step % args.save_interval == 0:
                trajectories[save_idx] = system.pos.detach().clone()
                save_idx += 1
                
            pbar.update(1)

# Save trajectories
torch.save(trajectories.cpu(), os.path.join(args.output_dir, f"trajectories{args.T}K.pt"))
print(f"Saved {n_save_steps} frames to {os.path.join(args.output_dir, f'trajectories{args.T}K.pt')}") 