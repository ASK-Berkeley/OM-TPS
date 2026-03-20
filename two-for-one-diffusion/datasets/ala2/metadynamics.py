import torch
import numpy as np
from tqdm import tqdm
from molecule_reader import load_ala2_data
from utils import center_zero, get_phi_angle, get_psi_angle
import argparse
import os
from actions import PICOSEC2TIMEU, TIMEFACTOR, BOLTZMAN
from torch.func import vmap

#Get metric - How far is the descriptor from actual target. Account for periodicity etc. Distances are non-periodic. Also acount for rescaling. 
#Periodicity from -pi to pi
def metric_from(R, C):
    d_uncor = R - C
    d_part = torch.where(d_uncor > np.pi, d_uncor-2*np.pi, d_uncor)
    d = torch.where(d_part <= -np.pi, d_part+2*np.pi, d_part)
    return d

def gaussian_bias(points, centers, sigma, height):
    points = points.unsqueeze(1)
    centers = centers.unsqueeze(0)
    """Compute 2D Gaussian function"""
    distances = metric_from(points, centers)
    return torch.sum(height * torch.exp(-0.5 * 
        (torch.sum((distances)**2, dim=-1)) / sigma**2), axis=0)


parser = argparse.ArgumentParser(description="Metadynamics simulation")
parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
parser.add_argument("--precision", type=str, default="float32", help="Precision to use")
parser.add_argument("--dt", type=float, default=1.0, help="Time step in femtoseconds")
parser.add_argument("--gamma", type=float, default=10.0, help="Friction coefficient in 1/picoseconds")
parser.add_argument("--T", type=float, default=300.0, help="Temperature in Kelvin")
parser.add_argument("--n_steps", type=int, default=10000, help="Number of simulation steps")
parser.add_argument("--save_interval", type=int, default=10, help="Save every nth step")
parser.add_argument("--burn_in", type=int, default=1000, help="Number of burn-in steps")
parser.add_argument("--output_dir", type=str, default="metadynamics_trajectories", help="Output directory")
parser.add_argument("--start_from", type=str, default="reactants", help="Start from reactants or products")
parser.add_argument("--path_length", type=int, default=2, help="Path length")
# Metadynamics specific parameters
parser.add_argument("--hill_height", type=float, default=0.3, help="Height of Gaussian hills")
parser.add_argument("--hill_width", type=float, default=0.3, help="Width of Gaussian hills")
parser.add_argument("--hill_frequency", type=int, default=100, help="Add new hill every N steps")
parser.add_argument("--cv_type", type=str, default="dihedral", help="Type of collective variable")
parser.add_argument("--checkpoint_interval", type=int, default=100_000, help="Checkpoint every N steps")
args = parser.parse_args()

reactants, products, sample_force, sample_forces_laplace, system = load_ala2_data(args)

# Set up Langevin parameters
dt = torch.tensor(args.dt / TIMEFACTOR, device=args.device, dtype=system.pos.dtype)
gamma = torch.tensor(args.gamma / PICOSEC2TIMEU, device=args.device, dtype=system.pos.dtype)
T = args.T
M = system.M
zeta = args.gamma * M
D = BOLTZMAN * args.T / zeta

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize velocities
v = torch.randn_like(system.pos) * torch.sqrt(D / dt)

# Initialize storage for trajectories and hills
n_save_steps = (args.n_steps - args.burn_in) // args.save_interval
trajectories = torch.zeros((n_save_steps, args.path_length, system.natoms, 3), 
                         device=args.device, dtype=system.pos.dtype)
hills = []  # Store (position, height) pairs for Gaussian hills

def compute_cv(pos):
    if args.cv_type == "dihedral":
        phi = get_phi_angle(pos.unsqueeze(0)).squeeze()
        psi = get_psi_angle(pos.unsqueeze(0)).squeeze()
        return torch.stack([phi, psi])
    else:
        raise ValueError(f"Unknown CV type: {args.cv_type}")

def get_bias_force(x, hills, hill_width):
    # If no hills, return zero force
    if len(hills) == 0:
        return torch.zeros_like(x)

    with torch.enable_grad():
    # Enable gradient computation
        x.requires_grad_(True)

        # Compute CVs for current position
        cv = compute_cv(x)

        # Compute total bias potential from all Gaussian hills

        total_bias = gaussian_bias(cv, torch.cat(hills, axis=0), hill_width, args.hill_height).sum()
        
        
        # Get gradient of total bias with respect to coordinates
        bias_force = -torch.autograd.grad(total_bias, x, allow_unused=True)[0]
    
    # Clean up
    #x.requires_grad_(False)
    
    return bias_force


force_func = lambda x: vmap(sample_force.compute)(x)

# Run Metadynamics
save_idx = 0
with torch.no_grad():
    with tqdm(total=args.n_steps, desc="Running Metadynamics") as pbar:
        for step in range(args.n_steps):
            # Compute physical forces
            _, forces = force_func(system.pos)
            
            # Add biasing forces from all hills at once
            bias_forces = get_bias_force(system.pos, hills, args.hill_width)
            
            # Update positions and velocities with both physical and biasing forces
            v = v + ((forces + bias_forces) / M - gamma * v) * dt + torch.randn_like(v) * torch.sqrt(2 * D * dt)
            system.pos = system.pos + v * dt
            
            # Add new hill if needed
            if step % args.hill_frequency == 0:
                # Compute collective variable
                cv = compute_cv(system.pos)
                hills.append((cv.detach().clone()))

            if step % args.checkpoint_interval == 0:
                torch.save(torch.cat(hills, dim=0).cpu(), os.path.join(args.output_dir, f"hills{step}.pt"))
                print(f"Saved {len(hills)} hills to {os.path.join(args.output_dir, f'hills{step}.pt')}") 
            
                
            pbar.update(1)
            pbar.set_postfix(cost=f"NO. hills: {len(hills)}")

# Save trajectories and hills
torch.save(torch.cat(hills, dim=0).cpu(), os.path.join(args.output_dir, "hills.pt"))
print(f"Saved {len(hills)} hills to {os.path.join(args.output_dir, 'hills.pt')}") 