import torch
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import argparse
from pathlib import Path

from utils import get_psi_angle, get_phi_angle

def main():
    parser = argparse.ArgumentParser(description="Analyze Ramachandran angles from trajectory")
    parser.add_argument("--trajectory", type=str, default="langevin_trajectories/trajectories1000.0K.pt", help="Path to trajectory file")
    parser.add_argument("--output", type=str, default="ramachandran.png", help="Output plot filename")
    args = parser.parse_args()

    # Load trajectory
    print("Loading trajectory...")
    trajectories = torch.load(args.trajectory)  

    print(trajectories.shape)  
    
    # Calculate dihedral angles
    print("Calculating dihedral angles...")
    phi_angle = get_phi_angle(trajectories).flatten()
    psi_angle = get_psi_angle(trajectories).flatten()

    # Load and calculate angles for all trajectories
    trajectories_dict = {}
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Load final trajectory
    final_traj = torch.load("result_data/expert-night-11/final_unwrapped.pt")
    trajectories_dict['final'] = {
        'phi': get_phi_angle(final_traj),
        'psi': get_psi_angle(final_traj)
    }
    print("Final trajectory shape:", final_traj.shape)
    
    # Load intermediate trajectories
    for i in range(7):
        traj = torch.load(f"result_data/expert-night-11/unwrapped_{i}.pt")
        trajectories_dict[f'step_{i}'] = {
            'phi': get_phi_angle(traj),
            'psi': get_psi_angle(traj)
        }
    print(f"Trajectory {i} shape:", traj.shape)

    plt.figure(figsize=(8, 6))
    
    # Plot density from Langevin trajectories
    plt.hexbin(phi_angle.detach().cpu(), psi_angle.detach().cpu(), 
               bins='log', cmap='viridis', gridsize=40, mincnt=1)
    plt.colorbar(label='log10(N)')
    
    # Plot all trajectories
    for idx, (name, angles) in enumerate(trajectories_dict.items()):
        plt.scatter(angles['phi'].detach().cpu(), angles['psi'].detach().cpu(),
                   c=colors[idx], alpha=0.5, s=10, 
                   label="Final path" if name == "final" else f"Step {name.split('_')[1]} path")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('φ (Phi)')
    plt.ylabel('ψ (Psi)')
    plt.title('Ramachandran Plot')
    
    # Set axis limits to show full -pi to pi range
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 