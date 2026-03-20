import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

#Get metric - How far is the descriptor from actual target. Account for periodicity etc. Distances are non-periodic. Also acount for rescaling. 
#Periodicity from -pi to pi
def metric_from(R, C):
    d_uncor = R - C
    d_part = torch.where(d_uncor > np.pi, d_uncor-2*np.pi, d_uncor)
    d = torch.where(d_part <= -np.pi, d_part+2*np.pi, d_part)
    return d

def dont_materialize(x, y, points, sigma, height):
    centers = torch.cat([x,y], dim=-1)
    points = points

    for i in range(centers.shape[0]):
        
    print(centers.shape, points.shape)
    """Compute 2D Gaussian function"""
    distances = metric_from(points, centers)

    return torch.sum(height * np.exp(-0.5 * (torch.sum(distances**2, dim=-1)) / sigma**2), axis=0)

def gaussian_2d(x, y, points, sigma, height):

    print(x.shape, y.shape)
    centers = torch.cat([x,y], dim=-1).unsqueeze(0)
    points = points.unsqueeze(1)
    print(centers.shape, points.shape)
    """Compute 2D Gaussian function"""
    distances = metric_from(points, centers)

    return torch.sum(height * np.exp(-0.5 * (torch.sum(distances**2, dim=-1)) / sigma**2), axis=0)

def plot_hills(hills_path, hill_width, hill_height, output_path=None):
    # Load hills data
    hills = torch.load(hills_path)


    print(hills.shape)
    # Create a grid for visualization
    x = np.linspace(-np.pi, np.pi, 200)  # phi angle
    y = np.linspace(-np.pi, np.pi, 200)  # psi angle
    X, Y = np.meshgrid(x, y, indexing='xy')
    X = torch.tensor(X.reshape(-1, 1))
    Y = torch.tensor(Y.reshape(-1, 1))
    
    # Initialize the total bias potential
    total_bias = np.zeros_like(X)
    
    # Add each hill to the total bias
    total_bias = gaussian_2d(X, Y, hills, hill_width, hill_height)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    print(X.shape, Y.shape, total_bias.shape)
    plt.contourf(X.reshape(200,200), Y.reshape(200,200), total_bias.reshape(200, 200), levels=50, cmap='viridis')
    plt.colorbar(label='Bias Potential')
    
    # Plot individual hill centers
    plt.scatter(hills[:, 0], hills[:, 1], c='red', s=20, alpha=0.5, label='Hill Centers')
    
    plt.xlabel('φ (rad)')
    plt.ylabel('ψ (rad)')
    plt.title('Metadynamics Hills')
    plt.grid(True)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metadynamics hills")
    parser.add_argument("--hills_path", type=str, default="metadynamics_trajectories/hills.pt", help="Path to hills.pt file")
    parser.add_argument("--hill_width", type=float, required=True, help="Width of Gaussian hills")
    parser.add_argument("--hill_height", type=float, required=True, help="Height of Gaussian hills")
    parser.add_argument("--output", type=str, default="metadynamics_trajectories/hills.png", help="Output path for the plot (optional)")
    
    args = parser.parse_args()
    plot_hills(args.hills_path, args.hill_width, args.hill_height, args.output) 