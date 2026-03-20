import argparse
from molecule_reader import load_ala2_data
import torch
from tqdm import tqdm
import os
from torch.func import vmap
def optimize_endpoints():
    # Set up arguments
    parser = argparse.ArgumentParser(description="Optimize ala2 endpoints")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--precision", type=str, default="float32", help="Precision to use")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimization steps")
    parser.add_argument("--path_length", type=int, default=2, help="Path length")
    args = parser.parse_args()

    # Load data
    args.path_length = 2  # We only need endpoints
    reactants, products, sample_force, sample_forces_laplace, system = load_ala2_data(args)

    v_forces = vmap(sample_force.compute)

    # Get initial coordinates
    print(reactants.coords.shape)
    initial_coords = torch.cat([
        torch.tensor(reactants.coords),
        torch.tensor(products.coords)
    ], dim=-1).to(args.device)
    initial_coords = initial_coords.permute(2,0,1)
    initial_coords.requires_grad = True

    # Set up optimizer
    optimizer = torch.optim.LBFGS(
        [initial_coords],
        history_size=10,
        tolerance_change=1e-6,
        max_iter=args.steps,
        line_search_fn="strong_wolfe"
    )

    # Optimization loop
    energies = []
    with tqdm(total=args.steps, desc="Optimizing endpoints") as pbar:
        def closure():
            optimizer.zero_grad()
            energy, forces = v_forces(initial_coords)
            initial_coords.grad = -forces
            energies.append(energy.sum().item())
            pbar.set_postfix(energy=f"{energy.sum().item():.4f}")
            pbar.update(1)
            return energy.sum()

        optimizer.step(closure)

    # Save optimized coordinates
    os.makedirs("data", exist_ok=True)
    optimized_coords = initial_coords.detach().cpu()
    
    # Save reactants and products separately
    print(optimized_coords.shape)
    print(optimized_coords[0].shape)
    reactants.coords = optimized_coords[0].unsqueeze(-1).numpy()
    products.coords = optimized_coords[1].unsqueeze(-1).numpy()
    
    reactants.write("data/reactants_opt.coor")
    products.write("data/product_opt.coor")
    
    print(f"Optimized endpoints saved to data/reactants_opt.coor and data/product_opt.coor")
    print(f"Final energy: {energies[-1]:.4f} kcal/mol")

if __name__ == "__main__":
    optimize_endpoints()
