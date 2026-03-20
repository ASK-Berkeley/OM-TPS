from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from datasets.ala2.sample_forces import SampleForces
from datasets.ala2.system import System
from torch.func import vmap, jvp, vjp, grad
import torch

import numpy as np

import os

class Args:
    batch_size = None
    precision = None
    device = None


def load_ala2_data(args):

    # Lets have even path length
    assert args.path_length % 2 == 0

    testdir = "datasets/ala2/data/"

    reactants = Molecule(os.path.join(testdir, "ala2nw.prmtop"))  # Reading the system topology
    reactants.read(os.path.join(testdir, "reactants_opt.coor"))  # Reading the initial simulation coordinates
    reactants.read(os.path.join(testdir, "input.xsc"))  # Reading the box dimensions
    #reactants.center(loc=(10.0,10.0,10.0))

    products = Molecule(os.path.join(testdir, "ala2nw.prmtop"))  # Reading the system topology
    products.read(os.path.join(testdir, "products_opt.coor"))  # Reading the initial simulation coordinates
    products.read(os.path.join(testdir, "input.xsc"))  # Reading the box dimensions
    #products.center(loc=(10.0,10.0,10.0))

    precision = torch.float32 if args.precision=="float32" else torch.float64


    system = System(reactants.numAtoms, nreplicas=args.path_length, precision=precision, device=args.device)
    ff = ForceField.create(reactants, os.path.join(testdir, "ala2nw.prmtop"))
    parameters = Parameters(ff, reactants, precision=precision, device=args.device)
    parameters.mapped_atom_types = parameters.mapped_atom_types.to(args.device)
    sample_forces = SampleForces(parameters, system.box, terms=SampleForces.terms)
    sample_forces_laplace = SampleForces(parameters, system.box, terms=SampleForces.laplace_terms)
    system.M = sample_forces.par.masses.unsqueeze(0)


   
    r_coords = torch.tensor(reactants.coords).repeat(1,1,args.path_length//2)
    p_coords = torch.tensor(products.coords).repeat(1,1,args.path_length//2)
    init_coordinates = torch.cat((r_coords,p_coords),dim=-1)
    init_coordinates_batchfirst = torch.permute(init_coordinates.detach(),(2,0,1))
    system.set_positions(init_coordinates.detach())
    system.set_box(reactants.box)
    system.M = sample_forces.par.masses.unsqueeze(0)
    #print(sample_forces.compute(torch.tensor(reactants.coords)))
    if system.pos.dtype != args.precision:
        system.pos = system.pos.to(dtype=precision)
    e, example_forces = vmap(sample_forces.compute)(system.pos)
    #print(example_forces[0,0,0])
    #sample_grad = torch.autograd.grad(example_forces[0,0,0], system.pos)
    #print(sample_grad)
    #dasda

    return reactants, products, sample_forces, sample_forces_laplace, system


def load_trp_cage_data(args):

    # Lets have even path length
    assert args.path_length % 2 == 0

    testdir = "data/trp_cage/"

    reactants = Molecule(os.path.join(testdir, "trp_unfold.prmtop"))  # Reading the system topology
    reactants.read(os.path.join(testdir, "trp_unfold_coor.crd"))  # Reading the initial simulation coordinates
    reactants.read(os.path.join(testdir, "input.xsc"))  # Reading the box dimensions
    #reactants.center(loc=(10.0,10.0,10.0))

    products = Molecule(os.path.join(testdir, "trp_folded.prmtop"))  # Reading the system topology
    products.read(os.path.join(testdir, "trp_folded.crd"))  # Reading the initial simulation coordinates
    products.read(os.path.join(testdir, "input.xsc"))  # Reading the box dimensions
    #products.center(loc=(10.0,10.0,10.0))

    precision = torch.float32 if args.precision=="float32" else torch.float64


    system = System(reactants.numAtoms, nreplicas=args.path_length, precision=precision, device=args.device)
    ff = ForceField.create(reactants, os.path.join(testdir, "trp_folded.prmtop"))
    parameters = Parameters(ff, reactants, precision=precision, device=args.device)
    parameters.mapped_atom_types = parameters.mapped_atom_types.to(args.device)
    sample_forces = SampleForces(parameters, system.box, terms=SampleForces.terms)
    sample_forces_laplace = SampleForces(parameters, system.box, terms=SampleForces.laplace_terms)
    system.M = sample_forces.par.masses.unsqueeze(0)


   
    r_coords = torch.tensor(reactants.coords).repeat(1,1,args.path_length//2)
    p_coords = torch.tensor(products.coords).repeat(1,1,args.path_length//2)
    init_coordinates = torch.cat((r_coords,p_coords),dim=-1)
    init_coordinates_batchfirst = torch.permute(init_coordinates.detach(),(2,0,1))
    #system.set_positions(init_coordinates.detach())
    system.set_positions(torch.load(os.path.join(testdir, "trp_cage_opt_endpoints.pt")).permute(1,2,0))
    system.set_box(reactants.box)
    system.M = sample_forces.par.masses.unsqueeze(0)
    #print(sample_forces.compute(torch.tensor(reactants.coords)))
    if system.pos.dtype != args.precision:
        system.pos = system.pos.to(dtype=precision)
    e, example_forces = vmap(sample_forces.compute)(system.pos)
    #print(system.pos)
    #sample_grad = torch.autograd.grad(example_forces[0,0,0], system.pos)
    #print(sample_grad)
    print(example_forces[0].mean(axis=1))
    #dasda

    return reactants, products, sample_forces, sample_forces_laplace, system

args = Args()
args.device = "cpu"
args.path_length = 20
args.initial_guess_method = "expand"
args.precision = "float32"
#load_chignolin_data(args)
"""
reactants, products, sample_forces, sample_forces_laplace, system = load_trp_cage_data(args)
step_size = 1e-6

def closure():
    lbfgs.zero_grad()
    e, f = vmap(sample_forces.compute)(system.pos)
    system.pos.grad = -f
    print(e.sum())
    return e.sum()


lbfgs = torch.optim.LBFGS([system.pos],
                    history_size=10, 
                    max_iter=4, 
                    line_search_fn="strong_wolfe")

for i in range(100):
    print(i)
    lbfgs.step(closure)

torch.save(system.pos.detach().cpu(), "trp_cage_opt_endpoints.pt")
"""