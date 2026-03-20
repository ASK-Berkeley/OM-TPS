"""
includes specific constants for ala2
"""

import torch

from torch.func import vmap
from torch.func import vjp as vjpfunc

# Constants to facilitate conversion
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR

class HutchinsonAction(torch.nn.Module):

    """
    Action that is the same as S2 Action but calculates laplacian using Hutchinson trick using 
    random vector of -1s and 1s.

    Units expected:
        - dt - femtoseconds
        - gamma - 1/picoseconds
        - M - atomic units (amu)
        - T - Kelvin

    Energy is then in kcal/mol
    """
    def __init__(self, force_func, dt=1.0, gamma=1.0, laplace_forces=None, M=None, T=None, N=1):
        super(HutchinsonAction, self).__init__()
        self.dt = dt / TIMEFACTOR
        self.gamma = gamma / PICOSEC2TIMEU
        self.N = N
        self.M = M
        self.T = T
        self.laplace_forces = laplace_forces
        self.force_func = force_func
        #Mass weighted 
        self.zeta = self.gamma * M
        self.D = BOLTZMAN * self.T / self.zeta
        
        def force_and_laplace(x: torch.tensor):
            res = 0

            energies, forces = self.force_func(x)
            #energies_laplace, forces_for_laplace = self.laplace_forces(x)

            for _ in range(self.N):
                # Generate random vector of -1s and 1s.
                v = torch.randint(0, 2, forces.shape, dtype=torch.float32) * 2 - 1
                v = v.to(x.device)

                # Calculate matrix vector product of Hessian and random vector
                Av, = torch.autograd.grad(forces, x, grad_outputs=v, retain_graph=True, create_graph=True)

                # Minus is from the sign of the forces
                res += - v*Av
            
            return forces, res / N, energies

        self.force_and_laplace = force_and_laplace

    
    def forward(self, path: torch.Tensor, forces: torch.Tensor = None):

        x_n = path[:-1]
        x_np = path[1:]
        f_n, laplace, energy = self.force_and_laplace(x_n)

        # Make sure the terms are [batch, 1] as is the third term
        first_term = torch.square((x_np - x_n)) * (self.zeta / 4 / self.dt)

        second_term = torch.square(f_n) * (
            self.dt / 4 / self.zeta
        )

        third_term = laplace * self.dt * BOLTZMAN * self.T / self.zeta / torch.tensor(2.0)
        assert(first_term.shape == second_term.shape and second_term.shape == third_term.shape)

        # Result is the action and is expected to have shape [batch, 1]
        return first_term.sum(axis = (1,2)).mean(), second_term.sum(axis = (1,2)).mean(), third_term.sum(axis = (1,2)).mean(), energy.sum()
    
class FDAction(torch.nn.Module):

    """
    Action that is the same as S2 Action but calculates laplacian using finite differences. Does not work for two reasons on Chignolin. 
    A: The laplace we are approximating is stll constant
    B: We are just displacing the entire molecule which must result in the same forces...

    Units expected:
        - dt - femtoseconds
        - gamma - 1/picoseconds
        - M - atomic units (amu)
        - T - Kelvin

    Energy is then in kcal/mol
    """
    def __init__(self, force_func, dt=1.0, gamma=1.0, laplace_forces=None, M=None, T=None, N=1):
        super(FDAction, self).__init__()
        self.dt = dt / TIMEFACTOR
        self.gamma = gamma / PICOSEC2TIMEU
        self.N = N
        self.M = M
        self.T = T
        self.laplace_forces = laplace_forces
        self.force_func = force_func
        #Mass weighted 
        self.zeta = self.gamma * M
        self.D = BOLTZMAN * self.T / self.zeta
        

    def forward(self, path: torch.Tensor, forces: torch.Tensor = None):

        x_n = path[:-1]
        x_np = path[1:]
        energy, all_force = self.force_func(path)

        f_n = all_force[:-1]
        f_np = all_force[1:]

        #print(x_n)

        h = torch.tensor(1e-5).to(x_n.device)
        e1 = torch.tensor([1,0,0]).reshape(1,1,3).to(x_n.device)
        e2 = torch.tensor([0,1,0]).reshape(1,1,3).to(x_n.device)
        e3 = torch.tensor([0,0,1]).reshape(1,1,3).to(x_n.device)
        #df_dx = (self.force_func(x_n + h*e1)[1] - self.force_func(x_n - h*e1)[1])/2/h
        #df_dy = (self.force_func(x_n + h*e2)[1] - self.force_func(x_n - h*e2)[1])/2/h
        #df_dz = (self.force_func(x_n + h*e3)[1] - self.force_func(x_n - h*e3)[1])/2/h
        df_dx = (self.force_func(x_n + h*e1)[1] - f_n)/h
        df_dy = (self.force_func(x_n + h*e2)[1] - f_n)/h
        df_dz = (self.force_func(x_n + h*e3)[1] - f_n)/h
        
        #print(x_np-x_n)
        #Sign from the force sign
        laplace = -(df_dx +  df_dy + df_dz)

        # Make sure the terms are [batch, 1] as is the third term
        first_term = torch.square((x_np - x_n)) * (self.zeta / 4 / self.dt)

        second_term = torch.square(f_n) * (
            self.dt / 4 / self.zeta
        )
        
        third_term = laplace * self.dt * BOLTZMAN * self.T / self.zeta / torch.tensor(2.0)
        #print(first_term, second_term, third_term)
        
        #print(first_term.shape, second_term.shape, third_term.shape)
        assert(first_term.shape == second_term.shape and second_term.shape == third_term.shape)
        #print(first_term.sum(), second_term.sum(), third_term.sum())
        # Result is the action and is expected to have shape [batch, 1]
        return first_term.sum(axis = (1,2)).mean(), second_term.sum(axis = (1,2)).mean(), third_term.sum(axis = (1,2)).mean(), energy.sum()


class ZeroTAction(torch.nn.Module):

    """
    Action that is the same as S2 Action but calculates laplacian using finite differences. Does not work for two reasons on Chignolin. 
    A: The laplace we are approximating is stll constant
    B: We are just displacing the entire molecule which must result in the same forces...

    Units expected:
        - dt - femtoseconds
        - gamma - 1/picoseconds
        - M - atomic units (amu)
        - T - Kelvin

    Energy is then in kcal/mol
    """
    def __init__(self, force_func, dt=1.0, gamma=1.0, laplace_forces=None, M=None, T=None, N=1):
        super(ZeroTAction, self).__init__()
        self.dt = dt / TIMEFACTOR
        self.gamma = gamma / PICOSEC2TIMEU
        self.N = N
        self.M = M
        self.T = T
        self.laplace_forces = laplace_forces
        self.force_func = force_func
        #Mass weighted 
        self.zeta = self.gamma * M
        self.D = BOLTZMAN * self.T / self.zeta
        

    def forward(self, path: torch.Tensor, forces: torch.Tensor = None):

        x_n = path[:-1]
        x_np = path[1:]
        energy, f_n = self.force_func(x_n)
        
        # Make sure the terms are [batch, 1] as is the third term
        first_term = torch.square((x_np - x_n)) * (self.zeta / 4 / self.dt)

        second_term = torch.square(f_n) * (
            self.dt / 4 / self.zeta
        )

        assert(first_term.shape == second_term.shape)

        return first_term.sum(axis = (1,2)).mean(), second_term.sum(axis = (1,2)).mean(), torch.tensor(0.0), energy
    


class ExpandingAction(torch.nn.Module):
    """
    Simple action with variable parameters used to generate initial guess.
    """

    def __init__(self, force_func):
        super(ExpandingAction, self).__init__()
        self.force_func = force_func

    
    def forward(self, path: torch.Tensor, k = None):

        x_n = path[:-1]
        x_np = path[1:]
        energies, f_n = self.force_func(x_n)

        first_term = torch.sum(torch.square((x_np - x_n)) * (k / 4), axis=-1)
        second_term = torch.sum(torch.square(f_n) * (
            1 / 4 / k
        ), axis=-1)

        return first_term.sum(), second_term.sum(), energies.sum()
    
    