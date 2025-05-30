from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import torch
from functorch import grad


class MullerBrownPotential(Calculator):
    """
    Muller-Brown potential calculator - wrapper around ASE Calculator
    """

    implemented_properties = ["energy", "forces"]

    nolabel = True

    def __init__(self, device, n_in=2, barrier=1.0, **kwargs):

        Calculator.__init__(self, **kwargs)
        """
        Args:
            args: Arguments object
            n_in (int): Number of input dimensions
            barrier (float): Barrier height
        """
        self.device = device
        self.n_in = n_in

        self.barrier = barrier
        prefactor = 1e-2

        barrier = 10.0 * self.barrier
        self.A = barrier * torch.tensor([-1.73, -0.87, -1.47, 0.13])
        self.alpha = prefactor * torch.tensor([-0.39, -0.39, -2.54, 0.273])
        self.a = torch.tensor([48, 32, 24, 16])

        self.beta = prefactor * torch.tensor([0, 0, 4.30, 0.23])
        self.b = torch.tensor([8, 16, 32, 24])

        self.gamma = prefactor * torch.tensor([-3.91, -3.91, -2.54, 0.273])

        self.D = 0.0
        self.d = 0.0

        self.E = 0.0
        self.e = 0.0

        self.spring = 0.1

        if self.n_in == 2:
            self.initial_point = torch.tensor([23.0, 30.0])
            self.final_point = torch.tensor([40.0, 9.0])
        elif self.n_in == 5:
            self.initial_point = torch.tensor([23.0, 25.0, 30.0, 25.0, 25.0])
            self.final_point = torch.tensor([40.0, 25.0, 9.0, 25.0, 25.0])

        self.Lx, self.Hx = -0, 50
        self.Ly, self.Hy = 0, 50

        self.U_min, self.U_max = -2 * barrier, 1 * barrier

        self.A = self.A.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.a = self.a.to(self.device)

        self.beta = self.beta.to(self.device)
        self.b = self.b.to(self.device)
        self.gamma = self.gamma.to(self.device)

        # self.initial_point = self.initial_point.to(device)
        # self.final_point = self.final_point.to(device)

    def U_split(self, x, y):
        """Simple potential that represents a transition avoiding a barrier for the 2D case.

        Args:
            x (Tensor): Position coordinate x
            y (Tensor): Position coordinate y

        """

        u = (
            torch.sum(
                self.A
                * torch.exp(
                    self.alpha * (x.unsqueeze(-1) - self.a) ** 2
                    + self.beta
                    * (x.unsqueeze(-1) - self.a)
                    * (y.unsqueeze(-1) - self.b)
                    + self.gamma * (y.unsqueeze(-1) - self.b) ** 2
                ),
                axis=-1,
            )
            + self.D * (x - self.d) ** 3
            - self.E * (y - self.e) ** 3
        )
        return u

    def get_energy(self, X):
        """Simple potential that represents a transition avoiding a barrier.

        Args:
            x (Tensor): Position of the particle
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, requires_grad=True)

        X = X.to(self.device)

        if self.n_in == 2:  # 2D case
            if len(X.shape) == 1:
                x = X[0]
                y = X[1]
            else:
                x = X[:, 0]
                y = X[:, 1]
            u = self.U_split(x, y)
        else:  # 5D case
            if len(X.shape) == 1:
                x1, x2, x3, x4, x5 = torch.split(X, 1, dim=0)
            else:
                x1, x2, x3, x4, x5 = torch.split(X, 1, dim=-1)

            x = x1
            y = x3
            u = self.U_split(x, y) + self.spring * (
                (x2 - 25.0) ** 2 + (x4 - 25.0) ** 2 + (x5 - 25.0) ** 2
            )
        return u

    def get_force(self, X):
        return -grad(self.get_energy)(X)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        assert natoms == 1

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))

        energies[0] = (
            self.get_energy(torch.tensor(self.atoms[0].position)).cpu().numpy()
        )
        forces[0] = self.get_force(torch.tensor(self.atoms[0].position)).cpu().numpy()

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["energies"] = energies

        self.results["free_energy"] = energy
        self.results["forces"] = forces


class DiffusionModel_MullerBrownPotential(MullerBrownPotential):
    """
    Muller-Brown potential calculator where the forces are computed with a trained diffusion model.
    Only forces are supported, not energies.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, t, device, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.t = t
        self.device = device
        self.model.to(device)
        self.force_func = lambda x: self.model.force_func(x[:, :2].to(device), t)

    def get_energy(self, X):
        # Dummy function to satisfy the Calculator interface
        return torch.tensor(0.0).to(self.device)

    def get_force(self, X):
        with torch.no_grad():
            if len(X.shape) == 1:
                X = X.unsqueeze(0).to(torch.float32)
            result = torch.zeros_like(X)
            result[:, :2] = self.force_func(X)
        return result
