import torch
import numpy as np
from torch.func import grad, grad_and_value, vmap


class SimpleMB:
    """
    Simple Muller-Brown Potential to test transition path optimization.
    """

    def __init__(self, device, n_in=2, barrier=1.0):
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

        self.total_potential = lambda X: torch.sum(self.U(X))

        self.force_func = lambda X: -grad_and_value(self.total_potential)(X)[0]

        self.to_(device)

    def to_(self, device):

        self.A = self.A.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.a = self.a.to(self.device)

        self.beta = self.beta.to(self.device)
        self.b = self.b.to(self.device)
        self.gamma = self.gamma.to(self.device)

        # self.initial_point = self.initial_point.to(device)
        # self.final_point = self.final_point.to(device)

    def U(self, X):
        """Simple potential that represents a transition avoiding a barrier.

        Args:
            x (Tensor): Position of the particle
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, requires_grad=True)

        if self.n_in == 2:
            if len(X.shape) == 1:
                x = X[0]
                y = X[1]
            else:
                x = X[:, 0]
                y = X[:, 1]
            u = self.U_split(x, y)
        else:
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

    def project2D(self, X):
        x = X[..., 0]
        y = X[..., 2]
        projected = torch.stack((x, y), dim=-1)
        return projected

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

    def laplace(self, X):
        """
        Compute the second derivatives of the potential at a given point X.
        I.e U_xx, U_yy. These will be added together in the action to form the laplacian.
        """
        assert self.n_in == 2
        x, y = torch.split(X, 1, dim=-1)

        U_xx = grad(
            lambda x, y: torch.sum(
                grad(lambda x, y: torch.sum(self.U_split(x, y)), argnums=(0, 1))(x, y)[
                    0
                ]
            ),
            argnums=0,
        )(x, y)
        U_yy = grad(
            lambda x, y: torch.sum(
                grad(lambda x, y: torch.sum(self.U_split(x, y)), argnums=(0, 1))(x, y)[
                    1
                ]
            ),
            argnums=1,
        )(x, y)

        return U_xx + U_yy

    def get_init_point(self, batch_size):

        basin1 = self.initial_point.repeat(batch_size, 1)
        basin2 = self.final_point.repeat(batch_size, 1)
        sim_init = torch.cat((basin1, basin2))
        return sim_init
