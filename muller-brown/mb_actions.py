import torch
from torch.func import vmap, grad, vjp


class TruncatedAction(torch.nn.Module):
    """OM Action with Hessian term ignored."""

    def __init__(self, force_func, dt, gamma, laplace_func=None, D=None):
        """
        Args:
            force_func: Force function
            dt: float, time step
            gamma: float, diffusion coefficient
            D: float, diffusion coefficient
        """
        super(TruncatedAction, self).__init__()
        self.force_func = force_func
        self.dt = dt
        self.gamma = gamma

    def forward(self, path: torch.Tensor):
        """
        Args: path of shape [L, 2] where L is path length
        """
        path_term = torch.square((path[1:] - path[:-1])) / (2 * self.dt)
        force_term = torch.square(self.force_func(path[:-1])) * (
            self.dt / (2 * self.gamma**2)
        )
        return (path_term + force_term).sum()


class S2Action(torch.nn.Module):
    """OM Action with Hessian calculated using autodiff - Hessian is exact."""

    def __init__(self, force_func=None, laplace_func=None, dt=1.0, gamma=1.0, D=1.0):
        """
        Args:
            force_func: Force function
            laplace_func: Laplace function
            dt: float, time step
            gamma: float, diffusion coefficient
            D: float, diffusion coefficient
        """
        super().__init__()
        self.force_func = vmap(force_func)
        self.laplace_func = laplace_func
        self.dt = dt
        self.gamma = gamma
        self.D = D

    def forward(self, path: torch.Tensor):
        """
        Args: path of shape [L, 2] where L is path length
        """

        path_term = torch.square((path[1:] - path[:-1])) / (2 * self.dt)
        force_term = torch.square(self.force_func(path[:-1])) * (
            self.dt / (2 * self.gamma**2)
        )

        hessian_term = self.laplace_func(path[:-1]) * (self.dt * self.D / self.gamma)

        result = torch.sum(path_term) + torch.sum(force_term) - torch.sum(hessian_term)
        return result


class HutchinsonAction(torch.nn.Module):
    """
    Action that is the same as S2 Action but calculates laplacian using Hutchinson trick using
    random vector of -1s and 1s.
    """

    def __init__(
        self,
        dt,
        gamma,
        D,
        N=1,
        sample_force_func=None,
        laplace_func=None,
        force_func=None,
        diffusion_model=False,
    ):
        super(HutchinsonAction, self).__init__()

        sample_force_func = (
            force_func if sample_force_func is None else sample_force_func
        )

        self.sample_force_func = sample_force_func
        self.dt = dt
        self.gamma = gamma
        self.D = D
        self.N = N
        self.true_laplace = laplace_func
        self.force_func = vmap(sample_force_func)

        def force_and_laplace(x: torch.tensor):
            res = 0

            if diffusion_model:
                x = x.unsqueeze(0)

            forces, vjp_func = vjp(self.sample_force_func, x)

            for _ in range(self.N):
                # Generate random vector of -1s and 1s.
                shape = x.shape
                v = torch.randint(0, 2, shape, dtype=torch.float32) * 2 - 1
                v = v.to(x.device)
                # Calculate matrix vector product of Hessian and random vector
                (Av,) = vjp_func(v)

                # Make it a scalar. Minus is from the sign of the forces
                res += -torch.sum(v * Av)

            return forces, res / N

        self.force_and_laplace = vmap(force_and_laplace, randomness="different")

    def forward(self, path: torch.Tensor):
        f_n, laplace = self.force_and_laplace(path[:-1])

        path_term = torch.square((path[1:] - path[:-1])) / (2 * self.dt)
        force_term = torch.square(f_n) * (self.dt / (2 * self.gamma**2))
        hessian_term = laplace.unsqueeze(-1) * (self.dt * self.D / self.gamma)

        result = torch.sum(path_term + force_term - hessian_term)
        return result
