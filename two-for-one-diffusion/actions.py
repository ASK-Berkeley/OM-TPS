import torch


class TruncatedAction(torch.nn.Module):
    """OM Action with force divergence term ignored."""

    def __init__(self, force_func, dt, gamma, D=None):
        """
        Args:
            force_func: Force function
            dt: float, time step
            gamma: torch.tensor, friction coefficient
            D: float, diffusion coefficient
        """
        super(TruncatedAction, self).__init__()
        self.force_func = force_func
        self.dt = dt
        self.gamma = gamma.unsqueeze(0).unsqueeze(-1)

    def forward(
        self,
        path: torch.Tensor,
        forces: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args: path of shape [B, P, N, 3], forces of shape [B, P, N, 3]
        """

        path_term = torch.square((path[:, 1:] - path[:, :-1])) / (2 * self.dt)
        force_term = torch.square(forces) * (self.dt / (2 * self.gamma**2))

        # mask out padded indices
        if mask is None:
            mask = torch.ones((path_term.shape[2],)).bool()

        return (
            path_term[:, :, mask].sum(),
            force_term[:, :, mask].sum(),
            torch.tensor(0).to(torch.float32),
        )


class HutchinsonAction(torch.nn.Module):
    """
    OM Action with force divergence term included. Calculates force divergence using Hutchinson trick using
    random vector of -1s and 1s.
    """

    def __init__(self, force_func, dt, gamma, D, N=1):
        super(HutchinsonAction, self).__init__()
        self.dt = dt
        self.gamma = gamma.unsqueeze(0).unsqueeze(-1)  # shape of [1, n_atoms, 1]
        self.D = D
        self.N = N
        self.force_func = force_func

        def force_and_divergence(x: torch.tensor, forces: torch.tensor = None):
            result = 0
            if forces is None:
                path_batch_flattened = x.reshape(-1, x.shape[-2], x.shape[-1])
                batch_forces = self.force_func(path_batch_flattened)
                # then reshape back to the original shape
                forces = batch_forces.reshape(
                    x.shape[0],
                    x.shape[1],
                    x.shape[2],
                    x.shape[3],
                )

            for _ in range(self.N):
                # Generate random vector of -1s and 1s.
                v = torch.randint(0, 2, forces.shape, dtype=torch.float32) * 2 - 1
                v = v.to(x.device)
                # Calculate matrix vector product of Hessian and random vector
                (Av,) = torch.autograd.grad(
                    forces, x, grad_outputs=v, retain_graph=True, create_graph=True
                )
                # Make it a scalar. Minus is because we want the energy Hessian, which is the negative force Jacobian.
                result += -torch.sum(v * Av)

            return forces, result / N

        self.force_and_divergence = force_and_divergence

    def forward(
        self,
        path: torch.Tensor,
        forces: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """
        Args: path of shape [B, P, N, 3], forces of shape [B, P, N, 3]
        """
        num_atoms = path.shape[-2]

        f_n, laplace = self.force_and_divergence(
            path[:, :-1],
            forces,
        )

        path_term = torch.square((path[:, 1:] - path[:, :-1])) / (2 * self.dt)

        force_term = torch.square(f_n) * (self.dt / (2 * self.gamma**2))

        divergence_term = laplace * self.dt * self.D / self.gamma

        # mask out padded indices
        if mask is None:
            mask = torch.ones((path_term.shape[2],)).bool()
        return (
            path_term[:, :, mask].sum(),
            force_term[:, :, mask].sum(),
            divergence_term[:, mask].sum(),
        )
