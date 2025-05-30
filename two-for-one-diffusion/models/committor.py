from torch_geometric.nn.models import GCN
import torch


class CommittorNN(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, h=None, t=None):
        # model returns per-residue contribution to the committor
        if h is not None and t is not None:
            per_atom_contrib = self.model(x, h, t, return_energy=True)
            # sum over all atoms and pass through sigmoid to get committor probability
            committor_prob = self.sigmoid(per_atom_contrib.sum(dim=(-2, -1)))
        else:
            committor_prob = self.sigmoid(self.model(x, return_energy=True))

        return committor_prob
