import torch
import torch.nn.functional as F
import torch.nn as nn


class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):

        # Delegate the initialization call to the parent class to define the
        # nn.Module properly
        super(GAT_gate, self).__init__()

        # Applies a linear transformation to the incoming data: y=xA^T+b
        self.W = nn.Linear(n_in_feature, n_out_feature)

        # Initializes the input atom weights
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))

        # Applies a linear transformation to the data: y=xA^T+b
        self.gate = nn.Linear(n_out_feature * 2, 1)

        # Set negative values to 0 applies a LeakyReLU to positives values
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):

        # Calculates the result of the layer in forward direction. X is the
        # input of the layer and the retval is the output.

        h = self.W(x)
        e = torch.einsum("ijl,ikl->ijk", (torch.matmul(h, self.A), h))
        e = e + e.permute((0, 2, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention * adj
        h_prime = F.relu(torch.einsum("aij,ajk->aik", (attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(
            1, 1, x.size(-1)
        )
        retval = coeff * x + (1 - coeff) * h_prime
        return retval
