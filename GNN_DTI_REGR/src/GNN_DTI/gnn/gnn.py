import torch
import torch.nn.functional as F
import torch.nn as nn
from GNN_DTI.gnn.layers import GAT_gate

N_ATOM_FEATURES = 28


class gnn(torch.nn.Module):

    # In this class the structure of the GNN is defined. It consists of n GAT
    # layers (for matrices A1 and A2) that will tell us the difference between
    # a binding pose and the separate structure and they are followed by m
    # fully connected layers (multi-layer perceptrons) that will help us
    # determine the binding affinity. These are connected to each other with a
    # ReLU function, except for the last one which uses a sigmoid

    def __init__(self, args):

        # The GNN is initialized and the features of the different layers are
        # extracted from the passed arguments
        super(gnn, self).__init__()
        num_graph_layer = args.num_graph_layer
        dim_graph_layer = args.dim_graph_layer
        num_FC_layer = args.num_FC_layer
        dim_FC_layer = args.dim_FC_layer
        self.dropout_rate = args.dropout_rate

        # Define a list with n positions (number of GAT layers) with the
        # dimension of each of them
        self.layers1 = [dim_graph_layer for i in range(num_graph_layer + 1)]

        # Initialize the GAT layers found in the layers.py file of the package
        # with the indicated dimensions
        self.gconv1 = nn.ModuleList(
            [
                GAT_gate(self.layers1[i], self.layers1[i + 1])
                for i in range(len(self.layers1) - 1)
            ]
        )

        # Initialize the fully connected layers found in the Pytorch
        # module with the indicated dimensions
        self.FC = nn.ModuleList(
            [
                nn.Linear(self.layers1[-1], dim_FC_layer)
                if i == 0
                else nn.Linear(dim_FC_layer, 1)
                if i == num_FC_layer - 1
                else nn.Linear(dim_FC_layer, dim_FC_layer)
                for i in range(num_FC_layer)
            ]
        )

        # We create a Tensor matrix with mu and dev values that will be
        # used to calculate the GAT2 input and define them as parameters
        # inside our GNN
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())

        # We create the function to embed the different feature vectors with
        # double size of the n_atom_features, since one half is for the
        # molecule and the other half for the protein
        self.embede = nn.Linear(
            2 * N_ATOM_FEATURES, dim_graph_layer, bias=False
        )

    def embede_graph(self, data):

        # This function does all the graph embedding so that it can be used as
        # input of a layer and runs the first part of the GNN, the GAT layers

        c_H, c_A1, c_A2, c_V = data

        # Embed features vectors
        c_H = self.embede(c_H)

        # Define the input matrix of GAT2 as exp((-(DMij - mu)^2)/dev). Then we
        # add the input Matrix of GAT1 that only contains the connections of
        # the covalent bonds
        c_A2 = (
            torch.exp(-torch.pow(c_A2 - self.mu.expand_as(c_A2), 2) / self.dev)
            + c_A1
        )

        regularization = torch.empty(len(self.gconv1), device=c_H.device)

        # Execute the GAT layers for the n layers defined above, obtaining,
        # between each of them, a new feature vector with the subtraction of
        # the two GATs of each layer (for A1 and for A2) and regularizing
        # with a defined dropout
        for k in range(len(self.gconv1)):
            c_H1 = self.gconv1[k](c_H, c_A1)
            c_H2 = self.gconv1[k](c_H, c_A2)
            c_H = c_H2 - c_H1
            c_H = F.dropout(c_H, p=self.dropout_rate, training=self.training)

        # The unsqueeze function adds a new dimension and we repeat the tensor
        # K times in the selected dimension. Then all the features are added,
        # obtaining a vector that represents the graph of the protein and the
        # ligand
        c_H = c_H * c_V.unsqueeze(-1).repeat(1, 1, c_H.size(-1))
        c_H = c_H.sum(1)

        return c_H

    def fully_connected(self, c_H):

        # This function runs all the fully connected layers of the GNN

        regularization = torch.empty(len(self.FC) * 1 - 1, device=c_H.device)

        # Execute the fully connected layers for the m layers defined above,
        # obtaining,between each of them, a new feature vector, regularizing
        # with a defined dropout and connecting with a ReLU function except
        # the last one in which there is no dropout and a Linear is
        # used
        for k in range(len(self.FC)):
            # c_H = self.FC[k](c_H)
            if k < len(self.FC) - 1:
                c_H = self.FC[k](c_H)
                c_H = F.dropout(
                    c_H, p=self.dropout_rate, training=self.training
                )
                c_H = F.relu(c_H)
            else:
                c_H = self.FC[k](c_H)

        # In this code, the output function is adapted to go from a
        # classification to a regression
        # TODO: Fix this
        # c_H = torch.special.exp2(c_H)
        # nn.Linear(1,1)
        # c_H = torch.special.exp2(c_H)

        return c_H

    def train_test_model(self, data):

        # Embede a graph to a vector and run GAT
        c_H = self.embede_graph(data)

        # Run fully connected NN
        c_H = self.fully_connected(c_H)

        c_H = c_H.view(-1)

        return c_H
