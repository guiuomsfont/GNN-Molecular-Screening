import numpy as np
import torch
import torch.nn as nn


def one_of_k_encoding(x, allowable_set):

    # Assigns entries that are not in the allowed set to the last element

    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(mol, atom_pos):

    # In this function we compute everything needed to create the 28-position
    # feature vector of the molecule
    # (1-10) Atom type -> C,N,O,S,F,P,Cl,Br,B,H (onehot)
    # (11-16) Degree of atom -> 0,1,2,3,4,5 (onehot)
    # (17-21) Number of hydrogen atoms attached -> 0,1,2,3,4 (onehot)
    # (22-27) Implicit valence electrons -> 1,2,3,4,5 (onehot)
    # (28) Aromatic -> 0 or 1
    # The return is a list of 28 positions

    atom = mol.GetAtomWithIdx(atom_pos)
    return np.array(
        one_of_k_encoding(
            atom.GetSymbol(),
            ["C", "N", "O", "S", "F", "P", "Cl", "Br", "B", "H"],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
        + [atom.GetIsAromatic()]
    )


def initialize_model(model, device, load_save_file=False):

    # Initialize the model with a dictionary if provided. Otherwise with ones
    # or with the function xavier_normal_

    if load_save_file:
        model.load_state_dict(torch.load(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal_(param)

    model.to(device)
    return model
