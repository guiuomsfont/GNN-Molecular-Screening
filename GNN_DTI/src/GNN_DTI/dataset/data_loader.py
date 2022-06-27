import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

random.seed(33)


class DTISampler(Sampler):

    # Create a custom sampler to use during the creation of the Dataloader
    # Class that is included in the torch library. It used to specify the
    # sequence of indices/keys used in data loading.
    # Usage: DataLoader(..., sampler = DTISampler(...), ...)
    def __init__(self, weights, num_samples, replacement=True):

        # Initialize the sampler with the necessary parameters
        weights = np.array(weights) / np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):

        # Create an iterator for the class
        retval = np.random.choice(
            len(self.weights),
            self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )
        return iter(retval.tolist())

    def __len__(self):

        # Used to get the size of the samples
        return self.num_samples


def DTICollate_fn(batch):

    # Create a custom collate_fn to use during the creation of the Dataloader
    # Class that is included in the torch library. It receives a list of tuples
    # if your __getitem__ function from a Dataset subclass returns a tuple, or
    # just a normal list if your Dataset subclass returns only one element. Its
    # main objective is to create the batch without spending much time
    # implementing it manually.
    # Usage: DataLoader(..., collate_fn = DTICollate_fn(...), ...)
    max_natoms = max([len(item["H"]) for item in batch if item is not None])

    H = np.zeros((len(batch), max_natoms, 56))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))
    Y = np.zeros((len(batch),))
    V = np.zeros((len(batch), max_natoms))
    keys = []

    for i in range(len(batch)):
        natom = len(batch[i]["H"])
        H[i, :natom] = batch[i]["H"]
        A1[i, :natom, :natom] = batch[i]["A1"]
        A2[i, :natom, :natom] = batch[i]["A2"]
        Y[i] = batch[i]["Y"]
        V[i, :natom] = batch[i]["V"]
        keys.append(batch[i]["key"])

    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    Y = torch.from_numpy(Y).float()
    V = torch.from_numpy(V).float()

    return H, A1, A2, Y, V, keys
