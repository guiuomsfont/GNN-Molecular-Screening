import pickle
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from GNN_DTI.utils import utils


def get_atom_feature(mol, is_ligand):

    # Here we create the feature vector for each atom in the molecule (calling
    # the atom_feature() function), where the first 28 positions of the vector
    # will be for the ligand and the last 28 for the protein.The return is a
    # list of vectors
    n_atoms = mol.GetNumAtoms()
    feature_vect = []
    for i in range(n_atoms):
        feature_vect.append(utils.atom_feature(mol, i))

    feature_vect = np.array(feature_vect)
    if is_ligand:
        feature_vect = np.concatenate(
            [feature_vect, np.zeros((n_atoms, 28))], 1
        )
    else:
        feature_vect = np.concatenate(
            [np.zeros((n_atoms, 28)), feature_vect], 1
        )

    return feature_vect


class MolDataset(Dataset):

    # Class used to create and save the dataset

    def __init__(self, keys, data_dir):

        # The keys contain the name of the protein next to the name of the
        # molecule joined by the character "_"
        # The data directory allows us to access the site where the entire
        # database is located
        self.keys = keys
        if data_dir[-1] != "/":
            data_dir = data_dir + "/"
        self.data_dir = data_dir

    def __len__(self):

        # Used to get the size of the dataset

        return len(self.keys)

    def __getitem__(self, idx_pos):

        # In this function, a protein-ligand pair is loaded by identifying it
        # with the index it occupies in the key list, reading it from memory,
        # and creating the feature vectors and distance matrix to obtain a list
        # of all the feature vectors of the ligands. protein and ligand atoms,
        # the two adjacency aggregation matrices, the target to be predicted, a
        # list of indices for aggregation, and the protein_ligand name (key)

        # Load ligand and receptor files
        key = self.keys[idx_pos]
        with open(self.data_dir + "/" + key, "rb") as pfile:
            mol1, mol2, real_val = pickle.load(pfile)

        if real_val > 0:
            real_val = np.log(real_val)
        else:
            real_val = np.log(1e-12)

        # Prepare ligand
        natoms_1 = mol1.GetNumAtoms()
        conformers_1 = mol1.GetConformers()[0]
        positions_1 = np.array(conformers_1.GetPositions())
        adj_1 = GetAdjacencyMatrix(mol1) + np.eye(natoms_1)
        feature_vects_1 = get_atom_feature(mol1, True)

        # Prepare protein
        natoms_2 = mol2.GetNumAtoms()
        conformers_2 = mol2.GetConformers()[0]
        positions_2 = np.array(conformers_2.GetPositions())
        adj_2 = GetAdjacencyMatrix(mol2) + np.eye(natoms_2)
        feature_vects_2 = get_atom_feature(mol2, False)

        # Make aggregation adjency matrix
        # In this part, the two matrices that will be provided as input to the
        # two GATs are created. The first matrix only contains the information
        # of the covalent bonds. Instead, the second contains this information
        # added to the distance matrix.
        feature_vects = np.concatenate([feature_vects_1, feature_vects_2], 0)
        agg_adj_1 = np.zeros((natoms_1 + natoms_2, natoms_1 + natoms_2))
        agg_adj_1[:natoms_1, :natoms_1] = adj_1
        agg_adj_1[natoms_1:, natoms_1:] = adj_2
        agg_adj_2 = np.copy(agg_adj_1)
        dist_matrix = distance_matrix(positions_1, positions_2)
        agg_adj_2[:natoms_1, natoms_1:] = np.copy(dist_matrix)
        agg_adj_2[natoms_1:, :natoms_1] = np.copy(np.transpose(dist_matrix))

        # Node indice for aggregation
        valid = np.zeros((natoms_1 + natoms_2,))
        valid[:natoms_1] = 1

        # Set the target value
        target = float(real_val)

        sample = {
            "H": feature_vects,
            "A1": agg_adj_1,
            "A2": agg_adj_2,
            "Y": target,
            "V": valid,
            "key": key,
        }

        return sample
