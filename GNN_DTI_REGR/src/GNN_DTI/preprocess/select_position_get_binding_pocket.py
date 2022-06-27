#! /usr/bin/env python

import os
import argparse
import pickle
import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolFromPDBBlock


parser = argparse.ArgumentParser()
parser.add_argument("receptor_file_path")
parser.add_argument("ligand_file_path")
parser.add_argument("receptor_name")
parser.add_argument("pickle_dir_mols")
parser.add_argument("pickle_dir_keys")
parser.add_argument("ligand_target")


args = parser.parse_args()


def create_box():

    receptor_name = args.receptor_name
    pickle_dir_mols = args.pickle_dir_mols
    pickle_dir_keys = args.pickle_dir_keys
    ligand_target = args.ligand_target

    with open("rescoring_list.txt") as f:
        scores = f.readlines()

    # In this part we obtain the ligands with their corresponding scores
    # obtained from the rescoring
    ligands_scores = PDBMolSupplier(
        scores, args.ligand_file_path, sanitize=False
    )

    # In this function we group the different positions of the same molecule
    # in their corresponding group
    separate_mols = [[ligands_scores[0]]]

    n_mols = 0
    counter = 1

    for i in range(1, len(ligands_scores)):
        if counter >= 9 or ligands_scores[i][0].GetProp(
            "_Name"
        ) != ligands_scores[i - 1][0].GetProp("_Name"):
            separate_mols.append([ligands_scores[i]])
            n_mols += 1
            counter = 1
        else:
            separate_mols[n_mols].append(ligands_scores[i])
            counter += 1

    # Here we get the best position of each group/molecule
    final_ligands = []
    last_dict = {}
    for group in separate_mols:
        best = -999
        final_mol = None
        for mol in group:
            if mol[1] > best:
                final_mol = mol[0]
                best = mol[1]
        mol_name = final_mol.GetProp("_Name")
        if mol_name in last_dict:
            final_mol.SetProp("_Name", mol_name + last_dict[mol_name])
            last_dict[mol_name] = last_dict[mol_name] + "I"
        else:
            last_dict[mol_name] = "I"
        final_ligands.append(final_mol)

    print("\n\nLengths:")
    print("Original scores: ", str(len(scores)))
    print("Read ligands: ", str(len(ligands_scores)))
    print("Distinguished mols: ", str(len(separate_mols)))
    print("Final selected mols: ", str(len(final_ligands)))

    receptor = MolFromPDBFile(args.receptor_file_path, sanitize=False)

    # Here we cut the binding pocket based on the minimum distance of a
    # protein atom from the drug molecules. All the atoms of the protein
    # that are less than X distance from our ligand remain. In this way
    # we eliminate all the parts of the protein that do not affect the
    # interaction.
    receptor_conformers = receptor.GetConformers()[0]
    receptor_positions = np.array(receptor_conformers.GetPositions())

    keys_list = []
    for i in range(len(final_ligands)):

        print("\n\nLigand " + final_ligands[i].GetProp("_Name") + "\n\n")

        edreceptor = Chem.EditableMol(receptor)

        if final_ligands[i] is not None:
            ligand_conformers = final_ligands[i].GetConformers()[0]
            ligand_positions = np.array(ligand_conformers.GetPositions())

            dist_matrix = distance_matrix(ligand_positions, receptor_positions)

            for j in reversed(range(len(receptor_positions))):
                inside = False
                for k in range(len(ligand_positions)):
                    distance = dist_matrix[k][j]
                    if distance < 8.0:
                        inside = True
                        break
                if not inside:
                    edreceptor.RemoveAtom(j)

            keys_list.append(
                receptor_name + "_" + final_ligands[i].GetProp("_Name")
            )

            with open(
                pickle_dir_mols
                + receptor_name
                + "_"
                + final_ligands[i].GetProp("_Name"),
                "wb",
            ) as f:
                pickle.dump([final_ligands[i], edreceptor.GetMol()], f)

    with open(
        pickle_dir_keys + receptor_name + "_" + ligand_target + ".pkl", "wb"
    ) as f:
        pickle.dump(keys_list, f)


# Function created to get all the molecules from a PDB file and their
# respective scores
def PDBMolSupplier(scores, file=None, sanitize=True):
    mols = []
    pos_i = 0

    with open(file, "r") as f:
        while f.tell() != os.fstat(f.fileno()).st_size:
            line = f.readline()
            print(line)
            if line.startswith("MODEL"):
                mol = []
                mol.append(line)
                line = f.readline()
                while not line.startswith("ENDMDL"):
                    mol.append(line)
                    line = f.readline()

                mol[-1] = mol[-1].rstrip()  # removes blank line at file end
                block = ",".join(mol).replace(",", "")
                m = MolFromPDBBlock(block, sanitize=sanitize)

                if m is not None:
                    mols.append([m, float(scores[pos_i])])
                else:
                    mols.append(None)

                pos_i += 1

    return mols


if __name__ == "__main__":
    create_box()
