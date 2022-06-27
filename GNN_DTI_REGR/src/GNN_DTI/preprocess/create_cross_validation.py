#! /usr/bin/env python

import os
import random
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_keys_path", help="Introduce the path to the data keys"
)
parser.add_argument(
    "k_folds", help="Introduce the number of folds", type=int, default=4
)
args = parser.parse_args()


def create_keys():

    # In this function different data sets are prepared to create a K-fold
    # validation

    data_keys_path = args.data_keys_path
    k_folds = args.k_folds

    if data_keys_path[-1] != "/":
        data_keys_path = data_keys_path + "/"

    mol_keys = os.listdir(data_keys_path)

    # All the different possible molecule names are obtained
    mol_names = []
    for i in range(len(mol_keys)):
        mol_name = mol_keys[i].split("_")[0]
        if mol_name not in mol_names:
            mol_names.append(mol_name)

    #  The different names are mixed
    mol_names_sample = random.sample(mol_names, len(mol_names))

    # Partitions are created based on the indicated k-folds and their
    # corresponding protein names are obtained.
    fold_len = int(len(mol_names_sample) / k_folds)

    mol_names_fold = []

    fold_pos = 0
    for i in range(k_folds - 1):
        mol_names_fold.append(
            mol_names_sample[fold_pos : (fold_pos + fold_len)]
        )
        fold_pos += fold_len
    mol_names_fold.append(mol_names_sample[fold_pos:])

    # All the files with the different keys that indicate the interactions with
    # each one of the molecules are read and they are put in their
    # corresponding list (k-fold group).
    for j in range(k_folds):
        print(str(j) + " Fold: ", mol_names_fold[j])

        mol_keys_fold = []
        for i in mol_names_fold[j]:
            keys = []
            with open(data_keys_path + i + "_actives.pkl", "rb") as pfile:
                keys = pickle.load(pfile)

            mol_keys_fold = mol_keys_fold + keys

            keys = []
            with open(data_keys_path + i + "_decoys.pkl", "rb") as pfile:
                keys = pickle.load(pfile)

            mol_keys_fold = mol_keys_fold + keys

        mol_keys_fold = random.sample(mol_keys_fold, len(mol_keys_fold))

        with open(
            data_keys_path + "../" + str(j) + "_keys.pkl", "wb"
        ) as pfile:
            pickle.dump(mol_keys_fold, pfile)

    # Once the lists with the different keys of the groups have been created,
    # the combinations are created for each of the k validations
    for j in range(k_folds):
        train_names = []
        for i in range(k_folds):
            if i != j:
                train_names = train_names + mol_names_fold[i]

        test_names = mol_names_fold[j]

        train_keys = []
        for i in train_names:
            keys = []
            with open(data_keys_path + i + "_actives.pkl", "rb") as pfile:
                keys = pickle.load(pfile)

            train_keys = train_keys + keys

            keys = []
            with open(data_keys_path + i + "_decoys.pkl", "rb") as pfile:
                keys = pickle.load(pfile)

            train_keys = train_keys + keys

        train_keys = random.sample(train_keys, len(train_keys))
        with open(
            data_keys_path + "../" + str(j) + "_train_keys.pkl", "wb"
        ) as pfile:
            pickle.dump(train_keys, pfile)

        test_keys = []
        for i in test_names:
            keys = []
            with open(data_keys_path + i + "_actives.pkl", "rb") as pfile:
                keys = pickle.load(pfile)

            test_keys = test_keys + keys

            keys = []
            with open(data_keys_path + i + "_decoys.pkl", "rb") as pfile:
                keys = pickle.load(pfile)

            test_keys = test_keys + keys

        test_keys = random.sample(test_keys, len(test_keys))
        with open(
            data_keys_path + "../" + str(j) + "_test_keys.pkl", "wb"
        ) as pfile:
            pickle.dump(test_keys, pfile)

        print("Mols in train set = " + str(len(train_keys)))
        print("Mols in test set = " + str(len(test_keys)))


if __name__ == "__main__":
    create_keys()
