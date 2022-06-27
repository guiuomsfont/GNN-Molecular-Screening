#! /usr/bin/env python

import os
import random
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_keys_path", help="Introduce the path to the data keys"
)
args = parser.parse_args()


def create_keys():

    # In this function all the molecules are taken, mixed and a train and a
    # test are created

    data_keys_path = args.data_keys_path

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

    # Partitions are created and their corresponding protein names are
    # obtained.
    mol_names_train = mol_names_sample[:74]
    mol_names_test = mol_names_sample[74:]

    print("Train set: ", mol_names_train)

    print("\nTest set: ", mol_names_test)

    # All the files with the different keys that indicate the interactions with
    # each one of the molecules are read and they are put in their
    # corresponding list (train or test).
    train_keys = []
    for i in mol_names_train:
        keys = []
        with open(data_keys_path + i + "_actives.pkl", "rb") as pfile:
            keys = pickle.load(pfile)

        train_keys = train_keys + keys

        keys = []
        with open(data_keys_path + i + "_decoys.pkl", "rb") as pfile:
            keys = pickle.load(pfile)

        train_keys = train_keys + keys

    train_keys = random.sample(train_keys, len(train_keys))
    with open(data_keys_path + "../train_keys.pkl", "wb") as pfile:
        pickle.dump(train_keys, pfile)

    test_keys = []
    for i in mol_names_test:
        keys = []
        with open(data_keys_path + i + "_actives.pkl", "rb") as pfile:
            keys = pickle.load(pfile)

        test_keys = test_keys + keys

        keys = []
        with open(data_keys_path + i + "_decoys.pkl", "rb") as pfile:
            keys = pickle.load(pfile)

        test_keys = test_keys + keys

    test_keys = random.sample(test_keys, len(test_keys))
    with open(data_keys_path + "../test_keys.pkl", "wb") as pfile:
        pickle.dump(test_keys, pfile)

    print("Mols in train set = " + str(len(train_keys)))
    print("Mols in test set = " + str(len(test_keys)))


if __name__ == "__main__":
    create_keys()
