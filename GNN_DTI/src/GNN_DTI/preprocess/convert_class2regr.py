#! /usr/bin/env python

import os
import pickle
import argparse
import rdkit
import copy


parser = argparse.ArgumentParser()
parser.add_argument("dude_path", help="Introduce the path to the dude data")
parser.add_argument("data_path", help="Introduce the path to the data")
args = parser.parse_args()


def convert_class2regr():

    for receptor in os.listdir(args.dude_path):
        with open(
            args.data_path + "/KEYS/" + receptor + "_actives.pkl", "rb"
        ) as fp:
            active_keys = pickle.load(fp)

        with open(
            args.data_path + "/KEYS/" + receptor + "_decoys.pkl", "rb"
        ) as fp:
            decoy_keys = pickle.load(fp)

        target_dict = get_target_dict(
            args.dude_path + "/" + receptor + "/actives_nM_chembl.ism",
            receptor,
        )

        for name_inter in active_keys:
            print(name_inter)
            if name_inter in target_dict:
                target = target_dict[name_inter]
            else:
                temp_name = copy.deepcopy(name_inter)
                while temp_name[-1] == "I":
                    temp_name = temp_name[:-1]
                if temp_name in target_dict:
                    target = target_dict[temp_name]
                else:
                    print("ERROR ERROR ERROR")
                    target = float(1000)

            with open(args.data_path + "/DATA/" + name_inter, "rb") as pfile:
                mol1, mol2 = pickle.load(pfile)

            with open(args.data_path + "/DATA/" + name_inter, "wb") as pfile:
                pickle.dump([mol1, mol2, target], pfile)

        for name_inter in decoy_keys:
            print(name_inter)
            with open(args.data_path + "/DATA/" + name_inter, "rb") as pfile:
                mol1, mol2 = pickle.load(pfile)

            with open(args.data_path + "/DATA/" + name_inter, "wb") as pfile:
                pickle.dump([mol1, mol2, float(1000000)], pfile)


def get_target_dict(path, receptor):
    with open(path) as f:
        lines = f.readlines()

    target = {}
    for line in lines:
        positions = line.split(" ")
        if receptor + "_" + positions[2] not in target:
            target[receptor + "_" + positions[2]] = float(positions[5])
        else:
            print("ERROR, target value repeated")

    return target


if __name__ == "__main__":
    convert_class2regr()
