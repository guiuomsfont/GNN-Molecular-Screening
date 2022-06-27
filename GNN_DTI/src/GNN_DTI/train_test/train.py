import os
import warnings
import argparse
import time
import pickle
import numpy as np
import torch.nn as nn
import torch
from tabulate import tabulate
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from GNN_DTI.gnn import gnn
from GNN_DTI.utils import utils
from GNN_DTI.dataset.dataset import MolDataset
from GNN_DTI.dataset.data_loader import DTICollate_fn, DTISampler

warnings.filterwarnings("ignore")

now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (
    now.tm_year,
    now.tm_mon,
    now.tm_mday,
    now.tm_hour,
    now.tm_min,
    now.tm_sec,
)
print(s)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default=10000)
parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
parser.add_argument(
    "--num_workers", help="number of workers", type=int, default=7
)
parser.add_argument(
    "--num_graph_layer", help="number of GNN layer", type=int, default=4
)
parser.add_argument(
    "--dim_graph_layer", help="dimension of GNN layer", type=int, default=140
)
parser.add_argument(
    "--num_FC_layer", help="number of FC layer", type=int, default=4
)
parser.add_argument(
    "--dim_FC_layer", help="dimension of FC layer", type=int, default=128
)
parser.add_argument(
    "--dude_data_fpath",
    help="file path of dude data",
    type=str,
    default="data/",
)
parser.add_argument(
    "--save_dir",
    help="save directory of model parameter",
    type=str,
    default="./save/",
)
parser.add_argument(
    "--initial_mu", help="initial value of mu", type=float, default=4.0
)
parser.add_argument(
    "--initial_dev", help="initial value of dev", type=float, default=1.0
)
parser.add_argument(
    "--dropout_rate", help="dropout_rate", type=float, default=0.0
)
parser.add_argument(
    "--train_keys", help="train keys", type=str, default="keys/train_keys.pkl"
)
parser.add_argument(
    "--test_keys", help="test keys", type=str, default="keys/test_keys.pkl"
)
args = parser.parse_args()
print(args)


def train_gnn():
    # Get all the hyper parameters through arguments
    num_epochs = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    dude_data_fpath = args.dude_data_fpath
    save_dir = args.save_dir

    # Make save dir if it doesn't exist
    if not os.path.isdir(save_dir):
        os.system("mkdir " + save_dir)

    # Read data that is stored in format of dictionary. Each key has
    # information about protein-ligand complex.
    with open(args.train_keys, "rb") as fp:
        train_keys = pickle.load(fp)
    with open(args.test_keys, "rb") as fp:
        test_keys = pickle.load(fp)

    # Print simple statistics about the data
    print("Number of train data: " + str(len(train_keys)))
    print("Number of test data: " + str(len(test_keys)))

    # Initialize the model
    model = gnn.gnn(args)
    print(
        "Number of parameters : ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Select the GPU that is going to be used
    device = torch.device("cuda", 1)
    model = utils.initialize_model(model, device)

    # Create the train and test dataset
    train_dataset = MolDataset(train_keys, dude_data_fpath)
    test_dataset = MolDataset(test_keys, dude_data_fpath)

    # Count active compounds
    num_train_chembl = len([0 for k in train_keys if "CHEMBL" in k])
    num_train_decoy = len([0 for k in train_keys if "CHEMBL" not in k])

    # Create the Dataloader for testing and for training. While training a
    # model, we want to pass samples in “minibatches”, reshuffle the data at
    # every epoch to reduce model overfitting, and use Python’s multiprocessing
    # to speed up data retrieval. DataLoader is an iterable that abstracts this
    # complexity for us

    train_weights = [
        1 / num_train_chembl if "CHEMBL" in k else 1 / num_train_decoy
        for k in train_keys
    ]

    train_sampler = DTISampler(
        train_weights, len(train_weights), replacement=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DTICollate_fn,
        sampler=train_sampler,
    )
    test_dataloader = DataLoader(
        test_dataset,
        int(batch_size / 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DTICollate_fn,
    )

    # Define the optimizer. Adam is used as a replacement for the optimizer for
    # gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define loss function (Binary Cross Entropy)
    loss_fn = nn.BCELoss()

    n_epochs_train = int(len(train_dataset) / batch_size)
    n_epochs_test = int(len(test_dataset) / (batch_size / 2))

    # Save the results of every epoch
    epoch_results = ""

    # Run the same process for every epoch of training
    for epoch in range(num_epochs):
        st = time.time()
        # Collect losses of each iteration
        train_losses = []
        test_losses = []

        # Collect true label of each iteration
        train_true = []
        test_true = []

        # Collect predicted label of each iteration
        train_pred = []
        test_pred = []

        # Train the model in i epoch
        # model.train() is used as a switch for some layers/parts of the model
        # that behave differently during training and inference (evaluating)
        # time. For example, Dropouts Layers, BatchNorm Layers etc. You need to
        # turn on them during model evaluation, and .train() does it.
        model.train()
        for i_batch, sample in enumerate(train_dataloader):

            print(
                "Train batch number: ", str(i_batch), "/", str(n_epochs_train)
            )
            model.zero_grad()
            H, A1, A2, Y, V, keys = sample
            H, A1, A2, Y, V = (
                H.to(device),
                A1.to(device),
                A2.to(device),
                Y.to(device),
                V.to(device),
            )

            # Train neural network
            pred = model.train_test_model((H, A1, A2, V))

            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

            # Collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().numpy())
            train_true.append(Y.data.cpu().numpy())
            train_pred.append(pred.data.cpu().numpy())

        # Evaluate the model in i epoch
        # model.eval() is used as a switch for some layers/parts of the model
        # that behave differently during training and inference (evaluating)
        # time. For example, Dropouts Layers, BatchNorm Layers etc. You need to
        # turn off them during model evaluation, and .eval() does it.
        model.eval()
        for i_batch, sample in enumerate(test_dataloader):

            print("Test batch number: ", str(i_batch), "/", str(n_epochs_test))
            model.zero_grad()
            H, A1, A2, Y, V, keys = sample
            H, A1, A2, Y, V = (
                H.to(device),
                A1.to(device),
                A2.to(device),
                Y.to(device),
                V.to(device),
            )

            # Test neural network
            pred = model.train_test_model((H, A1, A2, V))

            loss = loss_fn(pred, Y)

            # Collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().numpy())
            test_true.append(Y.data.cpu().numpy())
            test_pred.append(pred.data.cpu().numpy())

        train_losses = np.mean(np.array(train_losses))
        test_losses = np.mean(np.array(test_losses))

        train_pred = np.concatenate(np.array(train_pred), 0)
        test_pred = np.concatenate(np.array(test_pred), 0)

        # Warning ignored here
        train_true = np.concatenate(np.array(train_true), 0)
        test_true = np.concatenate(np.array(test_true), 0)

        with open(
            save_dir + "/save_" + str(epoch) + "_predictions.pkl", "wb"
        ) as fp:
            pickle.dump([test_pred, test_true], fp)

        train_roc = roc_auc_score(train_true, train_pred)
        test_roc = roc_auc_score(test_true, test_pred)

        train_pred = [int(i >= 0.5) for i in train_pred]
        test_pred = [int(i >= 0.5) for i in test_pred]

        train_acc = balanced_accuracy_score(train_true, train_pred)
        test_acc = balanced_accuracy_score(test_true, test_pred)
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(
            train_true, train_pred
        ).ravel()
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(
            test_true, test_pred
        ).ravel()
        end = time.time()

        # Print and save epoch results
        print(
            tabulate(
                [
                    [
                        "%.4f" % (epoch),
                        "%.4f" % (train_losses),
                        "%.4f" % (test_losses),
                        "%.4f" % (train_roc),
                        "%.4f" % (test_roc),
                        "%.4f" % (train_acc),
                        "%.4f" % (test_acc),
                        "%.4f" % (train_tn),
                        "%.4f" % (train_fp),
                        "%.4f" % (train_fn),
                        "%.4f" % (train_tp),
                        "%.4f" % (test_tn),
                        "%.4f" % (test_fp),
                        "%.4f" % (test_fn),
                        "%.4f" % (test_tp),
                        "%.4f" % (end - st),
                    ]
                ],
                headers=[
                    "Epoch",
                    "Train_losses",
                    "Test_losses",
                    "Train_roc",
                    "Test_roc",
                    "Train_bal_acc",
                    "Test_bal_acc",
                    "Train_TN",
                    "Train_FP",
                    "Train_FN",
                    "Train_TP",
                    "Test_TN",
                    "Test_FP",
                    "Test_FN",
                    "Test_TP",
                    "Time",
                ],
            )
        )
        print()

        epoch_results = (
            epoch_results
            + " "
            + str(epoch)
            + " "
            + str(round(train_losses, 3))
            + " "
            + str(round(test_losses, 3))
            + " "
            + str(round(train_roc, 3))
            + " "
            + str(round(test_roc, 3))
            + " "
            + str(round(train_acc, 3))
            + " "
            + str(round(test_acc, 3))
            + " "
            + str(round(train_tn, 3))
            + " "
            + str(round(train_fp, 3))
            + " "
            + str(round(train_fn, 3))
            + " "
            + str(round(train_tp, 3))
            + " "
            + str(round(test_tn, 3))
            + " "
            + str(round(test_fp, 3))
            + " "
            + str(round(test_fn, 3))
            + " "
            + str(round(test_tp, 3))
            + " "
            + str(end - st)
            + "\n"
        )

        textfile = open(save_dir + "/epoch_results.txt", "w")
        textfile.write(epoch_results)
        textfile.close()

        # Save the model
        name = save_dir + "/save_" + str(epoch) + ".pt"
        torch.save(model.state_dict(), name)


if __name__ == "__main__":
    train_gnn()
