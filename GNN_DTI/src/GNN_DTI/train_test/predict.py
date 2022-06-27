import warnings
import argparse
import time
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from GNN_DTI.gnn import gnn
from GNN_DTI.utils import utils
from GNN_DTI.dataset.dataset import MolDataset
from GNN_DTI.dataset.data_loader import DTICollate_fn

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
parser.add_argument(
    "--model_path", help="enter the path to the model used to predict"
)
parser.add_argument("--pred_keys", help="enter the keys to predict")
parser.add_argument("--data_fpath", help="file path of data")
parser.add_argument(
    "--dropout_rate", help="dropout_rate", type=float, default=0.0
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
    "--initial_mu", help="initial value of mu", type=float, default=4.0
)
parser.add_argument(
    "--initial_dev", help="initial value of dev", type=float, default=1.0
)
parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
parser.add_argument(
    "--num_workers", help="number of workers", type=int, default=15
)
parser.add_argument(
    "--save_dir",
    help="save directory of the results",
    type=str,
    default="./save/",
)


args = parser.parse_args()
print(args)


def predict_gnn():
    # Get arguments
    data_fpath = args.data_fpath
    model_path = args.model_path
    batch_size = args.batch_size

    # Read data that is stored in format of dictionary. Each key has
    # information about protein-ligand complex
    with open(args.pred_keys, "rb") as fp:
        pred_keys = pickle.load(fp)

    # Print simple statistics about the data
    print("Number of data to predict: " + str(len(pred_keys)))

    # Initialize the model
    model = gnn.gnn(args)
    print(
        "Number of parameters : ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Select the GPU that is going to be used
    device = torch.device("cuda", 1)
    model = utils.initialize_model(model, device)

    model.load_state_dict(torch.load(model_path))

    pred_dataset = MolDataset(pred_keys, data_fpath)

    pred_dataloader = DataLoader(
        pred_dataset,
        batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DTICollate_fn,
    )

    n_epochs_pred = int(len(pred_dataset) / batch_size)

    pred_pred = []
    pred_keys = []

    model.eval()
    for i_batch, sample in enumerate(pred_dataloader):

        print("Pred batch number: ", str(i_batch), "/", str(n_epochs_pred))

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

        pred_pred.append(pred.data.cpu().numpy())
        pred_keys = pred_keys + keys

    pred_pred = np.concatenate(np.array(pred_pred), 0)

    text_str = "NAME;PRED\n"
    for i in range(len(pred_pred)):
        text_str = (
            text_str + pred_keys[i] + ";" + str(round(pred_pred[i], 7)) + "\n"
        )

    textfile = open(args.save_dir + "/results.txt", "w")
    textfile.write(text_str)
    textfile.close()


if __name__ == "__main__":
    predict_gnn()
