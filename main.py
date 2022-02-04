import os
import argparse
import torch
import pytorch_lightning as pl
from torch_geometric import datasets
from torch_geometric.loader import DataLoader

from module import DegreeRegressor


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--log", type=bool, default=False)
parser.add_argument("--gpus", type=int, default=-1)
parser.add_argument("--num_workers", type=int, default=-1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=None)

parser.add_argument("--model", type=str, default="MPNN")
parser.add_argument("--aggr", type=str, default="add")
parser.add_argument("--hidden_features", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=1)

args = parser.parse_args()

if __name__ == "__main__":
    # Devices
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # Dataset
    if args.dataset == "MUTAG":
        dataset = datasets.TUDataset(args.root, args.dataset)
        in_features = 7
    elif args.dataset == "PROTEINS":
        dataset = datasets.TUDataset(args.root, args.dataset)
        in_features = 3
    elif args.dataset == "QM9":
        dataset = datasets.QM9(args.root)
        in_features = 11
    else:
        raise Exception("Data set not recognized:", args.dataset)
    out_features = 1

    # Model
    if args.model == "MPNN":
        from GNNs.message_passing import MPNN

        model = MPNN(
            in_features,
            args.hidden_features,
            out_features,
            args.num_layers,
            aggr=args.aggr,
        )
    else:
        raise Exception("Model not recognized:", args.model)

    dataloader = DataLoader(
        dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Logging
    if args.log:
        logger = pl.loggers.WandbLogger(
            project="DegreeRegressor",
            name="_".join([args.dataset, args.model]),
            config=args,
        )
    else:
        logger = None

    # Reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        deterministic = True
    else:
        deterministic = False

    model = DegreeRegressor(model, args.lr)

    # Let's go!
    print(model)
    print(args)
    trainer = pl.Trainer(
        gpus=args.gpus,
        logger=logger,
        max_epochs=args.epochs,
        deterministic=deterministic,
    )
    trainer.fit(model, dataloader)
