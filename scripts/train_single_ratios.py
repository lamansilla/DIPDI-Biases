import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from utils.datasets import get_dataset
from utils.hparams import get_hparams
from utils.models import get_model
from utils.train_utils import predict, train


def sample_and_split_data(train_metadata, args):
    ratio_m, ratio_f = map(int, args.ratio.split("-"))
    assert ratio_m + ratio_f == 100, "Invalid ratio"

    # Determine sample size for each gender
    n = train_metadata["a"].value_counts().min()
    n_m = int(n * ratio_m / 100)
    n_f = n - n_m

    def sample_data(group, n):
        return group.sample(n=n, replace=False, random_state=args.seed)

    train_m = train_metadata[train_metadata["a"] == 0].reset_index(drop=True)
    train_f = train_metadata[train_metadata["a"] == 1].reset_index(drop=True)

    train_m_sample = sample_data(train_m, n_m)
    train_f_sample = sample_data(train_f, n_f)

    train_metadata = pd.concat([train_m_sample, train_f_sample]).reset_index(drop=True)
    train_metadata = train_metadata.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split data into training and validation sets
    train_metadata, val_metadata = train_test_split(
        train_metadata,
        test_size=0.1,
        stratify=train_metadata["a"],
        random_state=args.seed,
    )

    train_metadata = train_metadata.reset_index(drop=True)
    val_metadata = val_metadata.reset_index(drop=True)

    return train_metadata, val_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--network", type=str)
    parser.add_argument("--task", type=str, choices=["regression", "classification"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ratio", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Arguments:", args.__dict__)

    hparams = get_hparams()
    print("Hyperparameters:", hparams)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = f"./metadata/{args.dataset}.csv"
    metadata = pd.read_csv(metadata_path)

    train_metadata = metadata[metadata["fold"] != args.fold].reset_index(drop=True)
    test_metadata = metadata[metadata["fold"] == args.fold].reset_index(drop=True)

    train_metadata, val_metadata = sample_and_split_data(train_metadata, args)

    print(f"Train size: {len(train_metadata)}")
    print(f"Val size: {len(val_metadata)}")
    print(f"Test size: {len(test_metadata)}")

    data_dir = "../datasets/"
    train_dataset = get_dataset(args.dataset, train_metadata, data_dir)
    val_dataset = get_dataset(args.dataset, val_metadata, data_dir)
    test_dataset = get_dataset(args.dataset, test_metadata, data_dir)

    n_epochs = train_dataset.N_EPOCHS
    patience = train_dataset.PATIENCE_LIMIT
    n_batches = train_dataset.N_BATCHES

    sampler = RandomSampler(train_dataset, num_samples=hparams["batch_size"] * n_batches)
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    model = get_model(
        "ERM",
        args.network,
        train_dataset.n_outputs,
        hparams,
        args.task,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_best.pt"

    print(f"Training...")
    train(model, train_loader, val_loader, log_dir, n_epochs, patience, checkpoint_path)

    print(f"Testing...")
    model.load(checkpoint_path)
    predictions = predict(model, test_loader)

    test_metadata["y_pred"] = predictions.astype(int).tolist()
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    test_metadata.to_csv(predictions_dir / "predictions.csv", index=False)

    targets = test_metadata["y"].values
    if args.task == "regression":
        print(f"MAE: {np.mean(np.abs(targets - predictions)):.4f}")
    else:
        print(f"Accuracy: {np.mean(targets == predictions):.4f}")

    print(f"Done!\n")
