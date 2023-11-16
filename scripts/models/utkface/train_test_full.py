import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from src.datasets import ImageDataset
from src.networks import VGGNet
from src.utils import generate_training_distribution
from torch.utils.data import DataLoader, RandomSampler


def train(fold, option, data_dir, meta_csv, output_dir, config):
    data = pd.read_csv(meta_csv)
    data = data[data["fold"] != fold]

    train_data = generate_training_distribution(data, option=option, random_state=42)
    train_data, val_data = train_test_split(
        train_data, test_size=0.1, random_state=42, stratify=train_data["a"]
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    model = VGGNet(pretrained=True, num_outputs=1)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_sampler = RandomSampler(
        train_data, num_samples=config["num_batches"] * config["batch_size"]
    )
    train_dataset = ImageDataset(train_data, data_dir)
    val_dataset = ImageDataset(val_data, data_dir)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    patience = 10
    early_stopping_counter = 0
    best_val_loss = float("inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch}/{config['num_epochs']-1} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Time: {epoch_time / 60:.2f} min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/fold{fold}_best.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


def test(fold, data_dir, meta_csv, checkpoint_dir, output_dir):
    data = pd.read_csv(meta_csv)
    test_data = data[data["fold"] == fold].reset_index(drop=True)

    model = VGGNet(pretrained=False, num_outputs=1)
    model.load_state_dict(torch.load(f"{checkpoint_dir}/fold{fold}_best.pt"))
    model.eval()

    test_dataset = ImageDataset(test_data, data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    with torch.no_grad():
        for inputs, _, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().squeeze())

    test_data["y_pred"] = predictions
    test_data.to_csv(f"{output_dir}/fold{fold}_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--option", type=str, choices=["male", "female", "mixed"])
    args = parser.parse_args()

    random.seed(args.fold)
    np.random.seed(args.fold)
    torch.manual_seed(args.fold)

    data_dir = "../datasets/"
    meta_csv = f"./metadata/utkface.csv"
    output_dir = Path(f"./models/utkface/full/{args.option}/")
    (output_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
    (output_dir / "output").mkdir(parents=True, exist_ok=True)

    config = {
        "num_epochs": 50,
        "batch_size": 32,
        "num_batches": 200,
        "lr": 1e-4,
    }

    print(f"Training fold {args.fold}")
    train(
        fold=args.fold,
        option=args.option,
        data_dir=data_dir,
        meta_csv=meta_csv,
        output_dir=output_dir / "checkpoint",
        config=config,
    )

    print(f"Testing fold {args.fold}")
    test(
        fold=args.fold,
        data_dir=data_dir,
        meta_csv=meta_csv,
        checkpoint_dir=output_dir / "checkpoint",
        output_dir=output_dir / "output",
    )
