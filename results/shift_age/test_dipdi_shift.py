import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.datasets import ImageDataset
from src.networks import VGGNet
from src.utils import generate_distribution_shift
from torch.utils.data import DataLoader


def test(fold, group, option, data, data_dir, checkpoint_dir, output_dir):
    model = VGGNet(pretrained=False, num_outputs=1)
    model.load_state_dict(
        torch.load(f"{checkpoint_dir}/{option}/checkpoint/fold{fold}_group{group}_best.pt")
    )
    model.eval()

    test_dataset = ImageDataset(data, data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    with torch.no_grad():
        for inputs, _, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().squeeze())

    data["y_pred"] = predictions
    data.to_csv(f"{output_dir}/fold{fold}_group{group}_{option}_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--shift", type=str, choices=["male", "female", "both"])
    args = parser.parse_args()

    random.seed(args.fold)
    np.random.seed(args.fold)
    torch.manual_seed(args.fold)

    data_dir = "../datasets/"
    meta_csv = f"./metadata/utkface.csv"
    checkpoint_dir = Path(f"./models/utkface/groups/")
    output_dir = Path(f"./models/utkface/dipdi_shift/{args.shift}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(meta_csv)
    data = data[data["fold"] == args.fold].reset_index(drop=True)
    younger_proportions = [0.5, 0.6, 0.7, 0.8, 0.9]
    num_groups = 4

    for younger_prop in younger_proportions:
        print(f"Testing fold {args.fold} with {younger_prop} younger")
        data_shift = generate_distribution_shift(
            data,
            shift=args.shift,
            desired_prop=younger_prop,
            num_samples=len(data),
        )
        (output_dir / f"{younger_prop}").mkdir(parents=True, exist_ok=True)

        for i in range(num_groups):
            test(
                fold=args.fold,
                group=i,
                option="male",
                data=data_shift,
                data_dir=data_dir,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir / f"{younger_prop}",
            )
            test(
                fold=args.fold,
                group=i,
                option="female",
                data=data_shift,
                data_dir=data_dir,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir / f"{younger_prop}",
            )
