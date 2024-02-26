import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from utils.datasets import get_dataset
from utils.dipdi_utils import DIPDI
from utils.hparams import get_hparams
from utils.models import get_model


def load_model(path, network, n_outputs, hparams, task, device):
    model = get_model(
        "ERM",
        network,
        n_outputs,
        hparams,
        task,
    )

    model.to(device)
    model.load(path)
    model.eval()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--network", type=str)
    parser.add_argument("--task", type=str, choices=["regression", "classification"])
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--n_groups", type=int, default=2)
    parser.add_argument("--ratios_A", type=str, nargs="+")
    parser.add_argument("--ratios_B", type=str, nargs="+")
    parser.add_argument("--ckpt_dir", type=str)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = "../datasets/"
    metadata_path = f"./metadata/{args.dataset}.csv"
    metadata = pd.read_csv(metadata_path)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = ckpt_dir / "{}/group_{}/fold_{}/run_{}/checkpoint/model_best.pt"

    models_A = random.sample(range(1, args.n_groups + 1), args.n_groups // 2)
    models_B = [i for i in range(1, args.n_groups + 1) if i not in groups_A]

    print("Computing DIPDI...")
    df_all = pd.DataFrame()

    for ratio_A, ratio_B in zip(args.ratios_A, args.ratios_B):
        print(f"Ratio A: {ratio_A}, Ratio B: {ratio_B}")
        for fold in range(1, args.n_folds + 1):
            test_metadata = metadata[metadata["fold"] == fold].reset_index(drop=True)
            test_dataset = get_dataset(args.dataset, test_metadata, data_dir)

            results = {
                "fold": [],
                "run": [],
                "ratio_A": [],
                "ratio_B": [],
                "dipdi": [],
            }

            for run in range(1, args.n_runs + 1):
                models_set_A_paths = [
                    str(ckpt_path).format(ratio_A, i, fold, run) for i in models_A
                ]
                models_set_B_paths = [
                    str(ckpt_path).format(ratio_B, i, fold, run) for i in models_B
                ]

                models_set_A = [
                    load_model(
                        path, args.network, test_dataset.n_outputs, hparams, args.task, device
                    )
                    for path in models_set_A_paths
                ]
                models_set_B = [
                    load_model(
                        path, args.network, test_dataset.n_outputs, hparams, args.task, device
                    )
                    for path in models_set_B_paths[::-1]
                ]
                dipdi = DIPDI(models_set_A, models_set_B, test_dataset, args.task)

                results["fold"].append(fold)
                results["run"].append(run)
                results["ratio_A"].append(ratio_A)
                results["ratio_B"].append(ratio_B)
                results["dipdi"].append(dipdi.item())
                df_all = pd.concat([df_all, pd.DataFrame(results)], axis=0)

                print(f"Fold: {fold}, Run: {run}, DIPDI: {dipdi.item():.4f}")

    df_all.to_csv(output_dir / f"dipdi_ratios_{args.dataset}.csv", index=False)

    print(f"Done!\n")
