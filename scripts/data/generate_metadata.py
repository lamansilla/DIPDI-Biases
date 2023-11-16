import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def generate_metadata_chestxray14(data_dir):
    print("Generating metadata for ChestX-ray14")

    chest14_dir = Path(os.path.join(data_dir, "ChestXray-NIHCC"))
    assert (chest14_dir / "Data_Entry_2017_v2020.csv").is_file()

    data = pd.read_csv(chest14_dir / "Data_Entry_2017_v2020.csv")

    data = data[data["Finding Labels"] == "No Finding"]
    data["subject_id"] = data["Patient ID"].astype(str)
    data = data.drop_duplicates(subset="subject_id", keep="first", ignore_index=True)

    data_female = data[data["Patient Gender"] == "F"]
    data_male = data[data["Patient Gender"] == "M"]
    data_male = data_male.sample(n=len(data_female), replace=False, random_state=42)
    data = pd.concat([data_male, data_female], ignore_index=True)

    data["filename"] = data["Image Index"].apply(
        lambda x: os.path.join("ChestXray-NIHCC/images", x)
    )
    data["a"] = data["Patient Gender"].map({"M": "Male", "F": "Female"})
    data["y"] = data["Patient Age"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (_, test_idx) in enumerate(skf.split(X=data.index, y=data["a"])):
        data.loc[test_idx, "fold"] = i
    data["fold"] = data["fold"].astype(int)

    return data


def generate_metadata_utkface(data_dir):
    print("Generating metadata for UTKFace")

    utkface_dir = Path(os.path.join(data_dir, "UTKFace"))

    ages = []
    genders = []
    races = []
    filenames = []

    for image_path in utkface_dir.glob("*.jpg"):
        filename = image_path.name
        parts = filename.split("_")
        if len(parts) != 4:
            continue
        age, gender, race, _ = parts
        ages.append(int(age))
        genders.append(int(gender))
        races.append(int(race))
        filenames.append(os.path.join("UTKFace", filename))

    data = pd.DataFrame(
        {
            "filename": filenames,
            "Age": ages,
            "Gender": genders,
            "Race": races,
        }
    )
    data = data[(data["Age"].gt(10)) & (data["Age"].lt(100))].reset_index(drop=True)

    data_female = data[data["Gender"] == 1]
    data_male = data[data["Gender"] == 0]
    data_male = data_male.sample(n=len(data_female), replace=False, random_state=42)
    data = pd.concat([data_male, data_female], ignore_index=True)

    data["a"] = data["Gender"].map({0: "Male", 1: "Female"})
    data["y"] = data["Age"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (_, test_idx) in enumerate(skf.split(X=data.index, y=data["a"])):
        data.loc[test_idx, "fold"] = i
    data["fold"] = data["fold"].astype(int)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["chestxray14", "utkface"])
    args = parser.parse_args()

    data_dir = "../datasets/"
    meta_dir = Path("./metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)

    dataset_metadata_generators = {
        "chestxray14": generate_metadata_chestxray14,
        "utkface": generate_metadata_utkface,
    }

    data = dataset_metadata_generators[args.dataset](data_dir)
    data.to_csv(meta_dir / f"{args.dataset}.csv", index=False)
