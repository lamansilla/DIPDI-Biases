from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import transforms


def get_dataset(dataset_name, data, data_dir):
    if dataset_name not in DATASETS:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return DATASETS[dataset_name](data, data_dir)


class _BaseDataset(Dataset):
    N_EPOCHS = 50
    PATIENCE_LIMIT = 5
    N_BATCHES = 200

    def __init__(self, metadata, data_dir, image_transform=None):
        self.paths = metadata["path"].apply(lambda x: Path(data_dir) / x).astype(str).tolist()
        self.targets = metadata["y"].tolist()
        self.image_transform = image_transform
        self.n_outputs = len(set(self.targets))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        input_data = Image.open(self.paths[idx]).convert("RGB")
        input_data = self.image_transform(input_data) if self.image_transform else input_data
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return input_data, target


class CelebA(_BaseDataset):
    def __init__(self, metadata, data_dir):
        image_transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(metadata, data_dir, image_transform)


class CelebAGray(_BaseDataset):
    def __init__(self, metadata, data_dir):
        image_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        super().__init__(metadata, data_dir, image_transform)


class ChestXray14(_BaseDataset):
    def __init__(self, metadata, data_dir):
        image_transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(metadata, data_dir, image_transform)
        self.n_outputs = 1


class UTKFace(_BaseDataset):
    def __init__(self, metadata, data_dir):
        image_transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(metadata, data_dir, image_transform)
        self.n_outputs = 1


DATASETS = {
    "ChestX-ray14": ChestXray14,
    "UTKFace": UTKFace,
    "CelebA": CelebA,
}
