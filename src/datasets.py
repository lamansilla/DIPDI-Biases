from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, data, data_dir, return_prior_labels=False):
        self.filenames = data["filename"].apply(lambda x: Path(data_dir) / x).astype(str).tolist()
        self.labels = data["y"].tolist()
        self.attributes = data["a"].tolist()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.prior_labels = data["y_prior"].tolist() if return_prior_labels else None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        attribute = self.attributes[index]

        if self.prior_labels:
            prior_label = torch.tensor(self.prior_labels[index], dtype=torch.float32)
            return image, label, attribute, prior_label

        return image, label, attribute
