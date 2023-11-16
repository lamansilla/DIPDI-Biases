import torch
from torch.utils.data import DataLoader


def DIPDI(models_A, models_B, dataset):
    device = next(models_A[0].parameters())

    D = len(dataset)
    outputs_A = torch.zeros(2, D).to(device)
    outputs_B = torch.zeros(2, D).to(device)

    outputs_A[0] = compute_outputs(models_A[0], dataset)
    outputs_A[1] = compute_outputs(models_A[1], dataset)
    outputs_B[0] = compute_outputs(models_B[0], dataset)
    outputs_B[1] = compute_outputs(models_B[1], dataset)

    N_AB = torch.norm(outputs_A[0] - outputs_B[0], p=2) * torch.norm(
        outputs_A[1] - outputs_B[1], p=2
    )
    N_AA = torch.norm(outputs_A[0] - outputs_A[1], p=2)
    N_BB = torch.norm(outputs_B[0] - outputs_B[1], p=2)

    return torch.log(N_AB / (N_AA * N_BB + 1e-8))


def compute_outputs(model, dataset):
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    model.eval()

    outputs = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            output = model(images).detach()
            outputs.append(output)

    outputs = torch.cat(outputs, dim=0).squeeze()

    return outputs
