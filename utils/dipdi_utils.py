import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def DIPDI(models_set_A, models_set_B, test_dataset, task, eps=1e-8):
    if task not in DISCREPANCY_FUNCTIONS:
        raise ValueError(f"Invalid task name: {task}")

    if len(models_set_A) != len(models_set_B):
        raise ValueError("Number of models in set A and set B must be equal")

    if len(models_set_A) % 2 != 0 or len(models_set_B) % 2 != 0:
        raise ValueError("Number of models in set A and set B must be even")

    discrepancy_function = DISCREPANCY_FUNCTIONS[task]
    device = next(models_set_A[0].parameters())

    m = len(models_set_A)
    n_obs = len(test_dataset)
    n_outputs = models_set_A[0].network.output.out_features

    outputs_A = torch.zeros(m, n_obs, n_outputs).to(device)
    outputs_B = torch.zeros(m, n_obs, n_outputs).to(device)

    # Compute outputs for models in set A
    for i, model in enumerate(models_set_A):
        outputs_A[i] = compute_outputs(model, test_dataset)

    # Compute outputs for models in set B
    for i, model in enumerate(models_set_B):
        outputs_B[i] = compute_outputs(model, test_dataset)

    N_AB = 1
    N_AA = 1
    N_BB = 1

    # Compute output discrepancy between sets A and B
    for i in range(m):
        N_AB *= discrepancy_function(outputs_A[i], outputs_B[i])

    # Compute output discrepancy within sets A and B
    for i in range(m // 2):
        N_AA *= discrepancy_function(outputs_A[i], outputs_A[i + m // 2])
        N_BB *= discrepancy_function(outputs_B[i], outputs_B[i + m // 2])

    return torch.log(N_AB / (N_AA * N_BB + eps)) * 2 / m


def compute_outputs(model, dataset):
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    model.eval()

    outputs = []
    with torch.no_grad():
        for input_data, *_ in dataloader:
            output = model.predict(input_data.to(device))
            outputs.append(output)

    return torch.cat(outputs, dim=0)


def jensen_shannon_distance(input, target):
    p = F.softmax(input, dim=1)
    q = F.softmax(target, dim=1)
    m = 0.5 * (p + q)

    kl_pm = F.kl_div(p.log(), m.log(), reduction="batchmean", log_target=True)
    kl_qm = F.kl_div(q.log(), m.log(), reduction="batchmean", log_target=True)

    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd**0.5


def mean_absolute_error(input, target):
    return F.l1_loss(input, target)


def mean_squared_error(input, target):
    return F.mse_loss(input, target)


DISCREPANCY_FUNCTIONS = {
    "regression": mean_absolute_error,
    "classification": jensen_shannon_distance,
}


if __name__ == "__main__":

    class Model(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear = torch.nn.Linear(input_size, output_size)

        def predict(self, x):
            return self.linear(x)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, input_size, output_size, n_obs):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.n_obs = n_obs
            self.data = torch.randn(n_obs, input_size)
            self.labels = torch.randint(0, output_size, (n_obs,))

        def __len__(self):
            return self.n_obs

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    torch.manual_seed(42)

    input_size = 10
    output_size = 2
    n_obs = 1000
    task = "classification"
    n_models = 4

    dataset = Dataset(input_size, output_size, n_obs)
    models_set_A = [Model(input_size, output_size) for _ in range(n_models)]
    models_set_B = [Model(input_size, output_size) for _ in range(n_models)]

    dipdi = DIPDI(models_set_A, models_set_B, dataset, task)
    print(dipdi)
