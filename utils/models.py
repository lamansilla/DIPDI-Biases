import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


def get_model(model_name, network_name, n_outputs, hparams, task):
    if model_name not in MODELS:
        raise ValueError(f"Invalid model name: {model_name}")

    if network_name not in NETWORKS:
        raise ValueError(f"Invalid network name: {network_name}")

    if task not in LOSS_FUNCTIONS:
        raise ValueError(f"Invalid task type: {task}")

    return MODELS[model_name](network_name, n_outputs, hparams, task)


class _BaseModel(nn.Module):
    def __init__(self, n_outputs, hparams, task):
        super(_BaseModel, self).__init__()
        self.n_outputs = n_outputs
        self.hparams = hparams
        self.task = task

    def predict(self, input_data):
        raise NotImplementedError

    def compute_loss(self, input_data, target):
        raise NotImplementedError

    def update(self, input_data, target):
        raise NotImplementedError


def initialize_densenet_model(n_outputs):
    model = models.densenet121(weights="IMAGENET1K_V1").features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.flatten = nn.Flatten()
    model.output = nn.Linear(1024, n_outputs)
    return model


def initialize_vgg_model(n_outputs):
    model = models.vgg19(weights="IMAGENET1K_V1").features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.flatten = nn.Flatten()
    model.output = nn.Linear(512, n_outputs)
    return model


def initialize_resnet_model(n_outputs):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model = nn.Sequential(*list(model.children())[:-2])
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.flatten = nn.Flatten()
    model.output = nn.Linear(2048, n_outputs)
    return model


class ERM(_BaseModel):
    def __init__(self, network_name, n_outputs, hparams, task):
        super(ERM, self).__init__(n_outputs, hparams, task)
        self.network = NETWORKS[network_name](n_outputs)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=hparams["lr"],
        )
        self.loss = LOSS_FUNCTIONS[task]

    def predict(self, input_data):
        return self.network(input_data)

    def compute_loss(self, output, target):
        if self.n_outputs == 1:
            target = target.reshape(output.shape).float()
        else:
            target = target.long()

        loss = self.loss(output, target)

        return loss if self.n_outputs != 1 else loss.squeeze(1)

    def update(self, input_data, target):
        output = self.predict(input_data)
        loss = self.compute_loss(output, target).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, checkpoint_path):
        torch.save(self.network.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.network.load_state_dict(torch.load(checkpoint_path))


MODELS = {
    "ERM": ERM,
}

NETWORKS = {
    "DenseNet": initialize_densenet_model,
    "VGG": initialize_vgg_model,
    "ResNet": initialize_resnet_model,
}

LOSS_FUNCTIONS = {
    "classification": nn.CrossEntropyLoss(reduction="none"),
    "regression": nn.L1Loss(reduction="none"),
}


if __name__ == "__main__":
    loss1 = nn.L1Loss(reduction="none")
    loss2 = nn.L1Loss(reduction="mean")

    x = torch.randn(10, 1)
    y = torch.randn(10, 1)

    print(loss1(x, y))
    print(loss1(x, y).sum() / 10)
    print(loss2(x, y))
