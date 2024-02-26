import time

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, val_loader, log_dir, n_epochs, patience, checkpoint_path):
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir)

    best_val_loss = float("inf")
    best_epoch = 0
    early_stopping_counter = 0
    start_time = time.time()

    n_train_batches = len(train_loader)
    n_val_data = len(val_loader.dataset)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for i, (input_data, target) in enumerate(train_loader):
            loss = model.update(input_data.to(device), target.to(device))
            train_loss += loss

        train_loss /= n_train_batches

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_data, target in val_loader:
                output = model.predict(input_data.to(device))
                loss = model.compute_loss(output, target.to(device)).sum().item()
                val_loss += loss

        val_loss /= n_val_data
        epoch_time = time.time() - start_time

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(
            f"Epoch [{epoch + 1}/{n_epochs}]: "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Time: {epoch_time / 60:.2f} min"
        )

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stopping_counter = 0

            # Save model checkpoint
            model.save(checkpoint_path)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    writer.close()
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Val loss: {best_val_loss:.4f}")


def predict(model, data_loader):
    device = next(model.parameters()).device
    model.eval()

    predictions = []
    with torch.no_grad():
        for input_data, _ in data_loader:
            output = model.predict(input_data.to(device))
            if output.shape[1] != 1:
                predicted = torch.softmax(output, dim=1).argmax(dim=1)
            else:
                predicted = output.squeeze(1)
            predictions.extend(predicted.cpu().numpy())

    return np.array(predictions, dtype=int)
