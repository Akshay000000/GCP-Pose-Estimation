"""
Utility functions: seeding, early stopping, logging, visualisation.
"""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class MetricLogger:
    """Simple epoch-level metric logger."""

    def __init__(self):
        self.history = {}

    def log(self, epoch: int, metrics: dict):
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(v)

        # Pretty print
        parts = [f"Epoch {epoch:>3d}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        print(" │ ".join(parts))

    def get(self, key):
        return self.history.get(key, [])
