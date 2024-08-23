import torch
import numpy as np
from tqdm import tqdm
from . import config
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from .model import YOLOv1
from .loss import SumSquaredErrorLoss
from .dataset import PascalVocDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True) # Check for NAN loss
    now = datetime.now()


    model = YOLOv1().to(device)
    loss_function = SumSquaredErrorLoss()
    optimizer = torch.optim.adam.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_set = PascalVocDataset("train", normalize=True)
    test_set = PascalVocDataset("test", normalize=True)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        # drop the last incomplete batch
        drop_last=True,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        # drop the last incomplete batch
        drop_last=True
    )
