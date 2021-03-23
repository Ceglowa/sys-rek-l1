import pytorch_lightning as pl
from torch import nn


class UserAutoEncoder(pl.LightningModule):
    def __init__(self):
        super(UserAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
