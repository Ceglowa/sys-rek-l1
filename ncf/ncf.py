import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

# based on https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    """

    def __init__(self, n_users, n_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=15)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=15)
        self.fc1 = nn.Linear(in_features=30, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=1)

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.MSELoss()(predicted_labels.squeeze(-1), labels.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.MSELoss()(predicted_labels.squeeze(-1), labels.view(-1, 1).float())

        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
