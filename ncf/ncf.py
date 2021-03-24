import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

# based on https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e


class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    """

    def __init__(self, embedding_size, first_layer_size, second_layer_size, dropout_prob, n_users, n_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_size)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_size)
        self.fc1 = nn.Linear(in_features=2 * embedding_size, out_features=first_layer_size)
        self.d1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(in_features=first_layer_size, out_features=second_layer_size)
        self.d2 = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(in_features=second_layer_size, out_features=1)

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        vector = self.d1(nn.ReLU()(self.fc1(vector)))
        vector = self.d2(nn.ReLU()(self.fc2(vector)))

        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.MSELoss()(predicted_labels.squeeze(-1), labels.float())
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.MSELoss()(predicted_labels.squeeze(-1), labels.float())
        mae = nn.L1Loss()(predicted_labels.squeeze(-1), labels.float())
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
