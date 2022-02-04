import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.utils import degree
import torchmetrics


class DegreeRegressor(pl.LightningModule):
    """Module that regresses node degree, based on a specific GNN model"""

    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

        self.train_metric = torchmetrics.MeanSquaredError()
        # self.valid_metric = torchmetrics.MeanSquaredError()

    def forward(self, graph):
        return self.model(graph.x, graph.edge_index)

    def training_step(self, graph):
        pred = self(graph).squeeze()
        y = degree(graph.edge_index[0]).float()

        loss = F.mse_loss(pred, y)
        self.train_metric(pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
