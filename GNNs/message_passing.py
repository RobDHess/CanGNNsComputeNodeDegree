import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MPNN(nn.Module):
    """Message passing model with initial MLP embedding and output head"""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        num_layers,
        act=nn.ReLU,
        aggr="add",
    ):
        super().__init__()

        self.embedder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
        )

        layers = []
        for i in range(num_layers):
            layers.append(
                MPNNLayer(hidden_features, hidden_features, hidden_features, act, aggr)
            )

        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x, edge_index):
        x = self.embedder(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.head(x)

        return x


class MPNNLayer(MessagePassing):
    """Message passing layer"""

    def __init__(self, in_features, hidden_features, out_features, act, aggr):
        super().__init__(aggr=aggr)

        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.message_net(torch.cat((x_i, x_j), dim=-1))

    def update(self, message, x):
        return self.update_net(torch.cat((x, message), dim=-1))


if __name__ == "__main__":
    model = MPNN(3, 128, 5, 7)
    print(model)
