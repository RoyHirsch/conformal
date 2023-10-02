import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 out_dim=1,
                 hidden_dim=None,
                 drop_rate=0,
                 norm=False,
                 criteria_name='mse'):

        super().__init__()
        self.norm = norm
        self.hidden_dim = hidden_dim
        self.criteria_name = criteria_name
        if norm:
            self.norm = nn.LayerNorm(input_dim)
        if hidden_dim == None:
            self.layers = nn.Linear(input_dim, out_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            if drop_rate:
                layers.append(nn.Dropout(p=drop_rate))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.layers = nn.Sequential(*layers)

        if criteria_name == 'bce':
            self.post = nn.Sigmoid()

    def forward(self, x):
        if self.criteria_name == 'bce':
            return self.post(self.layers(x))
        else:
            return self.layers(x)

