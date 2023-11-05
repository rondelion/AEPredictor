import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional
from torch import optim
from functools import partial


class SimplePredictor:
    # Predictor based on Perceptron
    def __init__(self, config):
        self.ae = None
        self.predictor = Perceptron(config['input_dim'],
                                    config['hidden_dim'],
                                    config['output_dim'])
        self.predictor.optimizer = optim.Adam(self.predictor.parameters())
        self.loss_func = nn.BCELoss()

    def learn(self, x, y):
        self.predictor.train()
        self.predictor.optimizer.zero_grad()
        output = self.predictor(x)
        loss = self.loss_func(output, y)
        loss.backward()
        self.predictor.optimizer.step()
        return loss.item()


class Perceptron(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Perceptron, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.seq = torch.nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid(),
                                       nn.Linear(hidden_dim, out_dim), nn.Sigmoid())
        self.optimizer = None

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        h = self.l2(h)
        y = torch.sigmoid(h)
        return y



