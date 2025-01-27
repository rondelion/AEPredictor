import torch
import torch.utils.data
import torch.nn as nn
from torch import optim


class SimplePredictor:
    # Predictor based on Perceptron
    def __init__(self, config):
        self.ae = None
        if "activation_func" in config:
            activation_func = eval(config["activation_func"])
        else:
            activation_func = nn.Sigmoid()
        self.predictor = Perceptron(config['input_dim'],
                                    config['hidden_dim'],
                                    config['output_dim'],
                                    activation_func)
        self.predictor.optimizer = optim.Adam(self.predictor.parameters())
        if "loss_func" in config:
            self.loss_func = eval(config["loss_func"])
        else:
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
    def __init__(self, in_dim, hidden_dim, out_dim, activation_func):
        super(Perceptron, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, out_dim)
        # self.seq = torch.nn.Sequential(nn.Linear(in_dim, hidden_dim), self.activation_func(),
        #                                nn.Linear(hidden_dim, out_dim), self.activation_func())
        self.optimizer = None
        self.activation_func = activation_func

    def forward(self, x):
        h = self.l1(x)
        h = self.relu(h)
        h = self.l2(h)
        y = self.activation_func(h)
        return y



