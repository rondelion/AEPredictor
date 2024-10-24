"""Training Perceptron-based Predictor
"""

import json
import argparse

import torch
from torch import cuda
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision

from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder
from RotateMNIST import RotatingMNIST


def regularization_term(regularization, encoded):
    if regularization == "L1":  # for sparseness
        return torch.mean(torch.abs(encoded))
    else:
        return 0.0


def train(model, train_loader, optimizer, loss_func, regularization, reg_param, dry_run):
    for batch_idx, data in enumerate(train_loader):
        # input_0 = data[0]          # main input t=0
        # input_shape = data[0].size()
        input_1 = data[-1]           # main input t=1
        # operation = data[1:-1]     # another input
        input_with_operation = data[1]
        optimizer.zero_grad()
        encoded, decoded = model(input_with_operation)

        loss = loss_func(decoded, input_1) + reg_param * regularization_term(regularization, encoded)
        loss.backward()
        optimizer.step()

        if dry_run:
            break


def test(model, epoch, test_loader, loss_func, writer):
    """Evaluates the trained model."""
    model.eval()
    # loss of the epoch
    loss_sum = 0.0
    loader_size = len(test_loader)
    with torch.no_grad():
        # cnt = 0
        for batch_idx, data in enumerate(test_loader):
            # input_0 = data[0]           # main input t=0
            input_shape = data[0].size()
            input_1 = data[-1]  # main input t=1
            # operation = data[1:-1]     # another input
            input_with_operation = data[1]
            # forward pass
            encoded, decoded = model(input_with_operation)
            loss = loss_func(decoded, input_1)
            # log losses
            loss_sum += loss.item()
            # cnt += 1
        writer.add_image('train_SimpleAEPredictor/test/inputs',
                         torchvision.utils.make_grid(torch.reshape(input_1, input_shape)),
                         epoch)
        writer.add_image('train_SimpleAEPredictor/test/outputs',
                         torchvision.utils.make_grid(torch.reshape(decoded, input_shape)),
                         epoch)
        writer.add_scalar('train_SimpleAEPredictor/test/loss', loss_sum / loader_size, epoch)
        print('Test Epoch: {} \t Average loss: {:.4f}'.format(epoch, loss_sum / loader_size))


def set_loaders(config, **kwargs):
    if config['type'] == 'RotatingMNIST':
        train_dataset = RotatingMNIST(config['data_path'], train=True, download=True)
        test_dataset = RotatingMNIST(config['data_path'], train=False, download=True)
        train_loader = DataLoader(train_dataset,  shuffle=True, batch_size=config['batch_size'], **kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'], **kwargs)
    else:
        print("Loader type" + "'" + config['type'] + "' is not supported!")
        exit(-1)
    return train_loader, test_loader


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='AEPredictor')
    parser.add_argument('--config', type=str, default='AEPredictor_simple.json', metavar='N',
                        help='Model configuration (default: AEPredictor_simple.json')
    parser.add_argument('--config-path', type=str, default='', metavar='N',
                        help='Path within the configuration (default: ""')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', help='File path for saving the current Model')

    args = parser.parse_args()

    with open(args.config) as config_file:
        ca_config = json.load(config_file)

    config = ca_config
    if args.config_path.strip() != '':
        config_path = args.config_path.strip().split('/')
        for i in range(len(config_path)):
            if config_path[i] in config:
                config = config[config_path[i]]
            else:
                print("Config doesn't match config path!")
                exit(1)

    torch.manual_seed(args.seed)
    cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config['model']["device"] = device

    kwargs = {}

    if use_cuda:
        kwargs.update({
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        })

    writer = SummaryWriter()

    train_loader, test_loader = set_loaders(config['dataset'], **kwargs)

    if config['model']['type'] == 'cerenaut_pt_simple_ae':
        model = SimpleAutoencoder(eval(config['model']['input_shape']), config['model'],
                                  eval(config['model']['output_shape']))
    else:
        raise NotImplementedError('Model not supported: ' + str(config['model']))

    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    loss_func = eval(config["loss_func"])

    regularization = config.get('regularization')
    reg_param = config.get('reg_param', 0.0)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, loss_func, regularization, reg_param, args.dry_run)
        test(model, epoch, test_loader, loss_func, writer)

    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
