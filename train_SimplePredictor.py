"""Training Perceptron-based Predictor
"""

import json
import argparse

import torch
from torch import cuda

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision

from SimplePredictor import SimplePredictor
from RotateMNIST import RotatingMNIST


def train(args, predictor, train_loader, epoch):
    loss_sum = 0.0
    for batch_idx, data in enumerate(train_loader):
        # input_0 = data[0]          # main input t=0
        # input_shape = data[0].size()
        input_1 = data[-1]           # main input t=1
        # operation = data[1:-1]     # another input
        input_with_operation = data[1]
        predictor.learn(input_with_operation, input_1)

        if args.dry_run:
            break


def test(predictor, epoch, test_loader, writer):
    """Evaluates the trained model."""
    predictor.predictor.eval()
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
            predicted = predictor.predictor(input_with_operation)
            loss = predictor.loss_func(predicted, input_1)
            # log losses
            loss_sum += loss.item()
            # cnt += 1
        writer.add_image('train_SimplePredictor/test/inputs',
                         torchvision.utils.make_grid(torch.reshape(input_1, input_shape)),
                         epoch)
        writer.add_image('train_SimplePredictor/test/outputs',
                         torchvision.utils.make_grid(torch.reshape(predicted, input_shape)),
                         epoch)
        writer.add_scalar('train_SimplePredictor/test/loss', loss_sum / loader_size, epoch)
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
    parser.add_argument('--config', type=str, default='train_SimplePredictor.json', metavar='N',
                        help='Model configuration (default: train_SimplePredictor.json')
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

    if config['model']['type'] == 'simple':
        sp = SimplePredictor(config['model'])
        model = sp.predictor
    else:
        raise NotImplementedError('Model not supported: ' + str(config['model']))

    for epoch in range(1, args.epochs + 1):
        train(args, sp, train_loader, epoch)
        test(sp, epoch, test_loader, writer)

    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
