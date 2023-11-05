"""Training program for the Variational Encoder Decoder
   derived from the Variational Autoencoder at
   https://github.com/arpastrana/neu_vae/
"""

import json
import argparse

import torch
from torch import cuda

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch import optim

import torchvision

from RotateMNIST import RotatingMNIST

# the Model to be trained
from neu_VED import VariationalEncoderDecoder


def train(args, ved, train_loader, optimizer, epoch):
    """Trains the model for one epoch."""
    ved.model.train()
    # loss of the epoch
    train_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        # input_0 = data[0]           # main input t=0
        # input_shape = data[0].size()
        input_1 = data[-1]            # main input t=1
        # operation = data[1:-1]      # another input
        input_with_operation = data[1]
        loss_dict = ved.learn(input_with_operation, input_1, optimizer)

        # log losses
        loss = loss_dict["loss"]
        train_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tBatch: {}\tLoss: {:.6f}'.format(epoch, batch_idx, train_loss / args.log_interval))

        if args.dry_run:
            break


def test(model, epoch, test_loader, writer):
    """Evaluates the trained model."""
    model.eval()
    # loss of the epoch
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kld_loss = 0.0

    with torch.no_grad():
        cnt = 0
        for batch_idx, data in enumerate(test_loader):
            # input_0 = data[0]           # main input t=0
            input_shape = data[0].size()
            input_1 = data[-1]  # main input t=1
            # operation = data[1:-1]     # another input
            input_with_operation = data[1]
            # forward pass
            x_hat, z_mean, z_logvar = model(input_with_operation, None)
            # loss
            loss_dict = model.loss(input_1, x_hat, z_mean, z_logvar)

            loss = loss_dict["loss"]
            recon_loss = loss_dict["recon_loss"]
            kld_loss = loss_dict["kld_loss"]

            # log losses
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kld_loss += kld_loss.item()
            cnt += 1

        test_loss /= cnt
        writer.add_image('train_VED/test/inputs', torchvision.utils.make_grid(torch.reshape(input_1, input_shape)), epoch)
        writer.add_image('train_VED/test/outputs', torchvision.utils.make_grid(torch.reshape(x_hat, input_shape)), epoch)
        writer.add_scalars('train_VED/test/loss',
                           {'test_loss': test_loss / cnt,
                            'test_recon_loss': test_recon_loss / cnt,
                            'test_kld_loss': test_kld_loss / cnt},
                           epoch)

        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


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
    parser.add_argument('--config', type=str, default='neu_VED_beta.json', metavar='N',
                        help='Model configuration (default: neu_VED_beta.json')
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

    if config['model']['type'] == 'neu_beta':
        ved = VariationalEncoderDecoder(config['model'])
        model = ved.model
        config['model']["model"] = model
    else:
        raise NotImplementedError('Model not supported: ' + str(config['model']))

    optimizer = eval("optim." + config['model']['optimizer'] +
                     "(model.parameters(), lr=config['model']['learning_rate'])")

    for epoch in range(1, args.epochs + 1):
        train(args, ved, train_loader, optimizer, epoch)
        test(model, epoch, test_loader, writer)

    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
