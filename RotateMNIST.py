import argparse
import numpy as np

import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RotatingMNIST(torch.utils.data.Dataset):

    def __init__(self, data, **kwargs):
        super(RotatingMNIST).__init__()
        transform = transforms.Compose([transforms.ToTensor()])    #, transforms.Normalize((0.5,), (0.5,))])
        self.mnist = MNIST(data, transform=transform, **kwargs)

    def __getitem__(self, idx):
        rot_time = np.random.randint(0, 5)
        input, _ = self.mnist.__getitem__(idx)
        output = TF.rotate(input, rot_time * 90)
        rotation = np.zeros(5, dtype=int)
        rotation[rot_time] = 1
        rotation = torch.from_numpy(rotation.astype(np.float32)).clone()
        input_with_rotation = torch.cat((torch.flatten(input), rotation))
        return input, input_with_rotation, rotation, torch.flatten(output)

    def __len__(self):
        return self.mnist.__len__()


def main():
    parser = argparse.ArgumentParser(description='RotateMNIST')
    parser.add_argument('--data', type=str, default='data', metavar='N',
                        help='MNIST data folder (default: data')
    args = parser.parse_args()
    kwargs = {'download': True}
    dataset = RotatingMNIST(args.data, **kwargs)
    kwargs = {'batch_size': 50}
    loader = DataLoader(dataset, shuffle=True, **kwargs)
    writer = SummaryWriter()
    for batch_idx, (input, rotation, output) in enumerate(loader):
        writer.add_image('test/inputs', torchvision.utils.make_grid(input), batch_idx)
        writer.add_image('test/outputs', torchvision.utils.make_grid(output), batch_idx)
        if batch_idx % 10 == 0:
            print(batch_idx)


if __name__ == '__main__':
    main()
