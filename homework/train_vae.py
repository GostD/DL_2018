import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from vae.vae import VAE, loss_function
from vae.trainer import Trainer

def main():
    # TODO your code here

    transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
    dataset_train = datasets.MNIST(root='data/', download=True,
                               transform=transform, train=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle=True,
                                             num_workers=4, pin_memory=True)

    dataset_test = datasets.MNIST(root='data/', download=True,
                                     transform=transform, train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=256, shuffle=True,
                                                   num_workers=4, pin_memory=True)

    vae_mod = VAE()

    trainer = Trainer(vae_mod, dataloader_train, dataloader_test, Adam(vae_mod.parameters(), lr=0.01),#, betas=(0.9, 0.999)),
                 loss_function, device='cpu')

    for epoch in range(12):
        trainer.train(epoch, 100)
        trainer.test(epoch, 256, 100)


if __name__ == '__main__':
    main()
