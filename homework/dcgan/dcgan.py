import torch.nn as nn

class DCGenerator(nn.Module):

    def __init__(self, image_size):
        super(DCGenerator, self).__init__()
        self.layer1 = nn.ConvTranspose2d(100, 1024, 4)
        self.norm1 = nn.BatchNorm2d(1024)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        self.activation3 = nn.ReLU()
        self.layer4 = nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1)
        self.activation4 = nn.Tanh()


    def forward(self, data):
        # TODO your code here
        x = self.activation1(self.norm1(self.layer1(data)))
        x = self.activation2(self.norm2(self.layer2(x)))
        x = self.activation3(self.norm3(self.layer3(x)))
        x = self.activation4(self.layer4(x))
        return x


class DCDiscriminator(nn.Module):

    def __init__(self, image_size):
        super(DCDiscriminator, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, 4, stride=2)
        self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        self.layer2 = nn.Conv2d(64, 128, 4, stride=2)
        self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        self.layer3 = nn.Conv2d(128, 256, 4, stride=2)
        self.activation3 = nn.LeakyReLU(0.2, inplace=True)
        self.layer4 = nn.Linear(1024, 1)
        self.activation4 = nn.Sigmoid()


    def forward(self, data):
        # TODO your code here
        x = self.activation1(self.layer1(data))
        x = self.activation2(self.layer2(x))
        x = self.activation3(self.layer3(x))
        x = self.activation4(self.layer4(x.reshape(x.shape[0], 1024)))
        return x
