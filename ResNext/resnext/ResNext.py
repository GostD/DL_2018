import torch

class BasicRes(torch.nn.Module):
    def __init__(self, dim_in, dim_out, stride=1, cardinality=32):
        super(BasicRes, self).__init__()
        self.l1 = torch.nn.Conv2d(dim_in, dim_in, 1)
        self.bn1 = torch.nn.BatchNorm2d(dim_in)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Conv2d(dim_in, dim_in, 3, padding=1, stride=stride, groups=cardinality)
        self.bn2 = torch.nn.BatchNorm2d(dim_in)
        self.l3 = torch.nn.Conv2d(dim_in, dim_out, 1)
        self.bn3 = torch.nn.BatchNorm2d(dim_out)

        self.ll = torch.nn.Conv2d(dim_in, dim_out, 1, stride=stride)

        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x):
        x_id = x
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        if self.dim_in != self.dim_out:
            x_id = self.ll(x_id)
            x_id = self.bn3(x_id)
        x = x + x_id
        x = self.relu(x)
        return x


class resNextBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, times, stride=1):
        super(resNextBlock, self).__init__()
        self.l1 = BasicRes(dim_in, dim_out, stride = stride)
        self.l2 = BasicRes(dim_out, dim_out)
        self.times = times

    def forward(self, x):
        x = self.l1(x)
        for i in range(1, self.times):
            x = self.l2(x)
        return x

class ResNext(torch.nn.Module):
    def __init__(self, num_classes=1000, avgOut = 7):
        super(ResNext, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = resNextBlock(64, 128, 3)
        self.layer2 = resNextBlock(128, 256, 4, 2)
        self.layer3 = resNextBlock(256, 512, 6, 2)
        self.layer4 = resNextBlock(512, 1024, 3, 2)

        self.avgpool = torch.nn.AvgPool2d(avgOut, stride=1)
        self.fc = torch.nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y_pred = self.fc(x)

        return y_pred

import math

 
def test_forward(m):
    imgSz = m
    num_classes = 20
    batch_size = 4
    tensor = torch.rand(batch_size, 3, imgSz, imgSz)
    net = ResNext(num_classes=num_classes, avgOut=math.ceil(imgSz/32))
    output = net(tensor)
    assert output.size() == (batch_size, num_classes)


test_forward(224)