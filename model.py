import torch.nn as nn
import torch
from module import RAN

class encoding_layers_1(nn.Module):
    def __init__(self, angle1):
        super(encoding_layers_1, self).__init__()

        self.angle1 = angle1
        self.act = nn.LeakyReLU(0.2)
        self.layers = self._make_layers()

    def narrow_rf(self, x):
        return torch.narrow(input=x, dim=3, start=self.angle1, length=int(x.shape[3]/7))

    def add(self, x1, x2):
        return torch.add(x1, x2)

    def _make_layers(self):
        #  x = self.narrow(x)
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_channels=1, out_channels=4, padding=(1, 1), kernel_size=3, stride=(1, 2)))
        layers.append(self.act)
        #  shortcut = x
        layers.append(nn.Conv2d(in_channels=4, out_channels=4, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=4, out_channels=4, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=4, out_channels=4, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        # x = self.add(x, shortcut)

        layers.append(nn.Conv2d(in_channels=4, out_channels=8, padding=(1, 1), kernel_size=3, stride=(1, 2)))
        layers.append(self.act)
        # shortcut = x
        layers.append(nn.Conv2d(in_channels=8, out_channels=8, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=8, out_channels=8, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=8, out_channels=8, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        # x = self.add(x, shortcut)

        layers.append(nn.Conv2d(in_channels=8, out_channels=16, padding=(1, 1), kernel_size=3, stride=(1, 2)))
        layers.append(self.act)
        # shortcut = x
        layers.append(nn.Conv2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=16, out_channels=16, padding=(1,1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        # x = self.add(x, shortcut)

        layers.append(nn.Conv2d(in_channels=16, out_channels=32, padding=(1, 1), kernel_size=3, stride=(1, 2)))
        layers.append(self.act)
        # shortcut = x
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(self.act)
        # x = self.add(x, shortcut)

        return layers

    def forward(self, x):
        x = self.narrow_rf(x)
        for i in range(0, 2):
            x = self.layers[i](x)
        shortcut = x
        for i in range(2, 8):
            x = self.layers[i](x)
        x = self.add(x, shortcut)

        for i in range(8, 10):
            x = self.layers[i](x)
        shortcut = x
        for i in range(10, 16):
            x = self.layers[i](x)
        x = self.add(x, shortcut)

        for i in range(16, 18):
            x = self.layers[i](x)
        shortcut = x
        for i in range(18, 24):
            x = self.layers[i](x)
        x = self.add(x, shortcut)

        for i in range(24, 26):
            x = self.layers[i](x)
        shortcut = x
        for i in range(26, 32):
            x = self.layers[i](x)
        x = self.add(x, shortcut)

        return x


class encoding_layers_2(nn.Module):
    def __init__(self):
        super(encoding_layers_2, self).__init__()

        self.norm_1 = RAN(128, 128, 128)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.layers = self._make_layers()

    def add(self, x1, x2):
        return torch.add(x1, x2)

    def maxpool(self, x, kernel_size, stride):
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)(x)

    def _make_layers(self):
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_channels=224, out_channels=128, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(128))
        layers.append(self.act)
        # shortcut = x
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(128))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=128, out_channels=128, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(128))
        layers.append(self.act)
        # x = self.add(x, shortcut)
        # x = self.maxpool(x, kernel_size=2, stride=(2, 2))

        layers.append(nn.Conv2d(in_channels=128, out_channels=256, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(256))
        layers.append(self.act)
        #        shortcut = x
        layers.append(nn.Conv2d(in_channels=256, out_channels=256, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(256))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=256, out_channels=256, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(256))
        layers.append(self.act)
        # x = self.add(x, shortcut)
        # x = self.maxpool(x, kernel_size=2, stride=(2, 2))

        layers.append(nn.Conv2d(in_channels=256, out_channels=512, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(512))
        layers.append(self.act)
        #        shortcut = x
        layers.append(nn.Conv2d(in_channels=512, out_channels=512, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(512))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=512, out_channels=512, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(512))
        layers.append(self.act)
        # x = self.add(x, shortcut)
        # x = self.maxpool(x, kernel_size=2, stride=(2, 2))

        layers.append(nn.Conv2d(in_channels=512, out_channels=1024, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(self.act)
        #        shortcut = x
        layers.append(nn.Conv2d(in_channels=1024, out_channels=1024, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(self.act)
        layers.append(nn.Conv2d(in_channels=1024, out_channels=1024, padding=(1, 1), kernel_size=3, stride=(1, 1)))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(self.act)
        # x = self.add(x, shortcut)
        # x = self.maxpool(x, kernel_size=2, stride=(2, 2))

        return layers

    def forward(self, x, roi_seg_info):
        x = self.layers[0](x)
        x = self.norm_1(x, roi_seg_info)
        x = self.layers[2](x)
        shortcut = x
        x = self.layers[3](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = self.layers[8](x)
        x = self.add(x, shortcut)
        x = self.maxpool(x, kernel_size=2, stride=(2, 2))
        x = self.dropout(x)

        for i in range(9, 12):
            x = self.layers[i](x)
        shortcut = x
        for i in range(12, 18):
            x = self.layers[i](x)
        x = self.add(x, shortcut)
        x = self.maxpool(x, kernel_size=2, stride=(2, 2))
        x = self.dropout(x)

        for i in range(18, 21):
            x = self.layers[i](x)
        shortcut = x
        for i in range(21, 27):
            x = self.layers[i](x)
        x = self.add(x, shortcut)
        x = self.maxpool(x, kernel_size=2, stride=(2, 2))
        x = self.dropout(x)

        for i in range(27, 30):
            x = self.layers[i](x)
        shortcut = x
        for i in range(30, 36):
            x = self.layers[i](x)
        x = self.add(x, shortcut)
        x = self.maxpool(x, kernel_size=2, stride=(2, 2))
        x = self.dropout(x)

        return x

class fully_connected_layers(nn.Module):
    def __init__(self):
        super(fully_connected_layers, self).__init__()

        self.dropout =nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU(0.2)
        self.layers = self._make_layers()

    def _make_layers(self):
        # 8 x 8 x 1024
        layers = nn.ModuleList()
        layers.append(nn.Linear(8 * 8 * 1024,1000))
        layers.append(self.act)
        layers.append(self.dropout)
        layers.append(nn.Linear(1000,100))
        layers.append(self.act)
        layers.append(self.dropout)
        layers.append(nn.Linear(100,10))
        layers.append(self.act)
        layers.append(self.dropout)
        layers.append(nn.Linear(10,1))
        return layers

    def forward(self, x):
        x = x.view(-1, 8 * 8 * 1024)
        for i in range(0, 10):
            x = self.layers[i](x)

        return x

class create_model(nn.Module):
    def __init__(self):
        super(create_model, self).__init__()

        self.layers1 = encoding_layers_1(0*2048)
        self.layers2 = encoding_layers_1(1*2048)
        self.layers3 = encoding_layers_1(2*2048)
        self.layers4 = encoding_layers_1(3*2048)
        self.layers5 = encoding_layers_1(4*2048)
        self.layers6 = encoding_layers_1(5*2048)
        self.layers7 = encoding_layers_1(6*2048)

        self.layers1234567 = encoding_layers_2()

        self.layers_output = fully_connected_layers()

    def concat(self, tensors, dim=1):
        return torch.cat(tensors=tensors, dim=dim)

    def maxpool(self, x, kernel_size, stride):
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)(x)

    def forward(self, x, seg_info):
        l1_1 = self.layers1(x)
        l1_2 = self.layers2(x)
        l1_3 = self.layers3(x)
        l1_4 = self.layers4(x)
        l1_5 = self.layers5(x)
        l1_6 = self.layers6(x)
        l1_7 = self.layers7(x)

        l2_pre = self.concat((l1_1, l1_2, l1_3, l1_4, l1_5, l1_6, l1_7))

        l2 = self.layers1234567(l2_pre, seg_info)

        loutput = self.layers_output(l2)

        return loutput


def dimension_check():

    x = torch.randn(2, 1, 128, 2048*7)
    seg_info = torch.randn(2,1,1,11)

    loutput = create_model()(x, seg_info)
    print(loutput.shape)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



#dimension_check()
#model = create_model()
#print(count_parameters(model))
