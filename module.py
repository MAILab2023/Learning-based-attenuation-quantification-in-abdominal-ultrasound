
import torch
import torch.nn as nn

class RAN(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()

        self.channel = C
        self.hight = H
        self.width = W

        self.param_free_norm = nn.InstanceNorm2d(self.channel, affine=False)

        self.mlp_gamma = nn.Sequential(
            nn.Linear(self.hight,self.hight),
            nn.ReLU(),
            nn.Linear(self.hight, self.hight),
            nn.ReLU(),
            nn.Linear(self.hight, self.hight),
            nn.ReLU(),
            nn.Linear(self.hight, self.hight),
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(self.hight,self.hight),
            nn.ReLU(),
            nn.Linear(self.hight, self.hight),
            nn.ReLU(),
            nn.Linear(self.hight, self.hight),
            nn.ReLU(),
            nn.Linear(self.hight, self.hight)
        )
        self.act = nn.ReLU()
        self.layers = self._make_layers()

    def _make_layers(self):
        # 1 x 1 x 11
        layers = nn.ModuleList()
        layers.append(nn.Linear(11,self.hight))
        layers.append(self.act)
        layers.append(nn.Linear(self.hight,self.hight))
        layers.append(self.act)
        layers.append(nn.Linear(self.hight,self.hight))
        layers.append(self.act)
        layers.append(nn.Linear(self.hight,self.hight))
        layers.append(self.act)

        return layers

    def forward(self, x, segmap):

        # Part 1.
        normalized = self.param_free_norm(x)

        # Part 2.
        segmap = segmap.view(-1, 1 * 1 * 11)
        for i in range(0, 8):
            segmap = self.layers[i](segmap)

        gamma = self.mlp_gamma(segmap)
        gamma = gamma.view(-1, 1, self.hight, 1)
        gamma = torch.cat(self.width*[gamma], 3)
        gamma = torch.cat(self.channel * [gamma], 1)

        beta = self.mlp_beta(segmap)
        beta = beta.view(-1, 1, self.hight, 1)
        beta = torch.cat(self.width*[beta], 3)
        beta = torch.cat(self.channel * [beta], 1)

        out = normalized * (1 + gamma) + beta

        return out


