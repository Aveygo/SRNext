import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange, Reduce

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class Next(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to("cuda").float()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1, bias=False),
        )

        self.next = Next(num_feat)

    def forward(self, x):
        return self.cab(x) * self.attention(x) + self.next(x)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, 1, 1, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Unit(nn.Module):
    def __init__(self, dim):
        super(Unit, self).__init__()
        
        self.ln1 = LayerNorm(dim, data_format="channels_first")
        self.ln2 = LayerNorm(dim, data_format="channels_first")

        self.cab = CAB(dim)
        self.mlu = Mlp(dim, dim * 1)

        self.drop = nn.Dropout2d(0.01)
        self.noise = GaussianNoise(0.1)

    def forward(self, x):
        #x = self.drop(self.noise(x))
        x = x + self.cab(self.ln1(x)) * 0.2
        #x = self.drop(self.noise(x))
        x = x + self.mlu(self.ln2(x)) * 0.2
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num=1):
        super(Block, self).__init__()
        self.blocks = nn.Sequential(*[Unit(dim) for i in range(num)])

    def forward(self, x):
        return self.blocks(x)

class UpShuffle2D(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, 3 * 16, kernel_size=3, stride=1, padding=1)
        self.up = torch.nn.PixelShuffle(4)

    def forward(self, x):
        return self.up(self.conv(x))

class UpSample(torch.nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.up = torch.nn.Upsample(scale_factor = 2, mode='bilinear')
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.act(self.conv(self.pad(self.up(x))))

class DownSample(torch.nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.act(self.conv(x))

class uNet(nn.Module):
    def __init__(self, dim=64):
        super(uNet, self).__init__()

        self.down1 = nn.Sequential(*[
            Block(dim*1), DownSample(dim*1, dim*2)
        ])

        self.down2 = nn.Sequential(*[
            Block(dim*2), DownSample(dim*2, dim*4)
        ])
        
        self.down3 = nn.Sequential(*[
            Block(dim*4), DownSample(dim*4, dim*8)
        ])

        self.mid = nn.Sequential(*[
            Block(dim*8)
        ])

        self.up1 = nn.Sequential(*[
            UpSample(dim*8, dim*4), Block(dim*4), 
        ])

        self.up2 = nn.Sequential(*[
            UpSample(dim*4, dim*2), Block(dim*2), 
        ])

        self.up3 = nn.Sequential(*[
            UpSample(dim*2, dim*1), Block(dim*1)
        ])
    
    def forward(self, x0):
        # Down layers
        x2 = self.down1(x0)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Main body
        x = self.mid(x4)

        # Up layers
        x = self.up1(x+x4)
        x = self.up2(x+x3)
        x = self.up3(x+x2)
        return x + x0


class uNextPrototypeV2(nn.Module):
    def __init__(self, dim=64):
        super(uNextPrototypeV2, self).__init__()
        self.dim = dim
        
        self.m_head = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.unets = nn.Sequential(*[uNet(dim) for i in range(4)])
        
        #self.upsampling = UpShuffle2D(dim)
        self.upsampling = torch.nn.Sequential(
            UpSample(dim, dim),
            UpSample(dim, dim),
            nn.Conv2d(dim, 3, 3, 1, 1, bias=False)
        )

        self.apply(self._init_weights)
    
    @property
    def input_shape(self):
        return (1, 3, 64, 64)

    @property
    def output_shape(self):
        return (1, 3, 256, 256)

    def forward(self, x_in):

        x = self.m_head(x_in)
        x = self.unets(x) + x
        x = self.upsampling(x)
        return F.upsample(x_in, scale_factor=4) - x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    net = uNextPrototypeV2().cuda()
    x = torch.randn((2, 3, 64, 64)).cuda()
    x = net(x)
    print(x.shape)