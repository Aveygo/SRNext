import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath

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
    def __init__(self, num_feat, squeeze_factor=16):
        super(CAB, self).__init__()
        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )
        self.next = Next(num_feat)

    def forward(self, x):
        return self.next(x)
    
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

class HNB(nn.Module):
    def __init__(self, dim):
        super(HNB, self).__init__()
        self.cab = CAB(dim)
        self.mlp = Mlp(dim, dim * 2)
        
    def forward(self, x):
        x = x + self.cab(x)
        x = x + self.mlp(x)
        return x

class Extract(nn.Module):
    def __init__(self, dim:int, block_idx:int):
        super().__init__()
        self.blocks = nn.Sequential(*[
            HNB(dim)
        ] * 6)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.block_idx = block_idx

    def forward(self, x):
        return x + self.conv(self.blocks(x))
        
        
class Upsample2D(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.up = torch.nn.Upsample(scale_factor = 2, mode='bilinear')
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.act(self.conv(self.pad(self.up(x))))

class UpShuffle2D(torch.nn.Module):
    def __init__(self, channels, scale=4):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, 3 * scale * scale, kernel_size=3, stride=1, padding=1)
        self.up = torch.nn.PixelShuffle(scale)

    def forward(self, x):
        return self.up(self.conv(x))
     
class SRNext(nn.Module):
    def __init__(self, embed_dim=60, num_in_ch=3, blocks=4, num_active=4):
        super(SRNext, self).__init__()
        self.num_active = num_active

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        self.body = nn.Sequential(*[
            Extract(dim=embed_dim, block_idx=idx) for idx in range(blocks)
        ])

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        self.out = UpShuffle2D(embed_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean)
        x = self.conv_first(x)
        x = x + self.conv_after_body(self.body(x))
        x = self.out(x)
        return x + self.mean
    
    @property
    def input_shape(self):
        return (1, 3, 64, 64)

    @property
    def output_shape(self):
        return (1, 3, 256, 256)
