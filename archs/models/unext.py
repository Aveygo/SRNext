import torch
import torch.nn as nn
from torch.nn import functional as F

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
    
class WMSA(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=head_dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        
        x = Rearrange('b h w c -> b c h w')(x)
        x = self.dwconv(x)
        x = Rearrange('b c h w -> b h w c')(x)
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        return x

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, head_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = Rearrange('b c h w -> b h w c')(x)
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        x = Rearrange('b h w c -> b c h w')(x)
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
    
class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim):
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        
        self.trans_block = Block(self.trans_dim, self.trans_dim, head_dim)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.drop = nn.Dropout2d(0)
        self.noise = GaussianNoise(0)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x, trans_x = self.drop(self.noise(conv_x)), self.drop(self.noise(trans_x))

        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x)

        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        return x + res*0.2

class Upsample2D(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = torch.nn.LeakyReLU()
        self.up = torch.nn.Upsample(scale_factor = 2, mode='bilinear')
        self.pad = torch.nn.ReflectionPad2d(1)
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.act(self.conv(self.pad(self.up(x))))
    
class uNextPrototype(nn.Module):
    def __init__(self, config=[2,2,2,2,2,2,2], dim=64, head_dim=32):
        super(uNextPrototype, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        
        self.m_head = nn.Sequential(*[nn.Conv2d(3, dim, 3, 1, 1, bias=False)])

        self.m_down1 = nn.Sequential(*[ConvTransBlock(dim//2, dim//2, self.head_dim) for i in range(config[0])] + [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)])
        self.m_down2 = nn.Sequential(*[ConvTransBlock(dim, dim, self.head_dim) for i in range(config[1])] + [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)])
        self.m_down3 = nn.Sequential(*[ConvTransBlock(2*dim, 2*dim, self.head_dim) for i in range(config[2])] + [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)])
        
        self.m_body = nn.Sequential(*[ConvTransBlock(4*dim, 4*dim, self.head_dim) for i in range(config[3])])

        self.m_up3 = nn.Sequential(*[nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),] + [ConvTransBlock(2*dim, 2*dim, self.head_dim) for i in range(config[4])])
        self.m_up2 = nn.Sequential(*[nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),] + [ConvTransBlock(dim, dim, self.head_dim) for i in range(config[5])])
        self.m_up1 = nn.Sequential(*[nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + [ConvTransBlock(dim//2, dim//2, self.head_dim) for i in range(config[6])])

        self.upsampling = torch.nn.Sequential(
            Upsample2D(dim),
            Upsample2D(dim),
        )

        self.m_tail = nn.Sequential(*[nn.Conv2d(dim, 3, 3, 1, 1, bias=False)])  
        self.apply(self._init_weights)
    
    @property
    def input_shape(self):
        return (1, 3, 64, 64)

    @property
    def output_shape(self):
        return (1, 3, 256, 256)

    def forward(self, x_in):

        x0 = self.m_head(x_in)

        # Down layers
        x2 = self.m_down1(x0)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        # Main body
        x = self.m_body(x4)

        # Up layers
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)

        # Upsampling
        x = self.upsampling(x+ x0)
        
        # Out
        return self.m_tail(x )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    net = uNextPrototype().cuda()
    x = torch.randn((2, 3, 64, 64)).cuda()
    x = net(x)
    print(x.shape)