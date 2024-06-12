import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


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
    
class HNB(nn.Module):
    def __init__(self, dim):
        super(HNB, self).__init__()
        self.mlp = Mlp(dim, dim * 2)
        
    def forward(self, x):
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
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, 3 * 16, kernel_size=3, stride=1, padding=1)
        self.up = torch.nn.PixelShuffle(4)

    def forward(self, x):
        return self.up(self.conv(x))
     
class Control(nn.Module):
    def __init__(self, embed_dim=60, num_in_ch=3, blocks=4, num_active=4):
        super(Control, self).__init__()
        self.num_active = num_active

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        self.body = nn.Sequential(*[
            Extract(dim=embed_dim, block_idx=idx) for idx in range(blocks)
        ])

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)

        #self.out = nn.Sequential(*[
        #    Upsample2D(embed_dim),
        #    Upsample2D(embed_dim),
        #    nn.Conv2d(embed_dim, 3, 3, 1, 1)
        #])

        self.out = UpShuffle2D(embed_dim)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):#(nn.Conv2d, )):
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


if __name__ == '__main__':
    from torchstat import stat
    model = Control()#.cuda()
    x = torch.randn((2, 3, 64, 64))#.cuda()
    x = model(x)
    print(x.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    stat(model, (3, 64, 64))

"""
Total params: 967,548
-------------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 37.12MB
Total MAdd: 7.92GMAdd
Total Flops: 3.96GFlops
Total MemR+W: 53.24MB
"""