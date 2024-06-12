from utils.recurrentgemma import torch as RG

import torch
from einops import rearrange
from torch import nn
from timm.models.layers import trunc_normal_

from torch.utils import checkpoint # TODO

Cache = dict[str, RG.modules.ResidualBlockCache]

class GemmaBlock(nn.Module):
    def __init__(self, dim:int, expand_ratio:int=3, num_heads:int=2):
        super().__init__()
        block_types = [
            RG.TemporalBlockType.RECURRENT,
            RG.TemporalBlockType.ATTENTION
        ]
        device = torch.device("cuda")
        dtype = torch.zeros(1, device=device).float().cuda().dtype
        self.blocks = nn.ModuleList([
            RG.modules.ResidualBlock(
                width=dim,
                mlp_expanded_width=expand_ratio * dim,
                num_heads=num_heads,
                attention_window_size=dim,
                temporal_block_type=block_type,
                lru_width=dim,
                final_w_init_variance_scale=2.0 / len(block_types),
                device=device,
                dtype=dtype
        )
        for block_type in block_types
    ])
    
    def forward(self, x, cache: Cache | None = None):
        pos = torch.arange(x.shape[1])
        pos = torch.repeat_interleave(pos[None], x.shape[0], dim=0)
        pos = pos.to(x.device)

        new_cache = {}
        for i, block in enumerate(self.blocks):
            block_name = f"blocks.{i}"
            block_cache = None if cache is None else cache[block_name]
            x, new_cache[block_name] = block(x, pos, block_cache)
        
        return x


class BiDiGemma(nn.Module):
    def __init__(self, dim: int, expand_ratio: int = 3, num_heads: int=4):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.forward_conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.backward_conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.ssm = GemmaBlock(dim=dim, expand_ratio=expand_ratio, num_heads=num_heads)

        self.softplus = nn.Softplus()
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, skip: torch.Tensor): # [batch_size, seq_len, dim]
        x = self.proj(self.norm(skip))

        # Excitation
        z = self.sigmoid(self.activation(skip).mean(dim=-1, keepdims=True))
        
        # Forward pass
        forwards = rearrange(x, "b s d -> b d s")
        forwards = self.forward_conv1d(forwards)
        forwards = rearrange(forwards, "b d s -> b s d")
        forwards = self.dropout(forwards)
        forwards = self.ssm(forwards)

        # Backward pass
        backwards = rearrange(x, "b s d -> b d s")
        backwards = self.backward_conv1d(backwards)
        backwards = rearrange(backwards, "b d s -> b s d")
        backwards = self.dropout(backwards)
        backwards = self.ssm(backwards)

        # Add excitations and skip
        return z * (forwards + backwards) + skip

class VisionGriffen(nn.Module):
    def __init__(self, dim: int, expand_ratio: int = 3, num_heads: int=2, image_size=(64, 64), patch_size=(8, 8)):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.image_size = image_size
        
        self.l1 = nn.Linear(patch_size[0] * patch_size[1], dim)
        self.gemma = BiDiGemma(dim, expand_ratio=expand_ratio, num_heads=num_heads)
        self.l2 = nn.Linear(dim, patch_size[0] * patch_size[1])
        self.gelu = nn.GELU()

    def to_patch_embedding(self, image:torch.Tensor):
        patches = image.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
        return patches.contiguous().view(image.shape[0], -1, self.patch_size[0] * self.patch_size[1])
    
    def from_patch_embedding(self, patches:torch.Tensor):
        patches = patches.view(patches.shape[0], self.dim, self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1], self.patch_size[0], self.patch_size[1])
        return patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, self.dim, self.image_size[0], self.image_size[1])
        
    def forward(self, x: torch.Tensor):
        x = self.to_patch_embedding(x)
        x = self.gelu(self.l1(x))
        x = self.gemma(x)
        x = self.gelu(self.l2(x))        
        x = self.from_patch_embedding(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, squeeze_factor=16):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, 1, 1, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, 1, 1, bias=False)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // squeeze_factor, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // squeeze_factor, hidden_features, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x0):
        x = self.fc1(x0)
        x = self.act(x)
        x = self.fc2(x)
        return x * self.ca(x0)

class Unit(nn.Module):
    def __init__(self, dim, image_size):
        super(Unit, self).__init__()

        self.cab = VisionGriffen(dim, image_size=image_size)
        self.mlp = Mlp(dim, dim * 1)

        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        #x = self.drop(x)
        x = x + self.cab(x) * 0.2
        #x = self.drop(x)
        x = x + self.mlp(x) * 0.2
        return x
    
class Block(nn.Module):
    def __init__(self, dim, image_size, num=1):
        super(Block, self).__init__()
        self.blocks = nn.Sequential(*[Unit(dim, image_size) for i in range(num)])

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
            Block(dim*1, image_size=(64, 64)), DownSample(dim*1, dim*2)
        ])

        self.down2 = nn.Sequential(*[
            Block(dim*2, image_size=(32, 32)), DownSample(dim*2, dim*4)
        ])
        
        self.down3 = nn.Sequential(*[
            Block(dim*4, image_size=(16, 16)), DownSample(dim*4, dim*8)
        ])

        self.mid = nn.Sequential(*[
            Block(dim*8, image_size=(8, 8))
        ])

        self.up1 = nn.Sequential(*[
            UpSample(dim*8, dim*4), Block(dim*4, image_size=(16, 16)), 
        ])

        self.up2 = nn.Sequential(*[
            UpSample(dim*4, dim*2), Block(dim*2, image_size=(32, 32)), 
        ])

        self.up3 = nn.Sequential(*[
            UpSample(dim*2, dim*1), Block(dim*1, image_size=(64, 64))
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
    

class GemmaIR(nn.Module):
    def __init__(self, dim=64):
        super(GemmaIR, self).__init__()
        self.dim = dim
        
        self.m_head = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.unets = nn.Sequential(*[uNet(dim) for i in range(1)])
        
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
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
