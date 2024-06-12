from methods.simple import Simple

from archs.models.srnext import SRNext

import torch

class Experiment(Simple):
    def __init__(self):
        generator = SRNext(embed_dim=60, blocks=4, num_active=4)
        generator.load_state_dict(torch.load("ckpts/SRNext.ckpt"), strict=False)
        super().__init__(generator)