from methods.simple import Simple
from archs.models.control import Control
import torch

class Experiment(Simple):
    def __init__(self):
        generator = Control()
        #generator.load_state_dict(torch.load("ckpts/Control_Simple_Latest.ckpt"), strict=False)

        super().__init__(generator, use_vgg=False)