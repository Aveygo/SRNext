from methods.simple import Simple
from archs.models.srnext import SRNext

class Experiment(Simple):
    def __init__(self):
        generator = SRNext(embed_dim=60, blocks=4, num_active=4)
        super().__init__(generator)