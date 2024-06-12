# Training libraries
from lightning.pytorch import Trainer

# The experiment
#from experiments.simple_swinir import Experiment
from experiments.simple_srnext import Experiment
#from experiments.simple_control import Experiment

# Datasets
#from archs.datasets.paired import PairedImageDataset
#dataset = PairedImageDataset(batch_size=32,use_cache=False, lq_pth="datasets/lq", hq_pth="datasets/hq", pre_crop=True).dataloader

from archs.datasets.testset import TestSet
dataset = TestSet(batch_size=64, src="datasets/FlickrAndDIV2k/", lq_size=64, hq_size=256, pre_crop=True, use_cache=True).dataloader

# Increase training speed by decreasing precision
#torch.set_float32_matmul_precision('medium')

# Training conifiguration, default is train forever
trainer = Trainer(max_epochs=-1, accumulate_grad_batches=1)

# Start training with our experiment and dataset
trainer.fit(model=Experiment(), train_dataloaders=dataset)