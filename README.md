# SRNext

This respository contains code to train & eval SRNext - A (potential) state of the art in lightweight super-resolution.

| Original | SwinIR | SRNext |
|---|---|---|
| ![Earth](samples/earth.png) | ![Earth](samples/swinir.png) | ![Earth](samples/srnext.png) |

Similar results, but up to 1.6x faster with 60% less utilised memory!

## Compared with SOTA

### PNSR

| Model  | Set14  | BSD100 | URBAN100 | MANGA109 |
|--------|--------|--------|----------|----------|
| IMDN   | 27.8040| 24.0524| 27.2880  | 23.4331  |
| CARN   | 28.0326| 24.0659| 27.3436  | 23.4023  |
| SwinIR | **28.2085**| 24.0346| **27.7790**  | 23.4692  |
| SRNext | 28.0364| **24.1766** | 27.1437  | **23.7909**  |

### SSIM

| Model  | Set14  | BSD100 | URBAN100 | MANGA109 |
|--------|--------|--------|----------|----------|
| IMDN   | 0.7508 | 0.7002 | 0.8201   | 0.8176   |
| CARN   | 0.7490 | 0.6730 | 0.7945   | 0.7885   |
| SwinIR | **0.7685** | 0.7080 | **0.8397**   | 0.8297   |
| SRNext | 0.7677 | **0.7103** | 0.8263   | **0.8338**   |

## Report

Find out more details in the [report](report/report.pdf)


## Quick Start

Install libraries
```bash
pip install -r requirements.txt
```

Start training (will go on forever, must Ctrl+C).
```bash
python3 train.py
```

Evaluate SRNext
```bash
python3 eval.py
```

Inference SRNext on an image
```
python3 inference.py sample.png sample_out.png 
```

## Setup

Datasets are primarily taken from kaggle:
[FLICKR2k](https://www.kaggle.com/datasets/daehoyang/flickr2k)
[DIV2k](https://www.kaggle.com/datasets/joe1995/div2k-dataset)
and combined manually.

Simiarly, test sets are from [here](https://www.kaggle.com/datasets/jesucristo/super-resolution-benchmarks).

## Structure

'Models' and 'Dataset' definitions are stored in the 'archs' directory, eg: The SRNext architecture is defined at archs/models/srnext.py

'Methods' takes models and trains them. They are very generalised don't initialise models, eg: methods/bootstrap.py

'Experiments' combines a method and their models. Eg, experiments/bootstrap_unext.py will take an untrained unext model and pretrained realesr model, then train unext to match the outputs of realesr.

Finally, to actually start the experiment, it is imported in train.py than ran with python.
