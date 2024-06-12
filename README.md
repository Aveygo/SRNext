# uNext -> SRNext
Efficient Upscaling - Computer Science Studio 2

This respository contains code to train & eval SRNext - A (potential) state of the art in lightweight super-resolution.

Datasets are primarily taken from kaggle:
[FLICKR2k](https://www.kaggle.com/datasets/daehoyang/flickr2k)
[DIV2k](https://www.kaggle.com/datasets/joe1995/div2k-dataset)
and combined manually.

Simiarly, test sets are from [here](https://www.kaggle.com/datasets/jesucristo/super-resolution-benchmarks).

## Quick Start

Install libraries
```bash
pip install -r requirements.txt
```

In another terminal, start training (will go on forever, must Ctrl+C).
```bash
python3 train.py
```

Evaluate SRNext
```bash
python3 eval.py
```

## Structure

'Models' and 'Dataset' definitions are stored in the 'archs' directory, eg: The SRNext architecture is defined at archs/models/srnext.py

'Methods' takes models and trains them. They are very generalised don't initialise models, eg: methods/bootstrap.py

'Experiments' combines a method and their models. Eg, experiments/bootstrap_unext.py will take an untrained unext model and pretrained realesr model, then train unext to match the outputs of realesr.

Finally, to actually start the experiment, it is imported in train.py than ran with python.
