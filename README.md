# Identifying Mislabeled Data using the Area Under the Margin Ranking
Implementation of the research paper Identifying Mislabeled Data using the Area Under the Margin Ranking.

Original paper: https://arxiv.org/pdf/2001.10528v4

This technique can be used to identify mislabeled or difficult samples in a dataset. These samples can then be relabeled or removed to improve the final performance of a model trained on the data.

## Setup

### 1. Virtual environment

Ensure you have Python installed, create a virtual environment and activate it.

### 2. Install PyTorch packages

With the virtual environment activated, run

```
pip install -r requirements_pytorch.txt [--index-url INDEX_URL]
```

The `--index-url` should only be specified if advised by https://pytorch.org/get-started/locally/.

### 3. Install remaining packages

Now run

```
pip install -r requirements_main.txt
```

to install the remaining packages.

## Project structure

- **identify_mislabeled_data.ipynb** is an example showing how to apply AUM Ranking to identify mislabeled samples in a dataset. It outputs TensorBoard logs to **runs/**, which can be viewed with `tensorboard --logdir runs/`.

- **aum_ranking.py** contains all the code specific to AUM Ranking.

- **models.py** defines the ResNet-32 model used in the AUM paper.
