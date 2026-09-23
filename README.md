# SPD Ricci Flow

This repository provides a self-contained reproduction of the Wisconsin
node-classification experiment for the SPD Ricci Flow model. 

The reported Wisconsin configuration uses the original graph and original
node features. It does not use graph rewiring, feature propagation, edge
weights, training-time augmentation, restart averaging, or an ensemble.

## Result

| Model | Accuracy (%) |
|---|---:|
| SPD Ricci Flow | **89.80 +/- 3.52** |

Values are the mean and population standard deviation over split and training
seeds `0,...,9`. 

## Setup

The recorded environment used Python 3.11, PyTorch 2.1.1 with CUDA 11.8,
PyTorch Geometric 2.4.0, and NumPy 1.26.4.

```bash
git clone https://github.com/Yuhan0524/SPDRicci.git
cd SPDRicci

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric downloads the Wisconsin WebKB data automatically on the
first run and stores it under `data/`, which is excluded from version control.

## Reproduce Wisconsin

Run on CUDA:

```bash
python reproduce_wisconsin.py --activation tgrelu --device cuda
```
