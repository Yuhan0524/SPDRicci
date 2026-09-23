# SPD Ricci Flow

This repository contains the SPD Ricci Flow implementation used for the
Wisconsin node-classification experiment.

<!--## Reported Result

| Model | Accuracy (%) |
|---|---:|
| SPD Ricci Flow | **89.80 +/- 3.52** |

The table reports the mean and population standard deviation over split and
training seeds `0,...,9`. The corresponding per-seed outputs are stored in
`results/wisconsin.json`.-->

## Setup

The experiment uses Python 3.11, PyTorch 2.1.1 with CUDA 11.8, PyTorch
Geometric 2.4.0, and NumPy 1.26.4.

```bash
git clone https://github.com/Yuhan0524/SPDRicci.git
cd SPDRicci

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric downloads the Wisconsin WebKB data into `data/` on the first
run.

## Run

```bash
python SPDRicci.py --dataset wisconsin --device cuda
```

The model and training hyperparameters are specified in
`configs/wisconsin.json`. Results are written to
`results/wisconsin_run.json` by default.
