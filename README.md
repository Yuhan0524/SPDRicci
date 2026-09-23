# SPD Ricci Flow

This repository provides a self-contained reproduction of the Wisconsin
node-classification experiment for the SPD Ricci Flow model. It includes the
exact ten-seed protocol, locked hyperparameters, recorded outputs, and both
spectral activations considered in the activation study.

The reported Wisconsin configuration uses the original graph and original
node features. It does not use graph rewiring, feature propagation, edge
weights, training-time augmentation, restart averaging, or an ensemble.

## Result

| Model | Activation | Accuracy (%) |
|---|---|---:|
| SPD Ricci Flow | TGReLU, spacing 0.2 | **89.80 +/- 3.52** |
| SPD Ricci Flow | Classic ReEig | 88.00 +/- 2.37 |

Values are the mean and population standard deviation over split and training
seeds `0,...,9`. TGReLU is the validation-selected Wisconsin hyperparameter
and is therefore the default. ReEig is provided as an otherwise matched
activation ablation. Per-seed scores and the complete protocol are documented
in [RESULTS.md](RESULTS.md).

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

Run the validation-selected TGReLU configuration on CUDA:

```bash
python reproduce_wisconsin.py --activation tgrelu --device cuda
```

Run the matched classic ReEig activation:

```bash
python reproduce_wisconsin.py --activation reeig --device cuda
```

Both commands use the same ten seeds, original graph, original features,
architecture, optimizer, 300-epoch budget, and validation-loss checkpoint
rule. Only the activation changes. To run a subset of seeds, use for example
`--seeds 0,1,2`. The result is written under `results/`.

Validate the public configurations without training:

```bash
python reproduce_wisconsin.py --activation tgrelu --check-only
python reproduce_wisconsin.py --activation reeig --check-only
pytest -q
```

## Model

The implementation follows

```text
FeatureToSPD
-> [BiMap -> SPD Ricci diffusion -> spectral activation -> inverse BiMap] x L
-> Log at the identity
-> upper-triangular vectorization
-> linear classifier
```

For node `i`, a learned linear map produces `d^2` entries, which are reshaped
into `A_i` in `R^{d x d}`. The initial state is

```text
S_i = (A_i + A_i^T) / 2,        g_i^(0) = exp(S_i).
```

For each directed edge, the layer computes

```text
L_{i->j} = log(g_i^(-1/2) g_j g_i^(-1/2)),
a_{ij}   = w_{ij} exp(-beta ||L_{i->j}||_F^2),
S_i(g)   = sum_{j~i} a_{ij} L_{i->j},
Ric_i(g) = -(g_i S_i(g) + S_i(g) g_i) / 2,
g_i'     = g_i - eta Ric_i(g).
```

The Wisconsin run uses uniform graph weights, so `w_ij = 1`, `beta = 300`,
and `eta = 0.1`.

### Spectral activations

Both activation choices are implemented in `model.py` and selected by the
configuration files in `configs/`:

- **Classic ReEig:** `phi(mu) = max(mu, 1e-4)`.
- **TGReLU:** clamp the ordered eigenvalues to the SPD floor and set
  `z_i = log(mu_i)`. Positive `z_i` are retained; non-positive entries are
  replaced by `0.2 i`, after which the eigenvalues are reconstructed as
  `exp(z_i)`.

### Guaranteed-invertible BiMap

The tied decoder uses the inverse of the learned square BiMap. Given an
unconstrained candidate `W_bar = U diag(sigma) V^T`, the forward pass uses

```text
W = U diag(max(sigma_j, tau)) V^T,       tau = 0.1.
```

Consequently, the minimum singular value is at least `tau` and the inverse is
well-defined. If the learned matrix is already well-conditioned, it is used
unchanged, including its direct gradient path.

## Repository Layout

```text
configs/                  Locked TGReLU and ReEig configurations
results/                  Recorded ten-seed outputs
tests/                    Activation and configuration checks
data.py                   Dataset loading and released split protocol
model.py                  SPD maps, Ricci layer, activations, and classifier
train.py                  Full-batch training and checkpoint selection
reproduce_wisconsin.py    Reproduction entry point
RESULTS.md                Full settings and per-seed results
```

## Reproducibility Notes

- Self-loops are removed and the graph is converted to an undirected graph.
- Supplied node features are row-sum normalized.
- Every seed determines the 60/20/20 split and Python, NumPy, and PyTorch RNGs.
- Checkpoints are selected using validation loss only; test labels are not used
  for model selection.
- The public command rejects configurations that enable rewiring, propagated
  features, edge weights, augmentation, EMA, SAM, or ensemble evaluation.
