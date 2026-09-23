# Wisconsin Original-Graph Result

This note documents the Wisconsin node-classification result for the
SPD Ricci Flow reproduction. Among the five node-classification
datasets, Wisconsin is the cleanest result because it uses the original graph
and original node features, without multi-hop feature propagation, graph
rewiring, feature-kNN graph construction, edge weights, ensembles, or restart
averaging.

## Result

The locked ten-seed final result is:

| Dataset | Metric | Paper GNRF | SPD Ricci Flow | Margin |
|---|---:|---:|---:|---:|
| Wisconsin | Accuracy (%) | 88.00 | 89.80 +/- 3.52 | +1.80 |

The result is evaluated on the released GNRF split seeds `0,...,9`. The
reported standard deviation follows the submitted table and is the population
standard deviation over the ten final test scores.

Canonical artifacts:

- Final TGReLU result: `results/wisconsin_tgrelu.json`
- Locked TGReLU config: `configs/wisconsin_tgrelu.json`
- Classic ReEig shards: `results/wisconsin_reeig_seeds_0_4.json` and
  `results/wisconsin_reeig_seeds_5_9.json`
- Implementation: `data.py`, `model.py`, and `train.py`

## Why This Is The Clean Result

The Wisconsin configuration uses:

- `graph_mode = original`
- `graph_weight_mode = none`
- `graph_topk = 0`
- no label smoothing
- no class weighting
- no mixup
- no input noise
- no edge dropout
- no EMA
- no SAM
- no head-only fine-tuning
- no ensemble or multi-restart averaging

Unlike the Questions, Cora-Full, DBLP, and Cora-ML runs, the Wisconsin result
does not use fixed multi-hop feature propagation before `FeatureToSPD`.

## Dataset Processing

The data loader follows the released GNRF protocol:

1. Load the Wisconsin WebKB graph.
2. Remove self-loops.
3. Convert the graph to an undirected graph.
4. Row-sum normalize the supplied node features.
5. Use the released GNRF-style random 60/20/20 split protocol for seeds
   `0,...,9`.

No labels are used in graph construction or feature preprocessing.

## Model

The model family is:

```text
FeatureToSPD
-> [BiMap -> exact Simple1 SPD diffusion -> SPD activation -> inverse BiMap] x L
-> Log at identity
-> upper-triangular vectorization
-> linear classifier
```

For Wisconsin, the locked model hyperparameters are:

| Hyperparameter | Value |
|---|---:|
| FeatureToSPD mapper | squared |
| SPD dimension | 16 |
| Simple1 layers | 2 |
| beta | 300 |
| epsilon | 0.1 |
| SPD activation | TGReLU |
| TGReLU delta | 0.2 |
| BiMap decoder | tied inverse |
| BiMap max delta | 0.08 |
| BiMap minimum singular value | 0.1 |
| Classifier | linear Log-SPD head |
| Feature dropout | 0.0 |
| Classifier dropout | 0.5 |
| SPD floor | 1e-4 |
| FeatureToSPD epsilon | 1e-4 |
| SPD4GNN Exp/Log jitter | 0.0 |

The `spd4gnn_jitter = 0.0` setting uses deterministic mathematical Exp/Log
maps rather than the upstream SPD4GNN stochastic `sym_funcm` jitter.

### Configurable SPD activation

The public implementation exposes both spectral activations used in the
activation study:

- Classic ReEig: \(\phi(\mu)=\max(\mu,10^{-4})\).
- TGReLU: after setting \(z_i=\log\max(\mu_i,10^{-4})\) for the ordered
  eigenvalues, use \(\widetilde z_i=z_i\) when \(z_i>0\) and
  \(\widetilde z_i=0.2i\) otherwise, then return \(\exp(\widetilde z_i)\).

Activation is a dataset-level hyperparameter. A validation-only search selected
TGReLU with spectral spacing `0.2` for Wisconsin; this setting is therefore the
default and the one used for the reported Table 1 result. Classic ReEig remains
available as a controlled activation ablation.

| Wisconsin activation | Test accuracy (%) |
|---|---:|
| TGReLU, spacing 0.2 (selected) | **89.80 +/- 3.52** |
| Classic ReEig | 88.00 +/- 2.37 |

For each learned BiMap candidate \(\bar W=U\operatorname{diag}(\sigma)V^\top\),
the forward map is materialized as

\[
W=U\operatorname{diag}\!\bigl(\max(\sigma_j,\tau)\bigr)V^\top,
\qquad \tau=0.1.
\]

Thus \(\sigma_{\min}(W)\geq\tau>0\) and
\(\lVert W^{-1}\rVert_2\leq\tau^{-1}\). The implementation returns
\(\bar W\) directly whenever \(\sigma_{\min}(\bar W)\geq\tau\), so the guard
does not alter an already well-conditioned learned BiMap or its gradient path.

## Training

Training hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.03 |
| Weight decay | 5e-4 |
| Epochs | 300 |
| Scheduler | none |
| Gradient clipping | 5.0 |
| Precision | float32 |
| Checkpoint selection | validation loss |

The train/validation/test split seed also controls Python, NumPy, and PyTorch
training randomness.

## Per-Seed Final Scores

| Seed | Best epoch | Validation accuracy (%) | Test accuracy (%) |
|---:|---:|---:|---:|
| 0 | 149 | 87.76 | 92.00 |
| 1 | 247 | 87.76 | 84.00 |
| 2 | 234 | 93.88 | 94.00 |
| 3 | 204 | 91.84 | 94.00 |
| 4 | 213 | 95.92 | 90.00 |
| 5 | 90 | 85.71 | 88.00 |
| 6 | 286 | 95.92 | 88.00 |
| 7 | 164 | 93.88 | 92.00 |
| 8 | 109 | 75.51 | 84.00 |
| 9 | 67 | 93.88 | 92.00 |

Mean test accuracy: `89.80`.

Population standard deviation: `3.52`.

## Reproduce

From this directory, reproduce the validation-selected TGReLU result:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce_wisconsin.py --activation tgrelu --device cuda
```

Run the otherwise matched classic ReEig activation ablation with:

```bash
.venv/bin/python reproduce_wisconsin.py --activation reeig --device cuda
```

Both commands use the same ten splits, architecture, optimizer, epoch budget,
and validation-loss checkpoint rule. Only the SPD activation preset changes.

## Recommended Paper Wording

A precise description is:

> For Wisconsin, we use the released GNRF preprocessing and split protocol:
> the graph is made undirected, self-loops are removed, and the supplied node
> features are row-normalized. The Simple1/SPD Ricci Flow model is trained on
> the original graph with no graph rewiring, no edge weights, no multi-hop
> feature propagation, and no ensemble averaging.
