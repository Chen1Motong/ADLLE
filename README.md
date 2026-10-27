# ADLLE

Code and data for the manuscript **"ADLLE: Adaptive Discriminant Locally Linear Embedding with Boundary-Sensitive Inter-class Neighborhood Selection"**.

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

All eight datasets used in the paper are included in this repository under `data/`:

| Folder | Dataset | Description |
|---|---|---|
| `data/ORL` | ORL (Olivetti faces) | Stored as `olivetti.npz`, the exact copy used in the paper (fetched via scikit-learn, `shuffle=True, random_state=42`) |
| `data/AR` | AR | 100 subjects, resized to 32×32 |
| `data/Yale` | Yale | 15 subjects, resized to 64×64 |
| `data/Yale B` | Extended Yale B | Resized to 64×64 |
| `data/UMIST` | UMIST | Resized to 64×64 |
| `data/PIE/Pose05_64x64_files` | CMU PIE (Pose 05) | 68 subjects × 49 images, 64×64 |
| `data/coil-20-proc` | COIL-20 (processed) | 20 classes × 72 images |
| `data/coil-100` | COIL-100 | 100 classes |

### Data provenance

Copies of the datasets are included in this repository solely to allow exact
reproduction of the results reported in the paper. They were obtained from
publicly available sources; all rights remain with the original authors and
distributors. Please cite the original dataset papers if you use them, and use
the data for academic research only.

## Usage

Each command below runs the full ADLLE pipeline on one dataset (data loading,
intrinsic/penalty graph construction, projection-matrix learning, and Top-1
accuracy evaluation with a nearest-neighbour classifier):

```bash
python ADLLE.py --dataset ORL      # choices: AR, COIL100, COIL20, ORL, PIE, UMIST, Yale, YaleB
```

The random seed is fixed (42) at the start of each run for reproducibility.

## Citation

If you use this code, please cite the manuscript above.

## Reproducibility

### Seeds and split indices

- Single-split runs (`ADLLE.py`) use seed 42.
- All multi-split evaluations (Tables 11–13 and the appendix) use ten random
  splits with seeds 0–9. The exact train/test sample indices of every split are
  released under `splits/`: one CSV per `(dataset, seed)`, with one row per
  sample (`role` = `train`/`test`, plus `sample_index` and the class label).
- The synthetic contamination study (Table 14) is generated on the fly by
  `experiments/run_outlier.py` with ten random seeds (0–9); no split files are
  involved.

### Preprocessing

Grayscale images, resized per dataset as in Table 1 of the paper. Features are
standardized with the scaler fitted on the training set only. No PCA is applied
to ADLLE. The DLLE/L baseline follows its original implementation, which
includes a PCA stage retaining 98% of the energy in the main accuracy
comparison.

### Parameter values and search ranges

Defaults: `k_max=300`, `k_min=2`, `eta=0.25`, `sigma=0.1`, `h=10`, `d=80`,
`k_1=3`. Search ranges: `eta` in {0.25, 0.5, 1.0}, `k_max` in
{100, 200, 500}, `k_min` in {5, 10}.

### Reproducing the revision tables

```bash
python experiments/run_stats.py            # Table 11 (+ exports splits/)
python experiments/run_sota.py             # Table 13 (SLE, AN-DWLLE, DARP)
python experiments/run_sota2.py            # Table 13 (remaining methods)
python experiments/run_outlier.py          # Table 14 (synthetic contamination)
python experiments/run_param_ablation.py   # appendix: eta / k_max / k_min
python experiments/extract_features.py     # Table 12 step 1 (needs torch/torchvision, not pinned)
python experiments/run_table12.py          # Table 12 step 2
```

Each script writes its CSV outputs to `experiments/../results/`. Runtime
environment: Python 3.8+ with the packages in `requirements.txt`.
