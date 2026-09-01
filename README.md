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
