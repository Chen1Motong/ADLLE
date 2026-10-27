"""Multi-split statistics runner for the ADLLE revision.

Runs every (dataset, seed, variant) cell, appends rows to results CSV, and
exports the exact train/test split indices per (dataset, seed) for release
(reviewer R2-5 "release the sample lists and splits").

Usage:
  python run_stats.py                       # full grid
  python run_stats.py --datasets Yale ORL   # subset
  python run_stats.py --seeds 0 1 2
"""

import argparse
import csv
import os
import time

import adle_core as ac

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
STATS_CSV = os.path.join(RESULTS, "stats.csv")
SPLIT_DIR = os.path.join(RESULTS, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

DATASETS = ["Yale", "ORL", "UMIST", "COIL20", "AR", "YaleB", "COIL100", "PIE"]
VARIANTS = ["adlle", "w_gaussian", "no_radius", "no_stopping", "dlle_l"]
SEEDS = list(range(10))

# DLLE/L baseline faithful to ver1.ipynb: k2 = 300 fixed neighbors per sample.
# no_stopping keeps k0=15 (the fixed-k component ablation of ADLLE itself).
DLLE_K2 = 300


def export_split(name, seed, tri, tei, sample_ids_unused):
    path = os.path.join(SPLIT_DIR, f"{name}_seed{seed}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role", "sample_index"])
        w.writerows([["train", int(i)] for i in tri])
        w.writerows([["test", int(i)] for i in tei])


def load_done():
    done = set()
    if os.path.exists(STATS_CSV):
        with open(STATS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], int(row["seed"]), row["variant"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    args = ap.parse_args()

    done = load_done()
    new_file = not os.path.exists(STATS_CSV)
    with open(STATS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["dataset", "seed", "variant", "acc", "n_train", "n_test", "seconds"])
        for name in args.datasets:
            for seed in args.seeds:
                t0 = time.time()
                Xtr, Xte, Ytr, Yte, tri, tei = ac.load_dataset(name, seed=seed)
                export_split(name, seed, tri, tei, None)
                print(f"[{name} seed{seed}] loaded {Xtr.shape}->{Xte.shape} "
                      f"({time.time()-t0:.1f}s)", flush=True)
                for v in args.variants:
                    if (name, seed, v) in done:
                        continue
                    params = dict(ac.PARAMS)
                    if v == "dlle_l":
                        params["k0"] = DLLE_K2
                    t = time.time()
                    acc = ac.embed_and_eval(Xtr, Ytr, Xte, Yte, v, params)
                    row = (name, seed, v, f"{acc:.2f}", len(Ytr), len(Yte),
                           f"{time.time()-t:.1f}")
                    w.writerow(row); f.flush()
                    print(f"  {v:12s} acc={acc:.2f}%  ({time.time()-t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
