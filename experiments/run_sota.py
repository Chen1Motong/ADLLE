"""Run the ported SLE (paper ref [20]) under the unified multi-split protocol."""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sota"))

import numpy as np

import adle_core as ac
import sle

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATASETS = ["ORL", "AR", "Yale", "YaleB", "UMIST", "PIE", "COIL20", "COIL100"]
SEEDS = range(10)


def main():
    out = os.path.join(RESULTS, "sota_sle.csv")
    new = not os.path.exists(out)
    done = set()
    if not new:
        for r in csv.DictReader(open(out, encoding="utf-8")):
            done.add((r["dataset"], int(r["seed"])))
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "seed", "acc", "best_beta", "best_k1", "n_classes", "seconds"])
        for name in DATASETS:
            for seed in SEEDS:
                if (name, seed) in done:
                    continue
                Xtr, Xte, Ytr, Yte, _, _ = ac.load_dataset(name, seed=seed)
                t = time.time()
                acc, cfg = sle.sle_embed(Xtr, Ytr, Xte, Yte,
                                         n_classes=len(np.unique(Ytr)),
                                         target_dim=80, seed=seed)
                w.writerow([name, seed, f"{acc:.2f}", cfg[0], cfg[1],
                            len(np.unique(Ytr)), f"{time.time()-t:.1f}"])
                f.flush()
                print(f"[{name} seed{seed}] SLE={acc:.2f}% cfg={cfg} "
                      f"({time.time()-t:.0f}s)", flush=True)

    # summary with ADLLE/DLLE/L reference columns
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    stats = {(r["dataset"], r["variant"]): [] for r in csv.DictReader(
        open(os.path.join(RESULTS, "stats.csv"), encoding="utf-8"))}
    for r in csv.DictReader(open(os.path.join(RESULTS, "stats.csv"), encoding="utf-8")):
        stats[(r["dataset"], r["variant"])].append(float(r["acc"]))
    lines = ["# SLE (ref [20]) vs ADLLE / DLLE-L, 10 random splits", "",
             "| Dataset | ADLLE | DLLE/L | SLE (grid best) |", "|---|---|---|---|"]
    for name in DATASETS:
        a = np.array(stats.get((name, "adlle"), [np.nan]))
        d = np.array(stats.get((name, "dlle_l"), [np.nan]))
        s = np.array([float(r["acc"]) for r in rows if r["dataset"] == name])
        lines.append(f"| {name} | {a.mean():.2f}±{a.std(ddof=1):.2f} | "
                     f"{d.mean():.2f}±{d.std(ddof=1):.2f} | "
                     f"{s.mean():.2f}±{s.std(ddof=1):.2f} |")
    txt = "\n".join(lines)
    open(os.path.join(RESULTS, "sota_summary.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
