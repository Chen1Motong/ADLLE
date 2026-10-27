"""η / kmax / kmin parameter ablation on full ADLLE (closes R3-7's list).

Sweeps (one parameter at a time, others at paper defaults):
  eta  in {0.25, 0.5, 1.0}
  kmax in {100, 200, 300, 500}
  kmin in {2, 5, 10}
8 datasets x 10 seeds, full ADLLE variant with parameter overrides.
"""
import csv
import os
import time

import adle_core as ac

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATASETS = ["ORL", "AR", "Yale", "YaleB", "UMIST", "PIE", "COIL20", "COIL100"]
SEEDS = range(10)
SWEEPS = {
    "eta": [0.25, 0.5, 1.0],
    "kmax": [100, 200, 300, 500],
    "kmin": [2, 5, 10],
}


def main():
    out = os.path.join(RESULTS, "param_ablation.csv")
    new = not os.path.exists(out)
    done = set()
    if not new:
        for r in csv.DictReader(open(out, encoding="utf-8")):
            done.add((r["dataset"], r["sweep"], r["value"], r["seed"]))
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "sweep", "value", "seed", "acc", "seconds"])
        for name in DATASETS:
            for seed in SEEDS:
                Xtr, Xte, Ytr, Yte, _, _ = ac.load_dataset(name, seed=seed)
                for sweep, values in SWEEPS.items():
                    for val in values:
                        if (name, sweep, val, seed) in done:
                            continue
                        params = dict(ac.PARAMS)
                        params[sweep] = val
                        t = time.time()
                        acc = ac.embed_and_eval(Xtr, Ytr, Xte, Yte, "adlle", params)
                        w.writerow([name, sweep, val, seed, f"{acc:.2f}",
                                    f"{time.time()-t:.1f}"])
                        f.flush()
                    print(f"[{name} seed{seed}] {sweep} sweep done", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
