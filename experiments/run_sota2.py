"""Run AN-DWLLE (ref [22]) and DARP (Info Fusion 2025) under the unified protocol.

DWLE: unsupervised, no projection matrix -> 5-NN-regressor out-of-sample map
      (same disclosed protocol as our LLE baseline), 1-NN classification.
DARP: supervised with projection matrix; PCA(100) pre-reduction for
      tractability (disclosed); grid over alpha in {0.01,0.1,1,10}, report best
      (same convention as SLE).
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sota"))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import adle_core as ac
import darp
import dwle

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATASETS = ["ORL", "AR", "Yale", "YaleB", "UMIST", "PIE", "COIL20", "COIL100"]
SEEDS = range(10)


def run_dwle(Xtr, Ytr, Xte, Yte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Yemb = dwle.embed(Xtr_s, alpha=0.5, kmin=3, kmax=20, target_dim=80)
    # standard LLE out-of-sample extension: reconstruct each test sample from
    # its k nearest training samples (weights sum to 1) and apply the same
    # weights to their training embeddings
    from sklearn.neighbors import NearestNeighbors
    k = 5
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr_s)
    dist, idx = nn.kneighbors(Xte_s)
    Yte_emb = np.zeros((len(Xte_s), Yemb.shape[1]))
    for i in range(len(Xte_s)):
        Z = Xtr_s[idx[i]] - Xte_s[i]
        G = Z @ Z.T
        G[np.arange(k), np.arange(k)] += 1e-8 * max(1e-3, np.trace(G))
        try:
            w = np.linalg.solve(G, np.ones(k))
        except np.linalg.LinAlgError:
            w = np.ones(k)
        w = np.maximum(w, 0)
        w /= w.sum() + 1e-12
        Yte_emb[i] = w @ Yemb[idx[i]]
    pred = Ytr[np.argmin(((Yemb[:, None, :] - Yte_emb[None, :, :]) ** 2).sum(2), axis=0)]
    return accuracy_score(Yte, pred) * 100


def run_darp(Xtr, Ytr, Xte, Yte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    n_comp = min(100, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(Xtr_s)
    acc, cfg = darp.darp_eval(pca.transform(Xtr_s), Ytr, pca.transform(Xte_s), Yte, d=80)
    return acc, cfg


def main():
    out_d = os.path.join(RESULTS, "sota_dwle.csv")
    out_p = os.path.join(RESULTS, "sota_darp.csv")
    dones = {}
    for path, key in [(out_d, "dwle"), (out_p, "darp")]:
        dones[key] = set()
        if os.path.exists(path):
            for r in csv.DictReader(open(path, encoding="utf-8")):
                dones[key].add((r["dataset"], int(r["seed"])))

    fd = open(out_d, "a", newline="", encoding="utf-8")
    fp = open(out_p, "a", newline="", encoding="utf-8")
    wd = csv.writer(fd)
    wp = csv.writer(fp)
    if fd.tell() == 0:
        wd.writerow(["dataset", "seed", "acc", "seconds"])
    if fp.tell() == 0:
        wp.writerow(["dataset", "seed", "acc", "alpha", "seconds"])

    for name in DATASETS:
        for seed in SEEDS:
            Xtr, Xte, Ytr, Yte, _, _ = ac.load_dataset(name, seed=seed)
            if (name, seed) not in dones["dwle"]:
                t = time.time()
                acc = run_dwle(Xtr, Ytr, Xte, Yte)
                wd.writerow([name, seed, f"{acc:.2f}", f"{time.time()-t:.1f}"])
                fd.flush()
                print(f"[{name} seed{seed}] DWLE={acc:.2f}% ({time.time()-t:.0f}s)", flush=True)
            if (name, seed) not in dones["darp"]:
                t = time.time()
                acc, cfg = run_darp(Xtr, Ytr, Xte, Yte)
                wp.writerow([name, seed, f"{acc:.2f}", cfg["alpha"] if cfg else "",
                             f"{time.time()-t:.1f}"])
                fp.flush()
                print(f"[{name} seed{seed}] DARP={acc:.2f}% alpha={cfg['alpha'] if cfg else '-'} "
                      f"({time.time()-t:.0f}s)", flush=True)
    fd.close()
    fp.close()

    # combined SOTA summary
    def col(path):
        m = {}
        if os.path.exists(path):
            for r in csv.DictReader(open(path, encoding="utf-8")):
                m.setdefault(r["dataset"], []).append(float(r["acc"]))
        return m

    stats = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "stats.csv"), encoding="utf-8")):
        stats.setdefault((r["dataset"], r["variant"]), []).append(float(r["acc"]))
    sle_m = col(os.path.join(RESULTS, "sota_sle.csv"))
    dw_m = col(os.path.join(RESULTS, "sota_dwle.csv"))
    dp_m = col(os.path.join(RESULTS, "sota_darp.csv"))

    lines = ["# SOTA comparisons under the unified 10-split protocol", "",
             "| Dataset | ADLLE | DLLE/L | SLE [20] | AN-DWLLE [22] | DARP (IF2025) |",
             "|---|---|---|---|---|---|"]
    for name in DATASETS:
        a = np.array(stats.get((name, "adlle"), [np.nan]))
        d = np.array(stats.get((name, "dlle_l"), [np.nan]))

        def fmt(v):
            v = np.array(v, dtype=float)
            return f"{v.mean():.2f}±{v.std(ddof=1):.2f}" if len(v) else "pending"
        lines.append(f"| {name} | {fmt(a)} | {fmt(d)} | {fmt(sle_m.get(name, []))} | "
                     f"{fmt(dw_m.get(name, []))} | {fmt(dp_m.get(name, []))} |")
    txt = "\n".join(lines)
    open(os.path.join(RESULTS, "sota_summary.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
