"""Controlled overlapping-classes + outlier experiment (R2-2d / R3-2).

Two synthetic settings:
  overlap  - two Gaussian classes with controllable separation
  multimode- each class has two sub-blobs (tests multi-modal safety radius)
Perturbations applied to the TRAINING set only:
  labels  - fraction p of training labels randomly flipped (class noise)
  gross   - p% extra far-away training points with random labels (gross outliers)
Records accuracy of adlle / dlle_l / no_radius and the mean selected
inter-class neighbor count of adlle (mechanism response).
"""
import csv
import os
import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import adle_core as ac

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
D, N_PER_CLASS = 100, 300
SEEDS = range(10)
PERTURB = [0.0, 0.05, 0.10, 0.20]


def make_split(seed, setting):
    rng = np.random.RandomState(seed)
    n = N_PER_CLASS
    c = 0.12  # per-dim half-separation
    if setting == "overlap":
        m0, m1 = -c * np.ones(D), +c * np.ones(D)
        X0 = m0 + rng.randn(n, D)
        X1 = m1 + rng.randn(n, D)
    else:  # multimode: two sub-blobs per class
        off = rng.randn(D) * 0.3
        X0 = np.vstack([(-c * np.ones(D) + rng.randn(n // 2, D)),
                        (-c * np.ones(D) + 2.5 * off / np.linalg.norm(off) + rng.randn(n // 2, D))])
        X1 = np.vstack([(+c * np.ones(D) + rng.randn(n // 2, D)),
                        (+c * np.ones(D) + 2.5 * off / np.linalg.norm(off) + rng.randn(n // 2, D))])
    X = np.vstack([X0, X1])
    Y = np.array([0] * n + [1] * n)
    idx = rng.permutation(len(Y))
    X, Y = X[idx], Y[idx]
    half = len(Y) // 2
    tr, te = idx[:half], idx[half:]
    Xtr, Xte = X[tr], X[te]
    Ytr, Yte = Y[tr], Y[te]
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), Ytr, Yte


def perturb(Xtr, Ytr, mode, p, seed):
    rng = np.random.RandomState(1000 + seed)
    X, Y = Xtr.copy(), Ytr.copy()
    if mode == "labels" and p > 0:
        k = int(p * len(Y))
        flip = rng.choice(len(Y), k, replace=False)
        Y[flip] = 1 - Y[flip]
    if mode == "gross" and p > 0:
        k = int(p * len(Y))
        Xg = rng.randn(k, X.shape[1]) * 6
        Yg = rng.randint(0, 2, k)
        X, Y = np.vstack([X, Xg]), np.concatenate([Y, Yg])
    return X, Y


def mean_degree(X, Y, params):
    Wp = ac.penalty_graph(X, Y, "adlle", kmax=params["kmax"], kmin=params["kmin"],
                          k0=params["k0"], eta=params["eta"], rr=params["rr"], k1=params["k1"])
    return float((Wp > 0).sum(1).mean())


def main():
    out = os.path.join(RESULTS, "outlier.csv")
    params = dict(ac.PARAMS)
    rows = []
    per_seed = []
    for setting in ["overlap", "multimode"]:
        for mode in ["labels", "gross"]:
            for p in PERTURB:
                accs = {"adlle": [], "dlle_l": [], "no_radius": []}
                degs = []
                for seed in SEEDS:
                    Xtr, Xte, Ytr, Yte = make_split(seed, setting)
                    Xp, Yp = perturb(Xtr, Ytr, mode, p, seed)
                    for v in accs:
                        pr = dict(params)
                        if v == "dlle_l":
                            pr["k0"] = 300
                        accs[v].append(ac.embed_and_eval(Xp, Yp, Xte, Yte, v, pr))
                    degs.append(mean_degree(Xp, Yp, params))
                r = [setting, mode, p,
                     f"{np.mean(accs['adlle']):.2f}", f"{np.mean(accs['dlle_l']):.2f}",
                     f"{np.mean(accs['no_radius']):.2f}", f"{np.mean(degs):.1f}"]
                for s_i in range(len(SEEDS)):
                    per_seed.append([setting, mode, p, s_i,
                                     f"{accs['adlle'][s_i]:.2f}", f"{accs['dlle_l'][s_i]:.2f}"])
                rows.append(r)
                print(f"{setting:9s} {mode:6s} p={p:4.0%}  adlle={r[3]}  "
                      f"dlle_l={r[4]}  no_radius={r[5]}  mean_k={r[6]}", flush=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "perturbation", "fraction", "adlle", "dlle_l",
                    "no_radius", "adlle_mean_selected_k"])
        w.writerows(rows)
    with open(out.replace("outlier.csv", "outlier_perseed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "perturbation", "fraction", "seed", "adlle", "dlle_l"])
        w.writerows(per_seed)
    print("saved", out)


if __name__ == "__main__":
    main()
