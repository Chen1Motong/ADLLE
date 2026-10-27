"""Table 12 rerun with multi-split protocol (R2-4 / R3-6).

Classifiers on frozen ResNet-18 features (mirroring the original protocol):
  1-NN          Euclidean nearest neighbour on standardized features
  LinearProbe   LogisticRegression(C=0.1, lbfgs, max_iter=5000)
  SVM           SVC(rbf, C=1.0, gamma='scale')
  ProtoNet      class means of L2-normalized features, Euclidean to prototypes
  MatchingNet   L2-normalized features, cosine attention over the support set
  Siamese-1NN   L2-normalized features, cosine-distance 1-NN
Ten random splits (same seeds/split rules as the ADLLE grid). The ADLLE column
is taken from results/stats.csv.
"""
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import adle_core as ac

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FEAT = os.path.join(RESULTS, "features")
SEEDS = range(10)
DATASETS = ["ORL", "AR", "Yale", "YaleB", "UMIST", "PIE", "COIL20", "COIL100"]


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)


def classify(name, Ftr, Ytr, Fte, Yte):
    out = {}
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr), sc.transform(Fte)

    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    out["1-NN"] = accuracy_score(Yte, Ytr[np.argmin(d, axis=0)]) * 100

    out["LinearProbe"] = LogisticRegression(max_iter=5000, C=0.1, random_state=42,
                                            n_jobs=-1).fit(A, Ytr).score(B, Yte) * 100
    out["SVM"] = SVC(kernel="rbf", C=1.0, gamma="scale",
                     random_state=42).fit(A, Ytr).score(B, Yte) * 100

    An, Bn = l2n(A), l2n(B)
    protos = np.vstack([An[Ytr == c].mean(0) for c in np.unique(Ytr)])
    pl = np.unique(Ytr)
    pd = np.sqrt(((protos[:, None, :] - Bn[None, :, :]) ** 2).sum(-1))
    out["ProtoNet"] = accuracy_score(Yte, pl[np.argmin(pd, axis=0)]) * 100

    sim = Bn @ An.T                                   # cosine similarity
    classes = np.unique(Ytr)
    # MatchingNet: softmax attention over the full support set (as in the
    # original protocol description; no FCE), scores aggregated per class
    attn = np.exp(sim - sim.max(axis=1, keepdims=True))
    attn = attn / attn.sum(axis=1, keepdims=True)
    scores = np.stack([attn[:, Ytr == c].sum(1) for c in classes], axis=1)
    out["MatchingNet"] = accuracy_score(Yte, classes[np.argmax(scores, axis=1)]) * 100

    out["Siamese-1NN"] = accuracy_score(Yte, Ytr[np.argmax(sim, axis=1)]) * 100
    return out


def main():
    # ADLLE reference column from the stats grid
    adlle = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "stats.csv"), encoding="utf-8")):
        if r["variant"] == "adlle":
            adlle.setdefault(r["dataset"], []).append(float(r["acc"]))

    out = os.path.join(RESULTS, "table12_multisplit.csv")
    cols = ["1-NN", "LinearProbe", "SVM", "ProtoNet", "MatchingNet", "Siamese-1NN"]
    new = not os.path.exists(out)
    done = set()
    if not new:
        for r in csv.DictReader(open(out, encoding="utf-8")):
            done.add((r["dataset"], int(r["seed"])))
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "seed"] + cols)
        for name in DATASETS:
            z = np.load(os.path.join(FEAT, f"{name}.npz"), allow_pickle=True)
            fmap = dict(zip(z["keys"].tolist(), z["features"]))
            for seed in SEEDS:
                if (name, seed) in done:
                    continue
                Xtr, Xte, Ytr, Yte, tri, tei = ac.load_dataset(name, seed=seed)
                # positional ids -> sample keys: the loader builds sample_ids in
                # the same order as its image list; recover via private helper
                ids = ac.sample_ids_for(name, seed)
                Ftr = np.vstack([fmap[k] for k in [ids[i] for i in tri]])
                Fte = np.vstack([fmap[k] for k in [ids[i] for i in tei]])
                res = classify(name, Ftr, Ytr, Fte, Yte)
                w.writerow([name, seed] + [f"{res[c]:.2f}" for c in cols])
                f.flush()
                print(f"[{name} seed{seed}] " +
                      " ".join(f"{c}={res[c]:.2f}" for c in cols), flush=True)

    # summary
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    lines = ["# Table 12 rerun: mean±std over 10 random splits", "",
             "| Dataset | ADLLE | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for name in DATASETS:
        a = np.array(adlle.get(name, [np.nan]))
        cells = [f"{a.mean():.2f}±{a.std(ddof=1):.2f}"]
        for c in cols:
            v = np.array([float(r[c]) for r in rows if r["dataset"] == name])
            cells.append(f"{v.mean():.2f}±{v.std(ddof=1):.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    with open(os.path.join(RESULTS, "table12_summary.md"), "w", encoding="utf-8") as f2:
        f2.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
