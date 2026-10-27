"""Core ADLLE / DLLE/L implementations with ablation switches for the revision.

All experiment variants live here; the published repo (Finally/ADLLE/ADLLE.py)
is imported read-only for dataset loaders and is never modified.

Variants (penalty-graph construction):
  adlle        - full method: safety-radius Gaussian kernel, adaptive start,
                 energy-gain stopping, uniform edge weights (as published)
  w_gaussian   - same selection, but Gaussian-t_{ij}-weighted edge weights
  no_radius    - safety radius removed: kernel exp(-d^2/2rr^2), start at k_min
  no_stopping  - fixed k0 neighbors per sample (no energy stopping), uniform
  dlle_l       - DLLE/L baseline: fixed k0 nearest inter-class neighbors, uniform
"""

import os
import sys
import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(_REPO))
import ADLLE as repo  # noqa: E402  (published code, used read-only)

PARAMS = dict(k1=3, k0=15, u=3, kmax=300, kmin=2, eta=0.25, d=80, rr=1e-1, h=10)


# ---------------------------------------------------------------------------
# Dataset loading with controllable randomness for multi-split evaluation.
# Rules (resize, split sizes) mirror the repo loaders exactly; the only change
# is that sample order within each class is permuted with the given seed so
# that different splits can be drawn. Split indices are returned for release.
# ---------------------------------------------------------------------------

def load_dataset(name, seed=0, data_root=None):
    """Return (X_train, X_test, Y_train, Y_test, train_idx, test_idx)."""
    if data_root is None:
        data_root = os.path.join(os.path.abspath(_REPO), "data")
    rng = random.Random(seed)
    nprng = np.random.RandomState(seed)

    images, labels, sample_ids = [], [], []
    folders = {"ORL": "ORL", "AR": "AR", "Yale": "Yale", "YaleB": "Yale B",
               "UMIST": "UMIST", "PIE": "PIE", "COIL20": "coil-20-proc",
               "COIL100": "coil-100"}
    base = os.path.join(data_root, folders[name])

    def add_image(path, label, size):
        import cv2
        img = cv2.imread(path, 0)
        if img is None:
            return
        if size is not None:
            img = cv2.resize(img, size)
        images.append(img.flatten())
        labels.append(label)
        sample_ids.append(path)

    if name == "ORL":
        data = np.load(os.path.join(base, "olivetti.npz"))
        X, Y = data["X"], data["Y"]
        per_class = {c: np.where(Y == c)[0].tolist() for c in np.unique(Y)}
        for c, idxs in per_class.items():
            nprng.shuffle(idxs)
            images.extend(X[idxs])
            labels.extend([c] * len(idxs))
            sample_ids.extend([f"olivetti#{i}" for i in idxs])
        Xa, Ya = np.array(images, dtype=float), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=5, test_n=5, sample_ids=sample_ids, seed=seed)

    if name == "AR":
        for lb, sd in enumerate(sorted(os.listdir(base))):
            for f in sorted(os.listdir(os.path.join(base, sd))):
                if f.endswith(".jpg"):
                    add_image(os.path.join(base, sd, f), lb, (32, 32))
        Xa, Ya = np.array(images), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=13, test_n=13, sample_ids=sample_ids, seed=seed)

    if name == "Yale":
        for lb, sd in enumerate(sorted(os.listdir(base), key=lambda s: int(s))):
            for f in sorted(os.listdir(os.path.join(base, sd))):
                if f.endswith(".bmp"):
                    add_image(os.path.join(base, sd, f), lb, (64, 64))
        Xa, Ya = np.array(images), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=6, test_n=5, sample_ids=sample_ids, seed=seed)

    if name == "YaleB":
        dirs = sorted(d for d in os.listdir(base) if d.startswith("yaleB"))
        for lb, sd in enumerate(dirs):
            pgms = sorted(f for f in os.listdir(os.path.join(base, sd)) if f.endswith(".pgm"))
            if len(pgms) < 20:
                continue
            chosen = rng.sample(pgms, 20)
            for f in chosen:
                add_image(os.path.join(base, sd, f), lb, (64, 64))
        Xa, Ya = np.array(images), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=10, test_n=10, sample_ids=sample_ids, seed=seed)

    if name == "UMIST":
        dirs = sorted(d for d in os.listdir(base) if d.startswith("1"))
        for lb, sd in enumerate(dirs):
            pgms = sorted(f for f in os.listdir(os.path.join(base, sd)) if f.endswith(".pgm"))
            if len(pgms) < 16:
                continue
            chosen = rng.sample(pgms, 16)
            for f in chosen:
                add_image(os.path.join(base, sd, f), lb, (64, 64))
        Xa, Ya = np.array(images), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=5, test_n=5, sample_ids=sample_ids, seed=seed)

    if name == "PIE":
        sub = os.path.join(base, "Pose05_64x64_files")
        n_subj, per = 68, 49
        keep = list(range(1, n_subj * per + 1))
        for i in keep:
            p = os.path.join(sub, f"{i}.jpg")
            if os.path.exists(p):
                add_image(p, (i - 1) // per, None)
        Xa, Ya = np.array(images), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=25, test_n=24, sample_ids=sample_ids, seed=seed)

    if name == "COIL20":
        for i in range(1, 21):
            files = [f"obj{i}__{j}.png" for j in range(72)
                     if os.path.exists(os.path.join(base, f"obj{i}__{j}.png"))]
            chosen = rng.sample(files, 20)
            for f in chosen:
                add_image(os.path.join(base, f), i - 1, (64, 64))
        Xa, Ya = np.array(images), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=10, test_n=10, sample_ids=sample_ids, seed=seed)

    if name == "COIL100":
        angles = list(range(0, 360, 5))
        for i in range(1, 101):
            files = [f"obj{i}__{k}.png" for k in angles
                     if os.path.exists(os.path.join(base, f"obj{i}__{k}.png"))]
            chosen = rng.sample(files, 20)
            for f in chosen:
                add_image(os.path.join(base, f), i - 1, (64, 64))
        Xa, Ya = np.array(images), np.array(labels)
        return _split_per_class(name, Xa, Ya, train_n=10, test_n=10, sample_ids=sample_ids, seed=seed)

    raise ValueError(f"unknown dataset {name}")


_ID_CACHE = {}


def sample_ids_for(name, seed):
    """Sample keys (file paths / olivetti indices) in loader order for a split."""
    if (name, seed) not in _ID_CACHE:
        load_dataset(name, seed=seed)
    return _ID_CACHE[(name, seed)]


def _split_per_class(name, X, Y, train_n, test_n, sample_ids, seed):
    """Permute within each class (seeded), then take the standard prefix split."""
    nprng = np.random.RandomState(seed)
    tr_idx, te_idx = [], []
    for c in np.unique(Y):
        idx = np.where(Y == c)[0]
        nprng.shuffle(idx)
        take = min(train_n + test_n, len(idx))
        tr_idx.extend(idx[:min(train_n, take)].tolist())
        if take > train_n:
            te_idx.extend(idx[train_n:take].tolist())
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    _ID_CACHE[(name, seed)] = list(sample_ids)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    return X_tr, X_te, Y[tr_idx], Y[te_idx], tr_idx, te_idx


# ---------------------------------------------------------------------------
# Penalty-graph variants
# ---------------------------------------------------------------------------

def intrinsic_graph(X, Y, k1):
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        same = np.where(Y == Y[i])[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        d = np.linalg.norm(X[i] - X[same], axis=1)
        k = min(k1, len(same))
        nb = same[np.argsort(d)[:k]]
        Z = X[nb] - X[i]
        G = Z @ Z.T + np.eye(k) * 1e-8
        try:
            Gi = np.linalg.inv(G)
            W[i, nb] = Gi.sum(axis=1) / Gi.sum()
        except np.linalg.LinAlgError:
            W[i, nb] = 1.0 / k
    M = W
    return M + M.T - M.T @ M


def penalty_graph(X, Y, variant, kmax=300, kmin=2, k0=15, eta=0.25, rr=0.1, k1=3):
    """Return the (symmetric, row-normalized-by-symmetrization) penalty graph."""
    n = X.shape[0]
    Wp = np.zeros((n, n))

    def add_edges(i, nb):
        w = 1.0 / len(nb)
        for j in nb:
            if Wp[i, j] == 0:
                Wp[i, j] = Wp[j, i] = w
            else:
                Wp[i, j] = Wp[j, i] = (w + Wp[i, j]) / 2

    def add_edges_gaussian(i, nb, tw):
        for j, wt in zip(nb, tw):
            if Wp[i, j] == 0:
                Wp[i, j] = Wp[j, i] = wt
            else:
                Wp[i, j] = Wp[j, i] = (wt + Wp[i, j]) / 2

    for i in range(n):
        other = np.where(Y != Y[i])[0]
        if len(other) < kmin:
            continue
        d = np.linalg.norm(X[other] - X[i], axis=1)
        order = np.argsort(d)
        nb_all, dist = other[order][:kmax], d[order][:kmax]
        if len(nb_all) < kmin:
            continue

        if variant == "dlle_l":
            add_edges(i, nb_all[:k0])
            continue

        if variant == "no_radius":
            # boundary reference removed: plain RBF on the distance. The raw
            # bandwidth rr is calibrated for deviations from the safety radius
            # (O(1)), not raw distances (O(10)) in standardized space, so use a
            # data-adaptive scale instead; otherwise the kernel underflows.
            sigma = np.median(dist) if np.median(dist) > 0 else 1.0
            t = np.exp(-(dist ** 2) / (2 * sigma ** 2))
            start = kmin  # without a boundary reference the kernel is monotone
        else:
            # safety radius: farthest of the k1 NEAREST intra-class neighbors
            # (faithful to the published implementation, not the global max)
            same = np.where(Y == Y[i])[0]
            same = same[same != i]
            if len(same):
                d_in = np.linalg.norm(X[same] - X[i], axis=1)
                k1_eff = min(k1, len(same))
                d_im = np.sort(d_in)[:k1_eff].max()
            else:
                d_im = 1.0
            t = np.exp(-((dist - d_im) ** 2) / (2 * rr ** 2))
            start = max(int(np.argmax(t)), kmin - 1) + 1

        if variant == "no_stopping":
            k = min(k0, len(nb_all))
            add_edges(i, nb_all[:k])
            continue

        # forward scan with energy-gain stopping (incremental, matches repo)
        best_k = kmin
        prev_e = 0.0
        for k in range(start, len(nb_all) + 1):
            w = t[:k]
            w = w / (w.sum() + 1e-10)
            e = float(np.sum(w * dist[:k] ** 2))
            if k > start and prev_e > 0 and (e - prev_e) / prev_e < eta:
                best_k = k - 1
                break
            prev_e, best_k = e, k

        if variant == "w_gaussian":
            tw = t[:best_k]
            tw = tw / (tw.sum() + 1e-10)
            add_edges_gaussian(i, nb_all[:best_k], tw)
        else:
            add_edges(i, nb_all[:best_k])

    return Wp


def embed_and_eval(X_tr, Y_tr, X_te, Y_te, variant, params=PARAMS):
    Win = intrinsic_graph(X_tr, Y_tr, params["k1"])
    if variant == "k_matched":
        # isolate the stopping MECHANISM from the neighborhood SIZE: first run
        # the full adaptive rule, then use its mean selected count as a fixed k
        Wp_a = penalty_graph(X_tr, Y_tr, "adlle", kmax=params["kmax"], kmin=params["kmin"],
                             k0=params["k0"], eta=params["eta"], rr=params["rr"],
                             k1=params["k1"])
        target = max(2, int(round((Wp_a > 0).sum(1).mean())))
        Wp = penalty_graph(X_tr, Y_tr, "no_stopping", kmax=params["kmax"], kmin=params["kmin"],
                           k0=target, eta=params["eta"], rr=params["rr"], k1=params["k1"])
    else:
        Wp = penalty_graph(X_tr, Y_tr, variant, kmax=params["kmax"], kmin=params["kmin"],
                           k0=params["k0"], eta=params["eta"], rr=params["rr"],
                           k1=params["k1"])
    D_in = np.diag(Win.sum(1)); L = D_in - Win
    D_p = np.diag(Wp.sum(1)); Lp = D_p - Wp
    obj = X_tr.T @ Lp @ X_tr - params["h"] * X_tr.T @ L @ X_tr
    ev, evec = np.linalg.eigh(obj)
    W = evec[:, np.argsort(ev)[::-1][:params["d"]]]
    P_tr, P_te = X_tr @ W, X_te @ W
    pred = np.zeros(len(P_te), dtype=Y_tr.dtype)
    for a in range(len(P_te)):
        pred[a] = Y_tr[np.argmin(np.sqrt(np.sum((P_tr - P_te[a]) ** 2, axis=1)))]
    return accuracy_score(Y_te, pred) * 100
