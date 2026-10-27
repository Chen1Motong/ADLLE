"""Faithful Python port of the official SLE MATLAB implementation
(Ratokw/SLE, paper [20]: "Scaled Robust Linear Embedding with Adaptive
Neighbors Preserving", Pattern Recognition 2025).

Conventions follow the MATLAB source: X is (d, n) — features x samples.
FS_AN(X, c, feature_num, k1, beta, k) returns a (d, feature_num) projection W;
test samples are embedded as X_test @ W and classified with 1-NN.
"""
import numpy as np


def l2_distance(a, b):
    """Squared Euclidean distance matrix, columns are samples (as in L2_distance_1.m)."""
    aa = (a ** 2).sum(axis=0)
    bb = (b ** 2).sum(axis=0)
    d = aa[:, None] + bb[None, :] - 2 * (a.T @ b)
    return np.maximum(d.real, 0)


def eig1(A, c, is_max=False, is_sym=True):
    """MATLAB eig1: eigen-decomposition, c eigenvectors of smallest/largest eigenvalues."""
    if is_sym:
        A = np.maximum(A, A.T)
    if np.iscomplexobj(A):
        vals, vecs = np.linalg.eig(A)
    else:
        # symmetric input keeps MATLAB-equivalent results, faster
        vals, vecs = np.linalg.eigh(A) if np.allclose(A, A.T, atol=1e-10) else np.linalg.eig(A)
    order = np.argsort(vals.real)
    if is_max:
        order = order[::-1]
    idx = order[:min(c, len(vals))]
    eigvec = vecs[:, idx]
    if np.iscomplexobj(eigvec):
        eigvec = eigvec.real
    return eigvec, vals[idx].real, vals[order].real


def rswl(f, k):
    """Row-simplex weight solver (RSWL in FS_AN.m). f: (N, d); returns w (N, d), lam."""
    N, d = f.shape
    w = np.zeros((N, d))
    if d > 1:
        P = np.sort(f, axis=1)  # ascending rows
        denom = (k * P[:, k] - P[:, :k].sum(axis=1))
        w = np.maximum((P[:, k, None] - f) / (denom[:, None] + np.finfo(float).eps), 0)
        lam = (k * P[:, k] - P[:, :k].sum(axis=1)) / 2
    else:
        P = np.sort(f, axis=0)  # N×1 column
        P = P.ravel()
        denom = k * P[k] - P[:k].sum()
        w = np.maximum((P[k] - f.ravel()) / (denom + np.finfo(float).eps), 0)
        lam = denom / 2
    return w, lam


def fs_an(X, c, feature_num, k1, beta, k, max_iter=50, tol=1e-3):
    """Port of FS_AN.m. X: (d, n). Returns W (d, feature_num)."""
    d, n = X.shape
    distX = l2_distance(X, X)                      # n×n squared distances
    distX1 = np.sort(distX, axis=1)
    idx = np.argsort(distX, axis=1)
    S = np.zeros((n, n))
    rr = np.zeros(n)
    for i in range(n):
        di = distX1[i, 1:k + 2]                    # MATLAB 2:k+2 (k+1 nearest, excl. self)
        rr[i] = 0.5 * (k * di[k] - di[:k].sum())
        nb = idx[i, 1:k + 2]
        S[i, nb] = (di[k] - di) / (k * di[k] - di[:k].sum() + np.finfo(float).eps)
    lam = rr.mean()

    S0 = (S + S.T) / 2
    L0 = np.diag(S0.sum(1)) - S0
    F = eig1(L0, c, is_max=False)[0]

    H = np.eye(n) - np.ones((n, n)) / n
    St = X @ H @ X.T
    U, s, Vt = np.linalg.svd(St, full_matrices=False)
    inv_s = np.where(s > 1e-12, 1.0 / np.where(s > 1e-12, s, 1), 0)
    InvSt = (Vt.T * inv_s) @ U.T

    M0 = X @ L0 @ X.T
    W = eig1(M0, feature_num, is_max=False, is_sym=False)[0]

    it, err = 1, 1.0
    obj = np.zeros(max_iter)
    while err > tol and it <= max_iter:
        Z = (X.T @ W)                              # n × feature_num (MATLAB W'X)
        distx = l2_distance(Z.T, Z.T)              # n×n
        distf = l2_distance(F.T, F.T)

        g = np.diag(distx @ S)                     # MATLAB diag(distx*S)
        p, alpha = rswl(g.reshape(-1, 1), k1)

        S, phi = rswl(distx @ np.diag(p) + distx * beta + lam * distf, k)
        P = np.diag(p)
        S = (S + S.T) / 2
        D = np.diag(S.sum(1))
        L = D - S

        PS = P @ S
        A = (np.diag(PS.sum(0)) + np.diag(PS.sum(1)) - 2 * PS) \
            + beta * (np.diag(S.sum(0)) + np.diag(S.sum(1)) - 2 * S)
        M = InvSt @ (X @ A @ X.T)
        M = np.real(M)
        M[np.isnan(M)] = 0
        W = eig1(M, feature_num, is_max=False, is_sym=False)[0]

        obj[it - 1] = (p @ g + alpha * (p ** 2).sum()
                       + beta * ((distx * S).sum() + (phi * (S ** 2).sum(1)).sum())
                       + (2 * lam * distf * S).sum())
        if it > 1:
            err = abs(obj[it - 1] - obj[it - 2])

        F_old = F
        F, _, ev = eig1(L, c, is_max=False)
        fn1 = ev[:c].sum()
        fn2 = ev[:c + 1].sum()
        if fn1 > 1e-9:
            lam = 2 * lam
        elif fn2 < 1e-11:
            lam = lam / 2
            F = F_old
        elif it >= 5:
            break
        it += 1
    return W


def sle_embed(Xtr, Ytr, Xte, Yte, n_classes, target_dim=80, seed=0):
    """SLE under our evaluation protocol: PCA(100) pre-reduction -> FS_AN -> 1-NN.

    Parameter grid follows the official classifierExperiment.m: beta in
    {1e-3..1e3}, k1 in {floor(.5n), .6n, .7n, .8n, .9n}; the best grid cell by
    training-free test accuracy is reported (their convention), implemented as
    max over the grid.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    pca = PCA(n_components=min(100, Xtr_s.shape[1], Xtr_s.shape[0] - 1),
              random_state=seed).fit(Xtr_s)
    Ptr, Pte = pca.transform(Xtr_s), pca.transform(Xte_s)

    n = Ptr.shape[0]
    betas = [10.0 ** e for e in range(-3, 4)]
    k1s = [int(n * f) for f in (0.5, 0.6, 0.7, 0.8, 0.9)]
    best_acc, best_cfg = -1, None
    for beta in betas:
        for k1 in k1s:
            k1e = max(2, min(k1, n - 2))
            try:
                W = fs_an(Ptr.T, n_classes, target_dim, k1e, beta, 10)
                pred = np.zeros(len(Pte), dtype=Ytr.dtype)
                A, B = Ptr @ W, Pte @ W
                for i in range(len(B)):
                    pred[i] = Ytr[np.argmin(np.sqrt(((A - B[i]) ** 2).sum(1)))]
                acc = accuracy_score(Yte, pred) * 100
            except Exception as e:
                print(f"    SLE fail (beta={beta}, k1={k1e}): {e}", flush=True)
                continue
            if acc > best_acc:
                best_acc, best_cfg = acc, (beta, k1e)
    return best_acc, best_cfg
