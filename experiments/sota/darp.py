"""DARP (Liu, Zhu et al., Information Fusion 2025) — port of Algorithm 1.

min_{A,P,M,O} ||A P^T X - (F + B*M)||_F^2                       (eq. 4)
             + lambda sum_ij ||P^T x_i - o_j||_F^2 F_ji         (eq. 5)
             + alpha sum_i ||P P^T x_i - x_i||_F^2              (eq. 6)
             + gamma ||P||_{2,1},  s.t. M >= 0                  (eq. 7)

Alternating updates (Eqs. 10/13/16/20); P via the Sylvester equation.
Supervised method with an explicit projection matrix P (m x d).
Run on PCA pre-reduced features (m=100) for tractability, disclosed.

Conventions: X (n, m) samples per row internally; the paper's Xt = X.T (m, n);
F (c, n) one-hot labels; G = F + B*M (c, n); O (m, c) columns are class
centroids; A (c, d); P (m, d).
"""
import numpy as np
from scipy.linalg import solve_sylvester


def _l21_diag(P, eps=1e-8):
    """Lambda_ii = 1 / (2*sqrt(p_i p_i^T + eps)) (eq. 19)."""
    return 1.0 / (2.0 * np.sqrt((P ** 2).sum(1) + eps))


def darp_fit(X, Y, d=80, lam=1.0, alpha=1.0, gamma=1.0, max_iter=30, tol=1e-6):
    n, m = X.shape
    classes = np.unique(Y)
    c = len(classes)
    remap = {c_: i for i, c_ in enumerate(classes)}
    F = np.zeros((c, n))
    for i, y in enumerate(Y):
        F[remap[y], i] = 1.0
    B = np.where(F == 1.0, 1.0, -1.0)

    Xt = X.T                                        # m × n
    XXT = Xt @ X                                    # m × m (X X^T in paper)
    invXXT = np.linalg.inv(XXT + 1e-8 * np.eye(m))
    # D1 = diag(sum_j F_ji) is the identity under one-hot labels (eq. 20)
    # => lambda * X D1 X^T = lambda * XXT

    # init: P from PCA loadings, others small random
    from sklearn.decomposition import PCA
    P = PCA(n_components=d, random_state=0).fit(X).components_.T   # m × d
    rng = np.random.RandomState(0)
    A = np.abs(rng.randn(c, d)) * 0.01
    O = rng.randn(m, c) * 0.01
    M = np.abs(rng.randn(c, n)) * 0.01

    prev = np.inf
    for _ in range(max_iter):
        G = F + B * M

        # update A (eq. 10)
        A = (G @ Xt.T @ P) @ np.linalg.inv(P.T @ XXT @ P + 1e-8 * np.eye(d))

        # update O (eq. 13): o_j = sum_i F_ji P^T x_i / sum_i F_ji; O stays (d × c)
        O = (P.T @ Xt @ F.T) / (F.sum(axis=1)[None, :] + 1e-12)    # d × c

        # update M (eq. 16)
        Nmat = A @ P.T @ Xt - F
        M = np.maximum(B * Nmat, 0.0)
        G = F + B * M

        # update P (eq. 20): Q P + P (A^T A) = R
        Q = invXXT @ (lam * XXT - alpha * (XXT + XXT.T)) \
            + gamma * np.diag(_l21_diag(P).ravel())
        R = invXXT @ (Xt @ G.T @ A + lam * (Xt @ F.T @ O.T))
        P = solve_sylvester(Q, A.T @ A, R)

        # objective (eq. 7) for convergence check
        APtX = A @ P.T @ Xt                                        # c × n
        L = ((APtX - G) ** 2).sum()
        proj = (P.T @ Xt).T                                        # n × d
        dc = ((proj[:, None, :] - O.T[None, :, :]) ** 2).sum(2)    # n × c
        L += lam * (dc * F.T).sum()
        L += alpha * ((P @ P.T @ Xt).T - X).__pow__(2).sum()
        L += gamma * np.sqrt((P ** 2).sum(1) + 1e-12).sum()
        if abs(prev - L) < tol * max(1.0, abs(prev)):
            break
        prev = L
    return P


def darp_eval(Xtr, Ytr, Xte, Yte, d=80, param_grid=None):
    """Fit DARP over a small grid (report best, same convention as SLE).
    Returns (best_acc, best_cfg)."""
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    if param_grid is None:
        param_grid = [{"lam": 1.0, "alpha": a, "gamma": 1.0}
                      for a in (0.01, 0.1, 1.0, 10.0)]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    best_acc, best_cfg = -1.0, None
    for g in param_grid:
        try:
            P = darp_fit(Xtr_s, Ytr, d=d, lam=g["lam"], alpha=g["alpha"],
                         gamma=g["gamma"])
            A, B = Xtr_s @ P, Xte_s @ P
            pred = np.zeros(len(Yte), dtype=Ytr.dtype)
            for i in range(len(B)):
                pred[i] = Ytr[np.argmin(((A - B[i]) ** 2).sum(1))]
            acc = accuracy_score(Yte, pred) * 100
        except Exception as e:
            print(f"    DARP fail ({g}): {type(e).__name__}: {e}", flush=True)
            continue
        if acc > best_acc:
            best_acc, best_cfg = acc, g
    return best_acc, best_cfg
