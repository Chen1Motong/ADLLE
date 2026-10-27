"""AN-DWLLE (paper ref [22], Zhang et al., TIMC 2023) — faithful port.

Unsupervised nonlinear LLE variant, no projection matrix; out-of-sample
mapping uses the same 5-NN regressor protocol as our LLE baseline (disclosed).
Pipeline per the paper:
  1) adaptive k per sample: candidate neighborhoods of size k (kmin..kmax)
     found by cosine similarity (eq. 6); pick k minimizing ||x_i - center||^2
     where center = (1/k) sum of the k neighbors (eq. 7);
  2) dual weight fusion: W' = alpha*W_E (LLE reconstruction weights, eq. 10)
     + (1-alpha)*W_S (neighbor-sequence weights, eq. 8);
  3) embedding Y = smallest eigenvectors of M = (I-W')^T (I-W') (eq. 11).
"""
import numpy as np


def cosine_topk(X, kmax):
    """Indices of the kmax most similar samples by cosine similarity (eq. 6)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -1)
    return np.argsort(-sim, axis=1)[:, :kmax]


def adaptive_k(X, nb, kmin, kmax):
    """Per-sample k minimizing ||x_i - center(candidate nbhd of size k)||^2 (eq. 7)."""
    n = X.shape[0]
    # cumulative sums of the kmax candidate neighbors (by cosine rank)
    cum = np.cumsum(X[nb], axis=1)              # n × kmax × D
    best_k = np.full(n, kmin, dtype=int)
    best_d = np.full(n, np.inf)
    xi2 = (X ** 2).sum(1)
    for t, k in enumerate(range(kmin, kmax + 1)):
        center = cum[:, t] / k
        d = xi2 - 2 * (X * center).sum(1) + (center ** 2).sum(1)
        better = d < best_d
        best_d[better] = d[better]
        best_k[better] = k
    return best_k


def lle_weights(X, nb, k_i):
    """LLE reconstruction weights over each sample's adaptive neighbors (eq. 10)."""
    n, D = X.shape
    W = np.zeros((n, n))
    for i in range(n):
        Z = X[nb[i, :k_i[i]]] - X[i]
        G = Z @ Z.T
        trace = np.trace(G)
        G[np.arange(k_i[i]), np.arange(k_i[i])] += max(1e-8, 1e-3 * trace)
        try:
            w = np.linalg.solve(G, np.ones(k_i[i]))
        except np.linalg.LinAlgError:
            w = np.ones(k_i[i]) / k_i[i]
        w = np.maximum(w, 0)
        w /= w.sum() + 1e-12
        W[i, nb[i, :k_i[i]]] = w
    return W


def embed(Xtr, alpha=0.5, kmin=3, kmax=20, target_dim=80):
    """Full AN-DWLLE on training data. Returns (n, target_dim) embedding."""
    nb = cosine_topk(Xtr, kmax)
    dist_all = np.linalg.norm(Xtr[:, None, :] - Xtr[None, :, :], axis=2) \
        if Xtr.shape[0] <= 400 else None
    if dist_all is None:
        # memory-safe squared distances
        sq = (Xtr ** 2).sum(1)
        dist_all = sq[:, None] + sq[None, :] - 2 * Xtr @ Xtr.T
        np.maximum(dist_all, 0, out=dist_all)
    dist_asc = np.take_along_axis(dist_all, nb, axis=1)
    k_i = adaptive_k(Xtr, nb, kmin, kmax)

    W_E = lle_weights(Xtr, nb, k_i)
    W_S = np.zeros_like(W_E)
    for i in range(Xtr.shape[0]):
        k = k_i[i]
        d = dist_asc[i, :k]
        W_S[i, nb[i, :k]] = d[::-1] / (d.sum() + 1e-12)
    Wp = alpha * W_E + (1 - alpha) * W_S

    M = (np.eye(n := Xtr.shape[0]) - Wp).T @ (np.eye(n) - Wp)
    vals, vecs = np.linalg.eigh(M)
    Y = vecs[:, 1:target_dim + 1]  # drop the trivial first component
    return Y
