"""
OBD-RAP: zero-shot KV-cache compression by pruning RoPE-aligned dimension pairs.

Pairs are scored in the bi-directionally whitened space of OBD-LLM,
W~ = L_g^T W L_x with C_x = L_x L_x^T and C_g = L_g L_g^T (K-FAC factors),
removed whole as in RAP, and the remaining rows are corrected with the
structured OBS update.

Weights use the HuggingFace layout [n_heads * head_dim, d_model]: row
h * head_dim + j is dimension j of head h. Retained indices are returned as
an [n_heads, 2s] tensor of per-head dimension indices, sorted ascending.
"""
from typing import List, Tuple

import torch

Pair = Tuple[int, int]


def get_rope_pairs(head_dim: int, strategy: str = "half_split") -> List[Pair]:
    """Dimension pairs rotated together by RoPE.

    half_split: (k, k + D/2), HuggingFace's rotate_half layout (use for HF models).
    adjacent:   (2k, 2k + 1), Meta's original interleaved layout.
    """
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    half = head_dim // 2
    if strategy == "half_split":
        return [(k, k + half) for k in range(half)]
    if strategy == "adjacent":
        return [(2 * k, 2 * k + 1) for k in range(half)]
    raise ValueError(f"unknown strategy {strategy!r}")


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------

def dampen_covariance(C: torch.Tensor, fraction: float = 0.1) -> torch.Tensor:
    """C + fraction * mean(diag C) * I."""
    eps = fraction * C.diagonal().mean()
    return C + eps * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)


def cholesky_with_retry(C: torch.Tensor, fraction: float = 0.1, max_retries: int = 5) -> torch.Tensor:
    """Lower Cholesky factor of the damped C; damping grows 10x on each failure,
    and the last attempt also symmetrizes C."""
    for attempt in range(max_retries + 1):
        M = C if attempt < max_retries else (C + C.T) / 2
        try:
            return torch.linalg.cholesky(dampen_covariance(M, fraction))
        except torch.linalg.LinAlgError:
            if attempt == max_retries:
                raise
            fraction *= 10.0


def compute_whitened_weight(W: torch.Tensor, L_g: torch.Tensor, L_x: torch.Tensor) -> torch.Tensor:
    """W~ = L_g^T W L_x. Under K-FAC, removing row i raises the loss by ||W~[i]||^2 / 2."""
    return L_g.T @ W @ L_x


# ---------------------------------------------------------------------------
# Pair scores
# ---------------------------------------------------------------------------

def pair_sum(row_values: torch.Tensor, n_heads: int, head_dim: int, pairs: List[Pair]) -> torch.Tensor:
    """For per-row values [H*D], return value[j] + value[j'] per head and pair: [H, n_pairs]."""
    v = row_values.view(n_heads, head_dim)
    p = torch.tensor(pairs, device=v.device)
    return v[:, p[:, 0]] + v[:, p[:, 1]]


def compute_obd_scores(W_tilde: torch.Tensor, n_heads: int, head_dim: int, pairs: List[Pair]) -> torch.Tensor:
    """sigma_p = ||W~[j]||^2 + ||W~[j']||^2 (row norms: we prune output dims). [H, n_pairs]."""
    return pair_sum(W_tilde.pow(2).sum(dim=1), n_heads, head_dim, pairs)


def compute_augmented_scores(
    obd_scores: torch.Tensor, W: torch.Tensor, C_g: torch.Tensor, C_x: torch.Tensor,
    n_heads: int, head_dim: int, pairs: List[Pair], n_pruned: int,
) -> torch.Tensor:
    """sigma_bar_p = sigma_p + n_pruned * ||C_g||_2 * ||C_x - blkdiag(C_x)||_2 * ||W[p]||_F^2.

    blkdiag keeps the 2x2 blocks of C_x at (h*D + j, h*D + j'), i.e. it indexes the
    input covariance with output-pair positions (unchanged from the original).
    The correction term is the same constant for every pair of a layer times ||W[p]||^2.
    """
    p = torch.tensor(pairs, device=C_x.device)
    offsets = torch.arange(n_heads, device=C_x.device).unsqueeze(1) * head_dim
    gi = (offsets + p[:, 0]).reshape(-1)
    gj = (offsets + p[:, 1]).reshape(-1)
    C_off = C_x.clone()
    for a in (gi, gj):
        for b in (gi, gj):
            C_off[a, b] = 0
    c_off = torch.linalg.matrix_norm(C_off, ord=2).item()
    c_g = torch.linalg.eigvalsh(C_g)[-1].item()
    w_sq = pair_sum(W.float().pow(2).sum(dim=1), n_heads, head_dim, pairs)
    return (obd_scores + n_pruned * c_g * c_off * w_sq).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Budget, selection, compensation, absorption
# ---------------------------------------------------------------------------

def adaptive_budget_allocation(group_scores: List[float], target_rho: float) -> List[float]:
    """RAP Algorithm 2: rho_i = rho * (1 - s_i / sum s) / (1 - 1/N), clipped to [0, 1]
    and rescaled so the mean is rho. Important groups get pruned less."""
    n = len(group_scores)
    total = sum(group_scores)
    if n <= 1 or total == 0:
        return [target_rho] * n
    rhos = [min(1.0, max(0.0, target_rho * (1 - s / total) / (1 - 1 / n))) for s in group_scores]
    mean = sum(rhos) / n
    if mean > 0:
        rhos = [min(1.0, max(0.0, r * target_rho / mean)) for r in rhos]
    return rhos


def select_pairs(scores: torch.Tensor, pairs: List[Pair], s_retain: int) -> torch.Tensor:
    """Keep the s_retain highest-scoring pairs of each head. Returns [H, 2s] dim indices."""
    top = torch.argsort(scores, dim=1, descending=True)[:, :s_retain]
    p = torch.tensor(pairs, device=scores.device)
    return p[top].reshape(scores.shape[0], -1).sort(dim=1).values


def flat_rows(idx: torch.Tensor, head_dim: int) -> torch.Tensor:
    """[H, k] per-head indices -> flat row indices into an [H*head_dim, :] matrix."""
    offsets = torch.arange(idx.shape[0], device=idx.device).unsqueeze(1) * head_dim
    return (offsets + idx).reshape(-1)


def obs_weight_update(W: torch.Tensor, C_g: torch.Tensor, idx: torch.Tensor,
                      head_dim: int, damping_fraction: float = 0.01) -> torch.Tensor:
    """Retained rows R after removing rows S, with OBS compensation.

    Under K-FAC (H = C_x kron C_g) the optimal correction for rows R when rows S are
    zeroed is dW_R = C_g[R,R]^-1 C_g[R,S] W_S (C_x cancels), so W_R + dW_R is returned.
    This equals LLM Surgeon's multi-row update (van der Ouderaa et al., eq. 27).
    """
    keep = torch.zeros(W.shape[0], dtype=torch.bool, device=W.device)
    keep[flat_rows(idx.to(W.device), head_dim)] = True
    R, S = keep.nonzero().squeeze(1), (~keep).nonzero().squeeze(1)
    W = W.float()
    if S.numel() == 0:
        return W[R]
    C = C_g.float()
    C_RR = C[R][:, R]
    C_RR = C_RR + damping_fraction * C_RR.diagonal().mean() * torch.eye(len(R), device=C.device)
    return W[R] + torch.linalg.solve(C_RR, C[R][:, S] @ W[S])


def absorb_q(W_q: torch.Tensor, idx_k: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Keep the Q rows that match the retained K dims, so Q K^T is the same sub-sum."""
    return W_q[flat_rows(idx_k.to(W_q.device), head_dim)]


def absorb_o(W_o: torch.Tensor, idx_v: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Keep the O columns that read the retained V dims."""
    return W_o[:, flat_rows(idx_v.to(W_o.device), head_dim)]


# ---------------------------------------------------------------------------
# RoPE on a subset of dimensions
# ---------------------------------------------------------------------------

def apply_rope_non_contiguous(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                              rap_indices: torch.Tensor) -> torch.Tensor:
    """HF rotate_half RoPE on retained dims only.

    x: [B, S, H, 2s] with the s pair leaders j first and their partners j + D/2 after
    (the order select_pairs produces for half_split pairs).
    cos, sin: [S, D] for the original head_dim D (HF repeats each frequency, so
    cos[j] == cos[j + D/2]). rap_indices: [H, 2s] original dim indices.
    """
    s = x.shape[-1] // 2
    lead = rap_indices[:, :s]
    c, sn = cos[:, lead], sin[:, lead]          # [S, H, s], broadcasts over batch
    x1, x2 = x[..., :s], x[..., s:]
    return torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn], dim=-1)
