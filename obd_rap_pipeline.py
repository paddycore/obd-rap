"""
OBD-RAP experiment pipeline for HuggingFace LLaMA-family models.

Same procedure as V5.ipynb (covariances -> adaptive budget -> pair pruning ->
OBS -> Q/O absorption -> attention-scale fix -> non-contiguous RoPE), plus the
iterative variant, as functions driven by one RunConfig.

Baselines run in the same setting (same model, calibration, protected layers,
KV-cache ratio and evaluation):
  rap         RAP without KD: K pairs scored by Fisher (or magnitude), V by
              per-head SVD absorbed into W_o, adaptive K/V budgets (RAP Sec. 4).
  obd_llm_kv  OBD-LLM low-rank decomposition of W_k and W_v only, rank set by the
              same KV-cache ratio (latent cache; K is rebuilt before RoPE).
              lr_groups splits the heads into groups with one latent each:
              1 = joint over all heads, n_heads = per head (cheaper to rebuild).

MHA only (num_key_value_heads == num_attention_heads); tested on transformers 4.57.
"""
import inspect
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

import obd_rap as O

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
LM_EVAL_TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "openbookqa")
METHODS = ("obd_rap", "rap", "obd_llm_kv")
_PATCH_VERSION = 2
SCORES = ("obd_bar", "obd", "fisher", "magnitude")


@dataclass
class RunConfig:
    """ratio: target KV-cache compression averaged over all layers. Protected layers are
    skipped, so the others are pruned at ratio * n_layers / n_prunable.
    n_rounds (obd_rap only): 1 = single-shot with adaptive per-layer budget; >1 = iterative,
    keeping the same fraction of pairs in every layer per round and re-collecting
    statistics between rounds.
    pair_align (obd_rap, rap): round kept pairs to a multiple of this; 4 keeps head widths
    a multiple of 8, which fused attention kernels need to run at full speed."""
    ratio: float = 0.3
    n_rounds: int = 1
    score_type: str = "obd_bar"      # obd_bar (V5), obd, fisher, magnitude
    use_obs: bool = True
    damping_fraction: float = 0.1    # Cholesky damping
    obs_damping: float = 1e-3        # OBS solve damping (V5 value)
    protect_first: int = 2
    protect_last: int = 3
    max_layer_rho: float = 0.9
    method: str = "obd_rap"          # obd_rap, rap, obd_llm_kv
    lr_groups: int = 1               # obd_llm_kv: head groups with one latent each
    pair_align: int = 1              # obd_rap, rap: kept pairs rounded to a multiple of this

    def name(self) -> str:
        r = f"rho{round(self.ratio * 100)}"
        if self.method == "obd_llm_kv":
            return f"{r}_obdllm_kv" + (f"_g{self.lr_groups}" if self.lr_groups > 1 else "")
        align = f"_a{self.pair_align}" if self.pair_align > 1 else ""
        if self.method == "rap":
            return f"{r}_rap_{self.score_type}{align}"
        mode = "single" if self.n_rounds == 1 else f"iter{self.n_rounds}"
        return f"{r}_{mode}_{self.score_type}" + ("" if self.use_obs else "_noobs") + align

    def label(self) -> str:
        if self.method == "rap":
            return f"RAP ({self.score_type}, no KD)"
        if self.method == "obd_llm_kv":
            return "OBD-LLM (K/V low-rank" + (f", {self.lr_groups} groups)" if self.lr_groups > 1 else ")")
        extra = [x for x, on in ((self.score_type, self.score_type != "obd_bar"),
                                 ("no OBS", not self.use_obs),
                                 (f"align {self.pair_align}", self.pair_align > 1)) if on]
        mode = "single-shot" if self.n_rounds == 1 else f"iterative x{self.n_rounds}"
        return f"OBD-RAP ({', '.join([mode] + extra)})"


def baseline_configs(ratio: float, **kw) -> List[RunConfig]:
    return [RunConfig(ratio=ratio, method="rap", score_type="fisher", use_obs=False, **kw),
            RunConfig(ratio=ratio, method="rap", score_type="magnitude", use_obs=False, **kw),
            RunConfig(ratio=ratio, method="obd_llm_kv", **kw)]


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

def model_dims(model):
    """(n_layers, n_heads, original head_dim); raises on GQA."""
    c = model.config
    n_kv = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    if n_kv != c.num_attention_heads:
        raise NotImplementedError(f"MHA only; got {c.num_attention_heads} query / {n_kv} KV heads")
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return c.num_hidden_layers, c.num_attention_heads, head_dim


def _device(model):
    return next(model.parameters()).device


def load_model(model_id: str, dtype=torch.float16, token: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", token=token, attn_implementation="sdpa")
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    model_dims(model)
    if any(getattr(l.self_attn, n).bias is not None for l in model.model.layers for n in PROJ):
        raise NotImplementedError("attention bias not supported")
    return model, tok


def snapshot_attention(model) -> List[Dict[str, torch.Tensor]]:
    """CPU copy of all q/k/v/o weights, used to reset the model between runs."""
    return [{n: getattr(l.self_attn, n).weight.detach().cpu().clone() for n in PROJ}
            for l in model.model.layers]


def _set_weight(linear: nn.Linear, w: torch.Tensor):
    ref = linear.weight
    linear.weight = nn.Parameter(w.to(ref.device, ref.dtype), requires_grad=False)
    linear.out_features, linear.in_features = w.shape


def _set_head_dim(attn, head_dim: int):
    attn.head_dim = head_dim
    attn.scaling = head_dim ** -0.5   # HF caches 1/sqrt(head_dim) at init


def restore_attention(model, snap):
    _, _, head_dim = model_dims(model)
    for layer, weights in zip(model.model.layers, snap):
        a = layer.self_attn
        for n, w in weights.items():
            _set_weight(getattr(a, n), w)
        _set_head_dim(a, head_dim)
        for attr in ("rap_indices_k", "rap_indices_v", "v_head_dim", "kv_latent_dims", "kv_latent_groups"):
            a.__dict__.pop(attr, None)


def _kv_dims(a):
    """Per-token K and V cache widths of one layer."""
    return getattr(a, "kv_latent_dims", None) or (a.k_proj.out_features, a.v_proj.out_features)


def _attn_params(a) -> int:
    """Attention parameters as deployed; low-rank K/V layers count their two factors
    (A: latent x d_model; B: block-diagonal over head groups, out x latent / groups)."""
    lat = getattr(a, "kv_latent_dims", None)
    g = getattr(a, "kv_latent_groups", 1)
    factored = lambda w, r: r * w.shape[1] + w.shape[0] * r // g
    kv = (a.k_proj.weight.numel() + a.v_proj.weight.numel() if lat is None else
          factored(a.k_proj.weight, lat[0]) + factored(a.v_proj.weight, lat[1]))
    return a.q_proj.weight.numel() + a.o_proj.weight.numel() + kv


def param_counts(model):
    raw = sum(getattr(l.self_attn, n).weight.numel() for l in model.model.layers for n in PROJ)
    attn = sum(_attn_params(l.self_attn) for l in model.model.layers)
    return sum(p.numel() for p in model.parameters()) - raw + attn, attn


def compression_stats(model, dense_params: int, dense_attn_params: int) -> dict:
    n_layers, n_heads, head_dim = model_dims(model)
    attns = [l.self_attn for l in model.model.layers]
    hds = [a.head_dim for a in attns]
    kv = sum(sum(_kv_dims(a)) for a in attns)
    total, attn = param_counts(model)
    return {
        "head_dims": hds,
        "v_head_dims": [getattr(a, "v_head_dim", a.head_dim) for a in attns],
        "avg_head_dim": sum(hds) / len(hds),
        "kv_cache_reduction": 1 - kv / (n_layers * 2 * n_heads * head_dim),
        "params_total": total,
        "params_reduction": 1 - total / dense_params,
        "attn_params_reduction": 1 - attn / dense_attn_params,
    }


# ---------------------------------------------------------------------------
# Patches: subset RoPE for pruned layers (rap_indices_k); V head size may differ
# from K (v_head_dim, used by the RAP baseline)
# ---------------------------------------------------------------------------

def install_rope_patch():
    import transformers.models.llama.modeling_llama as ml
    if getattr(ml, "_obd_rap_patched", 0) >= _PATCH_VERSION:   # re-patch over older versions
        return
    orig = ml.__dict__.setdefault("_orig_apply_rotary_pos_emb", ml.apply_rotary_pos_emb)

    def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        attn = inspect.currentframe().f_back.f_locals.get("self")
        idx = getattr(attn, "rap_indices_k", None)
        if idx is None:
            return orig(q, k, cos, sin, *args, **kwargs)
        cos, sin = cos.reshape(-1, *cos.shape[-2:])[0], sin.reshape(-1, *sin.shape[-2:])[0]
        rot = lambda t: O.apply_rope_non_contiguous(t.transpose(1, 2), cos, sin, idx).transpose(1, 2)
        return rot(q), rot(k)

    def attention_forward(self, hidden_states, position_embeddings, attention_mask,
                          past_key_values=None, cache_position=None, **kwargs):
        # LlamaAttention.forward (transformers 4.57) with a separate V head size.
        shape = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(*shape, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(*shape, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*shape, -1, getattr(self, "v_head_dim", self.head_dim)).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = ml.apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx,
                                          {"sin": sin, "cos": cos, "cache_position": cache_position})
        impl = self.config._attn_implementation
        attend = ml.eager_attention_forward if impl == "eager" else ml.ALL_ATTENTION_FUNCTIONS[impl]
        out, weights = attend(self, q, k, v, attention_mask, scaling=self.scaling,
                              dropout=0.0 if not self.training else self.attention_dropout, **kwargs)
        return self.o_proj(out.reshape(*shape, -1).contiguous()), weights

    ml.apply_rotary_pos_emb = apply_rotary_pos_emb
    ml.LlamaAttention.forward = attention_forward
    ml._obd_rap_patched = _PATCH_VERSION


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_wikitext_tokens(tok, split: str, join: str = "nonempty") -> torch.Tensor:
    """WikiText-2 raw as one token stream. "nonempty" drops blank lines (V5);
    "all" joins every line, as GPTQ/SparseGPT evaluation does."""
    from datasets import load_dataset
    lines = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)["text"]
    if join == "nonempty":
        lines = [t for t in lines if t.strip()]
    return tok("\n\n".join(lines), return_tensors="pt")["input_ids"][0]


def sample_calibration(tokens: torch.Tensor, n: int, seq_len: int, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    starts = random.sample(range(len(tokens) // seq_len), min(n, len(tokens) // seq_len))
    return [tokens[i * seq_len:(i + 1) * seq_len].unsqueeze(0) for i in starts]


# ---------------------------------------------------------------------------
# Covariances: C_x of the k/v input, C_g of the k and v output gradients
# ---------------------------------------------------------------------------

def _covariance_bytes(model) -> int:
    """float32 bytes for every layer's C_x, C_g(k) and C_g(v)."""
    total = 0
    for layer in model.model.layers:
        a = layer.self_attn
        total += a.k_proj.in_features ** 2 + a.k_proj.out_features ** 2 + a.v_proj.out_features ** 2
    return 4 * total


def _accum_device(model, headroom_gib: float):
    """Accumulate on the model's GPU when the covariances fit next to `headroom_gib`
    of free memory for activations; otherwise on the CPU."""
    dev = _device(model)
    if dev.type != "cuda":
        return torch.device("cpu")
    free, _ = torch.cuda.mem_get_info(dev)
    return dev if free > _covariance_bytes(model) + headroom_gib * 2 ** 30 else torch.device("cpu")


def collect_covariances(model, calib: List[torch.Tensor], log=print, log_every: int = 16,
                        accum_device=None, headroom_gib: float = 10.0):
    """C_x = E[x x^T] of the k/v input and C_g = E[g g^T] of the k and v output gradients,
    accumulated in float32 and returned on the CPU.

    Sums stay on the GPU when they fit (one 64 MB transfer per matrix at the end instead
    of one per sequence); `accum_device` overrides the choice. Weights stay frozen and
    only activations carry gradients, so no memory goes to weight gradients."""
    n_layers, _, _ = model_dims(model)
    dev = _device(model)
    acc_dev = torch.device(accum_device) if accum_device is not None else _accum_device(model, headroom_gib)
    cov = [{} for _ in range(n_layers)]
    counts = [0] * n_layers

    def add(l, key, t):
        m = t.detach().float().reshape(-1, t.shape[-1])
        prod = (m.T @ m).to(acc_dev)
        if key in cov[l]:
            cov[l][key].add_(prod)
        else:
            cov[l][key] = prod

    def on_input(l):
        def hook(mod, inp, out):
            add(l, "Cx", inp[0])          # k_proj and v_proj read the same input
            counts[l] += 1
        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, out: out.requires_grad_(True))]
    for l, layer in enumerate(model.model.layers):
        a = layer.self_attn
        hooks += [
            a.k_proj.register_forward_hook(on_input(l)),
            a.k_proj.register_full_backward_hook(lambda m, gi, go, l=l: add(l, "Cg_k", go[0])),
            a.v_proj.register_full_backward_hook(lambda m, gi, go, l=l: add(l, "Cg_v", go[0])),
        ]
    log(f"  accumulating covariances on {acc_dev}")
    t0 = time.time()
    try:
        for i, seq in enumerate(calib, 1):
            ids = seq.to(dev)
            model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
            if i % log_every == 0 or i == len(calib):
                log(f"  covariance {i}/{len(calib)} ({time.time() - t0:.0f}s)")
        out = {l: {k: (v / max(counts[l], 1)).cpu() for k, v in cov[l].items()} for l in range(n_layers)}
    finally:
        for h in hooks:
            h.remove()
        del cov
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def collect_fisher(model, calib: List[torch.Tensor], log=print, log_every: int = 16):
    """RAP's pair statistic (Eq. 6-7): per output row i of W_k and W_v,
    F[i] = E[sum_n (dL/dW[i, n])^2], with one calibration window per gradient.
    dL/dW = g^T x is formed from hooks, so no weight gradients are stored."""
    n_layers, _, _ = model_dims(model)
    dev = _device(model)
    fisher = [{} for _ in range(n_layers)]
    inputs = {}

    def save_input(l):
        def hook(mod, inp, out):
            inputs[l] = inp[0].detach()
        return hook

    def accumulate(l, key):
        def hook(mod, gi, go):
            x = inputs[l].reshape(-1, inputs[l].shape[-1]).float()
            g = go[0].detach().reshape(-1, go[0].shape[-1]).float()
            row = (g.T @ x).pow_(2).sum(dim=1)
            fisher[l][key] = fisher[l][key] + row if key in fisher[l] else row
        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, out: out.requires_grad_(True))]
    for l, layer in enumerate(model.model.layers):
        a = layer.self_attn
        hooks += [a.k_proj.register_forward_hook(save_input(l)),
                  a.k_proj.register_full_backward_hook(accumulate(l, "F_k")),
                  a.v_proj.register_full_backward_hook(accumulate(l, "F_v"))]
    t0 = time.time()
    try:
        for i, seq in enumerate(calib, 1):
            ids = seq.to(dev)
            model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
            inputs.clear()
            if i % log_every == 0 or i == len(calib):
                log(f"  fisher {i}/{len(calib)} ({time.time() - t0:.0f}s)")
    finally:
        for h in hooks:
            h.remove()
        inputs.clear()
    return {l: {k: (v / len(calib)).cpu() for k, v in fisher[l].items()} for l in range(n_layers)}


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

class _LayerFactors:
    """What compressing one layer needs, on `dev`: weights, covariances and Cholesky
    factors (only when the method uses them) and the K/V pair scores."""

    def __init__(self, attn, cov_l, fisher_l, n_heads, cfg: RunConfig, dev):
        self.n_heads, self.hd = n_heads, attn.head_dim
        self.pairs = O.get_rope_pairs(self.hd, "half_split")
        self.W = {p: getattr(attn, f"{p}_proj").weight.data.to(dev, torch.float32) for p in "qkvo"}
        whitened = cfg.method == "obd_llm_kv" or cfg.score_type in ("obd", "obd_bar")
        if whitened or (cfg.method == "obd_rap" and cfg.use_obs):
            self.Cx = cov_l["Cx"].to(dev)
            self.Cg = {"k": cov_l["Cg_k"].to(dev), "v": cov_l["Cg_v"].to(dev)}
        if whitened:
            self.Lx = O.cholesky_with_retry(self.Cx, cfg.damping_fraction)
            self.Lg = {p: O.cholesky_with_retry(self.Cg[p], cfg.damping_fraction) for p in "kv"}
        if cfg.method == "obd_llm_kv":
            return
        if cfg.score_type in ("obd", "obd_bar"):
            rows = {p: O.compute_whitened_weight(self.W[p], self.Lg[p], self.Lx).pow(2).sum(dim=1) for p in "kv"}
        elif cfg.score_type == "fisher":
            rows = {p: fisher_l[f"F_{p}"].to(dev) for p in "kv"}
        elif cfg.score_type == "magnitude":
            rows = {p: self.W[p].pow(2).sum(dim=1) for p in "kv"}
        else:
            raise ValueError(f"unknown score_type {cfg.score_type!r}")
        self.scores = {p: O.pair_sum(rows[p], n_heads, self.hd, self.pairs) for p in "kv"}


def _set_rap_indices(attn, p, idx):
    """Store retained dims; idx is relative to the current head, so compose it with
    earlier rounds to get original dims (needed for the RoPE frequencies)."""
    prev = getattr(attn, f"rap_indices_{p}", None)
    idx = idx.to(attn.q_proj.weight.device)
    setattr(attn, f"rap_indices_{p}", idx if prev is None else torch.gather(prev, 1, idx))


def _prune_layer(attn, f: _LayerFactors, s: int, cfg: RunConfig):
    """OBD-RAP: keep s pairs per head in K and V, OBS-correct, absorb into Q and O."""
    n_pairs = len(f.pairs)
    s = max(1, min(s, n_pairs))
    if s == n_pairs:
        return
    idx, new = {}, {}
    for p in "kv":
        scores = f.scores[p]
        if cfg.score_type == "obd_bar":
            scores = O.compute_augmented_scores(scores, f.W[p], f.Cg[p], f.Cx,
                                                f.n_heads, f.hd, f.pairs, n_pairs - s)
        idx[p] = O.select_pairs(scores, f.pairs, s)
        new[p] = (O.obs_weight_update(f.W[p], f.Cg[p], idx[p], f.hd, cfg.obs_damping) if cfg.use_obs
                  else f.W[p][O.flat_rows(idx[p], f.hd)])
    new["q"] = O.absorb_q(f.W["q"], idx["k"], f.hd)
    new["o"] = O.absorb_o(f.W["o"], idx["v"], f.hd)
    for p in "qkvo":
        _set_weight(getattr(attn, f"{p}_proj"), new[p])
    _set_head_dim(attn, 2 * s)
    for p in "kv":
        _set_rap_indices(attn, p, idx[p])


def _svd_heads(W_v, W_o, n_heads, hd, r):
    """Per-head rank-r SVD of W_v; the left factor is absorbed into W_o (RAP's V path)."""
    U, S, Vh = torch.linalg.svd(W_v.view(n_heads, hd, -1), full_matrices=False)
    new_v = (S[:, :r, None] * Vh[:, :r]).reshape(n_heads * r, -1)
    new_o = torch.einsum("ohd,hdr->ohr", W_o.view(W_o.shape[0], n_heads, hd), U[:, :, :r])
    return new_v, new_o.reshape(W_o.shape[0], n_heads * r)


def _rap_layer(attn, f: _LayerFactors, s_k: int, r_v: int):
    """RAP without KD: K keeps its s_k best pairs (absorbed into Q), V keeps r_v dims per
    head through truncated SVD (absorbed into O). K and V budgets differ."""
    s_k = max(1, min(s_k, len(f.pairs)))
    r_v = max(1, min(r_v, f.hd))
    idx = O.select_pairs(f.scores["k"], f.pairs, s_k)
    _set_weight(attn.k_proj, f.W["k"][O.flat_rows(idx, f.hd)])
    _set_weight(attn.q_proj, O.absorb_q(f.W["q"], idx, f.hd))
    if r_v < f.hd:
        new_v, new_o = _svd_heads(f.W["v"], f.W["o"], f.n_heads, f.hd, r_v)
        _set_weight(attn.v_proj, new_v)
        _set_weight(attn.o_proj, new_o)
    attn.v_head_dim = r_v
    _set_head_dim(attn, 2 * s_k)
    _set_rap_indices(attn, "k", idx)


def _obd_lowrank(W, Lg, Lx, r):
    """OBD-LLM rank-r approximation B @ A of W, truncated in the bi-whitened space (Eq. 17)."""
    U, S, Vh = torch.linalg.svd(Lg.T @ W @ Lx, full_matrices=False)
    root = S[:r].sqrt()
    B = torch.linalg.solve_triangular(Lg.T, U[:, :r] * root, upper=True)
    A = torch.linalg.solve_triangular(Lx.T, (root[:, None] * Vh[:r]).T, upper=True).T
    return B @ A


def _lowrank_layer(attn, f: _LayerFactors, r_group: int, groups: int, damping_fraction: float):
    """OBD-LLM on W_k and W_v, one rank-r_group latent per group of heads. Each group is
    whitened with its own block of C_g. The weights are stored rebuilt (B @ A), which gives
    the same outputs as the factored form; the cache width is recorded as groups * r_group."""
    rows = f.n_heads // groups * f.hd
    if r_group >= rows:
        return
    for p in "kv":
        if groups == 1:
            new = _obd_lowrank(f.W[p], f.Lg[p], f.Lx, r_group)
        else:
            new = torch.cat([_obd_lowrank(f.W[p][sl], O.cholesky_with_retry(f.Cg[p][sl, sl], damping_fraction),
                                          f.Lx, r_group)
                             for sl in (slice(g * rows, (g + 1) * rows) for g in range(groups))])
        _set_weight(getattr(attn, f"{p}_proj"), new)
    attn.kv_latent_dims = (groups * r_group,) * 2
    attn.kv_latent_groups = groups


def run_compression(model, cfg: RunConfig, dense_cov: dict, calib, log=print, fisher: Optional[dict] = None):
    """Compress in place. dense_cov (and fisher, for score_type='fisher') must come
    from the unpruned model."""
    if cfg.method not in METHODS or cfg.score_type not in SCORES:
        raise ValueError(f"unknown method/score: {cfg.method}/{cfg.score_type}")
    if cfg.method != "obd_rap" and cfg.n_rounds != 1:
        raise ValueError("baselines are single-shot")
    if cfg.score_type == "fisher" and fisher is None and cfg.method != "obd_llm_kv":
        raise ValueError("score_type='fisher' needs fisher=collect_fisher(model, calib)")
    install_rope_patch()
    n_layers, n_heads, head_dim = model_dims(model)
    if n_heads % cfg.lr_groups:
        raise ValueError("lr_groups must divide the number of heads")
    pairs = lambda x: max(cfg.pair_align, cfg.pair_align * round(x / cfg.pair_align))
    v_dims = lambda x: max(1, round(x)) if cfg.pair_align == 1 else 2 * pairs(x / 2)   # RAP's SVD'd V
    dev = _device(model)
    layers = model.model.layers
    prunable = [l for l in range(cfg.protect_first, n_layers - cfg.protect_last)]
    if not prunable or cfg.ratio <= 0:
        return {"eff_rho": 0.0, "rounds": []}
    eff_rho = min(cfg.ratio * n_layers / len(prunable), cfg.max_layer_rho)
    keep_per_round = (1 - eff_rho) ** (1 / cfg.n_rounds)
    info = {"eff_rho": eff_rho, "keep_per_round": keep_per_round, "rounds": []}

    cov, fish = dense_cov, fisher
    for r in range(cfg.n_rounds):
        t0 = time.time()
        if r > 0:
            cov = collect_covariances(model, calib, log=log)
            if cfg.score_type == "fisher":
                fish = collect_fisher(model, calib, log=log)
        factors = lambda l: _LayerFactors(layers[l].self_attn, cov[l], fish[l] if fish else None,
                                          n_heads, cfg, dev)
        if cfg.method == "obd_llm_kv":
            r_group = max(1, round((1 - eff_rho) * (n_heads // cfg.lr_groups) * head_dim))
            for l in prunable:
                _lowrank_layer(layers[l].self_attn, factors(l), r_group, cfg.lr_groups, cfg.damping_fraction)
        elif cfg.n_rounds == 1:
            # Adaptive budget over K and V groups from summed pair scores (RAP Alg. 2).
            # Factors are rebuilt per layer instead of kept, to bound GPU memory.
            groups = []
            for l in prunable:
                f = factors(l)
                groups += [f.scores["k"].sum().item(), f.scores["v"].sum().item()]
            rhos = O.adaptive_budget_allocation(groups, eff_rho)
            for i, l in enumerate(prunable):
                hd = layers[l].self_attn.head_dim
                s_k, s_v = (pairs(hd // 2 * (1 - rhos[2 * i + j])) for j in (0, 1))
                if cfg.method == "rap":
                    _rap_layer(layers[l].self_attn, factors(l), s_k, v_dims(hd * (1 - rhos[2 * i + 1])))
                else:   # OBD-RAP (V5): K and V share one head size, s = min(s_k, s_v)
                    _prune_layer(layers[l].self_attn, factors(l), min(s_k, s_v), cfg)
        else:
            for l in prunable:
                s = pairs(layers[l].self_attn.head_dim // 2 * keep_per_round)
                _prune_layer(layers[l].self_attn, factors(l), s, cfg)
        kv = sum(sum(_kv_dims(layers[l].self_attn)) for l in prunable) / (len(prunable) * 2 * n_heads)
        info["rounds"].append({"round": r + 1, "mean_kv_dim": kv, "seconds": round(time.time() - t0, 1)})
        log(f"  round {r + 1}/{cfg.n_rounds}: mean K/V dim per head in pruned layers {kv:.1f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def perplexity(model, tokens: torch.Tensor, seq_len: int, n_seqs: Optional[int] = None) -> float:
    """exp of the mean window loss over non-overlapping windows.
    Short protocol: seq_len=512, n_seqs=20. Standard: seq_len=2048, all windows."""
    dev = _device(model)
    n = tokens.numel() // seq_len
    n = min(n, n_seqs) if n_seqs else n
    windows = tokens[: n * seq_len].view(n, seq_len)
    loss = sum(model(input_ids=w[None].to(dev), labels=w[None].to(dev), use_cache=False).loss.float().item()
               for w in windows)
    return math.exp(loss / n)


def run_lm_eval(model, tok, tasks: Iterable[str] = LM_EVAL_TASKS, batch_size: int = 16,
                limit: Optional[int] = None) -> dict:
    """Zero-shot accuracy; "main" is acc_norm when the task has it, else acc."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    res = simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size),
                          tasks=list(tasks), num_fewshot=0, limit=limit)
    out = {}
    for task, metrics in res["results"].items():
        m = {k.split(",")[0]: v for k, v in metrics.items() if isinstance(v, (int, float))}
        out[task] = {"main": m.get("acc_norm", m.get("acc")), "acc": m.get("acc"), "acc_norm": m.get("acc_norm")}
    mains = [v["main"] for v in out.values() if v["main"] is not None]
    out["avg"] = sum(mains) / len(mains) if mains else None
    return out


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def env_info() -> dict:
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import lm_eval
        info["lm_eval"] = getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        pass
    return info


def config_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
