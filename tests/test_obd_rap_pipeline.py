"""Offline checks of obd_rap_pipeline on a tiny random LLaMA (no downloads).

Run from the repo root: python tests/test_obd_rap_pipeline.py   (needs torch + transformers 4.57.x)
"""
import copy
import math
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import obd_rap_pipeline as P

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=500, hidden_size=256, intermediate_size=512,
                  num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4,
                  max_position_embeddings=256, attn_implementation="sdpa")
model = LlamaForCausalLM(cfg).float().eval()
for p in model.parameters():
    p.requires_grad_(False)
model.config.use_cache = False
_, H, HD = P.model_dims(model)

calib = [torch.randint(0, 500, (1, 64)) for _ in range(8)]
x = torch.randint(0, 500, (2, 48))
dense_logits = model(input_ids=x).logits
snap = P.snapshot_attention(model)
dense_total, dense_attn = P.param_counts(model)

# 1. Patch with full identity indices == HF RoPE
P.install_rope_patch()
for layer in model.model.layers:
    layer.self_attn.rap_indices_k = torch.arange(HD).repeat(H, 1)
err = (model(input_ids=x).logits - dense_logits).abs().max().item()
print(f"[1] identity RoPE patch max err {err:.2e}")
assert err < 1e-4
P.restore_attention(model, snap)
assert (model(input_ids=x).logits - dense_logits).abs().max().item() < 1e-5

# 2. Covariances
cov = P.collect_covariances(model, calib, log=lambda *a: None)
assert cov[0]["Cx"].shape == (256, 256) and cov[0]["Cg_k"].shape == (256, 256)
assert all(torch.isfinite(cov[l][k]).all() for l in cov for k in cov[l])
assert cov[0]["Cg_k"].abs().sum() > 0
print("[2] covariances ok")


def reference_logits(pruned, protect):
    """Dense model with removed q/k/v rows zeroed and scaling set to the pruned
    head_dim: equals the pruned model exactly when no OBS update is applied."""
    ref = copy.deepcopy(dense_model)
    for l, (rl, pl) in enumerate(zip(ref.model.layers, pruned.model.layers)):
        pa, ra = pl.self_attn, rl.self_attn
        if not hasattr(pa, "rap_indices_k"):
            continue
        for name, idx in (("q_proj", pa.rap_indices_k), ("k_proj", pa.rap_indices_k),
                          ("v_proj", pa.rap_indices_v)):
            w = getattr(ra, name).weight.data
            keep = torch.zeros(H * HD, dtype=torch.bool)
            for h in range(H):
                keep[h * HD + idx[h]] = True
            w[~keep] = 0
        ra.scaling = pa.head_dim ** -0.5
    import transformers.models.llama.modeling_llama as ml
    saved = ml.apply_rotary_pos_emb
    ml.apply_rotary_pos_emb = ml._orig_apply_rotary_pos_emb
    try:
        return ref(input_ids=x).logits
    finally:
        ml.apply_rotary_pos_emb = saved


dense_model = copy.deepcopy(model)

# 3. Single-shot, no OBS: must equal the zeroed reference
for rounds in (1, 3):
    P.restore_attention(model, snap)
    rc = P.RunConfig(ratio=0.3, n_rounds=rounds, use_obs=False, protect_first=1, protect_last=1)
    info = P.run_compression(model, rc, cov, calib, log=lambda *a: None)
    out = model(input_ids=x).logits
    ref = reference_logits(model, None)
    err = (out - ref).abs().max().item()
    hds = [l.self_attn.head_dim for l in model.model.layers]
    print(f"[3] rounds={rounds} no-OBS vs zeroed reference max err {err:.2e}; head_dims {hds}")
    assert err < 1e-4, err
    gap = (out - dense_logits).abs().max().item()
    print(f"    pruned vs dense gap {gap:.2e}")
    assert gap > 1e-4
    # wrong RoPE frequencies must break the match: shift indices by one pair
    a0 = [l.self_attn for l in model.model.layers if hasattr(l.self_attn, "rap_indices_k")][0]
    good = a0.rap_indices_k.clone(); s2 = good.shape[1] // 2
    bad = good.clone(); bad[:, :s2] = (good[:, :s2] + 1) % (HD // 2); bad[:, s2:] = bad[:, :s2] + HD // 2
    a0.rap_indices_k = bad
    bad_err = (model(input_ids=x).logits - ref).abs().max().item()
    a0.rap_indices_k = good
    print(f"    with wrong RoPE indices err {bad_err:.2e}")
    assert bad_err > 1e-5
    # composed indices must be sorted, unique and half-split aligned
    for l in model.model.layers:
        a = l.self_attn
        if hasattr(a, "rap_indices_k"):
            idx = a.rap_indices_k
            s = idx.shape[1] // 2
            assert torch.equal(idx[:, s:], idx[:, :s] + HD // 2)
            assert (idx[:, :s] < HD // 2).all()

# 4. With OBS (default), all score types, plus generation with KV cache
for score_type in ("obd_bar", "obd"):
    for rounds in (1, 3):
        P.restore_attention(model, snap)
        rc = P.RunConfig(ratio=0.3, n_rounds=rounds, score_type=score_type,
                         protect_first=1, protect_last=1)
        P.run_compression(model, rc, cov, calib, log=lambda *a: None)
        ids = torch.randint(0, 500, (1, 64))
        loss = model(input_ids=ids, labels=ids).loss.item()
        assert torch.isfinite(torch.tensor(loss))
        model.config.use_cache = True
        gen = model.generate(input_ids=ids[:, :8], max_new_tokens=8, do_sample=False)
        model.config.use_cache = False
        st = P.compression_stats(model, dense_total, dense_attn)
        print(f"[4] {rc.name():22s} loss {loss:.3f} kv_red {st['kv_cache_reduction']:.3f} "
              f"params_red {st['params_reduction']:.3f} gen {tuple(gen.shape)}")

# 5. Restore gives back the dense model exactly
P.restore_attention(model, snap)
err = (model(input_ids=x).logits - dense_logits).abs().max().item()
print(f"[5] restore max err {err:.2e}")
assert err < 1e-5

# 6. Perplexity helper and GQA rejection
toks = torch.randint(0, 500, (1000,))
print(f"[6] ppl {P.perplexity(model, toks, 100, n_seqs=3):.1f}")
gqa = LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=64, intermediate_size=64,
                                   num_hidden_layers=1, num_attention_heads=4,
                                   num_key_value_heads=2))
try:
    P.model_dims(gqa)
    raise AssertionError("GQA not rejected")
except NotImplementedError:
    print("[6] GQA rejected")

# 7. Fisher statistic equals squared weight gradients summed over input dims
fisher = P.collect_fisher(model, calib, log=lambda *a: None)
ref = {}
for seq in calib:
    for l, layer in enumerate(model.model.layers):
        for p in ("k", "v"):
            getattr(layer.self_attn, f"{p}_proj").weight.requires_grad_(True)
    model.zero_grad()
    model(input_ids=seq, labels=seq).loss.backward()
    for l, layer in enumerate(model.model.layers):
        for p in ("k", "v"):
            w = getattr(layer.self_attn, f"{p}_proj").weight
            ref[(l, p)] = ref.get((l, p), 0) + w.grad.pow(2).sum(dim=1)
            w.grad = None
            w.requires_grad_(False)
err = max(((fisher[l][f"F_{p}"] - ref[(l, p)] / len(calib)).abs().max() / ref[(l, p)].abs().max()).item()
          for (l, p) in ref)
print(f"[7] fisher vs explicit weight gradients, max rel err {err:.2e}")
assert err < 1e-4

# 8. RAP baseline: K pairs + V per-head SVD absorbed into O == dense model with the
#    removed K dims zeroed and W_v replaced by its per-head rank-r reconstruction
for score in ("fisher", "magnitude"):
    P.restore_attention(model, snap)
    rc = P.RunConfig(ratio=0.3, method="rap", score_type=score, use_obs=False, protect_first=1, protect_last=1)
    P.run_compression(model, rc, cov, calib, log=lambda *a: None, fisher=fisher)
    ref_model = copy.deepcopy(dense_model)
    for rl, pl in zip(ref_model.model.layers, model.model.layers):
        pa, ra = pl.self_attn, rl.self_attn
        if not hasattr(pa, "rap_indices_k"):
            continue
        keep = torch.zeros(H * HD, dtype=torch.bool)
        for h in range(H):
            keep[h * HD + pa.rap_indices_k[h]] = True
        for name in ("q_proj", "k_proj"):
            getattr(ra, name).weight.data[~keep] = 0
        r = pa.v_head_dim
        U, S, Vh = torch.linalg.svd(ra.v_proj.weight.data.view(H, HD, -1), full_matrices=False)
        ra.v_proj.weight.data = (U[:, :, :r] @ (S[:, :r, None] * Vh[:, :r])).reshape(H * HD, -1)
        ra.scaling = pa.head_dim ** -0.5
    import transformers.models.llama.modeling_llama as ml
    saved = ml.apply_rotary_pos_emb
    ml.apply_rotary_pos_emb = ml._orig_apply_rotary_pos_emb
    try:
        ref_logits = ref_model(input_ids=x).logits
    finally:
        ml.apply_rotary_pos_emb = saved
    out = model(input_ids=x).logits
    err = (out - ref_logits).abs().max().item()
    st = P.compression_stats(model, dense_total, dense_attn)
    print(f"[8] {rc.label():26s} vs reference max err {err:.2e}; K dims {st['head_dims']} "
          f"V dims {st['v_head_dims']} kv_red {st['kv_cache_reduction']:.3f}")
    assert err < 1e-3, err
    assert abs(st["kv_cache_reduction"] - 0.3 * 6 / 4 * 4 / 6) < 0.08   # 4 of 6 layers pruned at eff rho
    model.config.use_cache = True
    model.generate(input_ids=x[:1, :8], max_new_tokens=6, do_sample=False)
    model.config.use_cache = False

# 9. OBD-LLM K/V low-rank: rank-limited weights, latent cache width recorded
P.restore_attention(model, snap)
rc = P.RunConfig(ratio=0.3, method="obd_llm_kv", protect_first=1, protect_last=1)
P.run_compression(model, rc, cov, calib, log=lambda *a: None)
a = model.model.layers[2].self_attn
r = a.kv_latent_dims[0]
rank = torch.linalg.matrix_rank(a.k_proj.weight.float()).item()
st = P.compression_stats(model, dense_total, dense_attn)
loss = model(input_ids=x, labels=x).loss.item()
print(f"[9] {rc.label()}: rank(W_k)={rank} <= r={r}; kv_red {st['kv_cache_reduction']:.3f} "
      f"attn_params_red {st['attn_params_reduction']:.3f} loss {loss:.3f}")
assert rank <= r and math.isfinite(loss)

# 10. Grouped low-rank: one latent per head group, each block rank-limited
for G in (2, H):
    P.restore_attention(model, snap)
    rc = P.RunConfig(ratio=0.3, method="obd_llm_kv", lr_groups=G, protect_first=1, protect_last=1)
    P.run_compression(model, rc, cov, calib, log=lambda *a: None)
    a = model.model.layers[2].self_attn
    r_total, rows = a.kv_latent_dims[0], H // G * HD
    for p in ("k_proj", "v_proj"):
        blocks = getattr(a, p).weight.float().split(rows)
        assert len(blocks) == G and all(torch.linalg.matrix_rank(b).item() <= r_total // G for b in blocks)
    d = a.k_proj.weight.shape[1]
    expect = a.q_proj.weight.numel() + a.o_proj.weight.numel() + 2 * (r_total * d + H * HD * r_total // G)
    assert P._attn_params(a) == expect
    loss = model(input_ids=x, labels=x).loss.item()
    assert math.isfinite(loss)
    print(f"[10] {rc.label()} [{rc.name()}]: latent {r_total} = {G} x {r_total // G}, loss {loss:.3f}")

# 11. pair_align=4: every compressed K/V head width is a multiple of 8
fisher = P.collect_fisher(model, calib, log=lambda *a: None) if "fisher" not in globals() else fisher
for kw in (dict(n_rounds=1), dict(n_rounds=3), dict(method="rap", score_type="fisher", use_obs=False)):
    P.restore_attention(model, snap)
    rc = P.RunConfig(ratio=0.3, pair_align=4, protect_first=1, protect_last=1, **kw)
    P.run_compression(model, rc, cov, calib, log=lambda *a: None, fisher=fisher)
    st = P.compression_stats(model, dense_total, dense_attn)
    assert all(h % 8 == 0 for h in st["head_dims"] + st["v_head_dims"]), st
    loss = model(input_ids=x, labels=x).loss.item()
    assert math.isfinite(loss)
    print(f"[11] {rc.label()} [{rc.name()}]: K dims {st['head_dims']} V dims {st['v_head_dims']}")
P.restore_attention(model, snap)
assert (model(input_ids=x).logits - dense_logits).abs().max().item() < 1e-5
print("ALL OK")
