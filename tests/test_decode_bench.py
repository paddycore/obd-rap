"""Each benchmarked decode step must compute the same output as plain attention with
the equivalent dense weights, so the timings compare equal work."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decode_bench as db

torch.manual_seed(0)
d, H, D, B, T = 64, 8, 16, 2, 12
X = torch.randn(B, T, d, dtype=torch.float64)


def check(name, m):
    m.prefill(X[:, :-1])
    out = m.step(X[:, -1])
    ref = db.reference_last_token(X, *m.full_weights(), H, D)
    err = ((out - ref).abs().max() / ref.abs().max()).item()
    print(f"{name:<14} rel err {err:.1e}")
    assert err < 1e-10, name


check("dense", db.PrunedAttention(d, H, D, D // 2, B, T, dtype=torch.float64))
check("obd_rap s=5", db.PrunedAttention(d, H, D, 5, B, T, dtype=torch.float64))
for G in (1, 2, 8):
    check(f"lowrank G={G}", db.LowRankAttention(d, H, D, 48, G, B, T, dtype=torch.float64))

# a wrong RoPE position must be caught
m = db.PrunedAttention(d, H, D, 5, B, T, dtype=torch.float64)
m.prefill(X[:, :-1]); m.cos, m.sin = m.cos.roll(1, 0), m.sin.roll(1, 0)
assert not torch.allclose(m.step(X[:, -1]), db.reference_last_token(X, *m.full_weights(), H, D))

rows = db.run_grid(batches=(1,), contexts=(64,), ratios=(0.3,), groups=(1, 8), dims=dict(d=64, H=8, D=16, ffn=128, n_layers=2),
                   device="cpu", dtype=torch.float32, iters=3, warmup=1, log=lambda *a: None)
assert len(rows) == 4 and all(r["attn_ms"] > 0 for r in rows)
kv = {r["method"]: r["kv_kb_per_token"] for r in rows}
print("kv per token:", kv)
print("ALL OK")
