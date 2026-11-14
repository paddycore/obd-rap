"""Draw the README figures from results/*.csv.  Run from the repo root:
    python scripts/make_figures.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

# Fixed categorical order (validated for colour-vision deficiency); every series also
# gets its own marker and a direct label, so identity never relies on colour alone.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, MUTED, GRID = "#222220", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150,
})


def label_end(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8.5, color=INK)


def ppl_vs_kv():
    main = pd.read_csv(os.path.join(RES, "main_results.csv"))
    groups = pd.read_csv(os.path.join(RES, "lowrank_groups.csv"))
    df = pd.concat([main, groups], ignore_index=True)
    dense = df.loc[df.name == "dense", "ppl_std2048"].item()
    series = [
        ("OBD-RAP (iterative x3)", "OBD-RAP iterative ×3", BLUE, "o", 0),
        ("OBD-RAP (single-shot)", "OBD-RAP single-shot", ORANGE, "s", 0),
        ("RAP (fisher, no KD)", "RAP (Fisher, no KD)", AQUA, "^", 0),
        ("OBD-LLM (K/V low-rank, 32 groups)", "Low-rank per head (G=32)", YELLOW, "D", 4),
        ("OBD-LLM (K/V low-rank, 8 groups)", "Low-rank 8 groups (G=8)", MAGENTA, "v", -2),
        ("OBD-LLM (K/V low-rank)", "Low-rank joint (G=1)", GREEN, "P", -6),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(dense, color=MUTED, lw=1, ls="--")
    ax.annotate(f"dense {dense:.2f}", (0, dense), xytext=(2, -10), textcoords="offset points", fontsize=8.5, color=MUTED)
    for method, label, color, marker, dy in series:
        d = df[df.method == method].sort_values("kv_cache_red")
        ax.plot(d.kv_cache_red, d.ppl_std2048, color=color, lw=2, marker=marker, ms=6, label=label,
                markeredgecolor="white", markeredgewidth=1.2)
        label_end(ax, d.kv_cache_red.iloc[-1], d.ppl_std2048.iloc[-1], label, dy)
    ax.set_yscale("log")
    ax.set_xlim(-1, 58)
    ax.set_xlabel("KV-cache reduction (%)")
    ax.set_ylabel("WikiText-2 perplexity (log scale, lower is better)")
    ax.set_title("Llama-2-7B, training-free KV compression at matched cache size", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ppl_vs_kv.png"))


def latency():
    df = pd.read_csv(os.path.join(RES, "decode_latency_rho40.csv"))
    series = [("dense", "Dense", MUTED, "o"), ("obd_rap", "OBD-RAP", BLUE, "s"),
              ("lowrank_G32", "Low-rank G=32", YELLOW, "D"), ("lowrank_G8", "Low-rank G=8", MAGENTA, "v"),
              ("lowrank_G1", "Low-rank G=1", GREEN, "P")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, B in zip(axes, (1, 32)):
        for m, label, color, marker in series:
            d = df[(df.batch == B) & (df.method == m)].sort_values("context")
            ax.plot(d.context, d.attn_ms, color=color, lw=2, marker=marker, ms=6, label=label,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1024, 4096, 16384], ["1K", "4K", "16K"])
        ax.set_xlabel("context length")
        ax.set_title(f"batch {B}", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("attention decode step (ms, log)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("One Llama-2-7B layer, 40% KV reduction, A100-80GB fp16 (CUDA graphs)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "decode_latency.png"))


def accuracy():
    df = pd.read_csv(os.path.join(RES, "main_results.csv"))
    keep = ["Full model", "OBD-LLM (K/V low-rank)", "OBD-RAP (iterative x3)", "OBD-RAP (single-shot)",
            "RAP (fisher, no KD)", "RAP (magnitude, no KD)"]
    d = df[(df.ratio.isin([0.0, 0.2])) & df.method.isin(keep)].set_index("method").loc[keep]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = range(len(d))
    ax.barh(list(y), d.avg_acc, color=BLUE, height=0.55)
    for i, v in zip(y, d.avg_acc):
        ax.annotate(f"{v:.1f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y), keep)
    ax.invert_yaxis()
    ax.set_xlim(0, 70)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("average zero-shot accuracy, 6 tasks (%)")
    ax.set_title("20% KV reduction (full model shown for reference)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "accuracy_20.png"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    ppl_vs_kv()
    latency()
    accuracy()
    print("figures written to", FIG)
