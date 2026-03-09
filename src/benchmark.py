import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, '.')
from src.matmul import distributed_matmul

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKER_COUNTS    = [1, 2, 4, 8]
STRONG_N         = 4096    # fixed matrix size for strong scaling
WEAK_ROWS_PER    = 512   # rows per worker for weak scaling

# ── Serial baseline ───────────────────────────────────────────────────────────
def serial_matmul(A: np.ndarray, B: np.ndarray) -> float:
    start = time.perf_counter()
    _ = A @ B
    return time.perf_counter() - start

# ── Timed distributed run ─────────────────────────────────────────────────────
def timed_distributed(A: np.ndarray, B: np.ndarray,
                       world_size: int) -> float:
    start = time.perf_counter()
    distributed_matmul(A, B, world_size)
    return time.perf_counter() - start

# ── Strong scaling ────────────────────────────────────────────────────────────
def run_strong_scaling() -> pd.DataFrame:
    """Fix N=STRONG_N, vary number of workers."""
    logger.info(f"Strong scaling — N={STRONG_N}, "
                f"workers={WORKER_COUNTS}")
    np.random.seed(42)
    A = np.random.randn(STRONG_N, STRONG_N).astype(np.float32)
    B = np.random.randn(STRONG_N, STRONG_N).astype(np.float32)

    t_serial = serial_matmul(A, B)
    logger.info(f"Serial baseline: {t_serial:.4f}s")

    results = []
    for w in WORKER_COUNTS:
        if STRONG_N % w != 0:
            logger.warning(f"Skipping w={w} — {STRONG_N} not divisible by {w}")
            continue
        t = timed_distributed(A, B, w)
        speedup    = t_serial / t
        efficiency = speedup / w
        logger.success(f"  workers={w}: {t:.4f}s  "
                       f"speedup={speedup:.2f}x  "
                       f"efficiency={efficiency:.2f}")
        results.append({
            "workers":    w,
            "time":       round(t, 4),
            "speedup":    round(speedup, 3),
            "efficiency": round(efficiency, 3),
        })

    return pd.DataFrame(results)

# ── Weak scaling ──────────────────────────────────────────────────────────────
def run_weak_scaling() -> pd.DataFrame:
    """Scale N with workers — each worker always gets WEAK_ROWS_PER rows."""
    logger.info(f"Weak scaling — {WEAK_ROWS_PER} rows/worker, "
                f"workers={WORKER_COUNTS}")

    results = []
    t_baseline = None

    for w in WORKER_COUNTS:
        N = WEAK_ROWS_PER * w
        np.random.seed(42)
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)

        t = timed_distributed(A, B, w)

        if t_baseline is None:
            t_baseline = t

        efficiency = t_baseline / t
        logger.success(f"  workers={w}, N={N}: {t:.4f}s  "
                       f"efficiency={efficiency:.2f}")
        results.append({
            "workers":    w,
            "N":          N,
            "time":       round(t, 4),
            "efficiency": round(efficiency, 3),
        })

    return pd.DataFrame(results)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(strong_df: pd.DataFrame, weak_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0F1117")

    for ax in axes:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#8B8FA8")
        ax.xaxis.label.set_color("#C4C6D4")
        ax.yaxis.label.set_color("#C4C6D4")
        ax.title.set_color("#E8E9F0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#23263A")

    # ── left: strong scaling speedup
    ax1 = axes[0]
    ax1.plot(strong_df["workers"], strong_df["speedup"],
             "o-", color="#7C6AF7", linewidth=2, markersize=8,
             label="Actual speedup")
    ax1.plot(strong_df["workers"], strong_df["workers"],
             "--", color="#8B8FA8", linewidth=1.5,
             label="Ideal linear speedup")
    ax1.set_xlabel("Number of workers")
    ax1.set_ylabel("Speedup (T_serial / T_parallel)")
    ax1.set_title(f"Strong Scaling (N={STRONG_N})")
    ax1.set_xticks(strong_df["workers"])
    ax1.legend(facecolor="#1A1D27", labelcolor="#C4C6D4",
               edgecolor="#23263A")
    ax1.grid(True, color="#23263A", linewidth=0.7)

    # ── right: weak scaling efficiency
    ax2 = axes[1]
    ax2.plot(weak_df["workers"], weak_df["efficiency"],
             "o-", color="#3ECFCF", linewidth=2, markersize=8,
             label="Parallel efficiency")
    ax2.axhline(y=1.0, color="#8B8FA8", linestyle="--",
                linewidth=1.5, label="Ideal (1.0)")
    ax2.set_xlabel("Number of workers")
    ax2.set_ylabel("Efficiency (T_1 / T_n)")
    ax2.set_title(f"Weak Scaling ({WEAK_ROWS_PER} rows/worker)")
    ax2.set_xticks(weak_df["workers"])
    ax2.set_ylim(0, 1.3)
    ax2.legend(facecolor="#1A1D27", labelcolor="#C4C6D4",
               edgecolor="#23263A")
    ax2.grid(True, color="#23263A", linewidth=0.7)

    plt.tight_layout()
    out = OUTPUT_DIR / "scaling_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    logger.success(f"Plots saved to {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    strong_df = run_strong_scaling()
    weak_df   = run_weak_scaling()

    print("\n── Strong Scaling Results ───────────────────────")
    print(strong_df.to_string(index=False))

    print("\n── Weak Scaling Results ─────────────────────────")
    print(weak_df.to_string(index=False))

    strong_df.to_csv(OUTPUT_DIR / "strong_scaling.csv", index=False)
    weak_df.to_csv(OUTPUT_DIR / "weak_scaling.csv",   index=False)

    plot_results(strong_df, weak_df)