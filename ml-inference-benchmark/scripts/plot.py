"""Gera gráficos e tabela markdown a partir de results/results.csv."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")
df["config"] = df["backend"] + " " + df["precision"]

for col, title, fname in [
    ("p50_ms", "Latência p50 (ms)", "latency_vs_batch.png"),
    ("p99_ms", "Latência p99 (ms)", "latency_p99_vs_batch.png"),
    ("throughput_img_s", "Throughput (img/s)", "throughput_vs_batch.png"),
    ("vram_mb", "VRAM (MB)", "vram_vs_batch.png"),
]:
    plt.figure(figsize=(7, 4.5))
    for cfg, g in df.groupby("config"):
        g = g.sort_values("batch")
        plt.plot(g["batch"], g[col], marker="o", label=cfg)
    plt.xscale("log", base=2)
    if col != "vram_mb":
        plt.yscale("log")
    plt.xlabel("Batch size"); plt.ylabel(title); plt.title(f"{title} — {df['gpu'].iloc[0]}")
    plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/{fname}", dpi=150); plt.close()

cols = ["config", "batch", "p50_ms", "p95_ms", "p99_ms", "throughput_img_s", "vram_mb", "ram_mb"]
open("results/table.md", "w").write(df.sort_values(["batch", "config"])[cols].to_markdown(index=False)
                                    if hasattr(df, "to_markdown") else df[cols].to_string(index=False))
print("[ok] gráficos e results/table.md gerados")
