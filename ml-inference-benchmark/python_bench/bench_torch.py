"""Baseline PyTorch. Mede host->host: H2D + forward + D2H (mesma definição dos benchmarks C++)."""
import argparse
import csv
import os
import time

import numpy as np
import psutil
import pynvml
import torch
import torchvision

COLS = ["backend", "precision", "batch", "iters", "mean_ms", "p50_ms", "p95_ms",
        "p99_ms", "throughput_img_s", "vram_mb", "ram_mb", "gpu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu = pynvml.nvmlDeviceGetName(h)
    gpu = gpu.decode() if isinstance(gpu, bytes) else gpu
    used_mb = lambda: pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
    baseline = used_mb()  # antes de criar o contexto CUDA

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    model = torchvision.models.resnet50(weights="DEFAULT").eval().cuda().to(dtype)
    proc = psutil.Process(os.getpid())

    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLS)
        for b in args.batches:
            x_host = torch.randn(b, 3, 224, 224).pin_memory()

            def step():
                x = x_host.cuda(non_blocking=True).to(dtype)
                y = model(x)
                return y.float().cpu()  # bloqueante: sincroniza

            with torch.inference_mode():
                for _ in range(args.warmup):
                    step()
                torch.cuda.synchronize()
                lat = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    lat.append((time.perf_counter() - t0) * 1e3)

            lat = np.array(lat)
            mean = lat.mean()
            row = ["pytorch", args.precision, b, args.iters, f"{mean:.4f}",
                   f"{np.percentile(lat, 50):.4f}", f"{np.percentile(lat, 95):.4f}",
                   f"{np.percentile(lat, 99):.4f}", f"{b * 1e3 / mean:.2f}",
                   f"{used_mb() - baseline:.1f}", f"{proc.memory_info().rss / 2**20:.1f}", gpu]
            w.writerow(row)
            f.flush()
            print(f"pytorch {args.precision} b={b:<3} p50={row[5]}ms p99={row[7]}ms "
                  f"thr={row[8]} img/s vram={row[9]}MB")


if __name__ == "__main__":
    main()
