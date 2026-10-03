#!/usr/bin/env bash
# Roda a matriz completa: backend x precisão x batch. Resultado em results/results.csv
set -euo pipefail
BATCHES="1 2 4 8 16 32 64"
OUT=results/results.csv
rm -f "$OUT"

for P in fp32 fp16; do
  python python_bench/bench_torch.py --precision $P --batches $BATCHES --out $OUT
done

for B in $BATCHES; do
  ./build/ort_bench --model models/resnet50.onnx      --precision fp32 --batch $B --out $OUT
  ./build/ort_bench --model models/resnet50_fp16.onnx --precision fp16 --batch $B --out $OUT
  ./build/trt_bench --engine models/resnet50_fp32.engine --precision fp32 --batch $B --out $OUT
  ./build/trt_bench --engine models/resnet50_fp16.engine --precision fp16 --batch $B --out $OUT
done
python scripts/plot.py
