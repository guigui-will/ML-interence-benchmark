#!/usr/bin/env bash
# Gera engines TensorRT FP32 e FP16 com perfil dinâmico de batch (1..64).
set -euo pipefail
ONNX=${1:-models/resnet50.onnx}
for P in fp32 fp16; do
  FLAG=""; [ "$P" = "fp16" ] && FLAG="--fp16"
  trtexec --onnx="$ONNX" --saveEngine="models/resnet50_${P}.engine" $FLAG \
    --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:64x3x224x224
done
