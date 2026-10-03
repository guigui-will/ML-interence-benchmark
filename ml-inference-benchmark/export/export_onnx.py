"""Exporta ResNet-50 (PyTorch) -> ONNX (FP32 e FP16) e valida paridade numérica."""
import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from onnxconverter_common import float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = torchvision.models.resnet50(weights="DEFAULT").eval()
    dummy = torch.randn(2, 3, 224, 224)

    fp32_path = f"{args.out_dir}/resnet50.onnx"
    torch.onnx.export(
        model, dummy, fp32_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"[ok] {fp32_path}")

    # FP16 (mantém entrada/saída em FP32 para a mesma interface dos benchmarks)
    fp16_path = f"{args.out_dir}/resnet50_fp16.onnx"
    m16 = float16.convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
    onnx.save(m16, fp16_path)
    print(f"[ok] {fp16_path}")

    # --- Paridade numérica PyTorch vs ONNX Runtime (CPU, para não depender de GPU) ---
    x = torch.randn(4, 3, 224, 224)
    with torch.inference_mode():
        ref = model(x).numpy()
    for path, tol in [(fp32_path, 1e-3), (fp16_path, 5e-2)]:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": x.numpy()})[0]
        max_diff = float(np.abs(out - ref).max())
        same_top1 = bool((out.argmax(1) == ref.argmax(1)).all())
        ok = np.allclose(out, ref, atol=tol, rtol=tol) and same_top1
        print(f"[{'PASS' if ok else 'FAIL'}] {path}: max|diff|={max_diff:.2e} top1_igual={same_top1}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
