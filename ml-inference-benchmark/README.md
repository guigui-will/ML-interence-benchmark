# ML Inference Benchmark: PyTorch vs ONNX Runtime vs TensorRT (C++)

Pipeline: `PyTorch → ONNX → (ONNX Runtime | TensorRT) → C++ → GPU`

Modelo: ResNet-50 (torchvision), entrada `Nx3x224x224`, batch dinâmico.

## Metodologia
- **Definição de latência (igual nos 3 backends):** host → host, ou seja, H2D + inferência + D2H + sincronização.
- Warm-up de 50 iterações; 300 iterações medidas; reporta mean, p50, p95, p99.
- **Throughput** = batch / latência média. **FPS** = throughput com batch = 1.
- **VRAM**: delta de `nvmlDeviceGetMemoryInfo` vs. baseline antes de criar o contexto CUDA
  (inclui contexto CUDA, pesos, workspace e ativações). **RAM**: VmRSS/RSS do processo ao final.
- Entradas aleatórias com seed fixa; nada mais rodando na GPU.
- Para resultados estáveis: `sudo nvidia-smi -lgc <min>,<max>` (trave os clocks).

## Como rodar
```bash
# 1) Docker (recomendado)
docker build -t ml-bench -f docker/Dockerfile .
docker run --gpus all -it --rm -v $PWD/results:/workspace/results ml-bench

# 2) Dentro do container
python export/export_onnx.py          # gera ONNX FP32/FP16 e valida paridade
bash scripts/build_engines.sh         # engines TensorRT FP32/FP16
bash scripts/run_all.sh               # benchmark completo + gráficos
```

Sem Docker: `pip install -r requirements.txt`, instale CUDA, cuDNN, TensorRT e o ONNX Runtime GPU, depois
`cmake -B build -DORT_ROOT=/caminho/onnxruntime -DTENSORRT_ROOT=/caminho/TensorRT && cmake --build build -j`.

## Resultados
> Preencha com a sua GPU após rodar. Os gráficos ficam em `results/*.png` e a tabela em `results/table.md`.

| Config | Batch | p50 (ms) | p99 (ms) | img/s | VRAM (MB) |
|--------|-------|----------|----------|-------|-----------|
| ...    |       |          |          |       |           |

## Análise (escreva a sua)
- Onde o TensorRT ganhou mais e por quê (fusão de camadas, FP16/Tensor Cores)?
- Onde o ganho foi pequeno (batch 1, overhead de cópia H2D/D2H dominando)?
- Custo em VRAM/RAM de cada backend.
- Próximos passos: INT8 com calibração (medir queda de acurácia), pinned memory + CUDA Graphs, YOLOv8 com NMS.
