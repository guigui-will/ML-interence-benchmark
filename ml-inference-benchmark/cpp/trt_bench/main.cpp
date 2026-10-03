// Benchmark C++ / TensorRT 10 (API enqueueV3). Mede host->host: H2D + inferência + D2H + sync.
// Uso: trt_bench --engine models/resnet50_fp16.engine --precision fp16 --batch 8
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "bench_utils.hpp"

#define CUDA_CHECK(x)                                                                  \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "CUDA error '%s' em %s:%d\n", cudaGetErrorString(e_),      \
                   __FILE__, __LINE__);                                                \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", m);
  }
};

int main(int argc, char** argv) {
  bench::Args a(argc, argv);
  const std::string engine_path = a.get("engine", "models/resnet50_fp16.engine");
  const std::string precision = a.get("precision", "fp16");
  const std::string out = a.get("out", "results/results.csv");
  const int B = a.geti("batch", 1), iters = a.geti("iters", 300), warmup = a.geti("warmup", 50);

  bench::Nvml nvml;
  const double vram_base = nvml.used_mb();

  std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "Engine não encontrado: " << engine_path << "\n"; return 1; }
  std::vector<char> blob(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(blob.data(), blob.size());

  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine) { std::cerr << "Falha ao desserializar o engine\n"; return 1; }
  std::unique_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext());

  if (!ctx->setInputShape("input", nvinfer1::Dims4{B, 3, 224, 224})) {
    std::cerr << "Batch " << B << " fora do perfil de otimização do engine\n";
    return 1;
  }

  const size_t in_elems = static_cast<size_t>(B) * 3 * 224 * 224;
  const size_t out_elems = static_cast<size_t>(B) * 1000;
  float *h_in, *h_out, *d_in, *d_out;
  CUDA_CHECK(cudaMallocHost(&h_in, in_elems * sizeof(float)));  // pinned
  CUDA_CHECK(cudaMallocHost(&h_out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_in, in_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.f, 1.f);
  for (size_t i = 0; i < in_elems; ++i) h_in[i] = dist(rng);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  ctx->setTensorAddress("input", d_in);
  ctx->setTensorAddress("output", d_out);

  auto step = [&] {
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, in_elems * sizeof(float), cudaMemcpyHostToDevice, stream));
    if (!ctx->enqueueV3(stream)) { std::cerr << "enqueueV3 falhou\n"; std::exit(1); }
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  for (int i = 0; i < warmup; ++i) step();

  bench::Row r;
  r.backend = "trt_cpp";
  r.precision = precision;
  r.batch = B;
  r.iters = iters;
  r.gpu = nvml.name();
  for (int i = 0; i < iters; ++i) {
    double t0 = bench::now_ms();
    step();
    r.lat_ms.push_back(bench::now_ms() - t0);
  }
  r.vram_mb = nvml.used_mb() - vram_base;
  r.ram_mb = bench::ram_mb();
  bench::append_csv(out, r);

  cudaStreamDestroy(stream);
  cudaFreeHost(h_in); cudaFreeHost(h_out); cudaFree(d_in); cudaFree(d_out);
  return 0;
}
