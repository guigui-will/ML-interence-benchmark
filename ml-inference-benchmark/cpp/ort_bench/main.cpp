// Benchmark C++ / ONNX Runtime (CUDA EP). Mede host->host (cópias H2D/D2H ocorrem dentro do Run).
// Uso: ort_bench --model models/resnet50.onnx --precision fp32 --batch 8 [--iters 300 --warmup 50 --out results/results.csv]
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <random>

#include "bench_utils.hpp"

int main(int argc, char** argv) {
  bench::Args a(argc, argv);
  const std::string model = a.get("model", "models/resnet50.onnx");
  const std::string precision = a.get("precision", "fp32");
  const std::string out = a.get("out", "results/results.csv");
  const int B = a.geti("batch", 1), iters = a.geti("iters", 300), warmup = a.geti("warmup", 50);

  bench::Nvml nvml;
  const double vram_base = nvml.used_mb();  // antes de criar o contexto CUDA

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_bench");
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  OrtCUDAProviderOptions cuda_opts{};
  cuda_opts.device_id = 0;
  opts.AppendExecutionProvider_CUDA(cuda_opts);
  Ort::Session session(env, model.c_str(), opts);

  const std::vector<int64_t> shape = {B, 3, 224, 224};
  std::vector<float> input(static_cast<size_t>(B) * 3 * 224 * 224);
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.f, 1.f);
  for (auto& v : input) v = dist(rng);

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                      shape.data(), shape.size());
  const char* in_names[] = {"input"};
  const char* out_names[] = {"output"};

  auto run = [&] {
    auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    (void)outs[0].GetTensorData<float>();  // saída já está no host; Run é síncrono
  };

  for (int i = 0; i < warmup; ++i) run();

  bench::Row r;
  r.backend = "ort_cpp";
  r.precision = precision;
  r.batch = B;
  r.iters = iters;
  r.gpu = nvml.name();
  for (int i = 0; i < iters; ++i) {
    double t0 = bench::now_ms();
    run();
    r.lat_ms.push_back(bench::now_ms() - t0);
  }
  r.vram_mb = nvml.used_mb() - vram_base;
  r.ram_mb = bench::ram_mb();
  bench::append_csv(out, r);
  return 0;
}
