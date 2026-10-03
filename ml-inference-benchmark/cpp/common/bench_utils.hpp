// Utilitários compartilhados: args, timer, percentis, RAM (VmRSS), VRAM (NVML), CSV.
#pragma once
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace bench {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string k = argv[i];
      if (k.rfind("--", 0) == 0) kv_[k.substr(2)] = argv[i + 1];
    }
  }
  std::string get(const std::string& k, const std::string& d = "") const {
    auto it = kv_.find(k);
    return it == kv_.end() ? d : it->second;
  }
  int geti(const std::string& k, int d) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? d : std::stoi(it->second);
  }

 private:
  std::map<std::string, std::string> kv_;
};

inline double ram_mb() {  // RSS atual do processo
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
      return kb / 1024.0;
    }
  }
  return 0.0;
}

class Nvml {  // VRAM usada na GPU 0 (do dispositivo inteiro; use o delta vs. baseline)
 public:
  Nvml() {
    nvmlInit_v2();
    nvmlDeviceGetHandleByIndex_v2(0, &dev_);
  }
  ~Nvml() { nvmlShutdown(); }
  double used_mb() const {
    nvmlMemory_t m;
    nvmlDeviceGetMemoryInfo(dev_, &m);
    return m.used / 1048576.0;
  }
  std::string name() const {
    char buf[96] = {0};
    nvmlDeviceGetName(dev_, buf, sizeof(buf));
    return buf;
  }

 private:
  nvmlDevice_t dev_;
};

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline double percentile(std::vector<double> v, double p) {  // nearest-rank
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(std::max(0.0, std::ceil(p / 100.0 * v.size()) - 1));
  return v[std::min(idx, v.size() - 1)];
}

struct Row {
  std::string backend, precision, gpu;
  int batch = 0, iters = 0;
  std::vector<double> lat_ms;
  double vram_mb = 0, ram_mb = 0;
};

inline void append_csv(const std::string& path, const Row& r) {
  bool fresh = true;
  { std::ifstream t(path); fresh = !t.good() || t.peek() == std::ifstream::traits_type::eof(); }
  std::ofstream f(path, std::ios::app);
  if (fresh)
    f << "backend,precision,batch,iters,mean_ms,p50_ms,p95_ms,p99_ms,"
         "throughput_img_s,vram_mb,ram_mb,gpu\n";
  double mean = std::accumulate(r.lat_ms.begin(), r.lat_ms.end(), 0.0) / r.lat_ms.size();
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.1f,%.1f,%s\n",
                r.backend.c_str(), r.precision.c_str(), r.batch, r.iters, mean,
                percentile(r.lat_ms, 50), percentile(r.lat_ms, 95), percentile(r.lat_ms, 99),
                r.batch * 1e3 / mean, r.vram_mb, r.ram_mb, r.gpu.c_str());
  f << buf;
  std::printf("%s %s b=%-3d p50=%.3fms p99=%.3fms thr=%.1f img/s vram=%.0fMB ram=%.0fMB\n",
              r.backend.c_str(), r.precision.c_str(), r.batch, percentile(r.lat_ms, 50),
              percentile(r.lat_ms, 99), r.batch * 1e3 / mean, r.vram_mb, r.ram_mb);
}

}  // namespace bench
