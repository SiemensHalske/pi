#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace powder::core {

enum class KernelBackend {
  Scalar,
  OpenMP,
  AVX2,
  AVX512,
};

using KernelFn = std::function<void(void*)>;
using BenchmarkFn = std::function<double(std::size_t)>;

struct KernelEntry {
  KernelFn fn;
  BenchmarkFn benchmark;
};

class KernelRegistry {
 public:
  void register_kernel(std::string name, KernelBackend backend, KernelFn fn, BenchmarkFn benchmark = {});
  [[nodiscard]] const KernelEntry* find(std::string_view name, KernelBackend backend) const;

 private:
  [[nodiscard]] static std::string key(std::string_view name, KernelBackend backend);
  std::unordered_map<std::string, KernelEntry> table_;
};

}  // namespace powder::core
