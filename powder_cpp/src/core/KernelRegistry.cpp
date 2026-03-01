#include "powder/core/KernelRegistry.hpp"

#include <stdexcept>

namespace powder::core {

void KernelRegistry::register_kernel(std::string name, KernelBackend backend, KernelFn fn, BenchmarkFn benchmark) {
  if (!fn) {
    throw std::runtime_error("kernel function must be valid");
  }
  table_[key(name, backend)] = KernelEntry{std::move(fn), std::move(benchmark)};
}

const KernelEntry* KernelRegistry::find(std::string_view name, KernelBackend backend) const {
  const auto it = table_.find(key(name, backend));
  if (it == table_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::string KernelRegistry::key(std::string_view name, KernelBackend backend) {
  return std::string(name) + "#" + std::to_string(static_cast<int>(backend));
}

}  // namespace powder::core
