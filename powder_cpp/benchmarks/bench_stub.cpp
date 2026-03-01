#include "powder/core/CpuFeatures.hpp"

#include <iostream>

int main() {
  const auto cpu = powder::core::detect_cpu_features();
  std::cout << "bench_stub cpu=" << cpu.to_string() << '\n';
  return 0;
}
