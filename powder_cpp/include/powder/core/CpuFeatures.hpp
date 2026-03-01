#pragma once

#include <string>

namespace powder::core {

struct CpuFeatures {
  bool arch_x86 = false;
  bool runtime_avx2 = false;
  bool runtime_avx512f = false;
  bool runtime_avx512dq = false;
  bool runtime_avx512bw = false;
  bool compile_avx2 = false;
  bool compile_avx512f = false;
  bool compile_fma = false;

  [[nodiscard]] std::string to_string() const;
};

[[nodiscard]] CpuFeatures detect_cpu_features();

}  // namespace powder::core
