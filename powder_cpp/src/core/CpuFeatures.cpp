#include "powder/core/CpuFeatures.hpp"

#include <sstream>

namespace powder::core {

CpuFeatures detect_cpu_features() {
  CpuFeatures features{};
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  features.arch_x86 = true;
#endif

#if defined(__AVX2__)
  features.compile_avx2 = true;
#endif

#if defined(__AVX512F__)
  features.compile_avx512f = true;
#endif

#if defined(__FMA__)
  features.compile_fma = true;
#endif

  if (features.arch_x86) {
    #if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    features.runtime_avx2 = __builtin_cpu_supports("avx2") != 0;
    features.runtime_avx512f = __builtin_cpu_supports("avx512f") != 0;
    features.runtime_avx512dq = __builtin_cpu_supports("avx512dq") != 0;
    features.runtime_avx512bw = __builtin_cpu_supports("avx512bw") != 0;
    #endif
  }

  return features;
}

std::string CpuFeatures::to_string() const {
  std::ostringstream out;
  out << "arch_x86=" << (arch_x86 ? "yes" : "no")
      << " runtime_avx2=" << (runtime_avx2 ? "yes" : "no")
      << " runtime_avx512f=" << (runtime_avx512f ? "yes" : "no")
      << " runtime_avx512dq=" << (runtime_avx512dq ? "yes" : "no")
      << " runtime_avx512bw=" << (runtime_avx512bw ? "yes" : "no")
      << " compile_avx2=" << (compile_avx2 ? "yes" : "no")
      << " compile_avx512f=" << (compile_avx512f ? "yes" : "no")
      << " compile_fma=" << (compile_fma ? "yes" : "no");
  return out.str();
}

}  // namespace powder::core
