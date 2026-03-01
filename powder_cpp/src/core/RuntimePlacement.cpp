#include "powder/core/RuntimePlacement.hpp"

#include <algorithm>
#include <cstddef>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__has_include)
#if __has_include(<numa.h>)
#define POWDERCPP_HAS_LIBNUMA_HEADER 1
#include <numa.h>
#else
#define POWDERCPP_HAS_LIBNUMA_HEADER 0
#endif
#else
#define POWDERCPP_HAS_LIBNUMA_HEADER 0
#endif

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#include <omp.h>
#endif

namespace powder::core {

bool set_current_thread_affinity_cpu(int cpu_index) {
#if defined(__linux__)
  if (cpu_index < 0) {
    return false;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu_index, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
  (void)cpu_index;
  return false;
#endif
}

void first_touch_zero(float* data, std::size_t count, int thread_count) {
  if (data == nullptr || count == 0) {
    return;
  }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
    data[static_cast<std::size_t>(i)] = 0.0F;
  }
#else
  (void)thread_count;
  std::fill(data, data + count, 0.0F);
#endif
}

bool numa_available_runtime() {
#if POWDERCPP_HAS_LIBNUMA_HEADER
  return numa_available() >= 0;
#else
  return false;
#endif
}

}  // namespace powder::core
