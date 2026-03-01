#pragma once

#include <cstddef>

namespace powder::core {

struct PlacementConfig {
  bool pin_threads = false;
  bool first_touch = true;
  int thread_count = 1;
};

[[nodiscard]] bool set_current_thread_affinity_cpu(int cpu_index);
void first_touch_zero(float* data, std::size_t count, int thread_count);
[[nodiscard]] bool numa_available_runtime();

}  // namespace powder::core
