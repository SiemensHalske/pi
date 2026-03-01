#include "powder/core/CpuFeatures.hpp"
#include "powder/core/DeterministicBootstrap.hpp"

#include <iostream>

int main() {
  powder::core::BootstrapOptions options{};
  options.requested_threads = 1;
  const auto state = powder::core::build_deterministic_state(options);

  if (state.tick_rate_hz != 60) {
    std::cerr << "tick_rate_hz mismatch\n";
    return 1;
  }
  if (state.substep_order.size() != 9U) {
    std::cerr << "substep_order mismatch\n";
    return 1;
  }

  const auto features = powder::core::detect_cpu_features();
  (void)features;
  return 0;
}
