#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace powder::core {

struct BootstrapOptions {
  std::string profile = "deterministic";
  bool headless = false;
  bool benchmark = false;
  int requested_threads = 0;
};

struct DeterministicState {
  std::uint64_t seed = 0;
  int tick_rate_hz = 60;
  double dt = 1.0 / 60.0;
  int grid_width = 512;
  int grid_height = 288;
  int active_threads = 1;
  std::vector<std::string> substep_order;
};

[[nodiscard]] DeterministicState build_deterministic_state(const BootstrapOptions& options);

}  // namespace powder::core
