#include "powder/app/Cli.hpp"

#include "powder/core/BuildInfo.hpp"
#include "powder/core/CpuFeatures.hpp"
#include "powder/core/DeterministicBootstrap.hpp"

#include <iostream>

namespace powder::app {

int run(const CliOptions& options) {
  powder::core::BootstrapOptions bootstrap_options{};
  bootstrap_options.profile = options.profile;
  bootstrap_options.headless = options.headless;
  bootstrap_options.benchmark = options.benchmark;
  bootstrap_options.requested_threads = options.threads;

  const auto state = powder::core::build_deterministic_state(bootstrap_options);
  const auto cpu = powder::core::detect_cpu_features();
  const auto build = powder::core::query_build_info();

  std::cout << "PowderCPP bootstrap" << '\n';
  std::cout << " project=" << build.project_name << " build=" << build.build_type
            << " openmp=" << (build.with_openmp ? "on" : "off") << '\n';
  std::cout << " profile=" << bootstrap_options.profile
            << " headless=" << (bootstrap_options.headless ? "true" : "false")
            << " benchmark=" << (bootstrap_options.benchmark ? "true" : "false")
            << '\n';
  std::cout << " threads=" << state.active_threads
            << " dt=" << state.dt
            << " grid=" << state.grid_width << "x" << state.grid_height
            << " seed=" << state.seed << '\n';
  std::cout << " cpu_features=" << cpu.to_string() << '\n';
  std::cout << " substeps=";
  for (std::size_t i = 0; i < state.substep_order.size(); ++i) {
    std::cout << state.substep_order[i];
    if (i + 1 < state.substep_order.size()) {
      std::cout << "->";
    }
  }
  std::cout << '\n';

  return 0;
}

}  // namespace powder::app
