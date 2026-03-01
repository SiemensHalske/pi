#include "powder/core/DeterministicBootstrap.hpp"

#include <thread>

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#include <omp.h>
#endif

namespace powder::core {

DeterministicState build_deterministic_state(const BootstrapOptions& options) {
  DeterministicState state{};
  state.seed = 0xC0FFEE1234ULL;
  state.tick_rate_hz = 60;
  state.dt = 1.0 / static_cast<double>(state.tick_rate_hz);
  state.grid_width = options.profile == "benchmark" ? 1024 : 512;
  state.grid_height = options.profile == "benchmark" ? 576 : 288;
  state.substep_order = {
      "external_forces",
      "pressure_solve",
      "velocity_advect",
      "scalar_advect",
      "diffusion",
      "phase_change",
      "chemistry",
      "ca_fallback",
      "boundary_conditions",
  };

  int threads = options.requested_threads;
  if (threads <= 0) {
    const auto hc = std::thread::hardware_concurrency();
    threads = hc == 0U ? 1 : static_cast<int>(hc);
  }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  omp_set_num_threads(threads);
  state.active_threads = omp_get_max_threads();
#else
  state.active_threads = threads;
#endif

  return state;
}

}  // namespace powder::core
