#pragma once

#include "powder/sim/CAGrid.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace powder::sim {

struct GranularRuleConfig {
  std::uint16_t sand_material = 4;
  std::uint16_t powder_material = 4;
  std::uint16_t slurry_material = 4;
};

struct GranularStepInput {
  const std::vector<float>* local_vel_x = nullptr;
  const std::vector<float>* local_vel_y = nullptr;
  std::size_t tick_index = 0;
};

[[nodiscard]] std::size_t step_granular_deterministic(CAGrid& grid, const GranularRuleConfig& rules,
                                                      const GranularStepInput& input);

}  // namespace powder::sim
