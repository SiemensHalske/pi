#pragma once

#include "powder/sim/WorldState.hpp"

#include <string>

namespace powder::sim {

void save_checkpoint_binary(const std::string& file_path, const WorldState& world);
[[nodiscard]] WorldState load_checkpoint_binary(const std::string& file_path);

}  // namespace powder::sim
