#include "powder/sim/WorldState.hpp"

namespace powder::sim {

void SpeciesFields::resize(std::size_t width, std::size_t height, std::size_t ghost) {
  o2.resize(width, height, ghost);
  fuel_vapor.resize(width, height, ghost);
  co2.resize(width, height, ghost);
  h2o.resize(width, height, ghost);
  soot.resize(width, height, ghost);
}

void StressFields::resize(std::size_t width, std::size_t height, std::size_t ghost) {
  sigma_xx.resize(width, height, ghost);
  sigma_yy.resize(width, height, ghost);
  tau_xy.resize(width, height, ghost);
  plastic_strain.resize(width, height, ghost);
  damage.resize(width, height, ghost);
}

std::size_t DebrisSoA::size() const noexcept {
  return x.size();
}

void WorldState::resize(std::size_t new_width, std::size_t new_height, std::size_t ghost_cells) {
  width = new_width;
  height = new_height;
  ghost = ghost_cells;

  pressure.resize(width, height, ghost);
  temperature.resize(width, height, ghost);
  enthalpy.resize(width, height, ghost);
  density.resize(width, height, ghost);
  phase_fraction.resize(width, height, ghost);

  velocity_u.resize(width, height, ghost);
  velocity_v.resize(width, height, ghost);

  species.resize(width, height, ghost);
  stress.resize(width, height, ghost);
}

WorldState create_world_state(std::size_t width, std::size_t height, std::size_t ghost_cells) {
  WorldState state{};
  state.resize(width, height, ghost_cells);
  return state;
}

}  // namespace powder::sim
