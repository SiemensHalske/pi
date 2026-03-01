#pragma once

#include "powder/core/Field2D.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace powder::sim {

struct SpeciesFields {
  powder::core::Field2D<float> o2;
  powder::core::Field2D<float> fuel_vapor;
  powder::core::Field2D<float> co2;
  powder::core::Field2D<float> h2o;
  powder::core::Field2D<float> soot;

  void resize(std::size_t width, std::size_t height, std::size_t ghost);
};

struct StressFields {
  powder::core::Field2D<float> sigma_xx;
  powder::core::Field2D<float> sigma_yy;
  powder::core::Field2D<float> tau_xy;
  powder::core::Field2D<float> plastic_strain;
  powder::core::Field2D<float> damage;

  void resize(std::size_t width, std::size_t height, std::size_t ghost);
};

struct DebrisSoA {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> vx;
  std::vector<float> vy;
  std::vector<float> radius;
  std::vector<float> mass;
  std::vector<std::uint32_t> material_id;

  [[nodiscard]] std::size_t size() const noexcept;
};

struct WorldState {
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t ghost = 0;

  powder::core::Field2D<float> pressure;
  powder::core::Field2D<float> temperature;
  powder::core::Field2D<float> enthalpy;
  powder::core::Field2D<float> density;
  powder::core::Field2D<float> phase_fraction;

  powder::core::FaceFieldU velocity_u;
  powder::core::FaceFieldV velocity_v;

  SpeciesFields species;
  StressFields stress;
  DebrisSoA debris;

  void resize(std::size_t new_width, std::size_t new_height, std::size_t ghost_cells = 2);
};

[[nodiscard]] WorldState create_world_state(std::size_t width, std::size_t height, std::size_t ghost_cells = 2);

}  // namespace powder::sim
