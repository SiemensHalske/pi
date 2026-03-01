#include "powder/core/Field2D.hpp"
#include "powder/sim/MaterialDB.hpp"
#include "powder/sim/ThermoChemStruct.hpp"
#include "powder/sim/WorldState.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

bool test_phase6_full_step() {
  auto world = powder::sim::create_world_state(18, 14, 2);
  powder::core::Field2D<std::uint32_t> material_id(world.width, world.height, world.ghost);
  powder::core::Field2D<float> mush_viscosity(world.width, world.height, world.ghost);

  for (std::size_t y = 0; y < world.height; ++y) {
    for (std::size_t x = 0; x < world.width; ++x) {
      std::uint32_t mid = 0U;
      if (x > 3U && x < 12U && y > 4U) {
        mid = static_cast<std::uint32_t>(powder::sim::MaterialId::Wood);
      }
      if (x >= 12U && y >= 6U) {
        mid = static_cast<std::uint32_t>(powder::sim::MaterialId::Steel);
      }
      if (x < 3U && y < 5U) {
        mid = static_cast<std::uint32_t>(powder::sim::MaterialId::Water);
      }
      material_id.at(x, y) = mid;

      world.temperature.at(x, y) = 320.0F + 5.0F * static_cast<float>(x) + 3.0F * static_cast<float>(y);
      world.density.at(x, y) = (mid == static_cast<std::uint32_t>(powder::sim::MaterialId::Water)) ? 980.0F : 750.0F;
      world.enthalpy.at(x, y) = world.temperature.at(x, y) * world.density.at(x, y);
      world.phase_fraction.at(x, y) = 0.0F;

      world.species.o2.at(x, y) = 0.24F;
      world.species.fuel_vapor.at(x, y) = (mid == static_cast<std::uint32_t>(powder::sim::MaterialId::Wood)) ? 0.05F : 0.002F;
      world.species.co2.at(x, y) = 0.0F;
      world.species.h2o.at(x, y) = 0.01F;
      world.species.soot.at(x, y) = 0.0F;

      world.stress.sigma_xx.at(x, y) = 0.0F;
      world.stress.sigma_yy.at(x, y) = 0.0F;
      world.stress.tau_xy.at(x, y) = 0.0F;
      world.stress.plastic_strain.at(x, y) = 0.0F;
      world.stress.damage.at(x, y) = 0.0F;
    }
  }

  for (std::size_t y = 0; y < world.velocity_u.value.height(); ++y) {
    for (std::size_t x = 0; x < world.velocity_u.value.width(); ++x) {
      world.velocity_u.value.at(x, y) = 0.25F * static_cast<float>(y) / static_cast<float>(world.velocity_u.value.height());
    }
  }
  for (std::size_t y = 0; y < world.velocity_v.value.height(); ++y) {
    for (std::size_t x = 0; x < world.velocity_v.value.width(); ++x) {
      world.velocity_v.value.at(x, y) = -0.15F * static_cast<float>(x) / static_cast<float>(world.velocity_v.value.width());
    }
  }

  powder::sim::ThermoChemConfig cfg{};
  cfg.dt = 1.0F / 180.0F;
  cfg.dx = 1.0F;
  cfg.dy = 1.0F;
  cfg.heat_max_iters = 32U;
  cfg.heat_tolerance = 1.0e-5F;
  cfg.melt_start_temperature = 900.0F;
  cfg.melt_end_temperature = 1400.0F;
  cfg.boil_temperature = 373.15F;
  cfg.leidenfrost_temperature = 500.0F;
  cfg.enable_hyperelastic = true;
  cfg.stoich_o2_per_fuel = 3.0F;
  cfg.pyrolysis_temperature = 500.0F;
  cfg.pyrolysis_rate = 0.06F;

  powder::sim::ThermoChemStats stats{};
  powder::sim::phase6_step(world, material_id, mush_viscosity, cfg, stats, 1);

  if (stats.heat_iterations == 0U || !std::isfinite(stats.heat_residual)) {
    return false;
  }
  if (!std::isfinite(stats.total_mass_before) || !std::isfinite(stats.total_mass_after) ||
      !std::isfinite(stats.total_energy_before) || !std::isfinite(stats.total_energy_after)) {
    return false;
  }

  bool has_structural_signal = false;
  bool has_chem_signal = (stats.reacted_fuel_mass > 0.0F) || (stats.generated_soot_mass > 0.0F);

  for (std::size_t y = 0; y < world.height; ++y) {
    for (std::size_t x = 0; x < world.width; ++x) {
      const float phi = world.phase_fraction.at(x, y);
      const float mu_mush = mush_viscosity.at(x, y);
      if (!(phi >= 0.0F && phi <= 1.0F)) {
        return false;
      }
      if (!(mu_mush >= cfg.mush_viscosity_min && mu_mush <= cfg.mush_viscosity_max + 1.0F)) {
        return false;
      }

      const float o2 = world.species.o2.at(x, y);
      const float fuel = world.species.fuel_vapor.at(x, y);
      const float co2 = world.species.co2.at(x, y);
      const float h2o = world.species.h2o.at(x, y);
      const float soot = world.species.soot.at(x, y);
      if (o2 < -1.0e-6F || fuel < -1.0e-6F || co2 < -1.0e-6F || h2o < -1.0e-6F || soot < -1.0e-6F) {
        return false;
      }

      const float sxx = world.stress.sigma_xx.at(x, y);
      const float syy = world.stress.sigma_yy.at(x, y);
      const float txy = world.stress.tau_xy.at(x, y);
      if (std::fabs(sxx) + std::fabs(syy) + std::fabs(txy) > 1.0e-4F) {
        has_structural_signal = true;
      }
    }
  }

  if (!has_structural_signal) {
    return false;
  }

  return has_chem_signal || (stats.vaporized_mass >= 0.0F && stats.condensed_mass >= 0.0F);
}

bool test_unified_reaction_api() {
  const powder::sim::SpeciesState s{
      0.5F,
      0.2F,
      0.0F,
      0.0F,
      0.0F,
  };
  powder::sim::ThermoChemConfig cfg{};
  cfg.dt = 1.0e-3F;
  cfg.stoich_o2_per_fuel = 2.8F;
  cfg.pyrolysis_temperature = 550.0F;
  cfg.pyrolysis_rate = 0.2F;

  const auto src = powder::sim::compute_reaction_sources(static_cast<std::uint32_t>(powder::sim::MaterialId::Wood),
                                                          900.0F,
                                                          0.2F,
                                                          s,
                                                          cfg,
                                                          cfg.dt,
                                                          2.0F);

  if (src.d_o2 >= 0.0F || src.d_co2 < 0.0F || src.d_h2o < 0.0F || src.d_temperature < 0.0F) {
    return false;
  }
  if (src.d_fuel_vapor <= -1.0F || src.d_fuel_vapor >= 1.0F) {
    return false;
  }
  if (src.d_soot < 0.0F || src.d_electrolyte < 0.0F) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!test_phase6_full_step()) {
    std::cerr << "phase6 full step failed\n";
    return 1;
  }
  if (!test_unified_reaction_api()) {
    std::cerr << "phase6 reaction api failed\n";
    return 1;
  }
  return 0;
}
