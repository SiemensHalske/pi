#pragma once

#include "powder/core/Field2D.hpp"
#include "powder/sim/WorldState.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace powder::sim {

struct ThermoChemConfig {
  float dt = 1.0F / 120.0F;
  float dx = 1.0F;
  float dy = 1.0F;

  std::size_t heat_max_iters = 20;
  float heat_tolerance = 1.0e-4F;

  float latent_heat = 2.2e5F;
  float melt_start_temperature = 1450.0F;
  float melt_end_temperature = 1600.0F;
  float mush_viscosity_min = 1.0F;
  float mush_viscosity_max = 1.0e6F;

  float boil_temperature = 373.15F;
  float leidenfrost_temperature = 573.15F;
  float htc_wet = 2200.0F;
  float htc_film = 220.0F;
  float vaporization_enthalpy = 2.256e6F;
  float condensation_enthalpy = 2.256e6F;
  float ambient_temperature = 300.0F;

  float structural_mu = 4.0e6F;
  float structural_lambda = 8.0e6F;
  float hardening_modulus = 4.0e5F;
  float damage_rate = 0.08F;
  float max_damage = 0.999F;

  bool enable_hyperelastic = true;
  float hyperelastic_temp_softening = 8.0e-4F;
  float reference_temperature = 300.0F;

  std::size_t chemistry_newton_iters = 6;
  float chemistry_newton_tol = 1.0e-6F;
  float reaction_heat_release = 4.4e7F;
  float edc_c_gamma = 2.13F;
  float edc_tau_floor = 1.0e-4F;
  float stoich_o2_per_fuel = 3.5F;

  float pyrolysis_temperature = 640.0F;
  float pyrolysis_rate = 0.08F;
  float soot_yield = 0.04F;
  float electrolyte_release = 0.005F;
};

struct ThermoChemStats {
  std::size_t heat_iterations = 0;
  bool heat_converged = false;
  float heat_residual = 0.0F;

  float total_mass_before = 0.0F;
  float total_mass_after = 0.0F;
  float total_energy_before = 0.0F;
  float total_energy_after = 0.0F;

  float vaporized_mass = 0.0F;
  float condensed_mass = 0.0F;
  float reacted_fuel_mass = 0.0F;
  float generated_soot_mass = 0.0F;
  float released_electrolyte = 0.0F;
};

struct SpeciesState {
  float o2 = 0.0F;
  float fuel_vapor = 0.0F;
  float co2 = 0.0F;
  float h2o = 0.0F;
  float soot = 0.0F;
};

struct ReactionPathway {
  float arrhenius_pre_exp = 0.0F;
  float arrhenius_activation_energy = 0.0F;
  float stoich_o2_per_fuel = 3.5F;
  float product_co2_yield = 2.75F;
  float product_h2o_yield = 1.80F;
  float soot_yield = 0.02F;
  float pyrolysis_temp = 640.0F;
  float pyrolysis_rate = 0.0F;
  float electrolyte_yield = 0.0F;
};

struct ReactionSources {
  float d_o2 = 0.0F;
  float d_fuel_vapor = 0.0F;
  float d_co2 = 0.0F;
  float d_h2o = 0.0F;
  float d_soot = 0.0F;
  float d_temperature = 0.0F;
  float d_electrolyte = 0.0F;
};

[[nodiscard]] ReactionPathway reaction_pathway_for_material(std::uint32_t material_id);

[[nodiscard]] ReactionSources compute_reaction_sources(std::uint32_t material_id,
                                                       float temperature,
                                                       float phase_fraction,
                                                       const SpeciesState& species,
                                                       const ThermoChemConfig& cfg,
                                                       float dt,
                                                       float turbulence_metric);

void solve_heat_equation_crank_nicolson(WorldState& world,
                                        const powder::core::Field2D<std::uint32_t>& material_id,
                                        const ThermoChemConfig& cfg,
                                        ThermoChemStats& stats,
                                        int thread_count);

void update_enthalpy_porosity(WorldState& world,
                              const powder::core::Field2D<std::uint32_t>& material_id,
                              powder::core::Field2D<float>& mush_viscosity,
                              const ThermoChemConfig& cfg,
                              int thread_count);

void apply_phase_change_and_leidenfrost(WorldState& world,
                                        const powder::core::Field2D<std::uint32_t>& material_id,
                                        const ThermoChemConfig& cfg,
                                        ThermoChemStats& stats,
                                        int thread_count);

void solve_elastoplastic_structure(WorldState& world,
                                   const powder::core::Field2D<std::uint32_t>& material_id,
                                   const ThermoChemConfig& cfg,
                                   int thread_count);

void apply_neo_hookean_thermomech(WorldState& world,
                                  const powder::core::Field2D<std::uint32_t>& material_id,
                                  const ThermoChemConfig& cfg,
                                  int thread_count);

void solve_stiff_chemistry(WorldState& world,
                           const powder::core::Field2D<std::uint32_t>& material_id,
                           const ThermoChemConfig& cfg,
                           ThermoChemStats& stats,
                           int thread_count);

void phase6_step(WorldState& world,
                 const powder::core::Field2D<std::uint32_t>& material_id,
                 powder::core::Field2D<float>& mush_viscosity,
                 const ThermoChemConfig& cfg,
                 ThermoChemStats& stats,
                 int thread_count);

void phase6_link_anchor();

}  // namespace powder::sim
