#include "powder/sim/ThermoChemStruct.hpp"

#include "powder/sim/MaterialDB.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#include <omp.h>
#endif

namespace powder::sim {
namespace {

constexpr float kGasConstant = 8.314462618F;

[[nodiscard]] inline float clamp01(float v) {
  return std::clamp(v, 0.0F, 1.0F);
}

[[nodiscard]] inline std::uint32_t clamp_material_id(std::uint32_t id) {
  const auto max_id = static_cast<std::uint32_t>(MaterialId::Count) - 1U;
  return std::min(id, max_id);
}

[[nodiscard]] inline const MaterialProperties& mat_props(std::uint32_t material_id) {
  return material_properties(static_cast<MaterialId>(clamp_material_id(material_id)));
}

[[nodiscard]] inline float safe_cp(const MaterialProperties& m) {
  return std::max(m.heat_capacity, 1.0F);
}

[[nodiscard]] inline float safe_density(float rho, const MaterialProperties& m) {
  return std::max(rho, std::max(1.0e-3F, 0.01F * m.density_ref));
}

[[nodiscard]] inline float local_turbulence_metric(const WorldState& world, std::size_t x, std::size_t y,
                                                   float dx, float dy) {
  const auto x_l = (x > 0U) ? (x - 1U) : x;
  const auto x_r = (x + 1U < world.width) ? (x + 1U) : x;
  const auto y_b = (y > 0U) ? (y - 1U) : y;
  const auto y_t = (y + 1U < world.height) ? (y + 1U) : y;

  const float uc_t = 0.5F * (world.velocity_u.value.at(x, y_t) + world.velocity_u.value.at(x + 1U, y_t));
  const float uc_b = 0.5F * (world.velocity_u.value.at(x, y_b) + world.velocity_u.value.at(x + 1U, y_b));
  const float vc_r = 0.5F * (world.velocity_v.value.at(x_r, y) + world.velocity_v.value.at(x_r, y + 1U));
  const float vc_l = 0.5F * (world.velocity_v.value.at(x_l, y) + world.velocity_v.value.at(x_l, y + 1U));

  const float dudx = (world.velocity_u.value.at(x + 1U, y) - world.velocity_u.value.at(x, y)) / dx;
  const float dvdy = (world.velocity_v.value.at(x, y + 1U) - world.velocity_v.value.at(x, y)) / dy;
  const float dudy = (uc_t - uc_b) / std::max((static_cast<float>(y_t) - static_cast<float>(y_b)) * dy, dy);
  const float dvdx = (vc_r - vc_l) / std::max((static_cast<float>(x_r) - static_cast<float>(x_l)) * dx, dx);

  const float sxx = dudx;
  const float syy = dvdy;
  const float sxy = 0.5F * (dudy + dvdx);
  const float vort = dvdx - dudy;
  return std::sqrt(std::max(0.0F, sxx * sxx + syy * syy + 2.0F * sxy * sxy + 0.25F * vort * vort));
}

void compute_total_mass_energy(const WorldState& world, float& total_mass, float& total_energy) {
  total_mass = 0.0F;
  total_energy = 0.0F;
  for (std::size_t y = 0; y < world.height; ++y) {
    for (std::size_t x = 0; x < world.width; ++x) {
      const float rho = std::max(0.0F, world.density.at(x, y));
      const float species_mass = std::max(0.0F, world.species.o2.at(x, y)) + std::max(0.0F, world.species.fuel_vapor.at(x, y)) +
                                 std::max(0.0F, world.species.co2.at(x, y)) + std::max(0.0F, world.species.h2o.at(x, y)) +
                                 std::max(0.0F, world.species.soot.at(x, y));
      total_mass += rho + species_mass;
      total_energy += world.enthalpy.at(x, y);
    }
  }
}

}  // namespace

ReactionPathway reaction_pathway_for_material(std::uint32_t material_id) {
  const auto id = static_cast<MaterialId>(clamp_material_id(material_id));
  const auto& m = material_properties(id);

  switch (id) {
    case MaterialId::Wood:
      return ReactionPathway{m.arrhenius_pre_exp, m.arrhenius_activation_energy, 1.5F, 1.6F, 0.7F, 0.06F, 560.0F, 0.11F, 0.020F};
    case MaterialId::Lava:
      return ReactionPathway{m.arrhenius_pre_exp, m.arrhenius_activation_energy, 0.9F, 0.2F, 0.1F, 0.02F, 900.0F, 0.02F, 0.001F};
    case MaterialId::Water:
      return ReactionPathway{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, std::numeric_limits<float>::max(), 0.0F, 0.0F};
    case MaterialId::Steel:
      return ReactionPathway{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1800.0F, 0.0F, 0.0F};
    case MaterialId::Sand:
      return ReactionPathway{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1400.0F, 0.0F, 0.0F};
    case MaterialId::Air:
    default:
      return ReactionPathway{m.arrhenius_pre_exp, m.arrhenius_activation_energy, 3.5F, 2.75F, 1.80F, 0.015F, 700.0F, 0.0F, 0.0F};
  }
}

ReactionSources compute_reaction_sources(std::uint32_t material_id,
                                         float temperature,
                                         float phase_fraction,
                                         const SpeciesState& species,
                                         const ThermoChemConfig& cfg,
                                         float dt,
                                         float turbulence_metric) {
  const auto pathway = reaction_pathway_for_material(material_id);
  ReactionSources src{};

  if (pathway.arrhenius_pre_exp > 0.0F && pathway.arrhenius_activation_energy > 0.0F && temperature > 1.0F) {
    const float k = pathway.arrhenius_pre_exp * std::exp(-pathway.arrhenius_activation_energy / (kGasConstant * temperature));
    const float tau_chem = 1.0F / std::max(k, 1.0e-12F);
    const float gamma_turb = cfg.edc_c_gamma * turbulence_metric / (turbulence_metric + 1.0F);
    const float limiter = clamp01(gamma_turb * dt / (tau_chem + cfg.edc_tau_floor));
    const float dt_eff = dt * limiter;

    const float stoich = std::max(pathway.stoich_o2_per_fuel, 1.0e-4F);
    const float fuel = std::max(species.fuel_vapor, 0.0F);
    const float o2 = std::max(species.o2, 0.0F);
    const float max_xi = std::max(0.0F, std::min(fuel, o2 / stoich));

    float xi = std::min(0.25F * max_xi, max_xi);
    for (std::size_t it = 0; it < cfg.chemistry_newton_iters; ++it) {
      const float fuel_r = std::max(0.0F, fuel - xi);
      const float o2_r = std::max(0.0F, o2 - stoich * xi);
      const float f = xi - dt_eff * k * fuel_r * o2_r;
      const float df = 1.0F + dt_eff * k * (o2_r + stoich * fuel_r);
      const float dxi = f / std::max(df, 1.0e-6F);
      xi = std::clamp(xi - dxi, 0.0F, max_xi);
      if (std::fabs(dxi) < cfg.chemistry_newton_tol) {
        break;
      }
    }

    src.d_fuel_vapor -= xi;
    src.d_o2 -= stoich * xi;
    src.d_co2 += pathway.product_co2_yield * xi;
    src.d_h2o += pathway.product_h2o_yield * xi;
    src.d_soot += pathway.soot_yield * xi;
    src.d_temperature += cfg.reaction_heat_release * xi;
  }

  if (temperature > pathway.pyrolysis_temp && pathway.pyrolysis_rate > 0.0F) {
    const float over = (temperature - pathway.pyrolysis_temp) / std::max(pathway.pyrolysis_temp, 1.0F);
    const float pyro = pathway.pyrolysis_rate * dt * std::max(0.0F, over) * (1.0F - clamp01(phase_fraction));
    src.d_fuel_vapor += pyro;
    src.d_soot += pathway.soot_yield * pyro;
    src.d_electrolyte += pathway.electrolyte_yield * pyro;
  }

  return src;
}

void solve_heat_equation_crank_nicolson(WorldState& world,
                                        const powder::core::Field2D<std::uint32_t>& material_id,
                                        const ThermoChemConfig& cfg,
                                        ThermoChemStats& stats,
                                        int thread_count) {
  powder::core::Field2D<float> old_t(world.width, world.height, world.ghost);
  for (std::size_t i = 0; i < old_t.size(); ++i) {
    old_t.raw()[i] = world.temperature.raw()[i];
  }

  const float inv_dx2 = 1.0F / (cfg.dx * cfg.dx);
  const float inv_dy2 = 1.0F / (cfg.dy * cfg.dy);

  stats.heat_iterations = 0;
  stats.heat_converged = false;
  stats.heat_residual = std::numeric_limits<float>::infinity();

  for (std::size_t iter = 0; iter < cfg.heat_max_iters; ++iter) {
    float max_delta = 0.0F;

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
    if (thread_count > 0) {
      omp_set_num_threads(thread_count);
    }
#pragma omp parallel for collapse(2) schedule(static) reduction(max : max_delta)
#endif
    for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(world.height); ++y) {
      for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(world.width); ++x) {
        const auto xi = static_cast<std::size_t>(x);
        const auto yi = static_cast<std::size_t>(y);

        const auto x_l = (xi > 0U) ? (xi - 1U) : xi;
        const auto x_r = (xi + 1U < world.width) ? (xi + 1U) : xi;
        const auto y_b = (yi > 0U) ? (yi - 1U) : yi;
        const auto y_t = (yi + 1U < world.height) ? (yi + 1U) : yi;

        const auto& m = mat_props(material_id.at(xi, yi));
        const float rho = safe_density(world.density.at(xi, yi), m);
        const float cp = safe_cp(m);
        const float diffusivity = std::max(0.0F, m.thermal_conductivity) / (rho * cp);
        const float alpha = 0.5F * cfg.dt * diffusivity;

        const float t_old = old_t.at(xi, yi);
        const float lap_old = (old_t.at(x_l, yi) - 2.0F * t_old + old_t.at(x_r, yi)) * inv_dx2 +
                              (old_t.at(xi, y_b) - 2.0F * t_old + old_t.at(xi, y_t)) * inv_dy2;
        const float rhs = t_old + alpha * lap_old;

        const float neighbor_sum = (world.temperature.at(x_l, yi) + world.temperature.at(x_r, yi)) * inv_dx2 +
                                   (world.temperature.at(xi, y_b) + world.temperature.at(xi, y_t)) * inv_dy2;
        const float denom = 1.0F + 2.0F * alpha * (inv_dx2 + inv_dy2);
        const float t_new = (rhs + alpha * neighbor_sum) / std::max(denom, 1.0e-6F);

        max_delta = std::max(max_delta, std::fabs(t_new - world.temperature.at(xi, yi)));
        world.temperature.at(xi, yi) = t_new;
      }
    }

    stats.heat_iterations = iter + 1U;
    stats.heat_residual = max_delta;
    if (max_delta <= cfg.heat_tolerance) {
      stats.heat_converged = true;
      break;
    }
  }
}

void update_enthalpy_porosity(WorldState& world,
                              const powder::core::Field2D<std::uint32_t>& material_id,
                              powder::core::Field2D<float>& mush_viscosity,
                              const ThermoChemConfig& cfg,
                              int thread_count) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(world.height); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(world.width); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const auto& m = mat_props(material_id.at(xi, yi));
      const float cp = safe_cp(m);
      const float rho = safe_density(world.density.at(xi, yi), m);
      const float t = world.temperature.at(xi, yi);

      const float melt_span = std::max(cfg.melt_end_temperature - cfg.melt_start_temperature, 1.0e-3F);
      const float melt = clamp01((t - cfg.melt_start_temperature) / melt_span);
      world.phase_fraction.at(xi, yi) = melt;
      world.enthalpy.at(xi, yi) = rho * (cp * t + cfg.latent_heat * melt);

      const float one_minus = 1.0F - melt;
      const float denom = melt * melt * melt + 1.0e-6F;
      const float pen = (one_minus * one_minus) / denom;
      const float pen_clamped = pen / (1.0F + pen);
      mush_viscosity.at(xi, yi) = cfg.mush_viscosity_min + (cfg.mush_viscosity_max - cfg.mush_viscosity_min) * pen_clamped;
    }
  }
}

void apply_phase_change_and_leidenfrost(WorldState& world,
                                        const powder::core::Field2D<std::uint32_t>& material_id,
                                        const ThermoChemConfig& cfg,
                                        ThermoChemStats& stats,
                                        int thread_count) {
  float vaporized_mass = 0.0F;
  float condensed_mass = 0.0F;

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : vaporized_mass, condensed_mass)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(world.height); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(world.width); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const auto& m = mat_props(material_id.at(xi, yi));

      float t = world.temperature.at(xi, yi);
      float rho = safe_density(world.density.at(xi, yi), m);
      const float cp = safe_cp(m);

      if (t > cfg.boil_temperature) {
        const bool leidenfrost = (t >= cfg.leidenfrost_temperature);
        const float htc = leidenfrost ? cfg.htc_film : cfg.htc_wet;
        const float q = htc * (t - cfg.boil_temperature) * cfg.dt;
        float evap = q / std::max(cfg.vaporization_enthalpy, 1.0e-3F);
        evap = std::min(evap, 0.1F * rho);
        evap = std::max(evap, 0.0F);

        rho = std::max(1.0e-4F, rho - evap);
        world.species.h2o.at(xi, yi) = std::max(0.0F, world.species.h2o.at(xi, yi) + evap);
        t -= (evap * cfg.vaporization_enthalpy) / std::max(rho * cp, 1.0e-3F);
        vaporized_mass += evap;
      } else {
        const float vapor = std::max(0.0F, world.species.h2o.at(xi, yi));
        if (vapor > 0.0F && t < cfg.boil_temperature) {
          const float q = cfg.htc_wet * (cfg.boil_temperature - t) * cfg.dt;
          float cond = 0.1F * q / std::max(cfg.condensation_enthalpy, 1.0e-3F);
          cond = std::min(cond, vapor);
          cond = std::max(cond, 0.0F);

          world.species.h2o.at(xi, yi) = vapor - cond;
          rho += cond;
          t += (cond * cfg.condensation_enthalpy) / std::max(rho * cp, 1.0e-3F);
          condensed_mass += cond;
        }
      }

      world.density.at(xi, yi) = rho;
      world.temperature.at(xi, yi) = std::max(1.0F, t);
    }
  }

  stats.vaporized_mass += vaporized_mass;
  stats.condensed_mass += condensed_mass;
}

void solve_elastoplastic_structure(WorldState& world,
                                   const powder::core::Field2D<std::uint32_t>& material_id,
                                   const ThermoChemConfig& cfg,
                                   int thread_count) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(world.height); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(world.width); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const auto& m = mat_props(material_id.at(xi, yi));
      const float yield_limit = std::max(0.0F, m.yield_limit);

      if (yield_limit <= 0.0F) {
        world.stress.sigma_xx.at(xi, yi) = 0.0F;
        world.stress.sigma_yy.at(xi, yi) = 0.0F;
        world.stress.tau_xy.at(xi, yi) = 0.0F;
        continue;
      }

      const auto x_l = (xi > 0U) ? (xi - 1U) : xi;
      const auto x_r = (xi + 1U < world.width) ? (xi + 1U) : xi;
      const auto y_b = (yi > 0U) ? (yi - 1U) : yi;
      const auto y_t = (yi + 1U < world.height) ? (yi + 1U) : yi;

      const float uc_t = 0.5F * (world.velocity_u.value.at(xi, y_t) + world.velocity_u.value.at(xi + 1U, y_t));
      const float uc_b = 0.5F * (world.velocity_u.value.at(xi, y_b) + world.velocity_u.value.at(xi + 1U, y_b));
      const float vc_r = 0.5F * (world.velocity_v.value.at(x_r, yi) + world.velocity_v.value.at(x_r, yi + 1U));
      const float vc_l = 0.5F * (world.velocity_v.value.at(x_l, yi) + world.velocity_v.value.at(x_l, yi + 1U));

      const float dudx = (world.velocity_u.value.at(xi + 1U, yi) - world.velocity_u.value.at(xi, yi)) / cfg.dx;
      const float dvdy = (world.velocity_v.value.at(xi, yi + 1U) - world.velocity_v.value.at(xi, yi)) / cfg.dy;
      const float dudy = (uc_t - uc_b) / std::max((static_cast<float>(y_t) - static_cast<float>(y_b)) * cfg.dy, cfg.dy);
      const float dvdx = (vc_r - vc_l) / std::max((static_cast<float>(x_r) - static_cast<float>(x_l)) * cfg.dx, cfg.dx);

      const float exx = dudx;
      const float eyy = dvdy;
      const float exy = 0.5F * (dudy + dvdx);
      const float tr_e = exx + eyy;

      float sxx = world.stress.sigma_xx.at(xi, yi) + cfg.dt * (2.0F * cfg.structural_mu * exx + cfg.structural_lambda * tr_e);
      float syy = world.stress.sigma_yy.at(xi, yi) + cfg.dt * (2.0F * cfg.structural_mu * eyy + cfg.structural_lambda * tr_e);
      float txy = world.stress.tau_xy.at(xi, yi) + cfg.dt * (2.0F * cfg.structural_mu * exy);

      const float vm = std::sqrt(std::max(0.0F, sxx * sxx - sxx * syy + syy * syy + 3.0F * txy * txy));
      const float p_strain = std::max(0.0F, world.stress.plastic_strain.at(xi, yi));
      const float y_eff = yield_limit + cfg.hardening_modulus * p_strain;

      if (vm > y_eff && vm > 1.0e-9F) {
        const float mean = 0.5F * (sxx + syy);
        const float scale = y_eff / vm;
        const float dev_x = sxx - mean;
        const float dev_y = syy - mean;
        sxx = mean + scale * dev_x;
        syy = mean + scale * dev_y;
        txy *= scale;

        const float dgamma = (vm - y_eff) / std::max(3.0F * cfg.structural_mu + cfg.hardening_modulus, 1.0e-6F);
        world.stress.plastic_strain.at(xi, yi) = p_strain + std::max(0.0F, dgamma);

        const float dmg = world.stress.damage.at(xi, yi) + cfg.dt * cfg.damage_rate * std::max(0.0F, vm / std::max(yield_limit, 1.0e-6F) - 1.0F);
        world.stress.damage.at(xi, yi) = std::clamp(dmg, 0.0F, cfg.max_damage);
      }

      const float health = 1.0F - world.stress.damage.at(xi, yi);
      world.stress.sigma_xx.at(xi, yi) = health * sxx;
      world.stress.sigma_yy.at(xi, yi) = health * syy;
      world.stress.tau_xy.at(xi, yi) = health * txy;
    }
  }
}

void apply_neo_hookean_thermomech(WorldState& world,
                                  const powder::core::Field2D<std::uint32_t>& material_id,
                                  const ThermoChemConfig& cfg,
                                  int thread_count) {
  if (!cfg.enable_hyperelastic) {
    return;
  }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(world.height); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(world.width); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const auto& m = mat_props(material_id.at(xi, yi));
      if (m.yield_limit <= 0.0F) {
        continue;
      }

      const auto x_l = (xi > 0U) ? (xi - 1U) : xi;
      const auto x_r = (xi + 1U < world.width) ? (xi + 1U) : xi;
      const auto y_b = (yi > 0U) ? (yi - 1U) : yi;
      const auto y_t = (yi + 1U < world.height) ? (yi + 1U) : yi;

      const float uc_t = 0.5F * (world.velocity_u.value.at(xi, y_t) + world.velocity_u.value.at(xi + 1U, y_t));
      const float uc_b = 0.5F * (world.velocity_u.value.at(xi, y_b) + world.velocity_u.value.at(xi + 1U, y_b));
      const float vc_r = 0.5F * (world.velocity_v.value.at(x_r, yi) + world.velocity_v.value.at(x_r, yi + 1U));
      const float vc_l = 0.5F * (world.velocity_v.value.at(x_l, yi) + world.velocity_v.value.at(x_l, yi + 1U));

      const float dudx = (world.velocity_u.value.at(xi + 1U, yi) - world.velocity_u.value.at(xi, yi)) / cfg.dx;
      const float dvdy = (world.velocity_v.value.at(xi, yi + 1U) - world.velocity_v.value.at(xi, yi)) / cfg.dy;
      const float dudy = (uc_t - uc_b) / std::max((static_cast<float>(y_t) - static_cast<float>(y_b)) * cfg.dy, cfg.dy);
      const float dvdx = (vc_r - vc_l) / std::max((static_cast<float>(x_r) - static_cast<float>(x_l)) * cfg.dx, cfg.dx);

      const float f11 = 1.0F + cfg.dt * dudx;
      const float f12 = cfg.dt * dudy;
      const float f21 = cfg.dt * dvdx;
      const float f22 = 1.0F + cfg.dt * dvdy;
      const float j = std::max(0.2F, f11 * f22 - f12 * f21);

      const float b11 = f11 * f11 + f12 * f12;
      const float b22 = f21 * f21 + f22 * f22;
      const float b12 = f11 * f21 + f12 * f22;
      const float jm23 = std::pow(j, -2.0F / 3.0F);

      const float bbar11 = b11 * jm23;
      const float bbar22 = b22 * jm23;
      const float bbar33 = 1.0F * jm23;
      const float tr_bbar = bbar11 + bbar22 + bbar33;

      const float dev11 = bbar11 - tr_bbar / 3.0F;
      const float dev22 = bbar22 - tr_bbar / 3.0F;
      const float dev12 = b12 * jm23;

      const float t = world.temperature.at(xi, yi);
      const float soften = std::exp(-cfg.hyperelastic_temp_softening * (t - cfg.reference_temperature));
      const float mu_t = cfg.structural_mu * soften;
      const float kappa_t = (cfg.structural_lambda + 2.0F * cfg.structural_mu / 3.0F) * soften;
      const float p = kappa_t * (j - 1.0F);

      const float sigma11 = -p + 2.0F * mu_t * dev11;
      const float sigma22 = -p + 2.0F * mu_t * dev22;
      const float sigma12 = 2.0F * mu_t * dev12;

      world.stress.sigma_xx.at(xi, yi) += 0.25F * sigma11;
      world.stress.sigma_yy.at(xi, yi) += 0.25F * sigma22;
      world.stress.tau_xy.at(xi, yi) += 0.25F * sigma12;
    }
  }
}

void solve_stiff_chemistry(WorldState& world,
                           const powder::core::Field2D<std::uint32_t>& material_id,
                           const ThermoChemConfig& cfg,
                           ThermoChemStats& stats,
                           int thread_count) {
  float reacted_fuel_mass = 0.0F;
  float generated_soot_mass = 0.0F;
  float released_electrolyte = 0.0F;

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : reacted_fuel_mass, generated_soot_mass, released_electrolyte)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(world.height); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(world.width); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const auto mid = material_id.at(xi, yi);
      const auto& m = mat_props(mid);

      const SpeciesState s{
          world.species.o2.at(xi, yi),
          world.species.fuel_vapor.at(xi, yi),
          world.species.co2.at(xi, yi),
          world.species.h2o.at(xi, yi),
          world.species.soot.at(xi, yi),
      };

      const float turb = local_turbulence_metric(world, xi, yi, cfg.dx, cfg.dy);
      const auto src = compute_reaction_sources(mid, world.temperature.at(xi, yi), world.phase_fraction.at(xi, yi), s, cfg, cfg.dt, turb);

      world.species.o2.at(xi, yi) = std::max(0.0F, s.o2 + src.d_o2);
      world.species.fuel_vapor.at(xi, yi) = std::max(0.0F, s.fuel_vapor + src.d_fuel_vapor);
      world.species.co2.at(xi, yi) = std::max(0.0F, s.co2 + src.d_co2);
      world.species.h2o.at(xi, yi) = std::max(0.0F, s.h2o + src.d_h2o);
      world.species.soot.at(xi, yi) = std::max(0.0F, s.soot + src.d_soot);

      const float cp = safe_cp(m);
      const float rho = safe_density(world.density.at(xi, yi), m);
      world.temperature.at(xi, yi) += src.d_temperature / std::max(rho * cp, 1.0e-3F);
      world.enthalpy.at(xi, yi) += src.d_temperature;

      reacted_fuel_mass += std::max(0.0F, -src.d_fuel_vapor);
      generated_soot_mass += std::max(0.0F, src.d_soot);
      released_electrolyte += std::max(0.0F, src.d_electrolyte);
    }
  }

  stats.reacted_fuel_mass += reacted_fuel_mass;
  stats.generated_soot_mass += generated_soot_mass;
  stats.released_electrolyte += released_electrolyte;
}

void phase6_step(WorldState& world,
                 const powder::core::Field2D<std::uint32_t>& material_id,
                 powder::core::Field2D<float>& mush_viscosity,
                 const ThermoChemConfig& cfg,
                 ThermoChemStats& stats,
                 int thread_count) {
  compute_total_mass_energy(world, stats.total_mass_before, stats.total_energy_before);

  solve_heat_equation_crank_nicolson(world, material_id, cfg, stats, thread_count);
  update_enthalpy_porosity(world, material_id, mush_viscosity, cfg, thread_count);
  apply_phase_change_and_leidenfrost(world, material_id, cfg, stats, thread_count);
  solve_elastoplastic_structure(world, material_id, cfg, thread_count);
  apply_neo_hookean_thermomech(world, material_id, cfg, thread_count);
  solve_stiff_chemistry(world, material_id, cfg, stats, thread_count);
  update_enthalpy_porosity(world, material_id, mush_viscosity, cfg, thread_count);

  compute_total_mass_energy(world, stats.total_mass_after, stats.total_energy_after);
}

void phase6_link_anchor() {}

}  // namespace powder::sim
