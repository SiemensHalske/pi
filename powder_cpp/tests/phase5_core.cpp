#include "powder/core/Field2D.hpp"
#include "powder/sim/FluidPDE.hpp"
#include "powder/sim/WorldState.hpp"

#include <cmath>
#include <iostream>

namespace {

bool test_mac_divergence_and_projection() {
  auto world = powder::sim::create_world_state(16, 16, 2);

  for (std::size_t y = 0; y < world.height; ++y) {
    for (std::size_t x = 0; x < world.width + 1U; ++x) {
      world.velocity_u.value.at(x, y) = 0.0F;
    }
  }
  for (std::size_t y = 0; y < world.height + 1U; ++y) {
    for (std::size_t x = 0; x < world.width; ++x) {
      world.velocity_v.value.at(x, y) = 0.0F;
    }
  }

  powder::sim::FluidNumericsConfig cfg{};
  cfg.dt = 1.0F / 60.0F;
  cfg.dx = 1.0F;
  cfg.dy = 1.0F;
  cfg.rbgs_max_iters = 80U;
  cfg.residual_tolerance = 1.0e-4F;

  powder::core::Field2D<float> div(world.width, world.height, world.ghost);
  powder::core::Field2D<float> p(world.width, world.height, world.ghost);

  for (std::size_t y = 0; y < div.height(); ++y) {
    for (std::size_t x = 0; x < div.width(); ++x) {
      const float sx = static_cast<float>(x) / static_cast<float>(div.width());
      const float sy = static_cast<float>(y) / static_cast<float>(div.height());
      div.at(x, y) = std::sin(6.28318530718F * sx) * std::cos(6.28318530718F * sy);
    }
  }

  const float inv_dx2 = 1.0F / (cfg.dx * cfg.dx);
  const float inv_dy2 = 1.0F / (cfg.dy * cfg.dy);
  const float before = powder::sim::simd::residual_l2(p, div, inv_dx2, inv_dy2, 1);
  const auto rbgs_stats = powder::sim::solve_poisson_rbgs(p, div, cfg.dx, cfg.dy, cfg.rbgs_max_iters, cfg.residual_tolerance, 1);
  const float after = powder::sim::simd::residual_l2(p, div, inv_dx2, inv_dy2, 1);

  const auto proj_stats = powder::sim::project_incompressible(world, div, p, cfg, 1, false);
  return rbgs_stats.iterations > 0U && proj_stats.iterations > 0U && std::isfinite(after) && (after < before);
}

bool test_maccormack_limiter_bounded() {
  powder::core::Field2D<float> src(12, 12, 2);
  powder::core::FaceFieldU u;
  powder::core::FaceFieldV v;
  u.resize(12, 12, 2);
  v.resize(12, 12, 2);

  for (std::size_t y = 0; y < src.height(); ++y) {
    for (std::size_t x = 0; x < src.width(); ++x) {
      src.at(x, y) = (x > 5U && y > 5U) ? 1.0F : 0.0F;
    }
  }

  for (std::size_t y = 0; y < u.value.height(); ++y) {
    for (std::size_t x = 0; x < u.value.width(); ++x) {
      u.value.at(x, y) = 0.5F;
    }
  }
  for (std::size_t y = 0; y < v.value.height(); ++y) {
    for (std::size_t x = 0; x < v.value.width(); ++x) {
      v.value.at(x, y) = 0.0F;
    }
  }

  powder::core::Field2D<float> dst(12, 12, 2);
  powder::sim::maccormack_advect_scalar_limited(src, u, v, dst, 0.5F, 1.0F, 1.0F, 1);

  float min_v = 1.0e9F;
  float max_v = -1.0e9F;
  for (std::size_t y = 0; y < dst.height(); ++y) {
    for (std::size_t x = 0; x < dst.width(); ++x) {
      min_v = std::fmin(min_v, dst.at(x, y));
      max_v = std::fmax(max_v, dst.at(x, y));
    }
  }

  return min_v >= -1.0e-4F && max_v <= 1.0001F;
}

bool test_forces_and_solver_paths() {
  auto world = powder::sim::create_world_state(10, 10, 2);

  for (std::size_t y = 0; y < world.height; ++y) {
    for (std::size_t x = 0; x < world.width; ++x) {
      world.temperature.at(x, y) = 320.0F;
      world.density.at(x, y) = 1.0F;
    }
  }

  powder::sim::FluidNumericsConfig cfg{};
  cfg.dt = 0.01F;
  cfg.gravity_y = -9.81F;
  cfg.buoyancy_alpha = 0.2F;
  cfg.ambient_temperature = 300.0F;
  cfg.linear_drag = 0.1F;

  powder::sim::ExternalForces ext{};
  ext.impulse_x = 2.0F;
  ext.impulse_y = 1.0F;
  ext.impulse_cell_x = 5;
  ext.impulse_cell_y = 5;
  ext.impulse_radius = 2.5F;

  powder::sim::accumulate_face_forces(world.velocity_u, world.velocity_v, world.temperature, world.density,
                                      ext, world.velocity_u, world.velocity_v, cfg, 1);

  powder::core::Field2D<float> div(world.width, world.height, world.ghost);
  powder::core::Field2D<float> p(world.width, world.height, world.ghost);

  const auto rbgs_stats = powder::sim::project_incompressible(world, div, p, cfg, 1, false);
  const auto mg_stats = powder::sim::project_incompressible(world, div, p, cfg, 1, true);

  return rbgs_stats.iterations > 0U && mg_stats.iterations > 0U;
}

bool test_simd_axpy() {
  powder::core::Field2D<float> x(8, 8, 2);
  powder::core::Field2D<float> y(8, 8, 2);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x.raw()[i] = 1.5F;
    y.raw()[i] = 1.0F;
  }
  powder::sim::simd::axpy(y, x, 2.0F, 1);

  const float expected = 4.0F;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (std::fabs(y.raw()[i] - expected) > 1.0e-5F) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!test_mac_divergence_and_projection()) {
    std::cerr << "phase5 projection failed\n";
    return 1;
  }
  if (!test_maccormack_limiter_bounded()) {
    std::cerr << "phase5 maccormack limiter failed\n";
    return 1;
  }
  if (!test_forces_and_solver_paths()) {
    std::cerr << "phase5 force/projection paths failed\n";
    return 1;
  }
  if (!test_simd_axpy()) {
    std::cerr << "phase5 simd axpy failed\n";
    return 1;
  }
  return 0;
}
