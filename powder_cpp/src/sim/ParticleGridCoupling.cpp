#include "powder/sim/ParticleGridCoupling.hpp"

#include "powder/core/MathPrimitives.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#include <omp.h>
#endif

namespace powder::sim {

namespace {

[[nodiscard]] float momentum_x(const DEMParticleSoA& particles) {
  float sum = 0.0F;
  for (std::size_t i = 0; i < particles.x.size(); ++i) {
    if (!particles.is_alive(i)) {
      continue;
    }
    sum += particles.mass[i] * particles.vx[i];
  }
  return sum;
}

[[nodiscard]] float momentum_y(const DEMParticleSoA& particles) {
  float sum = 0.0F;
  for (std::size_t i = 0; i < particles.x.size(); ++i) {
    if (!particles.is_alive(i)) {
      continue;
    }
    sum += particles.mass[i] * particles.vy[i];
  }
  return sum;
}

[[nodiscard]] float grid_sum(const powder::core::Field2D<float>& field) {
  float sum = 0.0F;
  const auto* p = field.raw();
  for (std::size_t i = 0; i < field.size(); ++i) {
    sum += p[i];
  }
  return sum;
}

}  // namespace

void p2g_deposit_thread_local(const DEMParticleSoA& particles,
                              powder::core::Field2D<float>& grid_mass,
                              powder::core::Field2D<float>& grid_mom_x,
                              powder::core::Field2D<float>& grid_mom_y,
                              int thread_count) {
  const auto n = grid_mass.size();
  const int workers = thread_count > 0 ? thread_count : 1;
  std::vector<std::vector<float>> local_mass(static_cast<std::size_t>(workers), std::vector<float>(n, 0.0F));
  std::vector<std::vector<float>> local_mx(static_cast<std::size_t>(workers), std::vector<float>(n, 0.0F));
  std::vector<std::vector<float>> local_my(static_cast<std::size_t>(workers), std::vector<float>(n, 0.0F));

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for (std::ptrdiff_t pi = 0; pi < static_cast<std::ptrdiff_t>(particles.x.size()); ++pi) {
      const auto i = static_cast<std::size_t>(pi);
      if (!particles.is_alive(i)) {
        continue;
      }
      const float px = particles.x[i];
      const float py = particles.y[i];
      const auto ix = static_cast<std::size_t>(std::max(0, static_cast<int>(px)));
      const auto iy = static_cast<std::size_t>(std::max(0, static_cast<int>(py)));
      const float fx = px - static_cast<float>(ix);
      const float fy = py - static_cast<float>(iy);

      const float w00 = (1.0F - fx) * (1.0F - fy);
      const float w10 = fx * (1.0F - fy);
      const float w01 = (1.0F - fx) * fy;
      const float w11 = fx * fy;

      const auto add = [&](std::size_t gx, std::size_t gy, float w) {
        if (gx >= grid_mass.width() || gy >= grid_mass.height()) {
          return;
        }
        const auto idx = grid_mass.index(gx, gy);
        local_mass[static_cast<std::size_t>(tid)][idx] += w * particles.mass[i];
        local_mx[static_cast<std::size_t>(tid)][idx] += w * particles.mass[i] * particles.vx[i];
        local_my[static_cast<std::size_t>(tid)][idx] += w * particles.mass[i] * particles.vy[i];
      };

      add(ix, iy, w00);
      add(ix + 1U, iy, w10);
      add(ix, iy + 1U, w01);
      add(ix + 1U, iy + 1U, w11);
    }
  }
#else
  (void)thread_count;
  for (std::size_t i = 0; i < particles.x.size(); ++i) {
    if (!particles.is_alive(i)) {
      continue;
    }
    const float px = particles.x[i];
    const float py = particles.y[i];
    const auto ix = static_cast<std::size_t>(std::max(0, static_cast<int>(px)));
    const auto iy = static_cast<std::size_t>(std::max(0, static_cast<int>(py)));
    const float fx = px - static_cast<float>(ix);
    const float fy = py - static_cast<float>(iy);

    const float w00 = (1.0F - fx) * (1.0F - fy);
    const float w10 = fx * (1.0F - fy);
    const float w01 = (1.0F - fx) * fy;
    const float w11 = fx * fy;

    const auto add = [&](std::size_t gx, std::size_t gy, float w) {
      if (gx >= grid_mass.width() || gy >= grid_mass.height()) {
        return;
      }
      const auto idx = grid_mass.index(gx, gy);
      local_mass[0][idx] += w * particles.mass[i];
      local_mx[0][idx] += w * particles.mass[i] * particles.vx[i];
      local_my[0][idx] += w * particles.mass[i] * particles.vy[i];
    };

    add(ix, iy, w00);
    add(ix + 1U, iy, w10);
    add(ix, iy + 1U, w01);
    add(ix + 1U, iy + 1U, w11);
  }
#endif

  auto* gm = grid_mass.raw();
  auto* gmx = grid_mom_x.raw();
  auto* gmy = grid_mom_y.raw();
  for (std::size_t i = 0; i < n; ++i) {
    gm[i] = 0.0F;
    gmx[i] = 0.0F;
    gmy[i] = 0.0F;
    for (int t = 0; t < workers; ++t) {
      gm[i] += local_mass[static_cast<std::size_t>(t)][i];
      gmx[i] += local_mx[static_cast<std::size_t>(t)][i];
      gmy[i] += local_my[static_cast<std::size_t>(t)][i];
    }
  }
}

void g2p_sample_velocity(const powder::core::Field2D<float>& grid_vel_x,
                         const powder::core::Field2D<float>& grid_vel_y,
                         DEMParticleSoA& particles,
                         float blend) {
  const float alpha = powder::core::clamp_branchless(blend, 0.0F, 1.0F);
  for (std::size_t i = 0; i < particles.x.size(); ++i) {
    if (!particles.is_alive(i)) {
      continue;
    }
    const float px = particles.x[i] + static_cast<float>(grid_vel_x.ghost());
    const float py = particles.y[i] + static_cast<float>(grid_vel_x.ghost());
    const float ux = powder::core::bilinear_sample(grid_vel_x.raw(), grid_vel_x.pitch(), px, py);
    const float uy = powder::core::bilinear_sample(grid_vel_y.raw(), grid_vel_y.pitch(), px, py);
    particles.vx[i] = (1.0F - alpha) * particles.vx[i] + alpha * ux;
    particles.vy[i] = (1.0F - alpha) * particles.vy[i] + alpha * uy;
  }
}

CouplingDiagnostics couple_particles_and_grid(DEMParticleSoA& particles,
                                              powder::core::Field2D<float>& grid_mass,
                                              powder::core::Field2D<float>& grid_mom_x,
                                              powder::core::Field2D<float>& grid_mom_y,
                                              const powder::core::Field2D<float>& grid_vel_x,
                                              const powder::core::Field2D<float>& grid_vel_y,
                                              float g2p_blend,
                                              int thread_count) {
  CouplingDiagnostics d{};
  d.particle_momentum_x_before = momentum_x(particles);
  d.particle_momentum_y_before = momentum_y(particles);

  p2g_deposit_thread_local(particles, grid_mass, grid_mom_x, grid_mom_y, thread_count);

  d.grid_momentum_x = grid_sum(grid_mom_x);
  d.grid_momentum_y = grid_sum(grid_mom_y);

  g2p_sample_velocity(grid_vel_x, grid_vel_y, particles, g2p_blend);

  d.particle_momentum_x_after = momentum_x(particles);
  d.particle_momentum_y_after = momentum_y(particles);
  return d;
}

}  // namespace powder::sim
