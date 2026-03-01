#pragma once

#include "powder/core/Field2D.hpp"
#include "powder/sim/DEM.hpp"

namespace powder::sim {

struct CouplingDiagnostics {
  float particle_momentum_x_before = 0.0F;
  float particle_momentum_y_before = 0.0F;
  float particle_momentum_x_after = 0.0F;
  float particle_momentum_y_after = 0.0F;
  float grid_momentum_x = 0.0F;
  float grid_momentum_y = 0.0F;
};

void p2g_deposit_thread_local(const DEMParticleSoA& particles,
                              powder::core::Field2D<float>& grid_mass,
                              powder::core::Field2D<float>& grid_mom_x,
                              powder::core::Field2D<float>& grid_mom_y,
                              int thread_count);

void g2p_sample_velocity(const powder::core::Field2D<float>& grid_vel_x,
                         const powder::core::Field2D<float>& grid_vel_y,
                         DEMParticleSoA& particles,
                         float blend);

[[nodiscard]] CouplingDiagnostics couple_particles_and_grid(DEMParticleSoA& particles,
                                                            powder::core::Field2D<float>& grid_mass,
                                                            powder::core::Field2D<float>& grid_mom_x,
                                                            powder::core::Field2D<float>& grid_mom_y,
                                                            const powder::core::Field2D<float>& grid_vel_x,
                                                            const powder::core::Field2D<float>& grid_vel_y,
                                                            float g2p_blend,
                                                            int thread_count);

}  // namespace powder::sim
