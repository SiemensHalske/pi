#pragma once

#include "powder/core/Field2D.hpp"
#include "powder/sim/WorldState.hpp"

#include <cstddef>

namespace powder::sim {

struct FluidNumericsConfig {
  float dt = 1.0F / 120.0F;
  float dx = 1.0F;
  float dy = 1.0F;

  float gravity_y = -9.81F;
  float buoyancy_alpha = 0.0F;
  float ambient_temperature = 300.0F;
  float linear_drag = 0.0F;

  std::size_t jacobi_max_iters = 80;
  std::size_t rbgs_max_iters = 24;
  std::size_t mg_max_levels = 5;
  std::size_t mg_pre_smooth = 2;
  std::size_t mg_post_smooth = 2;
  float residual_tolerance = 1.0e-5F;
};

struct SolverStats {
  std::size_t iterations = 0;
  float residual_l2 = 0.0F;
  bool converged = false;
  bool used_multigrid = false;
};

struct ExternalForces {
  float impulse_x = 0.0F;
  float impulse_y = 0.0F;
  std::size_t impulse_cell_x = 0;
  std::size_t impulse_cell_y = 0;
  float impulse_radius = 0.0F;
};

void apply_velocity_boundary_no_slip(powder::core::FaceFieldU& u, powder::core::FaceFieldV& v);
void apply_pressure_boundary_neumann(powder::core::Field2D<float>& pressure);

void compute_mac_divergence(const powder::core::FaceFieldU& u,
                            const powder::core::FaceFieldV& v,
                            powder::core::Field2D<float>& divergence,
                            float dx,
                            float dy,
                            int thread_count);

void subtract_pressure_gradient(powder::core::FaceFieldU& u,
                                powder::core::FaceFieldV& v,
                                const powder::core::Field2D<float>& pressure,
                                float dt,
                                float dx,
                                float dy,
                                int thread_count);

void semi_lagrangian_advect_scalar(const powder::core::Field2D<float>& src,
                                   const powder::core::FaceFieldU& u,
                                   const powder::core::FaceFieldV& v,
                                   powder::core::Field2D<float>& dst,
                                   float dt,
                                   float dx,
                                   float dy,
                                   int thread_count);

void maccormack_advect_scalar_limited(const powder::core::Field2D<float>& src,
                                      const powder::core::FaceFieldU& u,
                                      const powder::core::FaceFieldV& v,
                                      powder::core::Field2D<float>& dst,
                                      float dt,
                                      float dx,
                                      float dy,
                                      int thread_count);

void semi_lagrangian_advect_velocity(const powder::core::FaceFieldU& u_src,
                                     const powder::core::FaceFieldV& v_src,
                                     powder::core::FaceFieldU& u_dst,
                                     powder::core::FaceFieldV& v_dst,
                                     float dt,
                                     float dx,
                                     float dy,
                                     int thread_count);

void accumulate_face_forces(const powder::core::FaceFieldU& u_in,
                            const powder::core::FaceFieldV& v_in,
                            const powder::core::Field2D<float>& temperature,
                            const powder::core::Field2D<float>& density,
                            const ExternalForces& external,
                            powder::core::FaceFieldU& u_out,
                            powder::core::FaceFieldV& v_out,
                            const FluidNumericsConfig& cfg,
                            int thread_count);

SolverStats solve_poisson_jacobi(powder::core::Field2D<float>& pressure,
                                 const powder::core::Field2D<float>& rhs,
                                 float dx,
                                 float dy,
                                 std::size_t max_iters,
                                 float residual_tolerance,
                                 int thread_count);

SolverStats solve_poisson_rbgs(powder::core::Field2D<float>& pressure,
                               const powder::core::Field2D<float>& rhs,
                               float dx,
                               float dy,
                               std::size_t max_iters,
                               float residual_tolerance,
                               int thread_count);

SolverStats solve_poisson_multigrid(powder::core::Field2D<float>& pressure,
                                    const powder::core::Field2D<float>& rhs,
                                    float dx,
                                    float dy,
                                    const FluidNumericsConfig& cfg,
                                    int thread_count);

SolverStats project_incompressible(WorldState& world,
                                   powder::core::Field2D<float>& divergence_rhs,
                                   powder::core::Field2D<float>& pressure_scratch,
                                   const FluidNumericsConfig& cfg,
                                   int thread_count,
                                   bool use_multigrid);

float divergence_l2_norm(const powder::core::FaceFieldU& u,
                         const powder::core::FaceFieldV& v,
                         powder::core::Field2D<float>& scratch,
                         float dx,
                         float dy,
                         int thread_count);

namespace simd {

void laplacian_5pt(const powder::core::Field2D<float>& src,
                   powder::core::Field2D<float>& dst,
                   float inv_dx2,
                   float inv_dy2,
                   int thread_count);

void jacobi_step(const powder::core::Field2D<float>& x_prev,
                 const powder::core::Field2D<float>& b,
                 powder::core::Field2D<float>& x_next,
                 float alpha,
                 float beta_inverse,
                 int thread_count);

float residual_l2(const powder::core::Field2D<float>& x,
                  const powder::core::Field2D<float>& b,
                  float inv_dx2,
                  float inv_dy2,
                  int thread_count);

void axpy(powder::core::Field2D<float>& y,
          const powder::core::Field2D<float>& x,
          float a,
          int thread_count);

}  // namespace simd

void phase5_link_anchor();

}  // namespace powder::sim
