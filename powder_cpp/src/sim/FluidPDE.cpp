#include "powder/sim/FluidPDE.hpp"

#include "powder/core/MathPrimitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace powder::sim {
namespace {

[[nodiscard]] inline float sample_cell_centered(const powder::core::Field2D<float>& f, float x_center, float y_center) {
  const float px = x_center - 0.5F + static_cast<float>(f.ghost());
  const float py = y_center - 0.5F + static_cast<float>(f.ghost());
  const float x_clamped = powder::core::clamp_branchless(px, static_cast<float>(f.ghost()),
                                                          static_cast<float>(f.ghost() + f.width() - 1U));
  const float y_clamped = powder::core::clamp_branchless(py, static_cast<float>(f.ghost()),
                                                          static_cast<float>(f.ghost() + f.height() - 1U));
  return powder::core::bilinear_sample(f.raw(), f.pitch(), x_clamped, y_clamped);
}

[[nodiscard]] inline float sample_u_face(const powder::core::FaceFieldU& u, float x_face, float y_face) {
  const auto& uf = u.value;
  const float px = x_face + static_cast<float>(uf.ghost());
  const float py = y_face - 0.5F + static_cast<float>(uf.ghost());
  const float x_clamped = powder::core::clamp_branchless(px, static_cast<float>(uf.ghost()),
                                                          static_cast<float>(uf.ghost() + uf.width() - 1U));
  const float y_clamped = powder::core::clamp_branchless(py, static_cast<float>(uf.ghost()),
                                                          static_cast<float>(uf.ghost() + uf.height() - 1U));
  return powder::core::bilinear_sample(uf.raw(), uf.pitch(), x_clamped, y_clamped);
}

[[nodiscard]] inline float sample_v_face(const powder::core::FaceFieldV& v, float x_face, float y_face) {
  const auto& vf = v.value;
  const float px = x_face - 0.5F + static_cast<float>(vf.ghost());
  const float py = y_face + static_cast<float>(vf.ghost());
  const float x_clamped = powder::core::clamp_branchless(px, static_cast<float>(vf.ghost()),
                                                          static_cast<float>(vf.ghost() + vf.width() - 1U));
  const float y_clamped = powder::core::clamp_branchless(py, static_cast<float>(vf.ghost()),
                                                          static_cast<float>(vf.ghost() + vf.height() - 1U));
  return powder::core::bilinear_sample(vf.raw(), vf.pitch(), x_clamped, y_clamped);
}

[[nodiscard]] inline float centered_u(const powder::core::FaceFieldU& u, std::size_t x, std::size_t y) {
  return 0.5F * (u.value.at(x, y) + u.value.at(x + 1U, y));
}

[[nodiscard]] inline float centered_v(const powder::core::FaceFieldV& v, std::size_t x, std::size_t y) {
  return 0.5F * (v.value.at(x, y) + v.value.at(x, y + 1U));
}

void clear_interior(powder::core::Field2D<float>& f) {
  for (std::size_t y = 0; y < f.height(); ++y) {
    for (std::size_t x = 0; x < f.width(); ++x) {
      f.at(x, y) = 0.0F;
    }
  }
}

[[nodiscard]] float poisson_residual_l2(const powder::core::Field2D<float>& p,
                                        const powder::core::Field2D<float>& rhs,
                                        float dx,
                                        float dy,
                                        int thread_count) {
  const float inv_dx2 = 1.0F / (dx * dx);
  const float inv_dy2 = 1.0F / (dy * dy);
  const float diag = 2.0F * (inv_dx2 + inv_dy2);

  float sum = 0.0F;
  std::size_t count = p.width() * p.height();

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : sum)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(p.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(p.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const float c = p.at(xi, yi);
      const float l = p.at(xi - (xi > 0U ? 1U : 0U), yi);
      const float r = p.at(xi + (xi + 1U < p.width() ? 1U : 0U), yi);
      const float b = p.at(xi, yi - (yi > 0U ? 1U : 0U));
      const float t = p.at(xi, yi + (yi + 1U < p.height() ? 1U : 0U));
      const float lap = (l + r) * inv_dx2 + (b + t) * inv_dy2 - diag * c;
      const float res = rhs.at(xi, yi) - lap;
      sum += res * res;
    }
  }
  const float denom = static_cast<float>(std::max<std::size_t>(count, 1U));
  return std::sqrt(sum / denom);
}

struct MGLevel {
  powder::core::Field2D<float> p;
  powder::core::Field2D<float> rhs;
  powder::core::Field2D<float> residual;
  float dx = 1.0F;
  float dy = 1.0F;
};

void restrict_full_weighting(const powder::core::Field2D<float>& fine, powder::core::Field2D<float>& coarse) {
  clear_interior(coarse);
  for (std::size_t cy = 0; cy < coarse.height(); ++cy) {
    for (std::size_t cx = 0; cx < coarse.width(); ++cx) {
      const std::size_t fx = std::min(fine.width() - 1U, cx * 2U);
      const std::size_t fy = std::min(fine.height() - 1U, cy * 2U);
      float accum = 0.0F;
      float wsum = 0.0F;
      for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
          const int sx = static_cast<int>(fx) + ox;
          const int sy = static_cast<int>(fy) + oy;
          if (sx < 0 || sy < 0) {
            continue;
          }
          const auto sxu = static_cast<std::size_t>(sx);
          const auto syu = static_cast<std::size_t>(sy);
          if (sxu >= fine.width() || syu >= fine.height()) {
            continue;
          }
          const float wx = (ox == 0) ? 2.0F : 1.0F;
          const float wy = (oy == 0) ? 2.0F : 1.0F;
          const float w = wx * wy;
          accum += w * fine.at(sxu, syu);
          wsum += w;
        }
      }
      coarse.at(cx, cy) = (wsum > 0.0F) ? (accum / wsum) : fine.at(fx, fy);
    }
  }
}

void prolongate_bilinear_add(const powder::core::Field2D<float>& coarse, powder::core::Field2D<float>& fine) {
  for (std::size_t y = 0; y < fine.height(); ++y) {
    for (std::size_t x = 0; x < fine.width(); ++x) {
      const float cx = static_cast<float>(x) * 0.5F + 0.5F;
      const float cy = static_cast<float>(y) * 0.5F + 0.5F;
      fine.at(x, y) += sample_cell_centered(coarse, cx, cy);
    }
  }
}

void rbgs_smooth(powder::core::Field2D<float>& p,
                 const powder::core::Field2D<float>& rhs,
                 float dx,
                 float dy,
                 std::size_t sweeps,
                 int thread_count) {
  const float inv_dx2 = 1.0F / (dx * dx);
  const float inv_dy2 = 1.0F / (dy * dy);
  const float denom = 2.0F * (inv_dx2 + inv_dy2);

  for (std::size_t iter = 0; iter < sweeps; ++iter) {
    for (int parity = 0; parity < 2; ++parity) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
      if (thread_count > 0) {
        omp_set_num_threads(thread_count);
      }
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(p.height()); ++y) {
        for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(p.width()); ++x) {
          const auto xi = static_cast<std::size_t>(x);
          const auto yi = static_cast<std::size_t>(y);
          if (((xi + yi) & 1U) != static_cast<std::size_t>(parity)) {
            continue;
          }
          const std::size_t xl = (xi > 0U) ? xi - 1U : xi;
          const std::size_t xr = (xi + 1U < p.width()) ? xi + 1U : xi;
          const std::size_t yb = (yi > 0U) ? yi - 1U : yi;
          const std::size_t yt = (yi + 1U < p.height()) ? yi + 1U : yi;
          const float sum_lr = (p.at(xl, yi) + p.at(xr, yi)) * inv_dx2;
          const float sum_bt = (p.at(xi, yb) + p.at(xi, yt)) * inv_dy2;
          p.at(xi, yi) = (sum_lr + sum_bt - rhs.at(xi, yi)) / denom;
        }
      }
    }
    apply_pressure_boundary_neumann(p);
  }
}

void compute_poisson_residual(const powder::core::Field2D<float>& p,
                              const powder::core::Field2D<float>& rhs,
                              powder::core::Field2D<float>& res,
                              float dx,
                              float dy,
                              int thread_count) {
  const float inv_dx2 = 1.0F / (dx * dx);
  const float inv_dy2 = 1.0F / (dy * dy);
  const float diag = 2.0F * (inv_dx2 + inv_dy2);

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(p.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(p.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const std::size_t xl = (xi > 0U) ? xi - 1U : xi;
      const std::size_t xr = (xi + 1U < p.width()) ? xi + 1U : xi;
      const std::size_t yb = (yi > 0U) ? yi - 1U : yi;
      const std::size_t yt = (yi + 1U < p.height()) ? yi + 1U : yi;
      const float lap = (p.at(xl, yi) + p.at(xr, yi)) * inv_dx2 + (p.at(xi, yb) + p.at(xi, yt)) * inv_dy2 - diag * p.at(xi, yi);
      res.at(xi, yi) = rhs.at(xi, yi) - lap;
    }
  }
}

void multigrid_vcycle(std::vector<MGLevel>& levels, std::size_t level_index, const FluidNumericsConfig& cfg, int thread_count) {
  auto& lv = levels[level_index];
  rbgs_smooth(lv.p, lv.rhs, lv.dx, lv.dy, cfg.mg_pre_smooth, thread_count);

  const bool coarsest = (level_index + 1U == levels.size());
  if (coarsest) {
    rbgs_smooth(lv.p, lv.rhs, lv.dx, lv.dy, cfg.rbgs_max_iters, thread_count);
    return;
  }

  compute_poisson_residual(lv.p, lv.rhs, lv.residual, lv.dx, lv.dy, thread_count);

  auto& coarse = levels[level_index + 1U];
  clear_interior(coarse.p);
  restrict_full_weighting(lv.residual, coarse.rhs);
  apply_pressure_boundary_neumann(coarse.rhs);

  multigrid_vcycle(levels, level_index + 1U, cfg, thread_count);

  prolongate_bilinear_add(coarse.p, lv.p);
  apply_pressure_boundary_neumann(lv.p);
  rbgs_smooth(lv.p, lv.rhs, lv.dx, lv.dy, cfg.mg_post_smooth, thread_count);
}

}  // namespace

void apply_velocity_boundary_no_slip(powder::core::FaceFieldU& u, powder::core::FaceFieldV& v) {
  for (std::size_t y = 0; y < u.value.height(); ++y) {
    u.value.at(0, y) = 0.0F;
    u.value.at(u.value.width() - 1U, y) = 0.0F;
  }
  for (std::size_t x = 0; x < v.value.width(); ++x) {
    v.value.at(x, 0) = 0.0F;
    v.value.at(x, v.value.height() - 1U) = 0.0F;
  }
}

void apply_pressure_boundary_neumann(powder::core::Field2D<float>& pressure) {
  if (pressure.width() < 2U || pressure.height() < 2U) {
    return;
  }
  for (std::size_t y = 0; y < pressure.height(); ++y) {
    pressure.at(0, y) = pressure.at(1, y);
    pressure.at(pressure.width() - 1U, y) = pressure.at(pressure.width() - 2U, y);
  }
  for (std::size_t x = 0; x < pressure.width(); ++x) {
    pressure.at(x, 0) = pressure.at(x, 1);
    pressure.at(x, pressure.height() - 1U) = pressure.at(x, pressure.height() - 2U);
  }
}

void compute_mac_divergence(const powder::core::FaceFieldU& u,
                            const powder::core::FaceFieldV& v,
                            powder::core::Field2D<float>& divergence,
                            float dx,
                            float dy,
                            int thread_count) {
  const float inv_dx = 1.0F / dx;
  const float inv_dy = 1.0F / dy;

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(divergence.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(divergence.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const float du = (u.value.at(xi + 1U, yi) - u.value.at(xi, yi)) * inv_dx;
      const float dv = (v.value.at(xi, yi + 1U) - v.value.at(xi, yi)) * inv_dy;
      divergence.at(xi, yi) = du + dv;
    }
  }
}

void subtract_pressure_gradient(powder::core::FaceFieldU& u,
                                powder::core::FaceFieldV& v,
                                const powder::core::Field2D<float>& pressure,
                                float dt,
                                float dx,
                                float dy,
                                int thread_count) {
  const float sdx = dt / dx;
  const float sdy = dt / dy;

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(u.value.height()); ++y) {
    for (std::ptrdiff_t x = 1; x < static_cast<std::ptrdiff_t>(u.value.width()) - 1; ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      u.value.at(xi, yi) -= sdx * (pressure.at(xi, yi) - pressure.at(xi - 1U, yi));
    }
  }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 1; y < static_cast<std::ptrdiff_t>(v.value.height()) - 1; ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(v.value.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      v.value.at(xi, yi) -= sdy * (pressure.at(xi, yi) - pressure.at(xi, yi - 1U));
    }
  }
}

void semi_lagrangian_advect_scalar(const powder::core::Field2D<float>& src,
                                   const powder::core::FaceFieldU& u,
                                   const powder::core::FaceFieldV& v,
                                   powder::core::Field2D<float>& dst,
                                   float dt,
                                   float dx,
                                   float dy,
                                   int thread_count) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(src.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(src.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const float xc = static_cast<float>(xi) + 0.5F;
      const float yc = static_cast<float>(yi) + 0.5F;

      const float ux = centered_u(u, xi, yi);
      const float vy = centered_v(v, xi, yi);

      const float xb = xc - (dt / dx) * ux;
      const float yb = yc - (dt / dy) * vy;

      const float xcl = powder::core::clamp_branchless(xb, 0.5F, static_cast<float>(src.width()) - 0.5F);
      const float ycl = powder::core::clamp_branchless(yb, 0.5F, static_cast<float>(src.height()) - 0.5F);
      dst.at(xi, yi) = sample_cell_centered(src, xcl, ycl);
    }
  }
}

void maccormack_advect_scalar_limited(const powder::core::Field2D<float>& src,
                                      const powder::core::FaceFieldU& u,
                                      const powder::core::FaceFieldV& v,
                                      powder::core::Field2D<float>& dst,
                                      float dt,
                                      float dx,
                                      float dy,
                                      int thread_count) {
  powder::core::Field2D<float> phi_hat(src.width(), src.height(), src.ghost());
  powder::core::Field2D<float> phi_back(src.width(), src.height(), src.ghost());

  semi_lagrangian_advect_scalar(src, u, v, phi_hat, dt, dx, dy, thread_count);
  semi_lagrangian_advect_scalar(phi_hat, u, v, phi_back, -dt, dx, dy, thread_count);

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(src.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(src.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const float corr = 0.5F * (src.at(xi, yi) - phi_back.at(xi, yi));
      float value = phi_hat.at(xi, yi) + corr;

      float vmin = std::numeric_limits<float>::infinity();
      float vmax = -std::numeric_limits<float>::infinity();
      for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
          const int sx = static_cast<int>(xi) + ox;
          const int sy = static_cast<int>(yi) + oy;
          if (sx < 0 || sy < 0) {
            continue;
          }
          const auto sxu = static_cast<std::size_t>(sx);
          const auto syu = static_cast<std::size_t>(sy);
          if (sxu >= src.width() || syu >= src.height()) {
            continue;
          }
          const float sv = src.at(sxu, syu);
          vmin = std::fmin(vmin, sv);
          vmax = std::fmax(vmax, sv);
        }
      }
      value = powder::core::clamp_branchless(value, vmin, vmax);
      dst.at(xi, yi) = value;
    }
  }
}

void semi_lagrangian_advect_velocity(const powder::core::FaceFieldU& u_src,
                                     const powder::core::FaceFieldV& v_src,
                                     powder::core::FaceFieldU& u_dst,
                                     powder::core::FaceFieldV& v_dst,
                                     float dt,
                                     float dx,
                                     float dy,
                                     int thread_count) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(u_src.value.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(u_src.value.width()); ++x) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y) + 0.5F;
      const float ux = sample_u_face(u_src, fx, fy);
      const float vy = sample_v_face(v_src, fx, fy);
      const float xb = powder::core::clamp_branchless(fx - (dt / dx) * ux, 0.0F, static_cast<float>(u_src.value.width() - 1U));
      const float yb = powder::core::clamp_branchless(fy - (dt / dy) * vy, 0.5F, static_cast<float>(u_src.value.height()) - 0.5F);
      u_dst.value.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = sample_u_face(u_src, xb, yb);
    }
  }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(v_src.value.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(v_src.value.width()); ++x) {
      const float fx = static_cast<float>(x) + 0.5F;
      const float fy = static_cast<float>(y);
      const float ux = sample_u_face(u_src, fx, fy);
      const float vy = sample_v_face(v_src, fx, fy);
      const float xb = powder::core::clamp_branchless(fx - (dt / dx) * ux, 0.5F, static_cast<float>(v_src.value.width()) - 0.5F);
      const float yb = powder::core::clamp_branchless(fy - (dt / dy) * vy, 0.0F, static_cast<float>(v_src.value.height() - 1U));
      v_dst.value.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = sample_v_face(v_src, xb, yb);
    }
  }
}

void accumulate_face_forces(const powder::core::FaceFieldU& u_in,
                            const powder::core::FaceFieldV& v_in,
                            const powder::core::Field2D<float>& temperature,
                            const powder::core::Field2D<float>& density,
                            const ExternalForces& external,
                            powder::core::FaceFieldU& u_out,
                            powder::core::FaceFieldV& v_out,
                            const FluidNumericsConfig& cfg,
                            int thread_count) {
  const float dt = cfg.dt;

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(u_in.value.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(u_in.value.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const float drag = 1.0F - dt * cfg.linear_drag;
      u_out.value.at(xi, yi) = u_in.value.at(xi, yi) * drag;
    }
  }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(v_in.value.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(v_in.value.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      float buoy = 0.0F;
      if (xi < temperature.width() && yi < temperature.height()) {
        const float t = temperature.at(xi, yi);
        const float rho = std::fmax(1.0e-6F, density.at(xi, yi));
        buoy = cfg.buoyancy_alpha * (t - cfg.ambient_temperature) / rho;
      }
      const float drag = 1.0F - dt * cfg.linear_drag;
      v_out.value.at(xi, yi) = (v_in.value.at(xi, yi) + dt * (cfg.gravity_y + buoy)) * drag;
    }
  }

  if (external.impulse_radius > 0.0F) {
    const float r2 = external.impulse_radius * external.impulse_radius;
    for (std::size_t y = 0; y < u_out.value.height(); ++y) {
      for (std::size_t x = 0; x < u_out.value.width(); ++x) {
        const float dxp = static_cast<float>(x) - static_cast<float>(external.impulse_cell_x);
        const float dyp = static_cast<float>(y) - static_cast<float>(external.impulse_cell_y);
        const float d2 = dxp * dxp + dyp * dyp;
        if (d2 <= r2) {
          const float w = std::exp(-d2 / (2.0F * r2));
          u_out.value.at(x, y) += external.impulse_x * w;
        }
      }
    }
    for (std::size_t y = 0; y < v_out.value.height(); ++y) {
      for (std::size_t x = 0; x < v_out.value.width(); ++x) {
        const float dxp = static_cast<float>(x) - static_cast<float>(external.impulse_cell_x);
        const float dyp = static_cast<float>(y) - static_cast<float>(external.impulse_cell_y);
        const float d2 = dxp * dxp + dyp * dyp;
        if (d2 <= r2) {
          const float w = std::exp(-d2 / (2.0F * r2));
          v_out.value.at(x, y) += external.impulse_y * w;
        }
      }
    }
  }
}

SolverStats solve_poisson_jacobi(powder::core::Field2D<float>& pressure,
                                 const powder::core::Field2D<float>& rhs,
                                 float dx,
                                 float dy,
                                 std::size_t max_iters,
                                 float residual_tolerance,
                                 int thread_count) {
  SolverStats stats{};
  stats.used_multigrid = false;

  powder::core::Field2D<float> next(pressure.width(), pressure.height(), pressure.ghost());
  const float inv_dx2 = 1.0F / (dx * dx);
  const float inv_dy2 = 1.0F / (dy * dy);
  const float denom = 2.0F * (inv_dx2 + inv_dy2);

  for (std::size_t it = 0; it < max_iters; ++it) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
    if (thread_count > 0) {
      omp_set_num_threads(thread_count);
    }
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(pressure.height()); ++y) {
      for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(pressure.width()); ++x) {
        const auto xi = static_cast<std::size_t>(x);
        const auto yi = static_cast<std::size_t>(y);
        const std::size_t xl = (xi > 0U) ? xi - 1U : xi;
        const std::size_t xr = (xi + 1U < pressure.width()) ? xi + 1U : xi;
        const std::size_t yb = (yi > 0U) ? yi - 1U : yi;
        const std::size_t yt = (yi + 1U < pressure.height()) ? yi + 1U : yi;
        const float sum_lr = (pressure.at(xl, yi) + pressure.at(xr, yi)) * inv_dx2;
        const float sum_bt = (pressure.at(xi, yb) + pressure.at(xi, yt)) * inv_dy2;
        next.at(xi, yi) = (sum_lr + sum_bt - rhs.at(xi, yi)) / denom;
      }
    }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(pressure.height()); ++y) {
      for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(pressure.width()); ++x) {
        pressure.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) =
            next.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y));
      }
    }

    apply_pressure_boundary_neumann(pressure);
    stats.iterations = it + 1U;
    stats.residual_l2 = poisson_residual_l2(pressure, rhs, dx, dy, thread_count);
    if (stats.residual_l2 <= residual_tolerance) {
      stats.converged = true;
      break;
    }
  }
  if (!stats.converged) {
    stats.converged = stats.residual_l2 <= residual_tolerance;
  }
  return stats;
}

SolverStats solve_poisson_rbgs(powder::core::Field2D<float>& pressure,
                               const powder::core::Field2D<float>& rhs,
                               float dx,
                               float dy,
                               std::size_t max_iters,
                               float residual_tolerance,
                               int thread_count) {
  SolverStats stats{};
  stats.used_multigrid = false;

  for (std::size_t it = 0; it < max_iters; ++it) {
    rbgs_smooth(pressure, rhs, dx, dy, 1U, thread_count);
    stats.iterations = it + 1U;
    stats.residual_l2 = poisson_residual_l2(pressure, rhs, dx, dy, thread_count);
    if (stats.residual_l2 <= residual_tolerance) {
      stats.converged = true;
      break;
    }
  }
  if (!stats.converged) {
    stats.converged = stats.residual_l2 <= residual_tolerance;
  }
  return stats;
}

SolverStats solve_poisson_multigrid(powder::core::Field2D<float>& pressure,
                                    const powder::core::Field2D<float>& rhs,
                                    float dx,
                                    float dy,
                                    const FluidNumericsConfig& cfg,
                                    int thread_count) {
  SolverStats stats{};
  stats.used_multigrid = true;

  std::vector<MGLevel> levels;
  levels.reserve(cfg.mg_max_levels);

  std::size_t w = pressure.width();
  std::size_t h = pressure.height();
  float ldx = dx;
  float ldy = dy;

  while (levels.size() < cfg.mg_max_levels) {
    MGLevel level;
    level.p.resize(w, h, pressure.ghost());
    level.rhs.resize(w, h, pressure.ghost());
    level.residual.resize(w, h, pressure.ghost());
    level.dx = ldx;
    level.dy = ldy;
    levels.push_back(std::move(level));

    if (w <= 4U || h <= 4U) {
      break;
    }
    w = std::max<std::size_t>(2U, w / 2U);
    h = std::max<std::size_t>(2U, h / 2U);
    ldx *= 2.0F;
    ldy *= 2.0F;
  }

  for (std::size_t y = 0; y < pressure.height(); ++y) {
    for (std::size_t x = 0; x < pressure.width(); ++x) {
      levels[0].p.at(x, y) = pressure.at(x, y);
      levels[0].rhs.at(x, y) = rhs.at(x, y);
    }
  }

  constexpr std::size_t max_cycles = 12U;
  for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
    multigrid_vcycle(levels, 0U, cfg, thread_count);
    stats.iterations = cycle + 1U;
    stats.residual_l2 = poisson_residual_l2(levels[0].p, levels[0].rhs, dx, dy, thread_count);
    if (stats.residual_l2 <= cfg.residual_tolerance) {
      stats.converged = true;
      break;
    }
  }

  for (std::size_t y = 0; y < pressure.height(); ++y) {
    for (std::size_t x = 0; x < pressure.width(); ++x) {
      pressure.at(x, y) = levels[0].p.at(x, y);
    }
  }

  if (!stats.converged) {
    stats.converged = stats.residual_l2 <= cfg.residual_tolerance;
  }
  return stats;
}

SolverStats project_incompressible(WorldState& world,
                                   powder::core::Field2D<float>& divergence_rhs,
                                   powder::core::Field2D<float>& pressure_scratch,
                                   const FluidNumericsConfig& cfg,
                                   int thread_count,
                                   bool use_multigrid) {
  compute_mac_divergence(world.velocity_u, world.velocity_v, divergence_rhs, cfg.dx, cfg.dy, thread_count);

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(divergence_rhs.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(divergence_rhs.width()); ++x) {
      divergence_rhs.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) /= cfg.dt;
      pressure_scratch.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = 0.0F;
    }
  }

  SolverStats stats{};
  if (use_multigrid) {
    stats = solve_poisson_multigrid(pressure_scratch, divergence_rhs, cfg.dx, cfg.dy, cfg, thread_count);
  } else {
    stats = solve_poisson_rbgs(pressure_scratch, divergence_rhs, cfg.dx, cfg.dy,
                               cfg.rbgs_max_iters, cfg.residual_tolerance, thread_count);
  }

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(world.pressure.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(world.pressure.width()); ++x) {
      world.pressure.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) =
          pressure_scratch.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y));
    }
  }

  subtract_pressure_gradient(world.velocity_u, world.velocity_v, world.pressure, cfg.dt, cfg.dx, cfg.dy, thread_count);
  apply_velocity_boundary_no_slip(world.velocity_u, world.velocity_v);
  return stats;
}

float divergence_l2_norm(const powder::core::FaceFieldU& u,
                         const powder::core::FaceFieldV& v,
                         powder::core::Field2D<float>& scratch,
                         float dx,
                         float dy,
                         int thread_count) {
  compute_mac_divergence(u, v, scratch, dx, dy, thread_count);
  float sum = 0.0F;

#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : sum)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(scratch.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(scratch.width()); ++x) {
      const float d = scratch.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y));
      sum += d * d;
    }
  }

  const float denom = static_cast<float>(std::max<std::size_t>(1U, scratch.width() * scratch.height()));
  return std::sqrt(sum / denom);
}

namespace simd {

void laplacian_5pt(const powder::core::Field2D<float>& src,
                   powder::core::Field2D<float>& dst,
                   float inv_dx2,
                   float inv_dy2,
                   int thread_count) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(src.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(src.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const std::size_t xl = (xi > 0U) ? xi - 1U : xi;
      const std::size_t xr = (xi + 1U < src.width()) ? xi + 1U : xi;
      const std::size_t yb = (yi > 0U) ? yi - 1U : yi;
      const std::size_t yt = (yi + 1U < src.height()) ? yi + 1U : yi;
      const float c = src.at(xi, yi);
      const float lap = (src.at(xl, yi) - 2.0F * c + src.at(xr, yi)) * inv_dx2 +
                        (src.at(xi, yb) - 2.0F * c + src.at(xi, yt)) * inv_dy2;
      dst.at(xi, yi) = lap;
    }
  }
}

void jacobi_step(const powder::core::Field2D<float>& x_prev,
                 const powder::core::Field2D<float>& b,
                 powder::core::Field2D<float>& x_next,
                 float alpha,
                 float beta_inverse,
                 int thread_count) {
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(x_prev.height()); ++y) {
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(x_prev.width()); ++x) {
      const auto xi = static_cast<std::size_t>(x);
      const auto yi = static_cast<std::size_t>(y);
      const std::size_t xl = (xi > 0U) ? xi - 1U : xi;
      const std::size_t xr = (xi + 1U < x_prev.width()) ? xi + 1U : xi;
      const std::size_t yb = (yi > 0U) ? yi - 1U : yi;
      const std::size_t yt = (yi + 1U < x_prev.height()) ? yi + 1U : yi;
      const float neighbors = x_prev.at(xl, yi) + x_prev.at(xr, yi) + x_prev.at(xi, yb) + x_prev.at(xi, yt);
      x_next.at(xi, yi) = (neighbors + alpha * b.at(xi, yi)) * beta_inverse;
    }
  }
}

float residual_l2(const powder::core::Field2D<float>& x,
                  const powder::core::Field2D<float>& b,
                  float inv_dx2,
                  float inv_dy2,
                  int thread_count) {
  powder::core::Field2D<float> lap(x.width(), x.height(), x.ghost());
  laplacian_5pt(x, lap, inv_dx2, inv_dy2, thread_count);

  float sum = 0.0F;
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : sum)
#endif
  for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(x.height()); ++y) {
    for (std::ptrdiff_t xk = 0; xk < static_cast<std::ptrdiff_t>(x.width()); ++xk) {
      const auto xi = static_cast<std::size_t>(xk);
      const auto yi = static_cast<std::size_t>(y);
      const float r = b.at(xi, yi) - lap.at(xi, yi);
      sum += r * r;
    }
  }
  const float denom = static_cast<float>(std::max<std::size_t>(1U, x.width() * x.height()));
  return std::sqrt(sum / denom);
}

void axpy(powder::core::Field2D<float>& y,
          const powder::core::Field2D<float>& x,
          float a,
          int thread_count) {
  const std::size_t n = std::min(y.size(), x.size());
  float* yp = y.raw();
  const float* xp = x.raw();

#if defined(__AVX512F__)
  (void)thread_count;
  const __m512 av = _mm512_set1_ps(a);
  std::size_t i = 0;
  for (; i + 16U <= n; i += 16U) {
    __m512 yv = _mm512_loadu_ps(yp + i);
    const __m512 xv = _mm512_loadu_ps(xp + i);
    yv = _mm512_fmadd_ps(av, xv, yv);
    _mm512_storeu_ps(yp + i, yv);
  }
  const std::size_t rem = n - i;
  if (rem > 0U) {
    const __mmask16 mask = static_cast<__mmask16>((1U << rem) - 1U);
    __m512 yv = _mm512_maskz_loadu_ps(mask, yp + i);
    const __m512 xv = _mm512_maskz_loadu_ps(mask, xp + i);
    yv = _mm512_fmadd_ps(av, xv, yv);
    _mm512_mask_storeu_ps(yp + i, mask, yv);
  }
#else
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  if (thread_count > 0) {
    omp_set_num_threads(thread_count);
  }
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
    yp[static_cast<std::size_t>(i)] += a * xp[static_cast<std::size_t>(i)];
  }
#endif
}

}  // namespace simd

void phase5_link_anchor() {
  WorldState world = create_world_state(8, 8, 2);
  FluidNumericsConfig cfg{};
  cfg.dt = 1.0F / 60.0F;
  cfg.dx = 1.0F;
  cfg.dy = 1.0F;

  powder::core::Field2D<float> rhs(world.width, world.height, world.ghost);
  powder::core::Field2D<float> p(world.width, world.height, world.ghost);
  (void)project_incompressible(world, rhs, p, cfg, 1, true);
}

}  // namespace powder::sim
