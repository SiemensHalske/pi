#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace powder::core {

[[nodiscard]] inline float clamp_branchless(float value, float lo, float hi) {
  return std::fmin(hi, std::fmax(lo, value));
}

[[nodiscard]] inline float safe_reciprocal(float value, float epsilon = 1.0e-8F) {
  const auto mag = std::fmax(std::fabs(value), epsilon);
  return std::copysign(1.0F / mag, value);
}

[[nodiscard]] inline float slope_limiter_minmod(float a, float b) {
  if (a * b <= 0.0F) {
    return 0.0F;
  }
  return std::copysign(std::fmin(std::fabs(a), std::fabs(b)), a);
}

[[nodiscard]] inline float bilinear_sample(const float* field, std::size_t pitch, float x, float y) {
  const auto x0 = static_cast<std::size_t>(x);
  const auto y0 = static_cast<std::size_t>(y);
  const auto x1 = x0 + 1U;
  const auto y1 = y0 + 1U;

  const float tx = x - static_cast<float>(x0);
  const float ty = y - static_cast<float>(y0);

  const float v00 = field[y0 * pitch + x0];
  const float v10 = field[y0 * pitch + x1];
  const float v01 = field[y1 * pitch + x0];
  const float v11 = field[y1 * pitch + x1];

  const float a = v00 + tx * (v10 - v00);
  const float b = v01 + tx * (v11 - v01);
  return a + ty * (b - a);
}

[[nodiscard]] inline float grad_x_central(const float* field, std::size_t pitch, std::size_t x, std::size_t y, float dx) {
  const float left = field[y * pitch + (x - 1U)];
  const float right = field[y * pitch + (x + 1U)];
  return (right - left) * (0.5F / dx);
}

[[nodiscard]] inline float grad_y_central(const float* field, std::size_t pitch, std::size_t x, std::size_t y, float dy) {
  const float down = field[(y - 1U) * pitch + x];
  const float up = field[(y + 1U) * pitch + x];
  return (up - down) * (0.5F / dy);
}

[[nodiscard]] inline float laplacian_5pt(const float* field, std::size_t pitch, std::size_t x, std::size_t y,
                                         float dx, float dy) {
  const float c = field[y * pitch + x];
  const float l = field[y * pitch + (x - 1U)];
  const float r = field[y * pitch + (x + 1U)];
  const float d = field[(y - 1U) * pitch + x];
  const float u = field[(y + 1U) * pitch + x];
  const float idx2 = 1.0F / (dx * dx);
  const float idy2 = 1.0F / (dy * dy);
  return (l - 2.0F * c + r) * idx2 + (d - 2.0F * c + u) * idy2;
}

}  // namespace powder::core
