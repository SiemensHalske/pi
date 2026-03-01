#include "powder/sim/Granular.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace powder::sim {

namespace {

[[nodiscard]] bool is_granular(std::uint16_t mat, const GranularRuleConfig& rules) {
  return mat == rules.sand_material || mat == rules.powder_material || mat == rules.slurry_material;
}

[[nodiscard]] std::size_t flat(std::size_t w, std::size_t x, std::size_t y) {
  return y * w + x;
}

}  // namespace

std::size_t step_granular_deterministic(CAGrid& grid, const GranularRuleConfig& rules,
                                        const GranularStepInput& input) {
  const auto w = grid.width();
  const auto h = grid.height();
  if (w == 0 || h == 0) {
    return 0;
  }

  const std::size_t invalid = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> src_of_dest(w * h, invalid);
  std::vector<std::size_t> dst_of_src(w * h, invalid);

  const bool even_bias_right = (input.tick_index & 1U) == 0U;

  for (std::size_t y = h - 1; y < h; --y) {
    for (std::size_t x = 0; x < w; ++x) {
      if (!grid.occupied(x, y)) {
        continue;
      }
      const auto mat = grid.material(x, y);
      if (!is_granular(mat, rules)) {
        continue;
      }

      const auto src = flat(w, x, y);
      int preferred_dx[3] = {0, 0, 0};
      std::size_t candidates = 0;

      if (y + 1 < h) {
        preferred_dx[candidates++] = 0U;
        const float vx = input.local_vel_x != nullptr ? (*input.local_vel_x)[src] : 0.0F;
        const bool right_first = (vx > 0.0F) ^ (!even_bias_right);
        if (right_first) {
          preferred_dx[candidates++] = 1;
          preferred_dx[candidates++] = -1;
        } else {
          preferred_dx[candidates++] = -1;
          preferred_dx[candidates++] = 1;
        }
      }

      for (std::size_t c = 0; c < candidates; ++c) {
        std::size_t nx = x;
        if (preferred_dx[c] < 0) {
          if (x == 0) {
            continue;
          }
          nx = x - 1;
        } else if (preferred_dx[c] > 0) {
          if (x + 1 >= w) {
            continue;
          }
          nx = x + 1;
        }
        const auto ny = y + 1;
        if (grid.occupied(nx, ny)) {
          continue;
        }
        const auto dst = flat(w, nx, ny);
        if (src_of_dest[dst] == invalid) {
          src_of_dest[dst] = src;
          dst_of_src[src] = dst;
          break;
        }
      }
    }
    if (y == 0) {
      break;
    }
  }

  std::size_t moved = 0;
  for (std::size_t src = 0; src < dst_of_src.size(); ++src) {
    const auto dst = dst_of_src[src];
    if (dst == invalid) {
      continue;
    }
    const auto sx = src % w;
    const auto sy = src / w;
    const auto dx = dst % w;
    const auto dy = dst / w;
    const auto mat = grid.material(sx, sy);

    grid.set_occupied(sx, sy, false);
    grid.set_material(sx, sy, 0U);
    grid.set_occupied(dx, dy, true);
    grid.set_material(dx, dy, mat);
    ++moved;
  }

  return moved;
}

}  // namespace powder::sim
