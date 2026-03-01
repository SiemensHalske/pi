#include "powder/sim/CAGrid.hpp"

#include <algorithm>

namespace powder::sim {

void CAGrid::resize(std::size_t width, std::size_t height) {
  width_ = width;
  height_ = height;
  material_ids_.assign(width_ * height_, 0U);
  occupancy_bits_.assign((width_ * height_ + 63U) / 64U, 0ULL);
}

std::size_t CAGrid::index(std::size_t x, std::size_t y) const noexcept {
  return y * width_ + x;
}

bool CAGrid::in_bounds(std::size_t x, std::size_t y) const noexcept {
  return x < width_ && y < height_;
}

bool CAGrid::occupied(std::size_t x, std::size_t y) const noexcept {
  const auto id = index(x, y);
  const auto word = id >> 6U;
  const auto bit = id & 63U;
  return ((occupancy_bits_[word] >> bit) & 1ULL) != 0ULL;
}

void CAGrid::set_occupied(std::size_t x, std::size_t y, bool value) noexcept {
  const auto id = index(x, y);
  const auto word = id >> 6U;
  const auto bit = id & 63U;
  const auto mask = (1ULL << bit);
  if (value) {
    occupancy_bits_[word] |= mask;
  } else {
    occupancy_bits_[word] &= ~mask;
  }
}

std::uint16_t CAGrid::material(std::size_t x, std::size_t y) const noexcept {
  return material_ids_[index(x, y)];
}

void CAGrid::set_material(std::size_t x, std::size_t y, std::uint16_t material_id) noexcept {
  material_ids_[index(x, y)] = material_id;
}

}  // namespace powder::sim
