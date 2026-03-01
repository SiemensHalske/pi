#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace powder::sim {

class CAGrid {
 public:
  void resize(std::size_t width, std::size_t height);

  [[nodiscard]] std::size_t width() const noexcept { return width_; }
  [[nodiscard]] std::size_t height() const noexcept { return height_; }
  [[nodiscard]] std::size_t size() const noexcept { return width_ * height_; }

  [[nodiscard]] std::size_t index(std::size_t x, std::size_t y) const noexcept;
  [[nodiscard]] bool in_bounds(std::size_t x, std::size_t y) const noexcept;

  [[nodiscard]] bool occupied(std::size_t x, std::size_t y) const noexcept;
  void set_occupied(std::size_t x, std::size_t y, bool value) noexcept;

  [[nodiscard]] std::uint16_t material(std::size_t x, std::size_t y) const noexcept;
  void set_material(std::size_t x, std::size_t y, std::uint16_t material_id) noexcept;

  [[nodiscard]] const std::vector<std::uint16_t>& material_ids() const noexcept { return material_ids_; }

 private:
  std::size_t width_ = 0;
  std::size_t height_ = 0;
  std::vector<std::uint64_t> occupancy_bits_;
  std::vector<std::uint16_t> material_ids_;
};

}  // namespace powder::sim
