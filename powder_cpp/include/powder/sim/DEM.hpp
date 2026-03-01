#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace powder::sim {

struct DEMParticleSoA {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> vx;
  std::vector<float> vy;
  std::vector<float> omega;
  std::vector<float> radius;
  std::vector<float> mass;
  std::vector<float> inertia;
  std::vector<std::uint16_t> material_id;
  std::vector<std::uint8_t> alive;
  std::vector<std::size_t> free_list;

  void reserve(std::size_t capacity);
  [[nodiscard]] std::size_t create(float px, float py, float pradius, float pmass, std::uint16_t material);
  void destroy(std::size_t id);
  [[nodiscard]] bool is_alive(std::size_t id) const noexcept;
  [[nodiscard]] std::size_t alive_count() const noexcept;
};

}  // namespace powder::sim
