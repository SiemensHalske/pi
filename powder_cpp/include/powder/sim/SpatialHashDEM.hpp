#pragma once

#include "powder/sim/DEM.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace powder::sim {

struct DEMContact {
  std::size_t a = 0;
  std::size_t b = 0;
  float nx = 0.0F;
  float ny = 0.0F;
  float penetration = 0.0F;
};

class SpatialHash2D {
 public:
  explicit SpatialHash2D(float cell_size = 1.0F) : cell_size_(cell_size) {}

  void set_cell_size(float cell_size) noexcept { cell_size_ = cell_size; }
  [[nodiscard]] float cell_size() const noexcept { return cell_size_; }

  void build(const DEMParticleSoA& particles);
  [[nodiscard]] std::vector<DEMContact> generate_contacts(const DEMParticleSoA& particles) const;

 private:
  using Key = std::pair<int, int>;
  float cell_size_ = 1.0F;
  std::map<Key, std::vector<std::size_t>> buckets_;
};

struct ContactModelConfig {
  float k_normal = 1200.0F;
  float c_damping = 8.0F;
  float friction_mu = 0.45F;
};

void solve_contacts_penalty_dashpot(DEMParticleSoA& particles, const std::vector<DEMContact>& contacts,
                                    float dt, const ContactModelConfig& cfg);

}  // namespace powder::sim
