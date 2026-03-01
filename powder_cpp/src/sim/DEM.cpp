#include "powder/sim/DEM.hpp"

namespace powder::sim {

void DEMParticleSoA::reserve(std::size_t capacity) {
  x.reserve(capacity);
  y.reserve(capacity);
  vx.reserve(capacity);
  vy.reserve(capacity);
  omega.reserve(capacity);
  radius.reserve(capacity);
  mass.reserve(capacity);
  inertia.reserve(capacity);
  material_id.reserve(capacity);
  alive.reserve(capacity);
}

std::size_t DEMParticleSoA::create(float px, float py, float pradius, float pmass, std::uint16_t material) {
  std::size_t id = 0;
  if (!free_list.empty()) {
    id = free_list.back();
    free_list.pop_back();
    x[id] = px;
    y[id] = py;
    vx[id] = 0.0F;
    vy[id] = 0.0F;
    omega[id] = 0.0F;
    radius[id] = pradius;
    mass[id] = pmass;
    inertia[id] = 0.5F * pmass * pradius * pradius;
    material_id[id] = material;
    alive[id] = 1U;
    return id;
  }

  id = x.size();
  x.push_back(px);
  y.push_back(py);
  vx.push_back(0.0F);
  vy.push_back(0.0F);
  omega.push_back(0.0F);
  radius.push_back(pradius);
  mass.push_back(pmass);
  inertia.push_back(0.5F * pmass * pradius * pradius);
  material_id.push_back(material);
  alive.push_back(1U);
  return id;
}

void DEMParticleSoA::destroy(std::size_t id) {
  if (id >= alive.size() || alive[id] == 0U) {
    return;
  }
  alive[id] = 0U;
  free_list.push_back(id);
}

bool DEMParticleSoA::is_alive(std::size_t id) const noexcept {
  return id < alive.size() && alive[id] != 0U;
}

std::size_t DEMParticleSoA::alive_count() const noexcept {
  std::size_t count = 0;
  for (const auto a : alive) {
    count += static_cast<std::size_t>(a != 0U);
  }
  return count;
}

}  // namespace powder::sim
