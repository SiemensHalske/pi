#include "powder/core/Field2D.hpp"
#include "powder/sim/CAGrid.hpp"
#include "powder/sim/DEM.hpp"
#include "powder/sim/Granular.hpp"
#include "powder/sim/ParticleGridCoupling.hpp"
#include "powder/sim/SpatialHashDEM.hpp"

#include <cmath>
#include <iostream>

namespace {

bool test_ca_bitpack_and_material() {
  powder::sim::CAGrid g;
  g.resize(9, 7);
  g.set_occupied(3, 4, true);
  g.set_material(3, 4, 4U);
  return g.occupied(3, 4) && g.material(3, 4) == 4U;
}

bool test_granular_transport() {
  powder::sim::CAGrid g;
  g.resize(5, 5);
  g.set_occupied(2, 1, true);
  g.set_material(2, 1, 4U);

  powder::sim::GranularRuleConfig cfg{};
  powder::sim::GranularStepInput in{};
  in.tick_index = 0;
  const auto moved = powder::sim::step_granular_deterministic(g, cfg, in);
  return moved == 1U && g.occupied(2, 2);
}

bool test_dem_store_and_contacts() {
  powder::sim::DEMParticleSoA particles;
  const auto a = particles.create(1.0F, 1.0F, 0.5F, 1.0F, 4U);
  const auto b = particles.create(1.8F, 1.0F, 0.5F, 1.0F, 4U);
  if (a == b || particles.alive_count() != 2U) {
    return false;
  }

  powder::sim::SpatialHash2D hash(1.0F);
  hash.build(particles);
  const auto contacts = hash.generate_contacts(particles);
  if (contacts.empty()) {
    return false;
  }

  powder::sim::ContactModelConfig c{};
  const float vx_before = particles.vx[a];
  powder::sim::solve_contacts_penalty_dashpot(particles, contacts, 0.01F, c);
  return std::fabs(particles.vx[a] - vx_before) > 0.0F;
}

bool test_particle_grid_coupling() {
  powder::sim::DEMParticleSoA particles;
  const auto i = particles.create(2.2F, 2.4F, 0.4F, 2.0F, 4U);
  particles.vx[i] = 3.0F;
  particles.vy[i] = -1.0F;

  powder::core::Field2D<float> grid_mass(8, 8, 2);
  powder::core::Field2D<float> grid_mx(8, 8, 2);
  powder::core::Field2D<float> grid_my(8, 8, 2);
  powder::core::Field2D<float> grid_vx(8, 8, 2);
  powder::core::Field2D<float> grid_vy(8, 8, 2);

  grid_vx.at(2, 2) = 1.0F;
  grid_vy.at(2, 2) = 2.0F;

  const auto d = powder::sim::couple_particles_and_grid(particles, grid_mass, grid_mx, grid_my, grid_vx, grid_vy, 0.5F, 2);

  if (d.grid_momentum_x == 0.0F && d.grid_momentum_y == 0.0F) {
    return false;
  }
  return particles.vx[i] != 3.0F || particles.vy[i] != -1.0F;
}

}  // namespace

int main() {
  if (!test_ca_bitpack_and_material()) {
    std::cerr << "ca grid failed\n";
    return 1;
  }
  if (!test_granular_transport()) {
    std::cerr << "granular transport failed\n";
    return 1;
  }
  if (!test_dem_store_and_contacts()) {
    std::cerr << "dem/contact failed\n";
    return 1;
  }
  if (!test_particle_grid_coupling()) {
    std::cerr << "particle-grid coupling failed\n";
    return 1;
  }
  return 0;
}
