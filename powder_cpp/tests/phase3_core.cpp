#include "powder/sim/Checkpoint.hpp"
#include "powder/sim/FieldBuffers.hpp"
#include "powder/sim/MaterialDB.hpp"
#include "powder/sim/SimConfig.hpp"
#include "powder/sim/SubstepScheduler.hpp"
#include "powder/sim/WorldState.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool test_world_state_layout() {
  auto world = powder::sim::create_world_state(32, 20, 2);
  world.pressure.at(1, 1) = 42.0F;
  return world.pressure.at(1, 1) == 42.0F && world.velocity_u.value.width() == 33 && world.velocity_v.value.height() == 21;
}

bool test_material_table() {
  const auto& steel = powder::sim::material_properties(powder::sim::MaterialId::Steel);
  if (steel.yield_limit < 1.0e8F) {
    return false;
  }
  const auto id = powder::sim::material_id_from_name("lava");
  return id == powder::sim::MaterialId::Lava;
}

bool test_sim_config_overlay() {
  powder::sim::SimConfig base{};
  powder::sim::SimConfigStore store(base);

  const std::string toml = R"(
[sim]
width = 640
height = 360
dt = 0.01

[runtime]
max_threads = 4
deterministic = true
headless = true
)";

  store.apply_overlay_toml_text(toml);
  const auto cfg = store.current();
  return cfg.width == 640 && cfg.height == 360 && cfg.max_threads == 4 && cfg.headless;
}

bool test_scheduler_order() {
  powder::sim::SubstepScheduler scheduler;
  std::vector<std::string> order;

  for (const auto step : scheduler.order()) {
    scheduler.set_callback(step, [&order, step]() { order.emplace_back(powder::sim::substep_label(step)); });
  }

  scheduler.execute_once();
  if (order.size() != 9U) {
    return false;
  }
  return order.front() == "forces" && order.back() == "boundaries";
}

bool test_field_buffers() {
  powder::sim::FieldBuffer2D<float, powder::sim::BufferCount::Double> buffers;
  buffers.resize(16, 16, 2);
  buffers.read().at(0, 0) = 1.0F;
  buffers.write().at(0, 0) = 2.0F;
  buffers.advance();
  return buffers.read().at(0, 0) == 2.0F;
}

bool test_checkpoint_roundtrip() {
  auto world = powder::sim::create_world_state(12, 10, 2);
  world.temperature.at(2, 3) = 1234.5F;
  world.species.o2.at(1, 1) = 0.21F;
  world.debris.x = {1.0F, 2.0F};
  world.debris.y = {3.0F, 4.0F};
  world.debris.vx = {0.1F, 0.2F};
  world.debris.vy = {0.3F, 0.4F};
  world.debris.radius = {0.5F, 0.6F};
  world.debris.mass = {10.0F, 11.0F};
  world.debris.material_id = {1U, 2U};

  const auto path = std::filesystem::temp_directory_path() / "powdercpp_phase3.chk";
  powder::sim::save_checkpoint_binary(path.string(), world);
  const auto loaded = powder::sim::load_checkpoint_binary(path.string());
  std::filesystem::remove(path);

  return loaded.temperature.at(2, 3) == 1234.5F && loaded.species.o2.at(1, 1) == 0.21F && loaded.debris.size() == 2U;
}

}  // namespace

int main() {
  if (!test_world_state_layout()) {
    std::cerr << "world state test failed\n";
    return 1;
  }
  if (!test_material_table()) {
    std::cerr << "material table test failed\n";
    return 1;
  }
  if (!test_sim_config_overlay()) {
    std::cerr << "sim config test failed\n";
    return 1;
  }
  if (!test_scheduler_order()) {
    std::cerr << "scheduler test failed\n";
    return 1;
  }
  if (!test_field_buffers()) {
    std::cerr << "field buffer test failed\n";
    return 1;
  }
  if (!test_checkpoint_roundtrip()) {
    std::cerr << "checkpoint test failed\n";
    return 1;
  }

  return 0;
}
