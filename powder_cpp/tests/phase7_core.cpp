#include "powder/app/Console.hpp"
#include "powder/render/Renderer.hpp"
#include "powder/render/RuntimeShell.hpp"
#include "powder/script/LuaSandbox.hpp"
#include "powder/sim/WorldState.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>

namespace {

bool test_runtime_shell_headless() {
  powder::render::RuntimeShell shell({
      powder::render::RuntimeShellType::GLFW,
      true,
      640,
      360,
      "phase7",
      false,
  });
  if (!shell.initialize()) {
    return false;
  }
  const auto frame0 = shell.pump();
  const auto frame1 = shell.pump();
  if (!shell.is_headless()) {
    return false;
  }
  if (frame1.now_seconds < frame0.now_seconds) {
    return false;
  }
  if (frame1.delta_seconds < 0.0) {
    return false;
  }
  return true;
}

bool test_renderer_upload_and_present() {
  powder::render::RuntimeShell shell({
      powder::render::RuntimeShellType::GLFW,
      true,
      256,
      128,
      "phase7",
      false,
  });
  if (!shell.initialize()) {
    return false;
  }

  powder::render::Renderer renderer{};
  if (!renderer.initialize(shell, {})) {
    return false;
  }

  auto world = powder::sim::create_world_state(32, 24, 2);
  for (std::size_t y = 0; y < world.height; ++y) {
    for (std::size_t x = 0; x < world.width; ++x) {
      world.pressure.at(x, y) = static_cast<float>(x + y);
      world.temperature.at(x, y) = 295.0F + static_cast<float>(x);
      world.density.at(x, y) = 1.0F;
    }
  }

  const auto upload = renderer.upload(world);
  if (upload.total_bytes == 0U || upload.scalar_bytes == 0U || upload.velocity_bytes == 0U) {
    return false;
  }

  if (!renderer.render({&world, 1, 1.0 / 60.0})) {
    return false;
  }
  return renderer.presented_frames() == 1U;
}

bool test_console_registry_and_history() {
  powder::app::NativeConsole console{};
  console.register_command("add", 2, 2,
                           [](std::span<const std::string_view> args) {
                             const auto a = powder::app::NativeConsole::parse_i32(args[0]);
                             const auto b = powder::app::NativeConsole::parse_i32(args[1]);
                             if (!a.has_value() || !b.has_value()) {
                               return powder::app::ConsoleCommandResult{false, "parse error"};
                             }
                             return powder::app::ConsoleCommandResult{true, std::to_string(*a + *b)};
                           });

  const auto result = console.execute("add 2 5");
  if (!result.ok || result.message != "7") {
    return false;
  }

  const auto path = std::filesystem::temp_directory_path() / "powder_phase7_console_history.txt";
  if (!console.save_history(path.string())) {
    return false;
  }

  powder::app::NativeConsole loaded{};
  if (!loaded.load_history(path.string())) {
    return false;
  }
  return !loaded.history().empty();
}

bool test_lua_sandbox_stage_hooks() {
  auto world = powder::sim::create_world_state(12, 10, 2);
  world.temperature.at(0, 0) = 300.0F;

  powder::script::LuaSandbox sandbox({
      4096,
      2.0,
  });

  if (!sandbox.register_text_script("mutator",
                                    powder::script::ScriptStage::PreRender,
                                    powder::script::ScriptAccess::ReadWrite,
                                    "add temperature 0 0 20")) {
    return false;
  }

  if (!sandbox.register_text_script("readonly_fail",
                                    powder::script::ScriptStage::PreRender,
                                    powder::script::ScriptAccess::ReadOnly,
                                    "add temperature 0 0 1")) {
    return false;
  }

  const powder::script::ScriptHookContext context{
      powder::script::ScriptStage::PreRender,
      powder::script::ScriptAccess::ReadWrite,
      1,
      1.0 / 120.0,
      &world,
      &world,
  };

  const auto results = sandbox.invoke_stage(context);
  if (results.size() != 2U) {
    return false;
  }

  if (!results[0].ok) {
    return false;
  }
  if (results[1].ok) {
    return false;
  }

  return std::fabs(world.temperature.at(0, 0) - 320.0F) < 1.0e-4F;
}

}  // namespace

int main() {
  if (!test_runtime_shell_headless()) {
    std::cerr << "phase7 shell test failed\n";
    return 1;
  }
  if (!test_renderer_upload_and_present()) {
    std::cerr << "phase7 renderer test failed\n";
    return 1;
  }
  if (!test_console_registry_and_history()) {
    std::cerr << "phase7 console test failed\n";
    return 1;
  }
  if (!test_lua_sandbox_stage_hooks()) {
    std::cerr << "phase7 script test failed\n";
    return 1;
  }
  return 0;
}
