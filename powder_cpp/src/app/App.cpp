#include "powder/app/Cli.hpp"
#include "powder/app/Console.hpp"

#include "powder/core/BuildInfo.hpp"
#include "powder/core/CpuFeatures.hpp"
#include "powder/core/DeterministicBootstrap.hpp"
#include "powder/render/Renderer.hpp"
#include "powder/render/RuntimeShell.hpp"
#include "powder/script/LuaSandbox.hpp"
#include "powder/sim/WorldState.hpp"

#include <iostream>

namespace powder::app {

int run(const CliOptions& options) {
  powder::core::BootstrapOptions bootstrap_options{};
  bootstrap_options.profile = options.profile;
  bootstrap_options.headless = options.headless;
  bootstrap_options.benchmark = options.benchmark;
  bootstrap_options.requested_threads = options.threads;

  const auto state = powder::core::build_deterministic_state(bootstrap_options);
  const auto cpu = powder::core::detect_cpu_features();
  const auto build = powder::core::query_build_info();

  std::cout << "PowderCPP bootstrap" << '\n';
  std::cout << " project=" << build.project_name << " build=" << build.build_type
            << " openmp=" << (build.with_openmp ? "on" : "off") << '\n';
  std::cout << " profile=" << bootstrap_options.profile
            << " headless=" << (bootstrap_options.headless ? "true" : "false")
            << " benchmark=" << (bootstrap_options.benchmark ? "true" : "false")
            << '\n';
  std::cout << " threads=" << state.active_threads
            << " dt=" << state.dt
            << " grid=" << state.grid_width << "x" << state.grid_height
            << " seed=" << state.seed << '\n';
  std::cout << " cpu_features=" << cpu.to_string() << '\n';
  std::cout << " substeps=";
  for (std::size_t i = 0; i < state.substep_order.size(); ++i) {
    std::cout << state.substep_order[i];
    if (i + 1 < state.substep_order.size()) {
      std::cout << "->";
    }
  }
  std::cout << '\n';

  powder::render::RuntimeShell shell({
      powder::render::RuntimeShellType::GLFW,
      options.headless,
      state.grid_width,
      state.grid_height,
      "PowderCPP",
      true,
  });
  (void)shell.initialize();

  powder::render::Renderer renderer{};
  (void)renderer.initialize(shell, powder::render::RendererConfig{});

  auto world = powder::sim::create_world_state(static_cast<std::size_t>(state.grid_width),
                                                static_cast<std::size_t>(state.grid_height),
                                                2);
  world.temperature.at(0, 0) = 300.0F;

  powder::script::LuaSandbox sandbox({
      1U << 20U,
      1.0,
  });
  (void)sandbox.register_text_script("bootstrap_pre_render",
                                     powder::script::ScriptStage::PreRender,
                                     powder::script::ScriptAccess::ReadWrite,
                                     "add temperature 0 0 5");

  const powder::script::ScriptHookContext script_context{
      powder::script::ScriptStage::PreRender,
      powder::script::ScriptAccess::ReadWrite,
      0,
      state.dt,
      &world,
      &world,
  };
  const auto script_results = sandbox.invoke_stage(script_context);

  const auto upload = renderer.upload(world);
  (void)renderer.render({&world, 0, 0.0});

  NativeConsole console{};
  console.register_command("set_threads", 1, 1,
                           [](std::span<const std::string_view> args) {
                             if (!NativeConsole::parse_i32(args[0]).has_value()) {
                               return ConsoleCommandResult{false, "invalid integer"};
                             }
                             return ConsoleCommandResult{true, "ok"};
                           });
  const auto console_result = console.execute("set_threads 4");

  std::cout << " runtime_shell=" << (shell.is_headless() ? "headless" : "windowed")
            << " renderer_backend=" << static_cast<int>(renderer.backend())
            << " upload_bytes=" << upload.total_bytes
            << " script_hooks=" << sandbox.hook_count()
            << " script_results=" << script_results.size()
            << " console_ok=" << (console_result.ok ? "true" : "false")
            << '\n';

  return 0;
}

}  // namespace powder::app
