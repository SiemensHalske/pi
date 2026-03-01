#pragma once

#include "powder/sim/WorldState.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace powder::script {

enum class ScriptStage {
  PreStep,
  PostAdvection,
  PostProjection,
  PreRender,
  PostRender,
};

enum class ScriptAccess {
  ReadOnly,
  ReadWrite,
};

struct ScriptConfig {
  std::size_t memory_cap_bytes = 1U << 20U;
  double execution_budget_ms = 1.0;
};

struct ScriptHookContext {
  ScriptStage stage = ScriptStage::PreStep;
  ScriptAccess access = ScriptAccess::ReadOnly;
  std::uint64_t tick = 0;
  double dt = 0.0;
  const powder::sim::WorldState* world_ro = nullptr;
  powder::sim::WorldState* world_rw = nullptr;
};

struct ScriptResult {
  bool ok = false;
  std::string message;
  double elapsed_ms = 0.0;
};

using ScriptHookFn = std::function<ScriptResult(const ScriptHookContext&)>;

class LuaSandbox {
 public:
  LuaSandbox() = default;
  explicit LuaSandbox(ScriptConfig config);

  void configure(ScriptConfig config);
  [[nodiscard]] const ScriptConfig& config() const noexcept;

  [[nodiscard]] bool register_hook(std::string name, ScriptStage stage, ScriptAccess access, ScriptHookFn fn);
  [[nodiscard]] bool register_text_script(std::string name, ScriptStage stage, ScriptAccess access, std::string source);

  [[nodiscard]] std::vector<ScriptResult> invoke_stage(const ScriptHookContext& context) const;

  [[nodiscard]] std::size_t memory_used_bytes() const noexcept;
  [[nodiscard]] std::size_t hook_count() const noexcept;

 private:
  struct HookEntry {
    std::string name;
    ScriptStage stage = ScriptStage::PreStep;
    ScriptAccess access = ScriptAccess::ReadOnly;
    ScriptHookFn fn;
    std::size_t memory_bytes = 0;
  };

  ScriptResult run_guarded(const HookEntry& entry, const ScriptHookContext& context) const;

  ScriptConfig config_{};
  std::vector<HookEntry> hooks_;
  std::size_t memory_used_bytes_ = 0;
};

}  // namespace powder::script
