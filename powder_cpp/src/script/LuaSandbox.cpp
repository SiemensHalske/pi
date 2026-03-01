#include "powder/script/LuaSandbox.hpp"

#include <chrono>
#include <sstream>
#include <string>
#include <utility>

namespace powder::script {

namespace {

using clock = std::chrono::steady_clock;

[[nodiscard]] std::vector<std::string_view> tokenize(const std::string& source) {
  std::vector<std::string_view> tokens;
  std::size_t begin = 0;
  while (begin < source.size()) {
    while (begin < source.size() && source[begin] == ' ') {
      ++begin;
    }
    if (begin >= source.size()) {
      break;
    }
    std::size_t end = begin;
    while (end < source.size() && source[end] != ' ') {
      ++end;
    }
    tokens.emplace_back(source.data() + begin, end - begin);
    begin = end;
  }
  return tokens;
}

[[nodiscard]] bool parse_uindex(std::string_view token, std::size_t* out) {
  if (out == nullptr || token.empty()) {
    return false;
  }
  std::size_t value = 0;
  for (const char c : token) {
    if (c < '0' || c > '9') {
      return false;
    }
    value = value * 10U + static_cast<std::size_t>(c - '0');
  }
  *out = value;
  return true;
}

[[nodiscard]] bool parse_f32(std::string_view token, float* out) {
  if (out == nullptr) {
    return false;
  }
  std::string text(token);
  std::istringstream stream(text);
  float value = 0.0F;
  stream >> value;
  if (stream.fail()) {
    return false;
  }
  char remain = '\0';
  if (stream >> remain) {
    return false;
  }
  *out = value;
  return true;
}

[[nodiscard]] ScriptHookFn compile_text_script(std::string source, ScriptAccess access) {
  return [source = std::move(source), access](const ScriptHookContext& context) -> ScriptResult {
    const auto started = clock::now();
    const auto tokens = tokenize(source);
    if (tokens.empty()) {
      return ScriptResult{true, "", 0.0};
    }

    const auto elapsed_ms_now = [started]() {
      const auto elapsed = clock::now() - started;
      return std::chrono::duration<double, std::milli>(elapsed).count();
    };

    if (tokens[0] == "noop") {
      return ScriptResult{true, "", elapsed_ms_now()};
    }

    if (context.world_ro == nullptr) {
      return ScriptResult{false, "missing world context", elapsed_ms_now()};
    }

    if ((tokens[0] == "set" || tokens[0] == "add" || tokens[0] == "scale") && access == ScriptAccess::ReadOnly) {
      return ScriptResult{false, "mutation not allowed in read-only hook", elapsed_ms_now()};
    }

    if ((tokens[0] == "set" || tokens[0] == "add" || tokens[0] == "scale") && context.world_rw == nullptr) {
      return ScriptResult{false, "mutable world missing for write hook", elapsed_ms_now()};
    }

    if (tokens[0] == "set") {
      if (tokens.size() != 5U) {
        return ScriptResult{false, "set requires: set <field> <x> <y> <value>", elapsed_ms_now()};
      }
      std::size_t x = 0;
      std::size_t y = 0;
      float value = 0.0F;
      if (!parse_uindex(tokens[2], &x) || !parse_uindex(tokens[3], &y) || !parse_f32(tokens[4], &value)) {
        return ScriptResult{false, "invalid set arguments", elapsed_ms_now()};
      }
      if (x >= context.world_rw->width || y >= context.world_rw->height) {
        return ScriptResult{false, "set index out of bounds", elapsed_ms_now()};
      }
      if (tokens[1] == "temperature") {
        context.world_rw->temperature.at(x, y) = value;
      } else if (tokens[1] == "pressure") {
        context.world_rw->pressure.at(x, y) = value;
      } else if (tokens[1] == "density") {
        context.world_rw->density.at(x, y) = value;
      } else {
        return ScriptResult{false, "unknown field", elapsed_ms_now()};
      }
      return ScriptResult{true, "", elapsed_ms_now()};
    }

    if (tokens[0] == "add") {
      if (tokens.size() != 5U) {
        return ScriptResult{false, "add requires: add <field> <x> <y> <delta>", elapsed_ms_now()};
      }
      std::size_t x = 0;
      std::size_t y = 0;
      float delta = 0.0F;
      if (!parse_uindex(tokens[2], &x) || !parse_uindex(tokens[3], &y) || !parse_f32(tokens[4], &delta)) {
        return ScriptResult{false, "invalid add arguments", elapsed_ms_now()};
      }
      if (x >= context.world_rw->width || y >= context.world_rw->height) {
        return ScriptResult{false, "add index out of bounds", elapsed_ms_now()};
      }
      if (tokens[1] == "temperature") {
        context.world_rw->temperature.at(x, y) += delta;
      } else if (tokens[1] == "pressure") {
        context.world_rw->pressure.at(x, y) += delta;
      } else if (tokens[1] == "density") {
        context.world_rw->density.at(x, y) += delta;
      } else {
        return ScriptResult{false, "unknown field", elapsed_ms_now()};
      }
      return ScriptResult{true, "", elapsed_ms_now()};
    }

    if (tokens[0] == "scale") {
      if (tokens.size() != 3U) {
        return ScriptResult{false, "scale requires: scale <field> <factor>", elapsed_ms_now()};
      }
      float factor = 0.0F;
      if (!parse_f32(tokens[2], &factor)) {
        return ScriptResult{false, "invalid scale factor", elapsed_ms_now()};
      }

      for (std::size_t y = 0; y < context.world_rw->height; ++y) {
        for (std::size_t x = 0; x < context.world_rw->width; ++x) {
          if (tokens[1] == "temperature") {
            context.world_rw->temperature.at(x, y) *= factor;
          } else if (tokens[1] == "pressure") {
            context.world_rw->pressure.at(x, y) *= factor;
          } else if (tokens[1] == "density") {
            context.world_rw->density.at(x, y) *= factor;
          } else {
            return ScriptResult{false, "unknown field", elapsed_ms_now()};
          }
        }
      }
      return ScriptResult{true, "", elapsed_ms_now()};
    }

    return ScriptResult{false, "unsupported script opcode", elapsed_ms_now()};
  };
}

}  // namespace

LuaSandbox::LuaSandbox(ScriptConfig config) : config_(config) {}

void LuaSandbox::configure(ScriptConfig config) {
  config_ = config;
}

const ScriptConfig& LuaSandbox::config() const noexcept {
  return config_;
}

bool LuaSandbox::register_hook(std::string name, ScriptStage stage, ScriptAccess access, ScriptHookFn fn) {
  const std::size_t memory_cost = name.size();
  if (memory_used_bytes_ + memory_cost > config_.memory_cap_bytes) {
    return false;
  }

  hooks_.push_back(HookEntry{std::move(name), stage, access, std::move(fn), memory_cost});
  memory_used_bytes_ += memory_cost;
  return true;
}

bool LuaSandbox::register_text_script(std::string name, ScriptStage stage, ScriptAccess access, std::string source) {
  const std::size_t memory_cost = name.size() + source.size();
  if (memory_used_bytes_ + memory_cost > config_.memory_cap_bytes) {
    return false;
  }

  auto fn = compile_text_script(std::move(source), access);
  hooks_.push_back(HookEntry{std::move(name), stage, access, std::move(fn), memory_cost});
  memory_used_bytes_ += memory_cost;
  return true;
}

std::vector<ScriptResult> LuaSandbox::invoke_stage(const ScriptHookContext& context) const {
  std::vector<ScriptResult> results;
  for (const auto& hook : hooks_) {
    if (hook.stage != context.stage) {
      continue;
    }
    results.push_back(run_guarded(hook, context));
  }
  return results;
}

std::size_t LuaSandbox::memory_used_bytes() const noexcept {
  return memory_used_bytes_;
}

std::size_t LuaSandbox::hook_count() const noexcept {
  return hooks_.size();
}

ScriptResult LuaSandbox::run_guarded(const HookEntry& entry, const ScriptHookContext& context) const {
  ScriptHookContext guarded_context = context;
  guarded_context.access = entry.access;

  const auto started = clock::now();
  ScriptResult result = entry.fn(guarded_context);
  const auto elapsed = std::chrono::duration<double, std::milli>(clock::now() - started).count();
  result.elapsed_ms = elapsed;

  if (elapsed > config_.execution_budget_ms) {
    return ScriptResult{false, "execution budget exceeded", elapsed};
  }

  return result;
}

}  // namespace powder::script
