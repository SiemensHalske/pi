#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace powder::app {

struct ConsoleCommandResult {
  bool ok = false;
  std::string message;
};

using ConsoleCommandFn = std::function<ConsoleCommandResult(std::span<const std::string_view>)>;

class NativeConsole {
 public:
  NativeConsole() = default;

  void register_command(std::string name, std::size_t min_arity, std::size_t max_arity, ConsoleCommandFn fn);
  [[nodiscard]] ConsoleCommandResult execute(std::string_view line);

  [[nodiscard]] const std::vector<std::string>& history() const noexcept;
  [[nodiscard]] bool save_history(const std::string& path) const;
  [[nodiscard]] bool load_history(const std::string& path, std::size_t max_entries = 1024);

  [[nodiscard]] static std::optional<std::int32_t> parse_i32(std::string_view token);
  [[nodiscard]] static std::optional<float> parse_f32(std::string_view token);

 private:
  struct CommandEntry {
    std::size_t min_arity = 0;
    std::size_t max_arity = 0;
    ConsoleCommandFn fn;
  };

  std::unordered_map<std::string, CommandEntry> commands_;
  std::vector<std::string> history_;
};

}  // namespace powder::app
