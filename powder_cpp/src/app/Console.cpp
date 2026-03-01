#include "powder/app/Console.hpp"

#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace powder::app {

namespace {

[[nodiscard]] std::vector<std::string_view> split_tokens(const std::string& line) {
  std::vector<std::string_view> tokens;
  std::size_t begin = 0;
  while (begin < line.size()) {
    while (begin < line.size() && line[begin] == ' ') {
      ++begin;
    }
    if (begin >= line.size()) {
      break;
    }
    std::size_t end = begin;
    while (end < line.size() && line[end] != ' ') {
      ++end;
    }
    tokens.emplace_back(line.data() + begin, end - begin);
    begin = end;
  }
  return tokens;
}

}  // namespace

void NativeConsole::register_command(std::string name, std::size_t min_arity, std::size_t max_arity, ConsoleCommandFn fn) {
  if (name.empty()) {
    throw std::runtime_error("command name must not be empty");
  }
  if (min_arity > max_arity) {
    throw std::runtime_error("invalid arity bounds");
  }

  commands_[std::move(name)] = CommandEntry{min_arity, max_arity, std::move(fn)};
}

ConsoleCommandResult NativeConsole::execute(std::string_view line_view) {
  std::string line(line_view);
  history_.push_back(line);

  const auto tokens = split_tokens(line);
  if (tokens.empty()) {
    return ConsoleCommandResult{true, ""};
  }

  const std::string command_name(tokens.front());
  const auto it = commands_.find(command_name);
  if (it == commands_.end()) {
    return ConsoleCommandResult{false, "unknown command: " + command_name};
  }

  const auto argc = tokens.size() - 1;
  if (argc < it->second.min_arity || argc > it->second.max_arity) {
    return ConsoleCommandResult{false, "arity mismatch for command: " + command_name};
  }

  std::vector<std::string_view> args;
  args.reserve(argc);
  for (std::size_t index = 1; index < tokens.size(); ++index) {
    args.push_back(tokens[index]);
  }

  return it->second.fn(std::span<const std::string_view>(args.data(), args.size()));
}

const std::vector<std::string>& NativeConsole::history() const noexcept {
  return history_;
}

bool NativeConsole::save_history(const std::string& path) const {
  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    return false;
  }
  for (const auto& line : history_) {
    file << line << '\n';
  }
  return true;
}

bool NativeConsole::load_history(const std::string& path, std::size_t max_entries) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  history_.clear();
  std::string line;
  while (std::getline(file, line)) {
    if (history_.size() >= max_entries) {
      break;
    }
    history_.push_back(line);
  }
  return true;
}

std::optional<std::int32_t> NativeConsole::parse_i32(std::string_view token) {
  std::int32_t value = 0;
  const auto* begin = token.data();
  const auto* end = token.data() + token.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    return std::nullopt;
  }
  return value;
}

std::optional<float> NativeConsole::parse_f32(std::string_view token) {
  std::string text(token);
  std::istringstream stream(text);
  float value = 0.0F;
  stream >> value;
  if (stream.fail()) {
    return std::nullopt;
  }
  char remaining = '\0';
  if (stream >> remaining) {
    return std::nullopt;
  }
  return value;
}

}  // namespace powder::app
