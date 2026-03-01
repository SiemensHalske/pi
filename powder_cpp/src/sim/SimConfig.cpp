#include "powder/sim/SimConfig.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace powder::sim {

namespace {

[[nodiscard]] std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0) {
    s.erase(s.begin());
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.pop_back();
  }
  return s;
}

[[nodiscard]] bool parse_bool(const std::string& raw) {
  if (raw == "true") {
    return true;
  }
  if (raw == "false") {
    return false;
  }
  throw std::runtime_error("invalid bool token: " + raw);
}

[[nodiscard]] std::string strip_comment(const std::string& line) {
  const auto hash = line.find('#');
  if (hash == std::string::npos) {
    return line;
  }
  return line.substr(0, hash);
}

}  // namespace

SimConfigStore::SimConfigStore(SimConfig immutable_base) : base_(immutable_base), overlay_(immutable_base) {
  validate_or_throw(base_);
}

const SimConfig& SimConfigStore::base() const noexcept {
  return base_;
}

SimConfig SimConfigStore::current() const {
  return has_overlay_ ? overlay_ : base_;
}

void SimConfigStore::apply_overlay_toml_file(const std::string& file_path) {
  if (!std::filesystem::exists(file_path)) {
    throw std::runtime_error("overlay config file not found: " + file_path);
  }
  std::ifstream in(file_path);
  if (!in) {
    throw std::runtime_error("unable to open overlay config file: " + file_path);
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  apply_overlay_toml_text(buffer.str());
}

void SimConfigStore::apply_overlay_toml_text(const std::string& toml_text) {
  SimConfig candidate = base_;
  std::string active_section = "";
  std::unordered_set<std::string> seen_keys;

  std::istringstream stream(toml_text);
  std::string line;
  int line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    auto content = trim(strip_comment(line));
    if (content.empty()) {
      continue;
    }

    if (content.front() == '[' && content.back() == ']') {
      active_section = trim(content.substr(1, content.size() - 2));
      if (active_section != "sim" && active_section != "runtime") {
        throw std::runtime_error("unknown TOML section at line " + std::to_string(line_number));
      }
      continue;
    }

    const auto eq = content.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error("invalid TOML key/value at line " + std::to_string(line_number));
    }

    const auto key = trim(content.substr(0, eq));
    const auto value = trim(content.substr(eq + 1));

    if (key.empty()) {
      throw std::runtime_error("empty TOML key at line " + std::to_string(line_number));
    }

    const auto full_key = active_section.empty() ? key : (active_section + "." + key);
    if (seen_keys.contains(full_key)) {
      throw std::runtime_error("duplicate TOML key: " + full_key);
    }
    seen_keys.insert(full_key);

    if (full_key == "sim.schema_version") {
      candidate.schema_version = static_cast<std::uint32_t>(std::stoul(value));
    } else if (full_key == "sim.width") {
      candidate.width = std::stoi(value);
    } else if (full_key == "sim.height") {
      candidate.height = std::stoi(value);
    } else if (full_key == "sim.ghost_cells") {
      candidate.ghost_cells = std::stoi(value);
    } else if (full_key == "sim.dt") {
      candidate.dt = std::stof(value);
    } else if (full_key == "sim.dx") {
      candidate.dx = std::stof(value);
    } else if (full_key == "sim.dy") {
      candidate.dy = std::stof(value);
    } else if (full_key == "sim.cfl_limit") {
      candidate.cfl_limit = std::stof(value);
    } else if (full_key == "runtime.max_threads") {
      candidate.max_threads = std::stoi(value);
    } else if (full_key == "runtime.deterministic") {
      candidate.deterministic = parse_bool(value);
    } else if (full_key == "runtime.headless") {
      candidate.headless = parse_bool(value);
    } else {
      throw std::runtime_error("unknown TOML key: " + full_key);
    }
  }

  validate_or_throw(candidate);
  overlay_ = candidate;
  has_overlay_ = true;
}

void SimConfigStore::clear_overlay() noexcept {
  overlay_ = base_;
  has_overlay_ = false;
}

void SimConfigStore::validate_or_throw(const SimConfig& cfg) const {
  if (cfg.schema_version != 1U) {
    throw std::runtime_error("unsupported schema version");
  }
  if (cfg.width <= 0 || cfg.height <= 0) {
    throw std::runtime_error("invalid dimensions");
  }
  if (cfg.ghost_cells < 1) {
    throw std::runtime_error("ghost_cells must be >= 1");
  }
  if (cfg.max_threads < 1) {
    throw std::runtime_error("max_threads must be >= 1");
  }
  if (cfg.dt <= 0.0F || cfg.dx <= 0.0F || cfg.dy <= 0.0F) {
    throw std::runtime_error("dt/dx/dy must be positive");
  }
  if (cfg.cfl_limit <= 0.0F) {
    throw std::runtime_error("cfl_limit must be positive");
  }
}

}  // namespace powder::sim
