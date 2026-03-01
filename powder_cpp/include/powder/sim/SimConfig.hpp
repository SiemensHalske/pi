#pragma once

#include <cstdint>
#include <string>

namespace powder::sim {

struct SimConfig {
  std::uint32_t schema_version = 1;
  std::int32_t width = 512;
  std::int32_t height = 288;
  std::int32_t ghost_cells = 2;
  std::int32_t max_threads = 1;
  float dt = 1.0F / 60.0F;
  float dx = 0.05F;
  float dy = 0.05F;
  float cfl_limit = 0.5F;
  bool deterministic = true;
  bool headless = false;
};

class SimConfigStore {
 public:
  explicit SimConfigStore(SimConfig immutable_base);

  [[nodiscard]] const SimConfig& base() const noexcept;
  [[nodiscard]] SimConfig current() const;

  void apply_overlay_toml_file(const std::string& file_path);
  void apply_overlay_toml_text(const std::string& toml_text);
  void clear_overlay() noexcept;

 private:
  void validate_or_throw(const SimConfig& cfg) const;

  SimConfig base_;
  SimConfig overlay_;
  bool has_overlay_ = false;
};

}  // namespace powder::sim
