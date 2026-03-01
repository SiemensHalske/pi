#pragma once

#include <cstdint>
#include <string>

namespace powder::render {

enum class RuntimeShellType {
  NullHeadless,
  GLFW,
};

struct RuntimeShellConfig {
  RuntimeShellType preferred = RuntimeShellType::GLFW;
  bool headless = false;
  std::int32_t width = 1280;
  std::int32_t height = 720;
  std::string title = "PowderCPP";
  bool enable_vsync = true;
};

struct RuntimeShellFrame {
  double now_seconds = 0.0;
  double delta_seconds = 0.0;
  bool quit_requested = false;
};

class RuntimeShell {
 public:
  RuntimeShell() = default;
  explicit RuntimeShell(RuntimeShellConfig config);

  [[nodiscard]] bool initialize();
  [[nodiscard]] RuntimeShellFrame pump();
  void request_quit();

  [[nodiscard]] bool is_initialized() const noexcept;
  [[nodiscard]] bool is_headless() const noexcept;
  [[nodiscard]] RuntimeShellType type() const noexcept;
  [[nodiscard]] std::int32_t framebuffer_width() const noexcept;
  [[nodiscard]] std::int32_t framebuffer_height() const noexcept;

 private:
  RuntimeShellConfig config_{};
  bool initialized_ = false;
  bool quit_requested_ = false;
  double last_seconds_ = 0.0;
  RuntimeShellType active_type_ = RuntimeShellType::NullHeadless;
};

}  // namespace powder::render
