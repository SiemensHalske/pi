#include "powder/render/RuntimeShell.hpp"

#include <chrono>
#include <utility>

namespace powder::render {

namespace {

[[nodiscard]] double now_seconds() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  return static_cast<double>(micros) * 1.0e-6;
}

}  // namespace

RuntimeShell::RuntimeShell(RuntimeShellConfig config) : config_(std::move(config)) {}

bool RuntimeShell::initialize() {
  if (initialized_) {
    return true;
  }

  if (config_.headless) {
    active_type_ = RuntimeShellType::NullHeadless;
  } else {
    active_type_ = RuntimeShellType::NullHeadless;
  }

  last_seconds_ = now_seconds();
  initialized_ = true;
  return true;
}

RuntimeShellFrame RuntimeShell::pump() {
  RuntimeShellFrame frame{};
  const double current = now_seconds();
  frame.now_seconds = current;
  frame.delta_seconds = initialized_ ? (current - last_seconds_) : 0.0;
  if (frame.delta_seconds < 0.0) {
    frame.delta_seconds = 0.0;
  }
  frame.quit_requested = quit_requested_;
  last_seconds_ = current;
  return frame;
}

void RuntimeShell::request_quit() {
  quit_requested_ = true;
}

bool RuntimeShell::is_initialized() const noexcept {
  return initialized_;
}

bool RuntimeShell::is_headless() const noexcept {
  return config_.headless || active_type_ == RuntimeShellType::NullHeadless;
}

RuntimeShellType RuntimeShell::type() const noexcept {
  return active_type_;
}

std::int32_t RuntimeShell::framebuffer_width() const noexcept {
  return config_.width;
}

std::int32_t RuntimeShell::framebuffer_height() const noexcept {
  return config_.height;
}

}  // namespace powder::render
