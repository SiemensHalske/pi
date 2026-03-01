#pragma once

#include "powder/sim/WorldState.hpp"

#include <cstddef>
#include <cstdint>

namespace powder::render {

class RuntimeShell;

enum class RendererBackend {
  Null,
  OpenGLDebug,
  Vulkan,
};

struct RendererConfig {
  RendererBackend preferred = RendererBackend::OpenGLDebug;
  bool enable_vsync = true;
  std::size_t frame_pacing_queue = 2;
};

struct UploadStats {
  std::size_t scalar_bytes = 0;
  std::size_t velocity_bytes = 0;
  std::size_t species_bytes = 0;
  std::size_t stress_bytes = 0;
  std::size_t total_bytes = 0;
};

struct RenderFrameInput {
  const powder::sim::WorldState* world = nullptr;
  std::uint64_t frame_index = 0;
  double sim_time_seconds = 0.0;
};

class GpuUploadStager {
 public:
  [[nodiscard]] UploadStats stage_world(const powder::sim::WorldState& world);
  [[nodiscard]] std::size_t staging_bytes() const noexcept;

 private:
  std::size_t staging_bytes_ = 0;
};

class Renderer {
 public:
  Renderer() = default;

  [[nodiscard]] bool initialize(const RuntimeShell& shell, RendererConfig config);
  [[nodiscard]] UploadStats upload(const powder::sim::WorldState& world);
  [[nodiscard]] bool render(const RenderFrameInput& input);

  [[nodiscard]] RendererBackend backend() const noexcept;
  [[nodiscard]] bool is_initialized() const noexcept;
  [[nodiscard]] std::uint64_t presented_frames() const noexcept;

 private:
  RendererConfig config_{};
  RendererBackend backend_ = RendererBackend::Null;
  GpuUploadStager upload_stager_{};
  bool initialized_ = false;
  std::uint64_t presented_frames_ = 0;
};

}  // namespace powder::render
