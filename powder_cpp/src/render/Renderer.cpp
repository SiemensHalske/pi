#include "powder/render/Renderer.hpp"

#include "powder/render/RuntimeShell.hpp"

#include <algorithm>

namespace powder::render {

namespace {

template <typename T>
[[nodiscard]] std::size_t bytes_for_field(const powder::core::Field2D<T>& field) {
  return field.size() * sizeof(T);
}

[[nodiscard]] RendererBackend select_backend(RendererBackend preferred, const RuntimeShell& shell) {
  if (shell.is_headless()) {
    return RendererBackend::Null;
  }

  if (preferred == RendererBackend::Vulkan) {
    return RendererBackend::Vulkan;
  }

  if (preferred == RendererBackend::OpenGLDebug) {
    return RendererBackend::OpenGLDebug;
  }

  return RendererBackend::Null;
}

}  // namespace

UploadStats GpuUploadStager::stage_world(const powder::sim::WorldState& world) {
  UploadStats stats{};
  stats.scalar_bytes = bytes_for_field(world.pressure) +
                       bytes_for_field(world.temperature) +
                       bytes_for_field(world.enthalpy) +
                       bytes_for_field(world.density) +
                       bytes_for_field(world.phase_fraction);

  stats.velocity_bytes = bytes_for_field(world.velocity_u.value) +
                         bytes_for_field(world.velocity_v.value);

  stats.species_bytes = bytes_for_field(world.species.o2) +
                        bytes_for_field(world.species.fuel_vapor) +
                        bytes_for_field(world.species.co2) +
                        bytes_for_field(world.species.h2o) +
                        bytes_for_field(world.species.soot);

  stats.stress_bytes = bytes_for_field(world.stress.sigma_xx) +
                       bytes_for_field(world.stress.sigma_yy) +
                       bytes_for_field(world.stress.tau_xy) +
                       bytes_for_field(world.stress.plastic_strain) +
                       bytes_for_field(world.stress.damage);

  stats.total_bytes = stats.scalar_bytes + stats.velocity_bytes + stats.species_bytes + stats.stress_bytes;

  staging_bytes_ = std::max(staging_bytes_, stats.total_bytes);
  return stats;
}

std::size_t GpuUploadStager::staging_bytes() const noexcept {
  return staging_bytes_;
}

bool Renderer::initialize(const RuntimeShell& shell, RendererConfig config) {
  config_ = std::move(config);
  backend_ = select_backend(config_.preferred, shell);
  initialized_ = true;
  presented_frames_ = 0;
  return true;
}

UploadStats Renderer::upload(const powder::sim::WorldState& world) {
  return upload_stager_.stage_world(world);
}

bool Renderer::render(const RenderFrameInput& input) {
  if (!initialized_ || input.world == nullptr) {
    return false;
  }

  if (backend_ != RendererBackend::Null) {
    (void)upload(*input.world);
  }

  ++presented_frames_;
  return true;
}

RendererBackend Renderer::backend() const noexcept {
  return backend_;
}

bool Renderer::is_initialized() const noexcept {
  return initialized_;
}

std::uint64_t Renderer::presented_frames() const noexcept {
  return presented_frames_;
}

}  // namespace powder::render
