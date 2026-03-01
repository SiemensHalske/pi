#include "powder/core/Field2D.hpp"
#include "powder/core/KernelRegistry.hpp"
#include "powder/core/MathPrimitives.hpp"
#include "powder/core/Memory.hpp"
#include "powder/core/RuntimePlacement.hpp"

#include <cstdint>
#include <iostream>

namespace {

bool test_aligned_buffer() {
  powder::core::AlignedBuffer<float> buffer(1024, 64);
  const auto addr = reinterpret_cast<std::uintptr_t>(buffer.data());
  return (addr % 64U) == 0U;
}

bool test_field2d_pitch_and_ghost() {
  powder::core::Field2D<float> field(17, 9, 2);
  if (field.pitch() < 21) {
    return false;
  }
  field.at(0, 0) = 3.5F;
  field.at(16, 8) = 7.5F;
  return field.at(0, 0) == 3.5F && field.at(16, 8) == 7.5F;
}

bool test_math_primitives() {
  alignas(64) float grid[4 * 4] = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
  };

  const float clamped = powder::core::clamp_branchless(2.5F, 0.0F, 2.0F);
  if (clamped != 2.0F) {
    return false;
  }

  const float sampled = powder::core::bilinear_sample(grid, 4, 1.5F, 1.5F);
  if (sampled <= 0.0F) {
    return false;
  }

  const float lap = powder::core::laplacian_5pt(grid, 4, 2, 2, 1.0F, 1.0F);
  return lap < 1.0e-3F && lap > -1.0e-3F;
}

bool test_kernel_registry() {
  powder::core::KernelRegistry registry;
  int value = 0;
  registry.register_kernel("unit", powder::core::KernelBackend::Scalar,
                           [](void* ctx) {
                             auto* out = static_cast<int*>(ctx);
                             *out = 42;
                           },
                           [](std::size_t iters) {
                             return static_cast<double>(iters) * 0.1;
                           });

  const auto* entry = registry.find("unit", powder::core::KernelBackend::Scalar);
  if (entry == nullptr) {
    return false;
  }
  entry->fn(&value);
  if (value != 42) {
    return false;
  }
  if (!entry->benchmark) {
    return false;
  }
  return entry->benchmark(10) > 0.0;
}

bool test_runtime_placement() {
  powder::core::AlignedBuffer<float> field(2048, 64);
  powder::core::first_touch_zero(field.data(), field.size(), 2);
  for (std::size_t i = 0; i < field.size(); ++i) {
    if (field[i] != 0.0F) {
      return false;
    }
  }
  (void)powder::core::numa_available_runtime();
  return true;
}

}  // namespace

int main() {
  if (!test_aligned_buffer()) {
    std::cerr << "aligned buffer failed\n";
    return 1;
  }
  if (!test_field2d_pitch_and_ghost()) {
    std::cerr << "field2d failed\n";
    return 1;
  }
  if (!test_math_primitives()) {
    std::cerr << "math primitives failed\n";
    return 1;
  }
  if (!test_kernel_registry()) {
    std::cerr << "kernel registry failed\n";
    return 1;
  }
  if (!test_runtime_placement()) {
    std::cerr << "runtime placement failed\n";
    return 1;
  }

  return 0;
}
