#pragma once

#include "powder/core/Memory.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace powder::core {

[[nodiscard]] constexpr std::size_t lane_bytes_for_simd() {
#if defined(__AVX512F__)
  return 64;
#elif defined(__AVX2__)
  return 32;
#else
  return 16;
#endif
}

[[nodiscard]] inline std::size_t compute_padded_pitch(std::size_t width, std::size_t ghost_cells,
                                                      std::size_t element_bytes,
                                                      std::size_t alignment_bytes = kCacheLineBytes) {
  const auto logical = width + (2U * ghost_cells);
  const auto row_bytes = logical * element_bytes;
  const auto padded_bytes = align_up(row_bytes, alignment_bytes);
  return padded_bytes / element_bytes;
}

template <typename T>
class Field2D {
 public:
  Field2D() = default;

  Field2D(std::size_t width, std::size_t height, std::size_t ghost_cells = 2,
          std::size_t alignment = kCacheLineBytes) {
    resize(width, height, ghost_cells, alignment);
  }

  void resize(std::size_t width, std::size_t height, std::size_t ghost_cells = 2,
              std::size_t alignment = kCacheLineBytes) {
    if (width == 0 || height == 0) {
      throw std::runtime_error("Field2D dimensions must be non-zero");
    }
    width_ = width;
    height_ = height;
    ghost_ = ghost_cells;
    pitch_ = compute_padded_pitch(width_, ghost_, sizeof(T), alignment);
    total_rows_ = height_ + (2U * ghost_);
    storage_.resize(pitch_ * total_rows_, alignment);
  }

  [[nodiscard]] std::size_t width() const noexcept { return width_; }
  [[nodiscard]] std::size_t height() const noexcept { return height_; }
  [[nodiscard]] std::size_t ghost() const noexcept { return ghost_; }
  [[nodiscard]] std::size_t pitch() const noexcept { return pitch_; }
  [[nodiscard]] std::size_t total_rows() const noexcept { return total_rows_; }

  [[nodiscard]] std::size_t index(std::size_t x, std::size_t y) const noexcept {
    return (y + ghost_) * pitch_ + (x + ghost_);
  }

  [[nodiscard]] T& at(std::size_t x, std::size_t y) noexcept {
    return storage_[index(x, y)];
  }

  [[nodiscard]] const T& at(std::size_t x, std::size_t y) const noexcept {
    return storage_[index(x, y)];
  }

  [[nodiscard]] T* raw() noexcept { return storage_.data(); }
  [[nodiscard]] const T* raw() const noexcept { return storage_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }

 private:
  std::size_t width_ = 0;
  std::size_t height_ = 0;
  std::size_t ghost_ = 0;
  std::size_t pitch_ = 0;
  std::size_t total_rows_ = 0;
  AlignedBuffer<T> storage_;
};

struct FaceFieldU {
  Field2D<float> value;

  void resize(std::size_t cell_width, std::size_t cell_height, std::size_t ghost_cells = 2) {
    value.resize(cell_width + 1U, cell_height, ghost_cells);
  }
};

struct FaceFieldV {
  Field2D<float> value;

  void resize(std::size_t cell_width, std::size_t cell_height, std::size_t ghost_cells = 2) {
    value.resize(cell_width, cell_height + 1U, ghost_cells);
  }
};

}  // namespace powder::core
