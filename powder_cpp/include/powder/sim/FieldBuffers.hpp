#pragma once

#include "powder/core/Field2D.hpp"

#include <array>
#include <cstddef>

namespace powder::sim {

enum class BufferCount : std::size_t {
  Double = 2,
  Triple = 3,
};

template <typename T, BufferCount N>
class FieldBuffer2D {
 public:
  static constexpr std::size_t kCount = static_cast<std::size_t>(N);

  void resize(std::size_t width, std::size_t height, std::size_t ghost) {
    for (auto& buf : buffers_) {
      buf.resize(width, height, ghost);
    }
    read_index_ = 0;
    write_index_ = 1 % kCount;
  }

  [[nodiscard]] powder::core::Field2D<T>& read() noexcept { return buffers_[read_index_]; }
  [[nodiscard]] const powder::core::Field2D<T>& read() const noexcept { return buffers_[read_index_]; }
  [[nodiscard]] powder::core::Field2D<T>& write() noexcept { return buffers_[write_index_]; }
  [[nodiscard]] const powder::core::Field2D<T>& write() const noexcept { return buffers_[write_index_]; }

  void advance() noexcept {
    read_index_ = write_index_;
    write_index_ = (write_index_ + 1U) % kCount;
  }

 private:
  std::array<powder::core::Field2D<T>, kCount> buffers_{};
  std::size_t read_index_ = 0;
  std::size_t write_index_ = 1;
};

template <BufferCount N>
class MacVelocityBuffers {
 public:
  static constexpr std::size_t kCount = static_cast<std::size_t>(N);

  void resize(std::size_t width, std::size_t height, std::size_t ghost) {
    for (std::size_t i = 0; i < kCount; ++i) {
      u_[i].resize(width, height, ghost);
      v_[i].resize(width, height, ghost);
    }
    read_index_ = 0;
    write_index_ = 1 % kCount;
  }

  [[nodiscard]] powder::core::FaceFieldU& read_u() noexcept { return u_[read_index_]; }
  [[nodiscard]] powder::core::FaceFieldV& read_v() noexcept { return v_[read_index_]; }
  [[nodiscard]] powder::core::FaceFieldU& write_u() noexcept { return u_[write_index_]; }
  [[nodiscard]] powder::core::FaceFieldV& write_v() noexcept { return v_[write_index_]; }

  void advance() noexcept {
    read_index_ = write_index_;
    write_index_ = (write_index_ + 1U) % kCount;
  }

 private:
  std::array<powder::core::FaceFieldU, kCount> u_{};
  std::array<powder::core::FaceFieldV, kCount> v_{};
  std::size_t read_index_ = 0;
  std::size_t write_index_ = 1;
};

}  // namespace powder::sim
