#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace powder::core {

constexpr std::size_t kCacheLineBytes = 64;

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + (alignment - 1U)) & ~(alignment - 1U);
}

[[nodiscard]] void* aligned_allocate_bytes(std::size_t byte_count, std::size_t alignment);
void aligned_deallocate_bytes(void* ptr) noexcept;

template <typename T>
class AlignedBuffer {
 public:
  AlignedBuffer() = default;

  AlignedBuffer(std::size_t count, std::size_t alignment = kCacheLineBytes) {
    resize(count, alignment);
  }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept {
    *this = std::move(other);
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    data_ = other.data_;
    size_ = other.size_;
    alignment_ = other.alignment_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.alignment_ = kCacheLineBytes;
    return *this;
  }

  ~AlignedBuffer() {
    reset();
  }

  void resize(std::size_t count, std::size_t alignment = kCacheLineBytes) {
    reset();
    if (count == 0) {
      return;
    }
    if ((alignment & (alignment - 1U)) != 0U) {
      throw std::runtime_error("alignment must be a power of two");
    }
    const auto bytes = sizeof(T) * count;
    data_ = static_cast<T*>(aligned_allocate_bytes(bytes, alignment));
    size_ = count;
    alignment_ = alignment;
    if constexpr (std::is_trivially_default_constructible_v<T>) {
      for (std::size_t i = 0; i < size_; ++i) {
        data_[i] = T{};
      }
    } else {
      for (std::size_t i = 0; i < size_; ++i) {
        new (&data_[i]) T();
      }
    }
  }

  void reset() noexcept {
    if (data_ == nullptr) {
      return;
    }
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (std::size_t i = 0; i < size_; ++i) {
        data_[i].~T();
      }
    }
    aligned_deallocate_bytes(static_cast<void*>(data_));
    data_ = nullptr;
    size_ = 0;
    alignment_ = kCacheLineBytes;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }

  [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_[i]; }
  [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t alignment_ = kCacheLineBytes;
};

class FrameArena {
 public:
  explicit FrameArena(std::size_t capacity_bytes = 0);

  void reserve(std::size_t capacity_bytes);
  [[nodiscard]] void* allocate(std::size_t byte_count, std::size_t alignment = alignof(std::max_align_t));
  void reset() noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }
  [[nodiscard]] std::size_t used() const noexcept { return offset_; }

 private:
  std::vector<std::byte> storage_;
  std::size_t offset_ = 0;
};

class PersistentArena {
 public:
  explicit PersistentArena(std::size_t capacity_bytes = 0) : arena_(capacity_bytes) {}

  void reserve(std::size_t capacity_bytes) { arena_.reserve(capacity_bytes); }
  [[nodiscard]] void* allocate(std::size_t byte_count, std::size_t alignment = alignof(std::max_align_t)) {
    return arena_.allocate(byte_count, alignment);
  }

 private:
  FrameArena arena_;
};

template <typename T>
class ObjectPool {
 public:
  explicit ObjectPool(std::size_t initial_capacity = 0) {
    reserve(initial_capacity);
  }

  void reserve(std::size_t n) {
    if (n <= storage_.size()) {
      return;
    }
    const auto old = storage_.size();
    storage_.resize(n);
    alive_.resize(n, false);
    for (std::size_t i = old; i < n; ++i) {
      free_list_.push_back(i);
    }
  }

  template <typename... Args>
  [[nodiscard]] std::size_t create(Args&&... args) {
    if (free_list_.empty()) {
      reserve(storage_.empty() ? 64 : storage_.size() * 2);
    }
    const auto id = free_list_.back();
    free_list_.pop_back();
    storage_[id] = T(std::forward<Args>(args)...);
    alive_[id] = true;
    return id;
  }

  void destroy(std::size_t id) {
    if (id >= alive_.size() || !alive_[id]) {
      return;
    }
    alive_[id] = false;
    free_list_.push_back(id);
  }

  [[nodiscard]] T* try_get(std::size_t id) noexcept {
    if (id >= alive_.size() || !alive_[id]) {
      return nullptr;
    }
    return &storage_[id];
  }

  [[nodiscard]] const T* try_get(std::size_t id) const noexcept {
    if (id >= alive_.size() || !alive_[id]) {
      return nullptr;
    }
    return &storage_[id];
  }

 private:
  std::vector<T> storage_;
  std::vector<bool> alive_;
  std::vector<std::size_t> free_list_;
};

}  // namespace powder::core
