#include "powder/core/Memory.hpp"

#include <cstdlib>
#include <cstring>

namespace powder::core {

void* aligned_allocate_bytes(std::size_t byte_count, std::size_t alignment) {
  if (byte_count == 0) {
    return nullptr;
  }

#if defined(_MSC_VER)
  void* ptr = _aligned_malloc(byte_count, alignment);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
#else
  const auto padded_size = align_up(byte_count, alignment);
  void* ptr = nullptr;
  const int rc = posix_memalign(&ptr, alignment, padded_size);
  if (rc != 0 || ptr == nullptr) {
    throw std::bad_alloc();
  }
  std::memset(ptr, 0, padded_size);
  return ptr;
#endif
}

void aligned_deallocate_bytes(void* ptr) noexcept {
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

FrameArena::FrameArena(std::size_t capacity_bytes) {
  reserve(capacity_bytes);
}

void FrameArena::reserve(std::size_t capacity_bytes) {
  storage_.assign(capacity_bytes, std::byte{0});
  offset_ = 0;
}

void* FrameArena::allocate(std::size_t byte_count, std::size_t alignment) {
  const auto aligned_offset = align_up(offset_, alignment);
  const auto next = aligned_offset + byte_count;
  if (next > storage_.size()) {
    throw std::bad_alloc();
  }
  void* ptr = storage_.data() + aligned_offset;
  offset_ = next;
  return ptr;
}

void FrameArena::reset() noexcept {
  offset_ = 0;
}

}  // namespace powder::core
