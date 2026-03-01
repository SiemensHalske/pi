#include "powder/sim/Checkpoint.hpp"

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace powder::sim {

namespace {

constexpr std::uint32_t kMagic = 0x50435753U;  // PCWS
constexpr std::uint32_t kVersion = 1U;
constexpr std::uint32_t kEndianMarker = 0x01020304U;

struct CheckpointHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t endian_marker;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t ghost;
  std::uint64_t debris_count;
};

template <typename T>
[[nodiscard]] T to_little_endian(T value) {
  if constexpr (sizeof(T) == 1U) {
    return value;
  } else {
    if constexpr (std::endian::native == std::endian::little) {
      return value;
    }
    return std::byteswap(value);
  }
}

template <typename T>
[[nodiscard]] T from_little_endian(T value) {
  return to_little_endian(value);
}

template <typename T>
void write_integral(std::ofstream& out, T value) {
  const T le = to_little_endian(value);
  out.write(reinterpret_cast<const char*>(&le), static_cast<std::streamsize>(sizeof(T)));
}

template <typename T>
[[nodiscard]] T read_integral(std::ifstream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  if (!in) {
    throw std::runtime_error("failed to read checkpoint integral");
  }
  return from_little_endian(value);
}

void write_float(std::ofstream& out, float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  write_integral<std::uint32_t>(out, bits);
}

[[nodiscard]] float read_float(std::ifstream& in) {
  const auto bits = read_integral<std::uint32_t>(in);
  return std::bit_cast<float>(bits);
}

void write_field(std::ofstream& out, const powder::core::Field2D<float>& field) {
  for (std::size_t i = 0; i < field.size(); ++i) {
    write_float(out, field.raw()[i]);
  }
}

void read_field(std::ifstream& in, powder::core::Field2D<float>& field) {
  for (std::size_t i = 0; i < field.size(); ++i) {
    field.raw()[i] = read_float(in);
  }
}

void write_debris(std::ofstream& out, const DebrisSoA& debris) {
  for (std::size_t i = 0; i < debris.size(); ++i) {
    write_float(out, debris.x[i]);
    write_float(out, debris.y[i]);
    write_float(out, debris.vx[i]);
    write_float(out, debris.vy[i]);
    write_float(out, debris.radius[i]);
    write_float(out, debris.mass[i]);
    write_integral<std::uint32_t>(out, debris.material_id[i]);
  }
}

void read_debris(std::ifstream& in, DebrisSoA& debris, std::size_t count) {
  debris.x.resize(count);
  debris.y.resize(count);
  debris.vx.resize(count);
  debris.vy.resize(count);
  debris.radius.resize(count);
  debris.mass.resize(count);
  debris.material_id.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    debris.x[i] = read_float(in);
    debris.y[i] = read_float(in);
    debris.vx[i] = read_float(in);
    debris.vy[i] = read_float(in);
    debris.radius[i] = read_float(in);
    debris.mass[i] = read_float(in);
    debris.material_id[i] = read_integral<std::uint32_t>(in);
  }
}

}  // namespace

void save_checkpoint_binary(const std::string& file_path, const WorldState& world) {
  std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open checkpoint for write: " + file_path);
  }

  CheckpointHeader header{};
  header.magic = kMagic;
  header.version = kVersion;
  header.endian_marker = kEndianMarker;
  header.width = static_cast<std::uint32_t>(world.width);
  header.height = static_cast<std::uint32_t>(world.height);
  header.ghost = static_cast<std::uint32_t>(world.ghost);
  header.debris_count = static_cast<std::uint64_t>(world.debris.size());

  write_integral<std::uint32_t>(out, header.magic);
  write_integral<std::uint32_t>(out, header.version);
  write_integral<std::uint32_t>(out, header.endian_marker);
  write_integral<std::uint32_t>(out, header.width);
  write_integral<std::uint32_t>(out, header.height);
  write_integral<std::uint32_t>(out, header.ghost);
  write_integral<std::uint64_t>(out, header.debris_count);

  write_field(out, world.pressure);
  write_field(out, world.temperature);
  write_field(out, world.enthalpy);
  write_field(out, world.density);
  write_field(out, world.phase_fraction);

  write_field(out, world.velocity_u.value);
  write_field(out, world.velocity_v.value);

  write_field(out, world.species.o2);
  write_field(out, world.species.fuel_vapor);
  write_field(out, world.species.co2);
  write_field(out, world.species.h2o);
  write_field(out, world.species.soot);

  write_field(out, world.stress.sigma_xx);
  write_field(out, world.stress.sigma_yy);
  write_field(out, world.stress.tau_xy);
  write_field(out, world.stress.plastic_strain);
  write_field(out, world.stress.damage);

  write_debris(out, world.debris);
}

WorldState load_checkpoint_binary(const std::string& file_path) {
  std::ifstream in(file_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open checkpoint for read: " + file_path);
  }

  const auto magic = read_integral<std::uint32_t>(in);
  const auto version = read_integral<std::uint32_t>(in);
  const auto endian_marker = read_integral<std::uint32_t>(in);
  const auto width = read_integral<std::uint32_t>(in);
  const auto height = read_integral<std::uint32_t>(in);
  const auto ghost = read_integral<std::uint32_t>(in);
  const auto debris_count = read_integral<std::uint64_t>(in);

  if (magic != kMagic || version != kVersion || endian_marker != kEndianMarker) {
    throw std::runtime_error("invalid checkpoint header");
  }

  WorldState world = create_world_state(static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                                        static_cast<std::size_t>(ghost));

  read_field(in, world.pressure);
  read_field(in, world.temperature);
  read_field(in, world.enthalpy);
  read_field(in, world.density);
  read_field(in, world.phase_fraction);

  read_field(in, world.velocity_u.value);
  read_field(in, world.velocity_v.value);

  read_field(in, world.species.o2);
  read_field(in, world.species.fuel_vapor);
  read_field(in, world.species.co2);
  read_field(in, world.species.h2o);
  read_field(in, world.species.soot);

  read_field(in, world.stress.sigma_xx);
  read_field(in, world.stress.sigma_yy);
  read_field(in, world.stress.tau_xy);
  read_field(in, world.stress.plastic_strain);
  read_field(in, world.stress.damage);

  read_debris(in, world.debris, static_cast<std::size_t>(debris_count));
  return world;
}

}  // namespace powder::sim
