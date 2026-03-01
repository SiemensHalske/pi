#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace powder::sim {

enum class MaterialId : std::uint32_t {
  Air = 0,
  Water = 1,
  Steel = 2,
  Wood = 3,
  Sand = 4,
  Lava = 5,
  Count = 6,
};

struct MaterialProperties {
  std::string_view name;
  float eos_gamma;
  float viscosity;
  float thermal_conductivity;
  float heat_capacity;
  float density_ref;
  float yield_limit;
  float arrhenius_pre_exp;
  float arrhenius_activation_energy;
};

constexpr std::array<MaterialProperties, static_cast<std::size_t>(MaterialId::Count)> kMaterialTable{{
    {"air", 1.4F, 1.8e-5F, 0.026F, 1005.0F, 1.225F, 0.0F, 0.0F, 0.0F},
    {"water", 1.01F, 1.0e-3F, 0.60F, 4181.0F, 1000.0F, 0.0F, 0.0F, 0.0F},
    {"steel", 1.0F, 1.0e12F, 45.0F, 490.0F, 7850.0F, 250.0e6F, 1.0e7F, 2.5e5F},
    {"wood", 1.0F, 1.0e12F, 0.12F, 1700.0F, 700.0F, 40.0e6F, 2.2e6F, 1.1e5F},
    {"sand", 1.0F, 1.0e4F, 0.25F, 830.0F, 1600.0F, 3.0e6F, 0.0F, 0.0F},
    {"lava", 1.1F, 50.0F, 1.5F, 1200.0F, 2800.0F, 5.0e6F, 3.0e7F, 3.5e5F},
}};

[[nodiscard]] const MaterialProperties& material_properties(MaterialId id);
[[nodiscard]] MaterialId material_id_from_name(std::string_view name);

}  // namespace powder::sim
