#include "powder/sim/MaterialDB.hpp"

#include <stdexcept>

namespace powder::sim {

const MaterialProperties& material_properties(MaterialId id) {
  const auto index = static_cast<std::size_t>(id);
  if (index >= kMaterialTable.size()) {
    throw std::runtime_error("invalid material id");
  }
  return kMaterialTable[index];
}

MaterialId material_id_from_name(std::string_view name) {
  for (std::size_t i = 0; i < kMaterialTable.size(); ++i) {
    if (kMaterialTable[i].name == name) {
      return static_cast<MaterialId>(i);
    }
  }
  throw std::runtime_error("unknown material name: " + std::string(name));
}

}  // namespace powder::sim
