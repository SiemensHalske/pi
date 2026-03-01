#pragma once

#include <string>

namespace powder::core {

struct BuildInfo {
  std::string project_name;
  std::string build_type;
  bool with_openmp = false;
};

[[nodiscard]] BuildInfo query_build_info();

}  // namespace powder::core
