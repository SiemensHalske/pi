#include "powder/core/BuildInfo.hpp"

namespace powder::core {

BuildInfo query_build_info() {
  BuildInfo info{};
  info.project_name = "PowderCPP";
#if defined(POWDERCPP_BUILD_CONFIG)
  info.build_type = POWDERCPP_BUILD_CONFIG;
#else
  info.build_type = "unknown";
#endif
#if defined(POWDERCPP_WITH_OPENMP) && POWDERCPP_WITH_OPENMP
  info.with_openmp = true;
#else
  info.with_openmp = false;
#endif
  return info;
}

}  // namespace powder::core
