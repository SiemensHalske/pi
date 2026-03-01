#pragma once

#include <string>

namespace powder::app {

struct CliOptions {
  std::string profile = "deterministic";
  bool headless = false;
  bool benchmark = false;
  int threads = 0;
};

[[nodiscard]] CliOptions parse_cli(int argc, char** argv);
[[nodiscard]] int run(const CliOptions& options);

}  // namespace powder::app
