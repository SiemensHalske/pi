#include "powder/app/Cli.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace powder::app {

namespace {

[[nodiscard]] bool matches(std::string_view arg, std::string_view option) {
  return arg == option;
}

}  // namespace

CliOptions parse_cli(int argc, char** argv) {
  CliOptions options{};
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};

    if (matches(arg, "--headless")) {
      options.headless = true;
      continue;
    }

    if (matches(arg, "--benchmark")) {
      options.benchmark = true;
      continue;
    }

    if (matches(arg, "--profile")) {
      if (i + 1 >= argc) {
        throw std::runtime_error("--profile requires a value");
      }
      options.profile = argv[++i];
      continue;
    }

    if (matches(arg, "--threads")) {
      if (i + 1 >= argc) {
        throw std::runtime_error("--threads requires a value");
      }
      options.threads = std::stoi(argv[++i]);
      continue;
    }

    throw std::runtime_error("unknown option: " + std::string(arg));
  }

  if (options.benchmark && options.profile == "deterministic") {
    options.profile = "benchmark";
  }

  return options;
}

}  // namespace powder::app
