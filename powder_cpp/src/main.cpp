#include "powder/app/Cli.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const auto options = powder::app::parse_cli(argc, argv);
    return powder::app::run(options);
  } catch (const std::exception& ex) {
    std::cerr << "fatal: " << ex.what() << '\n';
    return 1;
  }
}
