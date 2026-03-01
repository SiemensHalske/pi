#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace powder::sim {

enum class Substep {
  Forces = 0,
  Advection = 1,
  Diffusion = 2,
  Projection = 3,
  Thermodynamics = 4,
  Chemistry = 5,
  Structure = 6,
  Particles = 7,
  Boundaries = 8,
};

using SubstepCallback = std::function<void()>;

class SubstepScheduler {
 public:
  SubstepScheduler();

  void set_callback(Substep step, SubstepCallback callback);
  void clear_callback(Substep step);

  void execute_once() const;
  [[nodiscard]] const std::array<Substep, 9>& order() const noexcept;
  [[nodiscard]] std::vector<std::string_view> order_labels() const;

 private:
  std::array<Substep, 9> order_;
  std::unordered_map<Substep, SubstepCallback> callbacks_;
};

[[nodiscard]] std::string_view substep_label(Substep step);

}  // namespace powder::sim
