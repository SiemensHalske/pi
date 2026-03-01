#include "powder/sim/SubstepScheduler.hpp"

#include <stdexcept>

namespace powder::sim {

SubstepScheduler::SubstepScheduler()
    : order_{Substep::Forces, Substep::Advection, Substep::Diffusion, Substep::Projection,
             Substep::Thermodynamics, Substep::Chemistry, Substep::Structure, Substep::Particles,
             Substep::Boundaries} {}

void SubstepScheduler::set_callback(Substep step, SubstepCallback callback) {
  if (!callback) {
    throw std::runtime_error("substep callback must be valid");
  }
  callbacks_[step] = std::move(callback);
}

void SubstepScheduler::clear_callback(Substep step) {
  callbacks_.erase(step);
}

void SubstepScheduler::execute_once() const {
  for (const auto step : order_) {
    const auto it = callbacks_.find(step);
    if (it != callbacks_.end()) {
      it->second();
    }
  }
}

const std::array<Substep, 9>& SubstepScheduler::order() const noexcept {
  return order_;
}

std::vector<std::string_view> SubstepScheduler::order_labels() const {
  std::vector<std::string_view> labels;
  labels.reserve(order_.size());
  for (const auto step : order_) {
    labels.push_back(substep_label(step));
  }
  return labels;
}

std::string_view substep_label(Substep step) {
  switch (step) {
    case Substep::Forces:
      return "forces";
    case Substep::Advection:
      return "advection";
    case Substep::Diffusion:
      return "diffusion";
    case Substep::Projection:
      return "projection";
    case Substep::Thermodynamics:
      return "thermodynamics";
    case Substep::Chemistry:
      return "chemistry";
    case Substep::Structure:
      return "structure";
    case Substep::Particles:
      return "particles";
    case Substep::Boundaries:
      return "boundaries";
  }
  return "unknown";
}

}  // namespace powder::sim
