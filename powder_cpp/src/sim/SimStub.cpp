#include "powder/sim/Checkpoint.hpp"
#include "powder/sim/FluidPDE.hpp"
#include "powder/sim/FieldBuffers.hpp"
#include "powder/sim/MaterialDB.hpp"
#include "powder/sim/SimConfig.hpp"
#include "powder/sim/SubstepScheduler.hpp"
#include "powder/sim/WorldState.hpp"

namespace powder::sim {

void phase3_link_anchor() {
	WorldState world = create_world_state(8, 8, 2);
	world.temperature.at(0, 0) = 300.0F;

	const auto& steel = material_properties(MaterialId::Steel);
	(void)steel;

	SimConfigStore store(SimConfig{});
	(void)store.current();

	SubstepScheduler scheduler;
	scheduler.execute_once();

	FieldBuffer2D<float, BufferCount::Double> scalar_buffers;
	scalar_buffers.resize(8, 8, 2);
	scalar_buffers.advance();

	phase5_link_anchor();

	(void)world;
}

}  // namespace powder::sim
