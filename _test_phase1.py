"""Phase 1 smoke test — run with: python _test_phase1.py"""
import os, sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame
pygame.init()
pygame.display.set_mode((10, 10))

# ---- import game module pieces without launching the main loop ----
# We compile-exec everything up to if __name__ == "__main__"
with open('game.py', 'r') as fh:
    src = fh.read()

# Split at the main guard
main_guard = "if __name__"
before_main = src[:src.index(main_guard)]

ns = {}
exec(compile(before_main, 'game.py', 'exec'), ns)

import numpy as np

# ── 1. PhysicsConstants ──────────────────────────────────────────────────────
PC = ns['PHYSICS']
assert abs(PC.dx - 0.05) < 1e-6,         f"dx={PC.dx}"
assert abs(PC.dt - 1/60) < 1e-6,         f"dt={PC.dt}"
assert abs(PC.g  - 9.81) < 1e-6,         f"g={PC.g}"
assert PC.dx_inv == 1.0 / PC.dx,         "dx_inv wrong"
print(f"[OK] PhysicsConstants: dx={PC.dx} m, dt={PC.dt:.4f} s, g={PC.g} m/s²")

# ── 2. BoundaryConditionType ─────────────────────────────────────────────────
BT = ns['BoundaryConditionType']
assert BT.NO_SLIP == 0
assert BT.OPEN    == 2
print(f"[OK] BoundaryConditionType members: {list(BT)}")

# ── 3. PowderPhysicsEngine._ensure_pde_state ─────────────────────────────────
PPE  = ns['PowderPhysicsEngine']
MATS = ns['MATERIALS']
MIDS = ns['MATERIAL_IDS']
eng  = PPE(MATS, MIDS)
eng._ensure_pde_state(96, 128)

assert isinstance(eng.vel_x, np.ndarray),         "vel_x not ndarray"
assert eng.vel_x.shape  == (96, 128),             f"vel_x shape {eng.vel_x.shape}"
assert eng.vel_x.dtype  == np.float32,            "vel_x dtype"
assert eng.pressure_pde.dtype == np.float32,      "pressure_pde dtype"
assert float(eng.pressure_pde[-1, 0]) > float(eng.pressure_pde[0, 0]), \
    "hydrostatic: bottom should be higher pressure than top"
print(f"[OK] PDE fields: vel_x{eng.vel_x.shape} float32")
print(f"     pressure_pde range: {eng.pressure_pde.min():.1f}..{eng.pressure_pde.max():.1f} Pa")

# ── 4. _ensure_thermal_state returns numpy ───────────────────────────────────
eng._ensure_thermal_state(96, 128)
assert isinstance(eng.temperature,  np.ndarray), "temperature not ndarray"
assert isinstance(eng.oxygen_level, np.ndarray), "oxygen_level not ndarray"
assert eng.temperature.dtype  == np.float32
assert eng.oxygen_level.dtype == np.float32
assert np.all(eng.oxygen_level == 1.0),          "oxygen_level should be 1.0 everywhere"
print(f"[OK] thermal fields: temperature{eng.temperature.shape}, oxygen{eng.oxygen_level.shape}")

# ── 5. _ensure_chemical_state vectorised reset ───────────────────────────────
grid = [[0]*128 for _ in range(96)]
grid[5][5] = 1   # non-air cell
eng._ensure_chemical_state(grid, 96, 128)
assert isinstance(eng.integrity, np.ndarray),    "integrity not ndarray"
assert eng.integrity[5, 5] == 1.0,               "non-air integrity"
assert eng.integrity[0, 0] == 1.0,               "air cell integrity"
print(f"[OK] chemical fields: integrity{eng.integrity.shape}")

# ── 6. step() fills substep_timings + last_cfl ───────────────────────────────
sim = ns['Simulation'](128, 96)
result = sim.physics.step(sim.grid, sim.rows, sim.cols, 0)
assert isinstance(result.timings, dict),          "timings not dict"
assert isinstance(sim.physics.last_cfl, float),   "last_cfl not float"
assert sim.physics.last_cfl == 0.0,               "CFL should be 0 with no velocity"
print(f"[OK] step() timings={list(result.timings.keys())}, CFL={sim.physics.last_cfl}")

# ── 7. _copy_field handles numpy ─────────────────────────────────────────────
arr = np.ones((4, 4), dtype=np.float32)
copied = sim._copy_field(arr)
assert isinstance(copied, np.ndarray),             "_copy_field numpy"
assert copied is not arr,                          "_copy_field must return new object"
arr[0, 0] = 99.0
assert copied[0, 0] == 1.0,                       "_copy_field is deep, not view"
print(f"[OK] _copy_field numpy support")

# ── 8. _capture_state / _restore_state ───────────────────────────────────────
state = sim._capture_state()
assert "vel_x" in state["physics"],               "vel_x missing from captured state"
assert isinstance(state["physics"]["vel_x"], np.ndarray), "captured vel_x should be ndarray"
sim._restore_state(state)
assert sim.physics.vel_x.shape == (96, 128),      "restored vel_x shape wrong"
print(f"[OK] _capture_state / _restore_state with PDE fields")

print("\n══ Phase 1 complete — all assertions passed ══")
