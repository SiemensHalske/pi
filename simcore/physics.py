from .world import *
from .world import _safe_load_json

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


@njit(cache=True)
def _jacobi_pressure_numba(
    p,
    rhs,
    solid_mask,
    active_mask,
    density_field,
    dx2,
    dy2,
    denom,
    iters_min,
    iters_max,
    residual_tol,
    boundary_mode,
    free_surface_enabled,
    air_density_cutoff,
):
    rows, cols = p.shape
    p_new = p.copy()
    residual = np.float32(0.0)
    iters_used = 0
    eps = np.float32(1.0e-12)
    rhs_scale = np.float32(0.0)
    active_count = 0
    for r in range(rows):
        for c in range(cols):
            if active_mask[r, c] and (not solid_mask[r, c]):
                rhs_scale += rhs[r, c] * rhs[r, c]
                active_count += 1
    rhs_scale = np.float32(np.sqrt(rhs_scale / max(1, active_count))) + eps

    for it in range(iters_max):
        for r in range(rows):
            for c in range(cols):
                if solid_mask[r, c]:
                    p_new[r, c] = np.float32(0.0)
                    continue
                if not active_mask[r, c]:
                    p_new[r, c] = np.float32(0.0)
                    continue

                pl = p[r, c - 1] if c > 0 else p[r, c]
                pr = p[r, c + 1] if c < (cols - 1) else p[r, c]
                pt = p[r - 1, c] if r > 0 else p[r, c]
                pb = p[r + 1, c] if r < (rows - 1) else p[r, c]
                p_new[r, c] = (((pl + pr) * dy2 + (pt + pb) * dx2 - rhs[r, c] * dx2 * dy2) / denom)

        if boundary_mode == 1:
            if cols > 1:
                for r in range(rows):
                    p_new[r, 0] = np.float32(0.0)
                    p_new[r, cols - 1] = np.float32(0.0)
            if rows > 1:
                for c in range(cols):
                    p_new[0, c] = np.float32(0.0)
                    p_new[rows - 1, c] = np.float32(0.0)
        else:
            if cols > 1:
                for r in range(rows):
                    p_new[r, 0] = p_new[r, 1]
                    p_new[r, cols - 1] = p_new[r, cols - 2]
            if rows > 1:
                for c in range(cols):
                    p_new[0, c] = p_new[1, c]
                    p_new[rows - 1, c] = p_new[rows - 2, c]

        if free_surface_enabled:
            for r in range(rows):
                for c in range(cols):
                    if density_field[r, c] <= air_density_cutoff:
                        p_new[r, c] *= np.float32(0.25)

        res_acc = np.float32(0.0)
        for r in range(rows):
            for c in range(cols):
                if active_mask[r, c] and (not solid_mask[r, c]):
                    pl = p_new[r, c - 1] if c > 0 else p_new[r, c]
                    pr = p_new[r, c + 1] if c < (cols - 1) else p_new[r, c]
                    pt = p_new[r - 1, c] if r > 0 else p_new[r, c]
                    pb = p_new[r + 1, c] if r < (rows - 1) else p_new[r, c]
                    lap = (pl - np.float32(2.0) * p_new[r, c] + pr) / dx2 + (pt - np.float32(2.0) * p_new[r, c] + pb) / dy2
                    rr = lap - rhs[r, c]
                    res_acc += rr * rr

        residual = np.float32(np.sqrt(res_acc / max(1, active_count)) / rhs_scale)
        p, p_new = p_new, p
        iters_used = it + 1
        if (iters_used >= iters_min) and (residual <= residual_tol):
            break

    return p, iters_used, residual

class PowderPhysicsEngine:
    """Dedicated physics engine for granular and fluid cellular behavior."""
    def __init__(self, materials, material_ids, config: PhysicsConfig | None = None):
        self.materials = materials
        self.material_ids = material_ids
        self.config = config or PhysicsConfig()
        self.random_manager = RandomManager(self.config)
        self.granular_config = GranularModelConfig()
        self.fluid_config = FluidModelConfig()
        self.structural_config = StructuralModelConfig()
        self.thermal_config = ThermalCombustionConfig()
        self.chemistry_config = ChemistryConfig()
        self.phase_change_config = PhaseChangeConfig()
        self.interaction_table = self._create_interaction_table()
        self.lateral_bias = []
        self.jammed_until = []
        self.fluid_pressure = []
        self.mix_ratio = []
        self.temperature = []
        self.oxygen_level = []
        self.burn_stage = []
        self.burn_progress = []
        self.smoke_density = []
        self.ignition_cooldown_until = []
        self.fire_lifetime = []
        self.moisture = []
        self.smoke_lifetime = []
        self.enthalpy_field = []
        self.liquid_fraction = []
        self.steam_density = []
        self.integrity = []
        self.saturation_level = []
        self.phase_state = []
        self.reaction_progress = []
        self.phase_cooldown_until = []
        self.phase_transition_progress = []
        self.phase_transition_target = []
        self.fuel_vapor = np.zeros((0, 0), dtype=np.float32)
        self.co2_density = np.zeros((0, 0), dtype=np.float32)
        self.co_density = np.zeros((0, 0), dtype=np.float32)
        self.h2o_vapor = np.zeros((0, 0), dtype=np.float32)
        self.mixture_fraction = np.zeros((0, 0), dtype=np.float32)
        self.turb_k = np.zeros((0, 0), dtype=np.float32)
        self.turb_eps = np.zeros((0, 0), dtype=np.float32)
        self.pyrolysis_progress = np.zeros((0, 0), dtype=np.float32)
        self.soot_mass_fraction = np.zeros((0, 0), dtype=np.float32)
        self.soot_number_density = np.zeros((0, 0), dtype=np.float32)
        self.h_plus = np.zeros((0, 0), dtype=np.float32)
        self.oh_minus = np.zeros((0, 0), dtype=np.float32)
        self.ph_field = np.zeros((0, 0), dtype=np.float32)
        self.catalyst_theta_fuel = np.zeros((0, 0), dtype=np.float32)
        self.catalyst_theta_o2 = np.zeros((0, 0), dtype=np.float32)
        self.disp_x = np.zeros((0, 0), dtype=np.float32)
        self.disp_y = np.zeros((0, 0), dtype=np.float32)
        self.struct_vel_x = np.zeros((0, 0), dtype=np.float32)
        self.struct_vel_y = np.zeros((0, 0), dtype=np.float32)
        self.sigma_xx = np.zeros((0, 0), dtype=np.float32)
        self.sigma_yy = np.zeros((0, 0), dtype=np.float32)
        self.tau_xy = np.zeros((0, 0), dtype=np.float32)
        self.plastic_strain = np.zeros((0, 0), dtype=np.float32)
        self.damage_field = np.zeros((0, 0), dtype=np.float32)
        self.pore_pressure = np.zeros((0, 0), dtype=np.float32)
        self.debris_particles: list[dict] = []
        # Phase 1 PDE fields (populated lazily by _ensure_pde_state)
        self.vel_x         = np.zeros((0, 0), dtype=np.float32)
        self.vel_y         = np.zeros((0, 0), dtype=np.float32)
        self.pressure_pde  = np.zeros((0, 0), dtype=np.float32)
        self.divergence_pde = np.zeros((0, 0), dtype=np.float32)
        self.density_field = np.zeros((0, 0), dtype=np.float32)
        self.active_mask   = np.zeros((0, 0), dtype=np.bool_)
        # Phase 2 MAC (staggered) storage:
        # mac_u[r, c+1] = horizontal face velocity u(i+1/2, j)
        # mac_v[r+1, c] = vertical face velocity   v(i, j+1/2)
        self.mac_u = np.zeros((0, 0), dtype=np.float32)
        self.mac_v = np.zeros((0, 0), dtype=np.float32)
        self.acoustic_pressure = np.zeros((0, 0), dtype=np.float32)
        self.acoustic_u = np.zeros((0, 0), dtype=np.float32)
        self.acoustic_v = np.zeros((0, 0), dtype=np.float32)
        self.porous_resistance = np.zeros((0, 0), dtype=np.float32)
        self.pending_detonations = []
        # Per-substep wall-clock timings in ms (Schritt 8)
        self.substep_timings: dict = {}
        self.last_cfl: float = 0.0
        self.pressure_solver_stats: dict = {}
        self.pde_validation_metrics: dict = {}
        self._numba_available = _NUMBA_AVAILABLE

    def _create_interaction_table(self):
        rules = self._load_interaction_rules_from_file()
        table = MaterialInteractionTable(rules)
        table.validate(self.materials)
        return table

    def _load_interaction_rules_from_file(self):
        payload = _safe_load_json(INTERACTION_MATRIX_FILE)
        if not payload or "rules" not in payload:
            return self._build_default_interaction_rules()

        resolved_rules = []
        for rule in payload["rules"]:
            resolved_rule = dict(rule)
            if "pair_names" in resolved_rule:
                pair_names = resolved_rule["pair_names"]
                if len(pair_names) != 2:
                    continue
                left = self.material_ids.get(str(pair_names[0]).strip().lower())
                right = self.material_ids.get(str(pair_names[1]).strip().lower())
                if left is None or right is None:
                    continue
                resolved_rule["pair"] = [left, right]

            if "pair" not in resolved_rule:
                continue
            resolved_rules.append(resolved_rule)

        if not resolved_rules:
            return self._build_default_interaction_rules()
        return resolved_rules

    def reload_interaction_table(self):
        self.interaction_table = self._create_interaction_table()

    def _build_default_interaction_rules(self):
        water = self.material_ids["water"]
        lava = self.material_ids["lava"]
        wall = self.material_ids["wall"]
        acid = self.material_ids["acid"]
        wood = self.material_ids["wood"]
        ash = self.material_ids["ash"]
        steam = self.material_ids["steam"]
        fire = self.material_ids.get("fire", -1)
        stone = self.material_ids.get("stone", -1)
        oil = self.material_ids.get("oil", -1)
        sand = self.material_ids["sand"]
        ice = self.material_ids["ice"]
        rules = [
            {
                "name": "water_lava_quench",
                "pair": (water, lava),
                "priority": 95,
                "conditions": {"min_temp": 80.0},
                "products": [steam, wall],
                "energy_delta": 14.0,
                "gas_release": 0.25,
                "residue": 0,
                "duration_ticks": 4,
            },
            {
                "name": "acid_wall_etch",
                "pair": (wall, acid),
                "priority": 90,
                "conditions": {"min_contact_progress": 0.2},
                "products": [sand, acid],
                "energy_delta": 3.0,
                "gas_release": 0.15,
                "residue": ash,
                "duration_ticks": 6,
            },
            {
                "name": "acid_wood_breakdown",
                "pair": (wood, acid),
                "priority": 85,
                "conditions": {"min_contact_progress": 0.1},
                "products": [ash, acid],
                "energy_delta": 1.5,
                "gas_release": 0.08,
                "residue": 0,
                "duration_ticks": 6,
            },
            {
                "name": "steam_condense",
                "pair": (water, steam),
                "priority": 60,
                "conditions": {"max_temp": 92.0},
                "products": [water, water],
                "energy_delta": -8.0,
                "gas_release": 0.0,
                "residue": 0,
                "duration_ticks": 5,
            },
            {
                "name": "acid_water_dilution",
                "pair": (water, acid),
                "priority": 72,
                "conditions": {},
                "products": [water, water],
                "energy_delta": 6.0,
                "gas_release": 0.06,
                "residue": 0,
                "duration_ticks": 14,
            },
            {
                "name": "acid_sand_etch",
                "pair": (sand, acid),
                "priority": 68,
                "conditions": {"min_contact_progress": 0.25},
                "products": [0, acid],
                "energy_delta": 2.0,
                "gas_release": 0.08,
                "residue": 0,
                "duration_ticks": 10,
            },
            {
                "name": "acid_ice_melt",
                "pair": (ice, acid),
                "priority": 80,
                "conditions": {},
                "products": [water, acid],
                "energy_delta": 5.0,
                "gas_release": 0.04,
                "residue": 0,
                "duration_ticks": 6,
            },
        ]
        # Fire + water quench → steam + air
        if fire >= 0:
            rules.append({
                "name": "fire_water_quench",
                "pair": (fire, water),
                "priority": 98,
                "conditions": {},
                "products": [steam, 0],
                "energy_delta": -10.0,
                "gas_release": 0.35,
                "residue": 0,
                "duration_ticks": 2,
            })
        # Lava + sand → stone + stone
        if stone >= 0:
            rules.append({
                "name": "lava_sand_solidify",
                "pair": (sand, lava),
                "priority": 88,
                "conditions": {"min_temp": 300.0},
                "products": [stone, stone],
                "energy_delta": -18.0,
                "gas_release": 0.05,
                "residue": 0,
                "duration_ticks": 8,
            })
            # Acid + stone → sand
            rules.append({
                "name": "acid_stone_etch",
                "pair": (stone, acid),
                "priority": 82,
                "conditions": {"min_contact_progress": 0.35},
                "products": [sand, acid],
                "energy_delta": 4.0,
                "gas_release": 0.1,
                "residue": 0,
                "duration_ticks": 12,
            })
        # Oil + acid → ash (breakdown)
        if oil >= 0:
            rules.append({
                "name": "acid_oil_breakdown",
                "pair": (oil, acid),
                "priority": 70,
                "conditions": {"min_contact_progress": 0.15},
                "products": [ash, acid],
                "energy_delta": 2.0,
                "gas_release": 0.12,
                "residue": 0,
                "duration_ticks": 10,
            })
        return rules

    def _in_bounds(self, row, col, rows, cols):
        return 0 <= row < rows and 0 <= col < cols

    def _get(self, mat_id):
        return self.materials[mat_id]

    def _mat_value(self, mat_data, key, default):
        return mat_data[key] if key in mat_data else default

    def _ensure_granular_state(self, rows, cols):
        if not (isinstance(self.lateral_bias, np.ndarray) and self.lateral_bias.shape == (rows, cols)):
            self.lateral_bias = np.zeros((rows, cols), dtype=np.int32)
            self.jammed_until = np.zeros((rows, cols), dtype=np.int32)

    def _ensure_fluid_state(self, rows, cols):
        if not (isinstance(self.fluid_pressure, np.ndarray) and self.fluid_pressure.shape == (rows, cols)):
            self.fluid_pressure = np.zeros((rows, cols), dtype=np.float32)
            self.mix_ratio      = np.zeros((rows, cols), dtype=np.float32)

    def _ensure_thermal_state(self, rows, cols):
        if not (isinstance(self.temperature, np.ndarray) and self.temperature.shape == (rows, cols)):
            amb = np.float32(self.thermal_config.ambient_temp)
            self.temperature             = np.full((rows, cols), amb,           dtype=np.float32)
            self.oxygen_level            = np.ones ((rows, cols),               dtype=np.float32)
            self.burn_stage              = np.zeros((rows, cols),               dtype=np.int32)
            self.burn_progress           = np.zeros((rows, cols),               dtype=np.float32)
            self.smoke_density           = np.zeros((rows, cols),               dtype=np.float32)
            self.ignition_cooldown_until = np.zeros((rows, cols),               dtype=np.int32)
            self.fire_lifetime           = np.zeros((rows, cols),               dtype=np.float32)
            self.moisture                = np.zeros((rows, cols),               dtype=np.float32)
            self.smoke_lifetime          = np.zeros((rows, cols),               dtype=np.float32)
            self.enthalpy_field          = np.zeros((rows, cols),               dtype=np.float32)
            self.liquid_fraction         = np.zeros((rows, cols),               dtype=np.float32)
            self.steam_density           = np.zeros((rows, cols),               dtype=np.float32)

    def _ensure_chemical_state(self, grid, rows, cols):
        if not (isinstance(self.integrity, np.ndarray) and self.integrity.shape == (rows, cols)):
            self.integrity                = np.ones ((rows, cols), dtype=np.float32)
            self.saturation_level         = np.zeros((rows, cols), dtype=np.float32)
            self.phase_state              = np.zeros((rows, cols), dtype=np.int32)
            self.reaction_progress        = np.zeros((rows, cols), dtype=np.float32)
            self.phase_cooldown_until     = np.zeros((rows, cols), dtype=np.int32)
            self.phase_transition_progress = np.zeros((rows, cols), dtype=np.float32)
            self.phase_transition_target  = np.zeros((rows, cols), dtype=np.int32)
            self.fuel_vapor               = np.zeros((rows, cols), dtype=np.float32)
            self.co2_density              = np.zeros((rows, cols), dtype=np.float32)
            self.co_density               = np.zeros((rows, cols), dtype=np.float32)
            self.h2o_vapor                = np.zeros((rows, cols), dtype=np.float32)
            self.mixture_fraction         = np.zeros((rows, cols), dtype=np.float32)
            self.turb_k                   = np.zeros((rows, cols), dtype=np.float32)
            self.turb_eps                 = np.zeros((rows, cols), dtype=np.float32)
            self.pyrolysis_progress       = np.zeros((rows, cols), dtype=np.float32)
            self.soot_mass_fraction       = np.zeros((rows, cols), dtype=np.float32)
            self.soot_number_density      = np.zeros((rows, cols), dtype=np.float32)
            self.h_plus                   = np.zeros((rows, cols), dtype=np.float32)
            self.oh_minus                 = np.zeros((rows, cols), dtype=np.float32)
            self.ph_field                 = np.full((rows, cols), np.float32(7.0), dtype=np.float32)
            self.catalyst_theta_fuel      = np.zeros((rows, cols), dtype=np.float32)
            self.catalyst_theta_o2        = np.zeros((rows, cols), dtype=np.float32)

        # Vectorised reset for air cells (grid==0) — avoids Python loop
        grid_np = np.asarray(grid, dtype=np.int32)
        air_mask = (grid_np == 0)
        self.integrity[air_mask]                 = 1.0
        self.reaction_progress[air_mask]         = 0.0
        self.phase_transition_progress[air_mask] = 0.0
        self.phase_transition_target[air_mask]   = 0
        self.pyrolysis_progress[air_mask]        = 0.0
        self.catalyst_theta_fuel[air_mask]       = np.float32(np.clip(self.catalyst_theta_fuel[air_mask], 0.0, 1.0))
        self.catalyst_theta_o2[air_mask]         = np.float32(np.clip(self.catalyst_theta_o2[air_mask], 0.0, 1.0))
        self.mixture_fraction[:] = np.clip(
            self.fuel_vapor / np.maximum(np.float32(1e-6), self.fuel_vapor + self.oxygen_level / np.float32(3.5)),
            0.0,
            1.0,
        ).astype(np.float32)

    def _ensure_structural_state(self, rows, cols):
        state_ok = (
            isinstance(self.disp_x, np.ndarray) and self.disp_x.shape == (rows, cols)
            and isinstance(self.disp_y, np.ndarray) and self.disp_y.shape == (rows, cols)
            and isinstance(self.struct_vel_x, np.ndarray) and self.struct_vel_x.shape == (rows, cols)
            and isinstance(self.struct_vel_y, np.ndarray) and self.struct_vel_y.shape == (rows, cols)
            and isinstance(self.sigma_xx, np.ndarray) and self.sigma_xx.shape == (rows, cols)
            and isinstance(self.sigma_yy, np.ndarray) and self.sigma_yy.shape == (rows, cols)
            and isinstance(self.tau_xy, np.ndarray) and self.tau_xy.shape == (rows, cols)
            and isinstance(self.plastic_strain, np.ndarray) and self.plastic_strain.shape == (rows, cols)
            and isinstance(self.damage_field, np.ndarray) and self.damage_field.shape == (rows, cols)
            and isinstance(self.pore_pressure, np.ndarray) and self.pore_pressure.shape == (rows, cols)
        )

        if not state_ok:
            self.disp_x = np.zeros((rows, cols), dtype=np.float32)
            self.disp_y = np.zeros((rows, cols), dtype=np.float32)
            self.struct_vel_x = np.zeros((rows, cols), dtype=np.float32)
            self.struct_vel_y = np.zeros((rows, cols), dtype=np.float32)
            self.sigma_xx = np.zeros((rows, cols), dtype=np.float32)
            self.sigma_yy = np.zeros((rows, cols), dtype=np.float32)
            self.tau_xy = np.zeros((rows, cols), dtype=np.float32)
            self.plastic_strain = np.zeros((rows, cols), dtype=np.float32)
            self.damage_field = np.zeros((rows, cols), dtype=np.float32)
            self.pore_pressure = np.zeros((rows, cols), dtype=np.float32)

    def _ensure_pde_state(self, rows, cols):
        """Allocate continuum fields and staggered MAC storage.

        References:
        - Stam 1999 / GDC03: stable semi-Lagrangian + projection.
        - Bridson 2007 §2.4: MAC discretization (u on vertical faces, v on
          horizontal faces, p at cell centers).
        """
        collocated_ok = (
            isinstance(self.vel_x, np.ndarray) and self.vel_x.shape == (rows, cols)
            and isinstance(self.vel_y, np.ndarray) and self.vel_y.shape == (rows, cols)
            and isinstance(self.pressure_pde, np.ndarray) and self.pressure_pde.shape == (rows, cols)
            and isinstance(self.divergence_pde, np.ndarray) and self.divergence_pde.shape == (rows, cols)
            and isinstance(self.density_field, np.ndarray) and self.density_field.shape == (rows, cols)
            and isinstance(self.active_mask, np.ndarray) and self.active_mask.shape == (rows, cols)
            and isinstance(self.acoustic_pressure, np.ndarray) and self.acoustic_pressure.shape == (rows, cols)
            and isinstance(self.acoustic_u, np.ndarray) and self.acoustic_u.shape == (rows, cols)
            and isinstance(self.acoustic_v, np.ndarray) and self.acoustic_v.shape == (rows, cols)
            and isinstance(self.porous_resistance, np.ndarray) and self.porous_resistance.shape == (rows, cols)
        )
        mac_ok = (
            isinstance(self.mac_u, np.ndarray) and self.mac_u.shape == (rows, cols + 1)
            and isinstance(self.mac_v, np.ndarray) and self.mac_v.shape == (rows + 1, cols)
        )

        if not collocated_ok:
            self.vel_x         = np.zeros((rows, cols), dtype=np.float32)  # m/s
            self.vel_y         = np.zeros((rows, cols), dtype=np.float32)  # m/s
            # Hydrostatic initialisation: p(i) = rho*g*h  where h = i*dy
            # Row 0 = top (atmosphere, p≈0); row rows-1 = bottom (max column weight).
            # Gravity acts in the +row direction so pressure rises with row index.
            row_idx            = np.arange(rows, dtype=np.float32).reshape(rows, 1)
            depth              = row_idx * np.float32(PHYSICS.dy)
            self.pressure_pde  = (np.float32(PHYSICS.rho_air) *
                                   np.float32(PHYSICS.g) * depth *
                                   np.ones((rows, cols), dtype=np.float32))
            self.divergence_pde = np.zeros((rows, cols), dtype=np.float32)
            self.density_field = np.full((rows, cols),
                                         np.float32(PHYSICS.rho_air),
                                         dtype=np.float32)
            # active_mask: True if cell needs PDE update this tick.
            # Sparse solver (Step 48) will grow this from non-trivial boundaries.
            self.active_mask   = np.ones((rows, cols), dtype=np.bool_)
            self.acoustic_pressure = np.zeros((rows, cols), dtype=np.float32)
            self.acoustic_u = np.zeros((rows, cols), dtype=np.float32)
            self.acoustic_v = np.zeros((rows, cols), dtype=np.float32)
            self.porous_resistance = np.zeros((rows, cols), dtype=np.float32)

        if not mac_ok:
            self.mac_u = np.zeros((rows, cols + 1), dtype=np.float32)
            self.mac_v = np.zeros((rows + 1, cols), dtype=np.float32)

    def _rebuild_density_and_solid_masks(self, grid, rows, cols):
        """Build density field from CA material IDs plus thermal expansion.

        Returns (solid_mask, nu_field, grid_np).
        """
        grid_np = np.asarray(grid, dtype=np.int32)
        density = np.full((rows, cols), np.float32(PHYSICS.rho_air), dtype=np.float32)
        solid_mask = np.zeros((rows, cols), dtype=np.bool_)
        nu_field = np.full(
            (rows, cols),
            np.float32(max(1e-6, self.fluid_config.pde_kinematic_viscosity)),
            dtype=np.float32,
        )

        for mat_id, mat_data in self.materials.items():
            mat_mask = (grid_np == int(mat_id))
            if not np.any(mat_mask):
                continue

            rho = np.float32(max(0.01, float(self._mat_value(mat_data, "density", PHYSICS.rho_air))))
            density[mat_mask] = rho

            mat_type = str(mat_data.get("type", "air"))
            if mat_type in ("solid", "powder"):
                solid_mask[mat_mask] = True

            mu_relative = float(self._mat_value(mat_data, "viscosity", 0.0))
            if mu_relative > 0.0:
                # Map gameplay viscosity to kinematic viscosity ν [m²/s].
                # ν = μ/ρ, with μ scaled from relative material viscosity.
                nu_value = np.float32(max(1e-6, (mu_relative * 1e-3) / max(float(rho), 1.0)))
                nu_field[mat_mask] = nu_value

        if isinstance(self.temperature, np.ndarray) and self.temperature.shape == (rows, cols):
            beta = np.float32(max(0.0, self.fluid_config.thermal_expansion_beta))
            t_ref = np.float32(self.thermal_config.ambient_temp)
            thermal_factor = 1.0 - beta * (self.temperature.astype(np.float32) - t_ref)
            thermal_factor = np.clip(thermal_factor, 0.2, 5.0).astype(np.float32)
            density *= thermal_factor

        self.density_field = density
        self.active_mask = ~solid_mask
        porous_mask = np.zeros((rows, cols), dtype=np.bool_)
        for mat_id, mat_data in self.materials.items():
            local_porosity = float(self._mat_value(mat_data, "porosity", 0.0))
            if local_porosity <= 0.0:
                continue
            mat_mask = (grid_np == int(mat_id))
            if not np.any(mat_mask):
                continue
            porous_mask[mat_mask] = True

        if np.any(porous_mask):
            porous_f = porous_mask.astype(np.float32)
            p_l = np.empty_like(porous_f)
            p_r = np.empty_like(porous_f)
            p_t = np.empty_like(porous_f)
            p_b = np.empty_like(porous_f)
            p_l[:, 0] = porous_f[:, 0]
            p_l[:, 1:] = porous_f[:, :-1]
            p_r[:, -1] = porous_f[:, -1]
            p_r[:, :-1] = porous_f[:, 1:]
            p_t[0, :] = porous_f[0, :]
            p_t[1:, :] = porous_f[:-1, :]
            p_b[-1, :] = porous_f[-1, :]
            p_b[:-1, :] = porous_f[1:, :]
            influence = np.clip((porous_f + p_l + p_r + p_t + p_b) * np.float32(0.2), 0.0, 1.0)
            influence[solid_mask] = 0.0
            self.porous_resistance = influence.astype(np.float32)
        else:
            self.porous_resistance.fill(0.0)

        return solid_mask, nu_field, grid_np, porous_mask

    def _apply_mac_boundaries(self, solid_mask):
        """Apply no-slip velocity boundary and solid-face blocking on MAC faces."""
        if self.mac_u.size == 0 or self.mac_v.size == 0:
            return

        btype = self.fluid_config.pde_boundary_type

        if btype in (BoundaryConditionType.OPEN, BoundaryConditionType.OUTLET):
            if self.mac_u.shape[1] > 2:
                self.mac_u[:, 0] = self.mac_u[:, 1]
                self.mac_u[:, -1] = self.mac_u[:, -2]
            else:
                self.mac_u[:, 0] = 0.0
                self.mac_u[:, -1] = 0.0
            if self.mac_v.shape[0] > 2:
                self.mac_v[0, :] = self.mac_v[1, :]
                self.mac_v[-1, :] = self.mac_v[-2, :]
            else:
                self.mac_v[0, :] = 0.0
                self.mac_v[-1, :] = 0.0
        elif btype == BoundaryConditionType.FREE_SLIP:
            self.mac_u[:, 0] = 0.0
            self.mac_u[:, -1] = 0.0
            self.mac_v[0, :] = 0.0
            self.mac_v[-1, :] = 0.0
            if self.mac_u.shape[0] > 2:
                self.mac_u[0, :] = self.mac_u[1, :]
                self.mac_u[-1, :] = self.mac_u[-2, :]
            if self.mac_v.shape[1] > 2:
                self.mac_v[:, 0] = self.mac_v[:, 1]
                self.mac_v[:, -1] = self.mac_v[:, -2]
        else:
            self.mac_u[:, 0] = 0.0
            self.mac_u[:, -1] = 0.0
            self.mac_v[0, :] = 0.0
            self.mac_v[-1, :] = 0.0

        rows, cols = solid_mask.shape
        if cols > 1:
            blocked_u = solid_mask[:, :-1] | solid_mask[:, 1:]
            self.mac_u[:, 1:-1][blocked_u] = 0.0
        if rows > 1:
            blocked_v = solid_mask[:-1, :] | solid_mask[1:, :]
            self.mac_v[1:-1, :][blocked_v] = 0.0

    def _apply_log_wind_inlet(self, solid_mask):
        if not self.fluid_config.wind_forcing_enabled:
            return
        rows, cols = solid_mask.shape
        if cols < 2:
            return

        u_ref = float(self.fluid_config.wind_reference_speed)
        z_ref = max(float(self.fluid_config.wind_reference_height), PHYSICS.dy)
        z0 = max(float(self.fluid_config.wind_roughness_length), 1e-4)
        d = max(0.0, float(self.fluid_config.wind_displacement_height))
        kappa = max(1e-4, float(self.fluid_config.wind_kappa))

        denom = math.log(max((z_ref - d) / z0, 1.0001))
        if denom <= 0.0:
            return
        u_star = u_ref * kappa / denom

        y_idx = np.arange(rows, dtype=np.float32)
        z = (np.float32(rows) - y_idx - np.float32(0.5)) * np.float32(PHYSICS.dy)
        z_eff = np.maximum(np.float32(z0 * 1.001), z - np.float32(d))
        profile = (np.float32(u_star / kappa) * np.log(z_eff / np.float32(z0))).astype(np.float32)
        profile = np.maximum(np.float32(0.0), profile)
        profile[solid_mask[:, 0]] = 0.0

        self.mac_u[:, 0] = profile
        self.mac_u[:, 1] = profile

    def _apply_porous_drag(self, solid_mask, rho_ref):
        if not self.fluid_config.porous_drag_enabled:
            return
        if self.porous_resistance.shape != self.vel_x.shape:
            return

        eps = np.clip(np.float32(self.fluid_config.porous_porosity), 0.05, 0.98)
        dp = np.float32(max(1e-4, self.fluid_config.porous_particle_diameter))
        rho = np.float32(max(1e-6, rho_ref))
        mu = np.float32(PHYSICS.mu_air)
        multiplier = np.float32(max(0.0, self.fluid_config.porous_drag_multiplier))

        k = (dp * dp * (eps ** 3)) / np.float32(150.0 * ((1.0 - eps) ** 2) + 1e-6)
        k1 = (dp * (eps ** 3)) / np.float32(1.75 * (1.0 - eps) + 1e-6)

        lin = (mu / (rho * max(1e-8, float(k)))) * multiplier
        speed = np.sqrt(self.vel_x * self.vel_x + self.vel_y * self.vel_y).astype(np.float32)
        quad = (speed / np.float32(max(1e-8, float(k1)))) * multiplier
        sigma = (np.float32(lin) + quad) * self.porous_resistance
        factor = np.exp(-sigma * np.float32(PHYSICS.dt)).astype(np.float32)
        factor[solid_mask] = 0.0

        self.vel_x *= factor
        self.vel_y *= factor

    def _build_detonation_divergence_source(self, rows, cols, rho_ref):
        src = np.zeros((rows, cols), dtype=np.float32)
        if not self.fluid_config.detonation_enabled or not self.pending_detonations:
            return src

        yy = np.arange(rows, dtype=np.float32)[:, None]
        xx = np.arange(cols, dtype=np.float32)[None, :]
        beta = np.float32(1.033)
        rho = np.float32(max(1e-6, rho_ref))
        dt = np.float32(max(1e-6, PHYSICS.dt))

        for row, col, strength in self.pending_detonations:
            E = np.float32(max(1.0, strength * self.fluid_config.detonation_energy_scale))
            radius_m = beta * ((E * dt * dt) / rho) ** np.float32(0.2)
            radius_cells = np.float32(max(1.5, radius_m * PHYSICS.dx_inv * self.fluid_config.detonation_radius_scale))

            dx = xx - np.float32(col)
            dy = yy - np.float32(row)
            rr = np.sqrt(dx * dx + dy * dy)
            falloff = np.clip(1.0 - rr / radius_cells, 0.0, 1.0).astype(np.float32)
            src += np.float32(self.fluid_config.detonation_divergence_scale) * falloff

            inv_r = 1.0 / np.maximum(rr, np.float32(0.5))
            impulse = np.float32(self.fluid_config.detonation_impulse_scale) * falloff
            self.vel_x += (impulse * dx * inv_r).astype(np.float32)
            self.vel_y += (impulse * dy * inv_r).astype(np.float32)

        self.pending_detonations.clear()
        return src

    def _apply_acoustic_substep_with_pml(self, solid_mask, rho_ref):
        if not self.fluid_config.acoustic_enabled:
            return
        rows, cols = solid_mask.shape
        if rows <= 1 or cols <= 1:
            return

        c = np.float32(max(0.1, self.fluid_config.acoustic_wave_speed))
        dt = np.float32(PHYSICS.dt)
        target_cfl = np.float32(max(0.1, self.fluid_config.acoustic_cfl_target))
        min_sub = int(max(1, self.fluid_config.acoustic_substeps_min))
        max_sub = int(max(min_sub, self.fluid_config.acoustic_substeps_max))
        dx_min = np.float32(min(PHYSICS.dx, PHYSICS.dy))
        cfl_sub = int(math.ceil(float(c * dt / max(1e-6, target_cfl * dx_min))))
        n_sub = max(min_sub, min(max_sub, cfl_sub))
        dt_sub = np.float32(dt / np.float32(n_sub))

        p = self.acoustic_pressure
        u = self.acoustic_u
        v = self.acoustic_v

        dx_inv = np.float32(PHYSICS.dx_inv)
        dy_inv = np.float32(PHYSICS.dy_inv)
        rho = np.float32(max(1e-6, rho_ref))

        thickness = int(max(1, self.fluid_config.acoustic_pml_thickness))
        pml_power = np.float32(max(1.0, self.fluid_config.acoustic_pml_power))
        pml_strength = np.float32(max(0.0, self.fluid_config.acoustic_pml_strength))

        y_idx = np.arange(rows, dtype=np.float32)[:, None]
        x_idx = np.arange(cols, dtype=np.float32)[None, :]
        d_left = x_idx
        d_right = (np.float32(cols - 1) - x_idx)
        d_top = y_idx
        d_bottom = (np.float32(rows - 1) - y_idx)
        d_edge = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom))
        eta = np.clip((np.float32(thickness) - d_edge) / np.float32(thickness), 0.0, 1.0).astype(np.float32)
        sigma = (pml_strength * (eta ** pml_power)).astype(np.float32)

        for _ in range(n_sub):
            p_l = np.empty_like(p)
            p_r = np.empty_like(p)
            p_t = np.empty_like(p)
            p_b = np.empty_like(p)
            p_l[:, 0] = p[:, 0]
            p_l[:, 1:] = p[:, :-1]
            p_r[:, -1] = p[:, -1]
            p_r[:, :-1] = p[:, 1:]
            p_t[0, :] = p[0, :]
            p_t[1:, :] = p[:-1, :]
            p_b[-1, :] = p[-1, :]
            p_b[:-1, :] = p[1:, :]

            grad_px = np.float32(0.5) * (p_r - p_l) * dx_inv
            grad_py = np.float32(0.5) * (p_b - p_t) * dy_inv
            u -= (dt_sub / rho) * grad_px
            v -= (dt_sub / rho) * grad_py

            u_l = np.empty_like(u)
            u_r = np.empty_like(u)
            v_t = np.empty_like(v)
            v_b = np.empty_like(v)
            u_l[:, 0] = u[:, 0]
            u_l[:, 1:] = u[:, :-1]
            u_r[:, -1] = u[:, -1]
            u_r[:, :-1] = u[:, 1:]
            v_t[0, :] = v[0, :]
            v_t[1:, :] = v[:-1, :]
            v_b[-1, :] = v[-1, :]
            v_b[:-1, :] = v[1:, :]
            div_uv = np.float32(0.5) * ((u_r - u_l) * dx_inv + (v_b - v_t) * dy_inv)
            p -= (rho * c * c * dt_sub) * div_uv

            damp = np.exp(-sigma * dt_sub).astype(np.float32)
            p *= damp
            u *= damp
            v *= damp
            p[solid_mask] = 0.0
            u[solid_mask] = 0.0
            v[solid_mask] = 0.0

        self.acoustic_pressure = p
        self.acoustic_u = u
        self.acoustic_v = v

        v_c = np.float32(max(0.0, self.fluid_config.acoustic_velocity_coupling))
        p_c = np.float32(max(0.0, self.fluid_config.acoustic_pressure_coupling))
        self.vel_x += v_c * u
        self.vel_y += v_c * v
        self.pressure_pde += p_c * p

    def _update_shock_failure(self, grid, rows, cols, counters, events):
        if not self.fluid_config.shock_failure_enabled:
            return
        if self.pressure_pde.shape != (rows, cols):
            return

        p = self.pressure_pde.astype(np.float32)
        p_l = np.empty_like(p)
        p_r = np.empty_like(p)
        p_t = np.empty_like(p)
        p_b = np.empty_like(p)
        p_l[:, 0] = p[:, 0]
        p_l[:, 1:] = p[:, :-1]
        p_r[:, -1] = p[:, -1]
        p_r[:, :-1] = p[:, 1:]
        p_t[0, :] = p[0, :]
        p_t[1:, :] = p[:-1, :]
        p_b[-1, :] = p[-1, :]
        p_b[:-1, :] = p[1:, :]
        grad_p = np.sqrt((np.float32(0.5) * (p_r - p_l) * np.float32(PHYSICS.dx_inv)) ** 2 +
                         (np.float32(0.5) * (p_b - p_t) * np.float32(PHYSICS.dy_inv)) ** 2).astype(np.float32)

        u = self.vel_x.astype(np.float32)
        v = self.vel_y.astype(np.float32)
        u_l = np.empty_like(u)
        u_r = np.empty_like(u)
        u_t = np.empty_like(u)
        u_b = np.empty_like(u)
        v_l = np.empty_like(v)
        v_r = np.empty_like(v)
        v_t = np.empty_like(v)
        v_b = np.empty_like(v)
        u_l[:, 0] = u[:, 0]
        u_l[:, 1:] = u[:, :-1]
        u_r[:, -1] = u[:, -1]
        u_r[:, :-1] = u[:, 1:]
        u_t[0, :] = u[0, :]
        u_t[1:, :] = u[:-1, :]
        u_b[-1, :] = u[-1, :]
        u_b[:-1, :] = u[1:, :]
        v_l[:, 0] = v[:, 0]
        v_l[:, 1:] = v[:, :-1]
        v_r[:, -1] = v[:, -1]
        v_r[:, :-1] = v[:, 1:]
        v_t[0, :] = v[0, :]
        v_t[1:, :] = v[:-1, :]
        v_b[-1, :] = v[-1, :]
        v_b[:-1, :] = v[1:, :]

        du_dx = np.float32(0.5) * (u_r - u_l) * np.float32(PHYSICS.dx_inv)
        dv_dy = np.float32(0.5) * (v_b - v_t) * np.float32(PHYSICS.dy_inv)
        du_dy = np.float32(0.5) * (u_b - u_t) * np.float32(PHYSICS.dy_inv)
        dv_dx = np.float32(0.5) * (v_r - v_l) * np.float32(PHYSICS.dx_inv)
        shear_rate = np.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2).astype(np.float32)

        grad_w = np.float32(max(0.0, self.fluid_config.shock_pressure_gradient_weight))
        shear_w = np.float32(max(0.0, self.fluid_config.shock_shear_weight))
        eq_stress = grad_w * grad_p + shear_w * np.float32(PHYSICS.rho_air) * shear_rate
        yield_thr = np.float32(max(1.0, self.fluid_config.shock_yield_threshold))
        damage_scale = np.float32(max(0.0, self.fluid_config.shock_damage_scale))
        spall_prob_scale = np.float32(max(0.0, self.fluid_config.spallation_probability_scale))

        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    continue
                mat_data = self._get(mat)
                mat_type = str(mat_data.get("type", "air"))
                if mat_type not in ("solid", "powder"):
                    continue

                overload = float(eq_stress[row][col] / yield_thr)
                if overload <= 1.0:
                    continue

                fragility = float(self._mat_value(mat_data, "shock_fragility", 1.0))
                if fragility <= 0.0:
                    continue

                damage = damage_scale * (overload - 1.0) * fragility
                self.integrity[row][col] = max(0.0, self.integrity[row][col] - damage)

                if self.integrity[row][col] > 0.05:
                    continue

                spall_product = int(self._mat_value(mat_data, "spall_product", self.material_ids.get("sand", 1)))
                p_spall = min(1.0, spall_prob_scale * (overload - 1.0) * fragility)
                if np.random.random() < p_spall:
                    if self._set_cell_material(grid, row, col, spall_product):
                        counters["changes"] += 1
                        events.append({"type": "shock_spall", "row": row, "col": col, "from": mat, "to": spall_product})

    def _mac_to_center_velocity(self):
        self.vel_x = 0.5 * (self.mac_u[:, :-1] + self.mac_u[:, 1:])
        self.vel_y = 0.5 * (self.mac_v[:-1, :] + self.mac_v[1:, :])

    def _center_to_mac_velocity(self):
        rows, cols = self.vel_x.shape
        self.mac_u.fill(0.0)
        self.mac_v.fill(0.0)
        if cols > 1:
            self.mac_u[:, 1:-1] = 0.5 * (self.vel_x[:, :-1] + self.vel_x[:, 1:])
        if rows > 1:
            self.mac_v[1:-1, :] = 0.5 * (self.vel_y[:-1, :] + self.vel_y[1:, :])

    def _build_mac_divergence(self, solid_mask):
        dx_inv = np.float32(PHYSICS.dx_inv)
        dy_inv = np.float32(PHYSICS.dy_inv)
        div = ((self.mac_u[:, 1:] - self.mac_u[:, :-1]) * dx_inv +
               (self.mac_v[1:, :] - self.mac_v[:-1, :]) * dy_inv).astype(np.float32)
        div[solid_mask] = 0.0
        self.divergence_pde = div
        return div

    def _dilate_mask(self, mask, passes=1):
        out = mask.astype(np.bool_, copy=True)
        for _ in range(max(0, int(passes))):
            pad = np.pad(out, 1, mode="constant", constant_values=False)
            out = (
                pad[1:-1, 1:-1]
                | pad[:-2, 1:-1]
                | pad[2:, 1:-1]
                | pad[1:-1, :-2]
                | pad[1:-1, 2:]
            )
        return out

    def _build_pressure_active_mask(self, rhs, solid_mask):
        if not bool(self.fluid_config.pde_sparse_pressure_enabled):
            active = ~solid_mask
            self.active_mask = active
            return active

        threshold = np.float32(max(0.0, float(self.fluid_config.pde_pressure_active_divergence_threshold)))
        active = (np.abs(rhs) >= threshold) & (~solid_mask)
        dilation = int(max(0, self.fluid_config.pde_pressure_active_dilation))
        if dilation > 0:
            active = self._dilate_mask(active, dilation) & (~solid_mask)

        active_cells = int(np.count_nonzero(active))
        min_cells = max(1, int(0.02 * active.size))
        if active_cells < min_cells:
            active = ~solid_mask

        self.active_mask = active
        return active

    def _get_pressure_iteration_budget(self):
        base_iters = int(max(1, self.fluid_config.pde_jacobi_iterations))
        min_iters = int(max(1, self.fluid_config.pde_pressure_iterations_min))
        max_iters = int(max(min_iters, self.fluid_config.pde_pressure_iterations_max))
        budget_ms = float(max(0.0, self.fluid_config.pde_pressure_budget_ms))
        iters_target = int(np.clip(base_iters, min_iters, max_iters))

        if bool(self.fluid_config.pde_pressure_budget_adapt):
            prev_fluid_ms = float(self.substep_timings.get("fluids", 0.0))
            if budget_ms > 0.0 and prev_fluid_ms > 0.0:
                ratio = np.clip(budget_ms / max(prev_fluid_ms, 1.0e-6), 0.4, 1.8)
                iters_target = int(np.clip(round(iters_target * ratio), min_iters, max_iters))

        return min_iters, max_iters, iters_target, budget_ms

    def _apply_multigrid_scaffold(self, p, rhs, solid_mask):
        if not bool(self.fluid_config.pde_multigrid_enabled):
            return p

        rows, cols = p.shape
        if rows < 8 or cols < 8:
            return p

        r2 = (rows // 2) * 2
        c2 = (cols // 2) * 2
        if r2 < 4 or c2 < 4:
            return p

        p_f = p[:r2, :c2]
        rhs_f = rhs[:r2, :c2]
        solid_f = solid_mask[:r2, :c2]

        p_c = (
            p_f[0::2, 0::2]
            + p_f[0::2, 1::2]
            + p_f[1::2, 0::2]
            + p_f[1::2, 1::2]
        ).astype(np.float32) * np.float32(0.25)
        rhs_c = (
            rhs_f[0::2, 0::2]
            + rhs_f[0::2, 1::2]
            + rhs_f[1::2, 0::2]
            + rhs_f[1::2, 1::2]
        ).astype(np.float32) * np.float32(0.25)
        solid_c = (
            solid_f[0::2, 0::2]
            | solid_f[0::2, 1::2]
            | solid_f[1::2, 0::2]
            | solid_f[1::2, 1::2]
        )

        dx2_c = np.float32((2.0 * PHYSICS.dx) * (2.0 * PHYSICS.dx))
        dy2_c = np.float32((2.0 * PHYSICS.dy) * (2.0 * PHYSICS.dy))
        denom_c = np.float32(2.0 * (dx2_c + dy2_c))
        smooth_iters = int(max(1, self.fluid_config.pde_multigrid_presmooth + self.fluid_config.pde_multigrid_postsmooth))

        for _ in range(smooth_iters):
            p_l = np.empty_like(p_c)
            p_r = np.empty_like(p_c)
            p_t = np.empty_like(p_c)
            p_b = np.empty_like(p_c)
            p_l[:, 0] = p_c[:, 0]
            p_l[:, 1:] = p_c[:, :-1]
            p_r[:, -1] = p_c[:, -1]
            p_r[:, :-1] = p_c[:, 1:]
            p_t[0, :] = p_c[0, :]
            p_t[1:, :] = p_c[:-1, :]
            p_b[-1, :] = p_c[-1, :]
            p_b[:-1, :] = p_c[1:, :]

            p_new = (((p_l + p_r) * dy2_c + (p_t + p_b) * dx2_c - rhs_c * dx2_c * dy2_c) / denom_c).astype(np.float32)
            p_new[solid_c] = 0.0
            p_c = p_new

        p_prolong = np.repeat(np.repeat(p_c, 2, axis=0), 2, axis=1).astype(np.float32)
        p_out = p.copy()
        p_out[:r2, :c2] = np.float32(0.5) * p_out[:r2, :c2] + np.float32(0.5) * p_prolong[:r2, :c2]
        p_out[solid_mask] = 0.0
        return p_out

    def _jacobi_pressure_solve(self, rhs, solid_mask, active_mask=None):
        """Solve ∇²p = rhs with adaptive Jacobi iterations and optional sparse/numba execution."""
        p = self.pressure_pde.astype(np.float32, copy=True)
        p = self._apply_multigrid_scaffold(p, rhs.astype(np.float32, copy=False), solid_mask)
        dx2 = np.float32(PHYSICS.dx * PHYSICS.dx)
        dy2 = np.float32(PHYSICS.dy * PHYSICS.dy)
        denom = np.float32(2.0 * (dx2 + dy2))
        min_iters, max_iters, iters_target, budget_ms = self._get_pressure_iteration_budget()
        active = (~solid_mask) if active_mask is None else active_mask.astype(np.bool_, copy=False)

        use_numba = bool(self.fluid_config.pde_numba_enabled) and self._numba_available
        hit_budget = False
        residual = np.float32(0.0)
        iters_used = 0
        start_t = time.perf_counter()

        btype = self.fluid_config.pde_boundary_type
        boundary_mode = 1 if btype in (BoundaryConditionType.OPEN, BoundaryConditionType.OUTLET) else 0
        residual_tol = np.float32(max(1.0e-8, float(self.fluid_config.pde_pressure_residual_tolerance)))
        iter_cap = int(np.clip(iters_target, min_iters, max_iters))

        if use_numba:
            p, iters_used, residual = _jacobi_pressure_numba(
                p,
                rhs.astype(np.float32, copy=False),
                solid_mask.astype(np.bool_, copy=False),
                active.astype(np.bool_, copy=False),
                self.density_field.astype(np.float32, copy=False),
                dx2,
                dy2,
                denom,
                int(min_iters),
                int(iter_cap),
                residual_tol,
                boundary_mode,
                bool(self.fluid_config.free_surface_enabled),
                np.float32(PHYSICS.rho_air * 1.05),
            )
        else:
            rhs_scale = np.sqrt(np.mean((rhs[active] ** 2).astype(np.float32))) if np.any(active) else np.float32(1.0)
            rhs_scale = np.float32(max(float(rhs_scale), 1.0e-12))

            for it in range(iter_cap):
                p_l = np.empty_like(p)
                p_r = np.empty_like(p)
                p_t = np.empty_like(p)
                p_b = np.empty_like(p)
                p_l[:, 0] = p[:, 0]
                p_l[:, 1:] = p[:, :-1]
                p_r[:, -1] = p[:, -1]
                p_r[:, :-1] = p[:, 1:]
                p_t[0, :] = p[0, :]
                p_t[1:, :] = p[:-1, :]
                p_b[-1, :] = p[-1, :]
                p_b[:-1, :] = p[1:, :]

                p_new = p.copy()
                update = (((p_l + p_r) * dy2 + (p_t + p_b) * dx2 - rhs * dx2 * dy2) / denom).astype(np.float32)
                p_new[active] = update[active]
                p_new[~active] = 0.0
                p_new[solid_mask] = 0.0

                if btype in (BoundaryConditionType.OPEN, BoundaryConditionType.OUTLET):
                    if p_new.shape[1] > 1:
                        p_new[:, 0] = 0.0
                        p_new[:, -1] = 0.0
                    if p_new.shape[0] > 1:
                        p_new[0, :] = 0.0
                        p_new[-1, :] = 0.0
                else:
                    if p_new.shape[1] > 1:
                        p_new[:, 0] = p_new[:, 1]
                        p_new[:, -1] = p_new[:, -2]
                    if p_new.shape[0] > 1:
                        p_new[0, :] = p_new[1, :]
                        p_new[-1, :] = p_new[-2, :]

                if self.fluid_config.free_surface_enabled:
                    air_mask = (self.density_field <= np.float32(PHYSICS.rho_air * 1.05))
                    p_new[air_mask] *= np.float32(0.25)

                p_l_n = np.empty_like(p_new)
                p_r_n = np.empty_like(p_new)
                p_t_n = np.empty_like(p_new)
                p_b_n = np.empty_like(p_new)
                p_l_n[:, 0] = p_new[:, 0]
                p_l_n[:, 1:] = p_new[:, :-1]
                p_r_n[:, -1] = p_new[:, -1]
                p_r_n[:, :-1] = p_new[:, 1:]
                p_t_n[0, :] = p_new[0, :]
                p_t_n[1:, :] = p_new[:-1, :]
                p_b_n[-1, :] = p_new[-1, :]
                p_b_n[:-1, :] = p_new[1:, :]

                if np.any(active):
                    lap = (((p_l_n - 2.0 * p_new + p_r_n) / dx2) + ((p_t_n - 2.0 * p_new + p_b_n) / dy2)).astype(np.float32)
                    rr = (lap - rhs)[active]
                    residual = np.float32(np.sqrt(np.mean(rr * rr)) / rhs_scale)
                else:
                    residual = np.float32(0.0)

                p = p_new
                iters_used = it + 1
                if iters_used >= min_iters and residual <= residual_tol:
                    break
                if budget_ms > 0.0 and (time.perf_counter() - start_t) * 1000.0 > budget_ms and iters_used >= min_iters:
                    hit_budget = True
                    break

        if budget_ms > 0.0 and (time.perf_counter() - start_t) * 1000.0 > budget_ms:
            hit_budget = True

        self.pressure_solver_stats = {
            "iterations": int(iters_used),
            "residual": float(residual),
            "active_cells": int(np.count_nonzero(active)),
            "active_fraction": float(np.count_nonzero(active) / max(1, active.size)),
            "hit_budget": bool(hit_budget),
            "budget_ms": float(budget_ms),
            "used_numba": bool(use_numba),
            "target_iterations": int(iter_cap),
            "min_iterations": int(min_iters),
            "max_iterations": int(max_iters),
        }
        self.pressure_pde = p

    def _project_mac_velocity(self, solid_mask, rho_ref):
        """Projection step: u = u* - dt/rho * grad(p)."""
        dt_over_rho = np.float32(PHYSICS.dt / max(1e-6, rho_ref))
        dx_inv = np.float32(PHYSICS.dx_inv)
        dy_inv = np.float32(PHYSICS.dy_inv)

        if self.pressure_pde.shape[1] > 1:
            self.mac_u[:, 1:-1] -= dt_over_rho * (self.pressure_pde[:, 1:] - self.pressure_pde[:, :-1]) * dx_inv
        if self.pressure_pde.shape[0] > 1:
            self.mac_v[1:-1, :] -= dt_over_rho * (self.pressure_pde[1:, :] - self.pressure_pde[:-1, :]) * dy_inv

        self._apply_mac_boundaries(solid_mask)

    def _bilinear_sample(self, field, x, y):
        rows, cols = field.shape
        x = np.clip(x, 0.0, np.float32(cols - 1))
        y = np.clip(y, 0.0, np.float32(rows - 1))

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, cols - 1)
        y1 = np.clip(y0 + 1, 0, rows - 1)

        sx = (x - x0).astype(np.float32)
        sy = (y - y0).astype(np.float32)

        q00 = field[y0, x0]
        q10 = field[y0, x1]
        q01 = field[y1, x0]
        q11 = field[y1, x1]

        w00 = (1.0 - sx) * (1.0 - sy)
        w10 = sx * (1.0 - sy)
        w01 = (1.0 - sx) * sy
        w11 = sx * sy
        return (q00 * w00 + q10 * w10 + q01 * w01 + q11 * w11).astype(np.float32)

    def _advect_centered_field(self, field, vel_x, vel_y, direction=1.0):
        rows, cols = field.shape
        x_coords = np.arange(cols, dtype=np.float32)[None, :]
        y_coords = np.arange(rows, dtype=np.float32)[:, None]
        x = np.broadcast_to(x_coords, (rows, cols)).astype(np.float32)
        y = np.broadcast_to(y_coords, (rows, cols)).astype(np.float32)

        x_back = x - np.float32(direction * PHYSICS.dt * PHYSICS.dx_inv) * vel_x
        y_back = y - np.float32(direction * PHYSICS.dt * PHYSICS.dy_inv) * vel_y
        return self._bilinear_sample(field, x_back, y_back)

    def _advect_velocity_bfecc(self, solid_mask):
        u0 = self.vel_x.astype(np.float32, copy=True)
        v0 = self.vel_y.astype(np.float32, copy=True)

        if self.fluid_config.pde_use_bfecc:
            u_hat = self._advect_centered_field(u0, u0, v0, direction=1.0)
            v_hat = self._advect_centered_field(v0, u0, v0, direction=1.0)

            u_check = self._advect_centered_field(u_hat, u_hat, v_hat, direction=-1.0)
            v_check = self._advect_centered_field(v_hat, u_hat, v_hat, direction=-1.0)

            u_corr = u0 + 0.5 * (u0 - u_check)
            v_corr = v0 + 0.5 * (v0 - v_check)

            self.vel_x = self._advect_centered_field(u_corr.astype(np.float32), u0, v0, direction=1.0)
            self.vel_y = self._advect_centered_field(v_corr.astype(np.float32), u0, v0, direction=1.0)
        else:
            self.vel_x = self._advect_centered_field(u0, u0, v0, direction=1.0)
            self.vel_y = self._advect_centered_field(v0, u0, v0, direction=1.0)

        self.vel_x[solid_mask] = 0.0
        self.vel_y[solid_mask] = 0.0

    def _diffuse_velocity_implicit(self, nu_scalar, solid_mask):
        a = np.float32(max(0.0, nu_scalar) * PHYSICS.dt * PHYSICS.dx_inv * PHYSICS.dx_inv)
        if a <= 0.0:
            return

        iters = int(max(1, self.fluid_config.pde_viscosity_iterations))
        inv_beta = np.float32(1.0 / (1.0 + 4.0 * a))

        u0 = self.vel_x.astype(np.float32, copy=True)
        v0 = self.vel_y.astype(np.float32, copy=True)
        u = u0.copy()
        v = v0.copy()

        for _ in range(iters):
            u_l = np.empty_like(u)
            u_r = np.empty_like(u)
            u_t = np.empty_like(u)
            u_b = np.empty_like(u)
            u_l[:, 0] = u[:, 0]
            u_l[:, 1:] = u[:, :-1]
            u_r[:, -1] = u[:, -1]
            u_r[:, :-1] = u[:, 1:]
            u_t[0, :] = u[0, :]
            u_t[1:, :] = u[:-1, :]
            u_b[-1, :] = u[-1, :]
            u_b[:-1, :] = u[1:, :]
            u = (u0 + a * (u_l + u_r + u_t + u_b)) * inv_beta

            v_l = np.empty_like(v)
            v_r = np.empty_like(v)
            v_t = np.empty_like(v)
            v_b = np.empty_like(v)
            v_l[:, 0] = v[:, 0]
            v_l[:, 1:] = v[:, :-1]
            v_r[:, -1] = v[:, -1]
            v_r[:, :-1] = v[:, 1:]
            v_t[0, :] = v[0, :]
            v_t[1:, :] = v[:-1, :]
            v_b[-1, :] = v[-1, :]
            v_b[:-1, :] = v[1:, :]
            v = (v0 + a * (v_l + v_r + v_t + v_b)) * inv_beta

            u[solid_mask] = 0.0
            v[solid_mask] = 0.0

        self.vel_x = u.astype(np.float32)
        self.vel_y = v.astype(np.float32)

    def _apply_buoyancy_force(self, solid_mask, rho_ref):
        if not isinstance(self.temperature, np.ndarray) or self.temperature.shape != self.vel_x.shape:
            return

        beta = np.float32(max(0.0, self.fluid_config.thermal_expansion_beta))
        t_ref = np.float32(self.thermal_config.ambient_temp)
        g = np.float32(PHYSICS.g)
        rho = self.density_field.astype(np.float32)
        rho_ref = np.float32(max(1e-6, rho_ref))
        buoyancy_scale = np.float32(self.fluid_config.pde_buoyancy_scale)

        thermal_term = -g * beta * (self.temperature.astype(np.float32) - t_ref)
        density_term = -g * ((rho - rho_ref) / rho_ref)
        force_y = buoyancy_scale * (thermal_term + density_term)
        force_y[solid_mask] = 0.0
        self.vel_y += np.float32(PHYSICS.dt) * force_y

    def _apply_vorticity_confinement(self, solid_mask):
        eps = np.float32(max(0.0, self.fluid_config.pde_vorticity_eps))
        if eps <= 0.0:
            return

        dx_inv = np.float32(PHYSICS.dx_inv)
        dy_inv = np.float32(PHYSICS.dy_inv)

        u = self.vel_x.astype(np.float32)
        v = self.vel_y.astype(np.float32)

        v_l = np.empty_like(v)
        v_r = np.empty_like(v)
        u_t = np.empty_like(u)
        u_b = np.empty_like(u)
        v_l[:, 0] = v[:, 0]
        v_l[:, 1:] = v[:, :-1]
        v_r[:, -1] = v[:, -1]
        v_r[:, :-1] = v[:, 1:]
        u_t[0, :] = u[0, :]
        u_t[1:, :] = u[:-1, :]
        u_b[-1, :] = u[-1, :]
        u_b[:-1, :] = u[1:, :]

        omega = 0.5 * ((v_r - v_l) * dx_inv - (u_b - u_t) * dy_inv)
        omega_abs = np.abs(omega)

        w_l = np.empty_like(omega_abs)
        w_r = np.empty_like(omega_abs)
        w_t = np.empty_like(omega_abs)
        w_b = np.empty_like(omega_abs)
        w_l[:, 0] = omega_abs[:, 0]
        w_l[:, 1:] = omega_abs[:, :-1]
        w_r[:, -1] = omega_abs[:, -1]
        w_r[:, :-1] = omega_abs[:, 1:]
        w_t[0, :] = omega_abs[0, :]
        w_t[1:, :] = omega_abs[:-1, :]
        w_b[-1, :] = omega_abs[-1, :]
        w_b[:-1, :] = omega_abs[1:, :]

        grad_x = 0.5 * (w_r - w_l) * dx_inv
        grad_y = 0.5 * (w_b - w_t) * dy_inv
        grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y) + np.float32(1e-6)

        n_x = grad_x / grad_mag
        n_y = grad_y / grad_mag

        force_x = eps * n_y * omega
        force_y = -eps * n_x * omega
        force_x[solid_mask] = 0.0
        force_y[solid_mask] = 0.0

        self.vel_x += np.float32(PHYSICS.dt) * force_x
        self.vel_y += np.float32(PHYSICS.dt) * force_y

    def _advect_scalar_pde_fields(self, solid_mask):
        if not self.fluid_config.pde_scalar_advection:
            return

        if isinstance(self.temperature, np.ndarray) and self.temperature.shape == self.vel_x.shape:
            self.temperature = self._advect_centered_field(
                self.temperature.astype(np.float32),
                self.vel_x,
                self.vel_y,
                direction=1.0,
            )
        if isinstance(self.smoke_density, np.ndarray) and self.smoke_density.shape == self.vel_x.shape:
            smoke_new = self._advect_centered_field(
                self.smoke_density.astype(np.float32),
                self.vel_x,
                self.vel_y,
                direction=1.0,
            )
            self.smoke_density = np.clip(smoke_new, 0.0, 1.0).astype(np.float32)
        if isinstance(self.oxygen_level, np.ndarray) and self.oxygen_level.shape == self.vel_x.shape:
            oxygen_new = self._advect_centered_field(
                self.oxygen_level.astype(np.float32),
                self.vel_x,
                self.vel_y,
                direction=1.0,
            )
            self.oxygen_level = np.clip(oxygen_new, 0.0, 1.0).astype(np.float32)

        self.temperature[solid_mask] = self.temperature[solid_mask]

    def _stage_pde_fluids(self, grid, rows, cols):
        """Phase 2 pipeline (Schritte 9-14):
        9b MAC setup, 13 buoyancy, 14 vorticity confinement,
        11 viscosity diffusion, 12/BFECC advection,
        9 Poisson solve, 10 projection.
        """
        if not self.fluid_config.pde_enabled:
            return

        pde_start = time.perf_counter()

        self._ensure_pde_state(rows, cols)
        self._ensure_thermal_state(rows, cols)

        solid_mask, nu_field, _, _ = self._rebuild_density_and_solid_masks(grid, rows, cols)
        active_density = self.density_field[~solid_mask]
        rho_ref = float(np.mean(active_density)) if active_density.size else float(PHYSICS.rho_air)
        nu_scalar = float(np.mean(nu_field[~solid_mask])) if np.any(~solid_mask) else float(self.fluid_config.pde_kinematic_viscosity)

        # Build staggered face velocities from collocated state
        self._center_to_mac_velocity()
        self._apply_mac_boundaries(solid_mask)
        self._apply_log_wind_inlet(solid_mask)
        self._mac_to_center_velocity()

        # External forces before pressure solve
        self._apply_buoyancy_force(solid_mask, rho_ref)
        self._apply_vorticity_confinement(solid_mask)
        self._apply_porous_drag(solid_mask, rho_ref)
        self._apply_acoustic_substep_with_pml(solid_mask, rho_ref)

        # Implicit viscosity (Jacobi)
        self._diffuse_velocity_implicit(nu_scalar, solid_mask)

        # Semi-Lagrangian / BFECC advection
        self._advect_velocity_bfecc(solid_mask)

        # Projection
        self._center_to_mac_velocity()
        self._apply_mac_boundaries(solid_mask)
        self._apply_log_wind_inlet(solid_mask)
        div = self._build_mac_divergence(solid_mask)
        div_src = self._build_detonation_divergence_source(rows, cols, rho_ref)
        rhs = (np.float32(rho_ref / max(PHYSICS.dt, 1e-6)) * (div - div_src)).astype(np.float32)
        active_mask = self._build_pressure_active_mask(rhs, solid_mask)
        self._jacobi_pressure_solve(rhs, solid_mask, active_mask=active_mask)
        self._project_mac_velocity(solid_mask, rho_ref)
        self._apply_log_wind_inlet(solid_mask)
        self._mac_to_center_velocity()

        # Passive scalar advection through divergence-free u
        self._advect_scalar_pde_fields(solid_mask)

        post_div = self._build_mac_divergence(solid_mask)
        if np.any(~solid_mask):
            div_fluid = post_div[~solid_mask]
            div_rms = float(np.sqrt(np.mean((div_fluid * div_fluid).astype(np.float32))))
            div_max = float(np.max(np.abs(div_fluid)))
        else:
            div_rms = 0.0
            div_max = 0.0

        self.pde_validation_metrics = {
            "divergence_rms": div_rms,
            "divergence_max": div_max,
            "pressure_residual": float(self.pressure_solver_stats.get("residual", 0.0)),
            "pressure_iterations": int(self.pressure_solver_stats.get("iterations", 0)),
            "pressure_active_fraction": float(self.pressure_solver_stats.get("active_fraction", 0.0)),
            "pde_stage_ms": float((time.perf_counter() - pde_start) * 1000.0),
        }

    def _set_cell_material(self, grid, row, col, new_mat):
        old_mat = grid[row][col]
        if old_mat == new_mat:
            return False

        grid[row][col] = new_mat
        self.integrity[row][col] = 1.0
        self.reaction_progress[row][col] = 0.0
        self.phase_state[row][col] = self._phase_state_from_material(new_mat)
        self.phase_transition_progress[row][col] = 0.0
        self.phase_transition_target[row][col] = 0
        self.fuel_vapor[row][col] = np.float32(0.0)
        self.pyrolysis_progress[row][col] = np.float32(0.0)
        self.soot_mass_fraction[row][col] = np.float32(0.0)
        self.soot_number_density[row][col] = np.float32(0.0)
        self.catalyst_theta_fuel[row][col] = np.float32(0.0)
        self.catalyst_theta_o2[row][col] = np.float32(0.0)
        mat_data = self._get(new_mat)
        acid_strength = np.float32(max(0.0, float(self._mat_value(mat_data, "acid_strength", 0.0))))
        base_strength = np.float32(max(0.0, float(self._mat_value(mat_data, "base_strength", 0.0))))
        conc_scale = np.float32(self.chemistry_config.electrolyte_concentration_scale)
        self.h_plus[row][col] = np.float32(1.0e-7 + acid_strength * conc_scale)
        self.oh_minus[row][col] = np.float32(1.0e-7 + base_strength * conc_scale)
        self.ph_field[row][col] = np.float32(np.clip(-np.log10(max(float(self.h_plus[row][col]), 1.0e-12)), 0.0, 14.0))

        if new_mat == 0:
            self.saturation_level[row][col] = 0.0
            self.burn_stage[row][col] = 0
            self.burn_progress[row][col] = 0.0

        return True

    def _phase_state_from_material(self, mat_id):
        phase_name = self._get(mat_id)["type"] if mat_id in self.materials else "air"
        phase_map = {"air": 0, "powder": 1, "solid": 2, "liquid": 3, "gas": 4}
        return phase_map.get(phase_name, 0)

    def apply_spawn_state(self, grid, row, col, mat_id, rows, cols):
        self._ensure_thermal_state(rows, cols)
        self._ensure_fluid_state(rows, cols)
        self._ensure_chemical_state(grid, rows, cols)

        mat_data = self._get(mat_id)
        initial_temp = self._mat_value(mat_data, "initial_temp", None)
        if initial_temp is None:
            initial_temp = self.thermal_config.ambient_temp

        self.temperature[row][col] = float(initial_temp)
        self.integrity[row][col] = 1.0
        self.reaction_progress[row][col] = 0.0
        self.phase_state[row][col] = self._phase_state_from_material(mat_id)
        self.saturation_level[row][col] = 0.0
        self.phase_transition_progress[row][col] = 0.0
        self.phase_transition_target[row][col] = 0
        self.fuel_vapor[row][col] = np.float32(0.0)
        self.pyrolysis_progress[row][col] = np.float32(0.0)
        self.soot_mass_fraction[row][col] = np.float32(0.0)
        self.soot_number_density[row][col] = np.float32(0.0)
        self.catalyst_theta_fuel[row][col] = np.float32(0.0)
        self.catalyst_theta_o2[row][col] = np.float32(0.0)

        acid_strength = np.float32(max(0.0, float(self._mat_value(mat_data, "acid_strength", 0.0))))
        base_strength = np.float32(max(0.0, float(self._mat_value(mat_data, "base_strength", 0.0))))
        conc_scale = np.float32(self.chemistry_config.electrolyte_concentration_scale)
        self.h_plus[row][col] = np.float32(1.0e-7 + acid_strength * conc_scale)
        self.oh_minus[row][col] = np.float32(1.0e-7 + base_strength * conc_scale)
        self.ph_field[row][col] = np.float32(np.clip(-np.log10(max(float(self.h_plus[row][col]), 1.0e-12)), 0.0, 14.0))

        if mat_id == 0:
            self.burn_stage[row][col] = 0
            self.burn_progress[row][col] = 0.0
            self.smoke_density[row][col] = 0.0

    def _is_combustible(self, mat_data):
        return self._mat_value(mat_data, "burn_rate", 0.0) > 0.0

    def _neighbor_is_flaming(self, row, col, rows, cols):
        for y in range(max(0, row - 1), min(rows, row + 2)):
            for x in range(max(0, col - 1), min(cols, col + 2)):
                if y == row and x == col:
                    continue
                if self.burn_stage[y][x] == 3:
                    return True
        return False

    def _neighbor_is_hot(self, row, col, rows, cols, threshold):
        """True if any cardinal neighbor's temperature >= threshold."""
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if self._in_bounds(nr, nc, rows, cols) and self.temperature[nr][nc] >= threshold:
                return True
        return False

    def _cell_is_jammed(self, row, col, tick_index):
        return self.jammed_until[row][col] > tick_index

    def _set_jam(self, row, col, tick_index, rng, mat_data):
        jam_scale = self._mat_value(mat_data, "jam_chance", 1.0)
        jam_ticks = int(
            self.granular_config.min_jam_ticks +
            ((self.granular_config.max_jam_ticks - self.granular_config.min_jam_ticks) * jam_scale * rng.random())
        )
        self.jammed_until[row][col] = tick_index + max(1, jam_ticks)

    def _set_bias(self, source_row, source_col, target_row, target_col):
        self.lateral_bias[target_row][target_col] = self.lateral_bias[source_row][source_col]
        self.lateral_bias[source_row][source_col] = 0

    def _set_bias_from_direction(self, source_row, source_col, target_row, target_col):
        direction = target_col - source_col
        if direction < 0:
            self.lateral_bias[target_row][target_col] = -1
        elif direction > 0:
            self.lateral_bias[target_row][target_col] = 1
        else:
            self.lateral_bias[target_row][target_col] = int(self.lateral_bias[source_row][source_col] * self.granular_config.inertia_decay)
        self.lateral_bias[source_row][source_col] = 0

    def _particle_state_grids(self):
        return [
            self.temperature,
            self.burn_stage,
            self.burn_progress,
            self.ignition_cooldown_until,
            self.integrity,
            self.saturation_level,
            self.phase_state,
            self.reaction_progress,
            self.phase_cooldown_until,
            self.phase_transition_progress,
            self.phase_transition_target,
            self.mix_ratio,
        ]

    def _state_grid_ready(self, state_grid, rows, cols):
        return len(state_grid) == rows and (rows == 0 or len(state_grid[0]) == cols)

    def _state_grid_cols(self, state_grid):
        if isinstance(state_grid, np.ndarray):
            if state_grid.ndim < 2 or state_grid.shape[0] == 0:
                return 0
            return int(state_grid.shape[1])
        if len(state_grid) == 0:
            return 0
        return len(state_grid[0])

    def _state_has_cell(self, state_grid, row, col):
        if isinstance(state_grid, np.ndarray):
            return state_grid.ndim >= 2 and row < state_grid.shape[0] and col < state_grid.shape[1]
        return row < len(state_grid) and col < (len(state_grid[row]) if row < len(state_grid) else 0)

    def _reset_empty_state(self, row, col):
        if self._state_has_cell(self.temperature, row, col):
            self.temperature[row][col] = self.thermal_config.ambient_temp
        if self._state_has_cell(self.burn_stage, row, col):
            self.burn_stage[row][col] = 0
        if self._state_has_cell(self.burn_progress, row, col):
            self.burn_progress[row][col] = 0.0
        if self._state_has_cell(self.ignition_cooldown_until, row, col):
            self.ignition_cooldown_until[row][col] = 0
        if self._state_has_cell(self.integrity, row, col):
            self.integrity[row][col] = 1.0
        if self._state_has_cell(self.saturation_level, row, col):
            self.saturation_level[row][col] = 0.0
        if self._state_has_cell(self.phase_state, row, col):
            self.phase_state[row][col] = 0
        if self._state_has_cell(self.reaction_progress, row, col):
            self.reaction_progress[row][col] = 0.0
        if self._state_has_cell(self.phase_cooldown_until, row, col):
            self.phase_cooldown_until[row][col] = 0
        if self._state_has_cell(self.phase_transition_progress, row, col):
            self.phase_transition_progress[row][col] = 0.0
        if self._state_has_cell(self.phase_transition_target, row, col):
            self.phase_transition_target[row][col] = 0
        if self._state_has_cell(self.mix_ratio, row, col):
            self.mix_ratio[row][col] = 0.0

    def _move_particle_state(self, row, col, target_row, target_col, rows, cols):
        for state_grid in self._particle_state_grids():
            if not self._state_grid_ready(state_grid, rows, cols):
                continue
            state_grid[target_row][target_col] = state_grid[row][col]
        self._reset_empty_state(row, col)

    def _swap_particle_state(self, row, col, target_row, target_col, rows, cols):
        for state_grid in self._particle_state_grids():
            if not self._state_grid_ready(state_grid, rows, cols):
                continue
            source_value = state_grid[row][col]
            state_grid[row][col] = state_grid[target_row][target_col]
            state_grid[target_row][target_col] = source_value

    def _move(self, grid, row, col, target_row, target_col, moved, mat):
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        self._move_particle_state(row, col, target_row, target_col, rows, cols)
        grid[row][col] = 0
        grid[target_row][target_col] = mat
        moved[target_row][target_col] = True
        return True

    def _swap(self, grid, row, col, target_row, target_col, moved, mat):
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        self._swap_particle_state(row, col, target_row, target_col, rows, cols)
        target_mat = grid[target_row][target_col]
        grid[target_row][target_col] = mat
        grid[row][col] = target_mat
        moved[target_row][target_col] = True
        moved[row][col] = True
        return True

    def _move_with_state(self, grid, row, col, target_row, target_col, moved, mat):
        self._set_bias_from_direction(row, col, target_row, target_col)
        self.jammed_until[target_row][target_col] = 0
        self.jammed_until[row][col] = 0
        return self._move(grid, row, col, target_row, target_col, moved, mat)

    def _swap_with_state(self, grid, row, col, target_row, target_col, moved, mat):
        source_bias = self.lateral_bias[row][col]
        target_bias = self.lateral_bias[target_row][target_col]
        self.lateral_bias[target_row][target_col] = source_bias
        self.lateral_bias[row][col] = target_bias
        self.jammed_until[target_row][target_col] = 0
        self.jammed_until[row][col] = 0
        return self._swap(grid, row, col, target_row, target_col, moved, mat)

    def _sink_probability(self, mover, target):
        density_diff = mover["density"] - target["density"]
        if density_diff <= 0:
            return 0.0
        base = 0.015 + min(0.08, density_diff / 10000)
        damping = 1.0 - (target["viscosity"] * 0.6) - mover["drag"]
        return max(0.01, min(0.18, base * max(0.2, damping)))

    def _friction_mobility(self, mat_data, rng, moving=False):
        static_friction = self._mat_value(mat_data, "friction_static", 0.58)
        dynamic_friction = self._mat_value(mat_data, "friction_dynamic", 0.38)
        threshold = dynamic_friction if moving else static_friction
        return rng.random() >= threshold

    def _effective_viscosity(self, mat_data, shear_rate=1.0):
        base_viscosity = self._mat_value(mat_data, "viscosity", 0.2)
        if not self.fluid_config.enable_non_newtonian:
            return base_viscosity

        shear_index = self._mat_value(mat_data, "shear_index", 1.0)
        if abs(shear_index - 1.0) < 1e-6:
            return base_viscosity

        effective = base_viscosity * (max(0.2, shear_rate) ** (shear_index - 1.0))
        return max(0.03, min(4.0, effective))

    def _flow_span_from_viscosity(self, viscosity):
        scaled = int(self.fluid_config.flow_span_base / (1.0 + (viscosity * 3.0)))
        return max(1, min(self.fluid_config.flow_span_base, scaled))

    def _local_shear_rate(self, grid, row, col, rows, cols):
        occupied = 0
        liquid_neighbors = 0
        for y in range(max(0, row - 1), min(rows, row + 2)):
            for x in range(max(0, col - 1), min(cols, col + 2)):
                if y == row and x == col:
                    continue
                if grid[y][x] != 0:
                    occupied += 1
                    if self._get(grid[y][x])["type"] == "liquid":
                        liquid_neighbors += 1
        return 1.0 + ((occupied - liquid_neighbors) * 0.1)

    def _compute_column_heights(self, grid, rows, cols):
        heights = [0 for _ in range(cols)]
        for col in range(cols):
            first_filled = rows
            for row in range(rows):
                if grid[row][col] != 0:
                    first_filled = row
                    break
            heights[col] = rows - first_filled
        return heights

    def _estimate_local_slope(self, column_heights, col):
        left_height = column_heights[col - 1] if col - 1 >= 0 else column_heights[col]
        right_height = column_heights[col + 1] if col + 1 < len(column_heights) else column_heights[col]
        height_diff = abs(right_height - left_height)
        return math.degrees(math.atan2(height_diff, 2.0))

    def _powder_should_avalanche(self, mat_data, local_slope_angle, rng):
        base_repose = self._mat_value(mat_data, "repose_angle", self.granular_config.avalanche_threshold)
        theta_start = max(0.0, base_repose - self.granular_config.theta_start_offset)
        if local_slope_angle <= theta_start:
            return False
        overdrive = min(1.0, (local_slope_angle - theta_start) / max(1.0, 90 - theta_start))
        return rng.random() < (0.18 + overdrive * 0.7)

    def _maybe_apply_jam(self, grid, row, col, rows, cols, rng, mat_data):
        occupied_neighbors = 0
        for y in range(max(0, row - 1), min(rows, row + 2)):
            for x in range(max(0, col - 1), min(cols, col + 2)):
                if y == row and x == col:
                    continue
                if grid[y][x] != 0:
                    occupied_neighbors += 1

        if occupied_neighbors < 5:
            return False

        jam_chance = self.granular_config.jam_probability_scale * (occupied_neighbors / 8.0)
        jam_chance *= self._mat_value(mat_data, "jam_chance", 1.0)
        if rng.random() < jam_chance:
            return True
        return False

    def _try_move_or_displace(self, grid, row, col, target_row, target_col, moved, mat, rows, cols, rng, counters):
        if not self._in_bounds(target_row, target_col, rows, cols):
            return False

        target_mat = grid[target_row][target_col]
        if target_mat == 0:
            counters["changes"] += 2
            return self._move_with_state(grid, row, col, target_row, target_col, moved, mat)

        mover_data = self._get(mat)
        target_data = self._get(target_mat)

        if target_data["type"] == "solid":
            return False

        if mover_data["density"] > target_data["density"]:
            if rng.random() < self._sink_probability(mover_data, target_data):
                counters["changes"] += 2
                return self._swap_with_state(grid, row, col, target_row, target_col, moved, mat)
        return False

    def _update_powder(self, grid, row, col, moved, mat, rows, cols, rng, counters, tick_index, column_heights):
        mat_data = self._get(mat)
        if self._cell_is_jammed(row, col, tick_index):
            return

        is_moving = abs(self.lateral_bias[row][col]) > 0
        if not self._friction_mobility(mat_data, rng, moving=is_moving):
            return

        if self._try_move_or_displace(grid, row, col, row + 1, col, moved, mat, rows, cols, rng, counters):
            return

        bias = self.lateral_bias[row][col]
        if bias < 0:
            directions = [-1, 1]
        elif bias > 0:
            directions = [1, -1]
        else:
            directions = [-1, 1]
            rng.shuffle(directions)

        for direction in directions:
            if self._friction_mobility(mat_data, rng, moving=True) and self._try_move_or_displace(grid, row, col, row + 1, col + direction, moved, mat, rows, cols, rng, counters):
                return

        if self._maybe_apply_jam(grid, row, col, rows, cols, rng, mat_data):
            self._set_jam(row, col, tick_index, rng, mat_data)
            return

        if row + 1 >= rows:
            return

        support_below = grid[row + 1][col] != 0
        if not support_below:
            return

        local_slope_angle = self._estimate_local_slope(column_heights, col)
        if not self._powder_should_avalanche(mat_data, local_slope_angle, rng):
            return

        if bias == 0:
            rng.shuffle(directions)
        for direction in directions:
            side_col = col + direction
            below_side_row = row + 1
            below_side_col = col + direction

            if not self._in_bounds(row, side_col, rows, cols) or not self._in_bounds(below_side_row, below_side_col, rows, cols):
                continue

            side_empty = grid[row][side_col] == 0
            below_side_occupied = grid[below_side_row][below_side_col] != 0
            if side_empty and below_side_occupied:
                counters["changes"] += 2
                self._move_with_state(grid, row, col, row, side_col, moved, mat)
                return

    def _update_liquid(self, grid, row, col, moved, mat, rows, cols, rng, counters):
        mat_data = self._get(mat)
        shear_rate = self._local_shear_rate(grid, row, col, rows, cols)
        effective_viscosity = self._effective_viscosity(mat_data, shear_rate)
        flow_span = self._flow_span_from_viscosity(effective_viscosity)

        if self._try_move_or_displace(grid, row, col, row + 1, col, moved, mat, rows, cols, rng, counters):
            return

        left_pressure = self.fluid_pressure[row][col - 1] if col - 1 >= 0 else self.fluid_pressure[row][col]
        right_pressure = self.fluid_pressure[row][col + 1] if col + 1 < cols else self.fluid_pressure[row][col]
        pressure_delta = right_pressure - left_pressure

        if pressure_delta > 0.05:
            directions = [-1, 1]
        elif pressure_delta < -0.05:
            directions = [1, -1]
        else:
            directions = [-1, 1]
            rng.shuffle(directions)

        if col <= 1:
            directions = [1, -1]
        elif col >= cols - 2:
            directions = [-1, 1]

        for direction in directions:
            if self._try_move_or_displace(grid, row, col, row + 1, col + direction, moved, mat, rows, cols, rng, counters):
                return

        spread = max(1, min(self._mat_value(mat_data, "dispersion", 3), flow_span))
        if col <= 1 or col >= cols - 2:
            spread = max(1, int(spread * (1.0 - self.fluid_config.boundary_damping)))
        for direction in directions:
            for distance in range(1, spread + 1):
                target_col = col + (direction * distance)
                if not self._in_bounds(row, target_col, rows, cols):
                    break

                if self._try_move_or_displace(grid, row, col, row, target_col, moved, mat, rows, cols, rng, counters):
                    return

                if grid[row][target_col] != 0:
                    break

    def _update_gas(self, grid, row, col, moved, mat, rows, cols, rng, counters):
        if rng.random() < self.fluid_config.gas_turbulence:
            jitter_direction = rng.choice([-1, 1])
            if self._try_move_or_displace(grid, row, col, row - 1, col + jitter_direction, moved, mat, rows, cols, rng, counters):
                return

        if self._try_move_or_displace(grid, row, col, row - 1, col, moved, mat, rows, cols, rng, counters):
            return

        directions = [-1, 1]
        rng.shuffle(directions)
        for direction in directions:
            if self._try_move_or_displace(grid, row, col, row - 1, col + direction, moved, mat, rows, cols, rng, counters):
                return

        for direction in directions:
            if self._try_move_or_displace(grid, row, col, row, col + direction, moved, mat, rows, cols, rng, counters):
                return

    def _update_hydrostatic_balance(self, grid, rows, cols):
        self._ensure_fluid_state(rows, cols)
        for col in range(cols):
            pressure_head = 0.0
            for row in range(rows - 1, -1, -1):
                mat = grid[row][col]
                if mat != 0 and self._get(mat)["type"] == "liquid":
                    density = self._mat_value(self._get(mat), "density", 1000)
                    pressure_head += 1.0 * (density / 1000.0)
                    self.fluid_pressure[row][col] = pressure_head
                else:
                    self.fluid_pressure[row][col] = 0.0

        smoothing = max(0.0, min(1.0, self.fluid_config.pressure_smoothing))
        if smoothing <= 0.0:
            return

        for row in range(rows):
            for col in range(1, cols - 1):
                if grid[row][col] == 0:
                    continue
                center = self.fluid_pressure[row][col]
                neighbor_avg = (
                    self.fluid_pressure[row][col - 1]
                    + center
                    + self.fluid_pressure[row][col + 1]
                ) / 3.0
                self.fluid_pressure[row][col] = center + ((neighbor_avg - center) * smoothing)

    def _update_density_sorting(self, grid, rows, cols, moved, rng, counters):
        for row in range(rows - 2, -1, -1):
            cols_order = list(range(cols))
            rng.shuffle(cols_order)
            for col in cols_order:
                if moved[row][col]:
                    continue
                mat = grid[row][col]
                if mat == 0 or self._get(mat)["type"] != "liquid":
                    continue

                below = grid[row + 1][col]
                if below == 0 or self._get(below)["type"] != "liquid":
                    continue

                density_here = self._mat_value(self._get(mat), "density", 1000)
                density_below = self._mat_value(self._get(below), "density", 1000)
                if density_here > density_below:
                    density_gap = min(1.0, (density_here - density_below) / 1000.0)
                    swap_prob = self.fluid_config.density_swap_rate * density_gap
                    if rng.random() < swap_prob:
                        counters["changes"] += 2
                        self._swap_with_state(grid, row, col, row + 1, col, moved, mat)

    def _update_fluid_flow(self, grid, rows, cols, moved, rng, counters):
        for row in range(rows - 1, -1, -1):
            cols_order = list(range(cols))
            rng.shuffle(cols_order)
            for col in cols_order:
                if moved[row][col]:
                    continue

                mat = grid[row][col]
                if mat == 0:
                    continue

                mat_type = self._get(mat)["type"]
                if mat_type == "liquid":
                    self._update_liquid(grid, row, col, moved, mat, rows, cols, rng, counters)
                elif mat_type == "gas":
                    self._update_gas(grid, row, col, moved, mat, rows, cols, rng, counters)

    def _update_mixing(self, grid, rows, cols, rng):
        for row in range(rows):
            for col in range(cols):
                self.mix_ratio[row][col] = max(0.0, self.mix_ratio[row][col] - self.fluid_config.mix_decay)

        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0 or self._get(mat)["type"] != "liquid":
                    continue

                for n_row, n_col in ((row + 1, col), (row, col + 1)):
                    if not self._in_bounds(n_row, n_col, rows, cols):
                        continue
                    n_mat = grid[n_row][n_col]
                    if n_mat == 0 or self._get(n_mat)["type"] != "liquid" or n_mat == mat:
                        continue

                    here_compat = self._mat_value(self._get(mat), "mix_compatibility", 0.25)
                    there_compat = self._mat_value(self._get(n_mat), "mix_compatibility", 0.25)
                    mix_chance = max(0.02, min(0.6, (here_compat + there_compat) * 0.12))
                    if rng.random() < mix_chance:
                        self.mix_ratio[row][col] = min(1.0, self.mix_ratio[row][col] + 0.2)
                        self.mix_ratio[n_row][n_col] = min(1.0, self.mix_ratio[n_row][n_col] + 0.2)

    def _stage_mechanics(self, grid, rows, cols, moved, rng, counters, tick_index):
        self._ensure_granular_state(rows, cols)
        column_heights = self._compute_column_heights(grid, rows, cols)
        for row in range(rows - 2, -1, -1):
            cols_order = list(range(cols))
            rng.shuffle(cols_order)

            for col in cols_order:
                if moved[row][col]:
                    continue

                mat = grid[row][col]
                if mat == 0:
                    continue

                mat_type = self._get(mat)["type"]
                if mat_type == "powder":
                    self._update_powder(grid, row, col, moved, mat, rows, cols, rng, counters, tick_index, column_heights)
        return moved

    def _stage_fluids(self, grid, rows, cols, moved, rng, counters):
        self._ensure_fluid_state(rows, cols)
        self._stage_pde_fluids(grid, rows, cols)
        self._update_hydrostatic_balance(grid, rows, cols)
        self._update_density_sorting(grid, rows, cols, moved, rng, counters)
        self._update_fluid_flow(grid, rows, cols, moved, rng, counters)
        self._update_mixing(grid, rows, cols, rng)

    def _solve_tridiagonal(self, a, b, c, d):
        n = len(d)
        if n == 0:
            return d

        cp = np.zeros(n, dtype=np.float32)
        dp = np.zeros(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)

        denom = np.float32(max(1e-6, b[0]))
        cp[0] = c[0] / denom
        dp[0] = d[0] / denom

        for i in range(1, n):
            denom = np.float32(max(1e-6, b[i] - a[i] * cp[i - 1]))
            cp[i] = c[i] / denom if i < (n - 1) else np.float32(0.0)
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        out[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            out[i] = dp[i] - cp[i] * out[i + 1]

        return out

    def _laplacian_5pt(self, field):
        left = np.empty_like(field)
        right = np.empty_like(field)
        up = np.empty_like(field)
        down = np.empty_like(field)

        left[:, 0] = field[:, 0]
        left[:, 1:] = field[:, :-1]
        right[:, -1] = field[:, -1]
        right[:, :-1] = field[:, 1:]
        up[0, :] = field[0, :]
        up[1:, :] = field[:-1, :]
        down[-1, :] = field[-1, :]
        down[:-1, :] = field[1:, :]

        return (left + right + up + down - 4.0 * field).astype(np.float32)

    def _build_structural_material_fields(self, grid_np, rows, cols):
        solid_mask = np.zeros((rows, cols), dtype=np.bool_)
        brittle_mask = np.zeros((rows, cols), dtype=np.bool_)

        rho = np.full((rows, cols), np.float32(PHYSICS.rho_air), dtype=np.float32)
        young = np.full((rows, cols), np.float32(2.0e7), dtype=np.float32)
        nu = np.full((rows, cols), np.float32(0.28), dtype=np.float32)
        alpha = np.full((rows, cols), np.float32(1.2e-5), dtype=np.float32)
        sigma_y = np.full((rows, cols), np.float32(2.0e5), dtype=np.float32)
        cohesion = np.full((rows, cols), np.float32(2.5e4), dtype=np.float32)
        friction = np.full((rows, cols), np.float32(30.0), dtype=np.float32)
        pore_coeff = np.full((rows, cols), np.float32(0.25), dtype=np.float32)
        neo_c1 = np.full((rows, cols), np.float32(1.0e5), dtype=np.float32)
        neo_d1 = np.full((rows, cols), np.float32(1.0e6), dtype=np.float32)
        hardening = np.full((rows, cols), np.float32(0.02), dtype=np.float32)
        deg_start = np.full((rows, cols), np.float32(500.0), dtype=np.float32)
        deg_end = np.full((rows, cols), np.float32(1200.0), dtype=np.float32)

        for mat_id, mat_data in self.materials.items():
            mask = (grid_np == int(mat_id))
            if not np.any(mask):
                continue
            m_type = str(mat_data.get("type", "air"))
            if m_type in ("solid", "powder"):
                solid_mask[mask] = True

            rho[mask] = np.float32(max(1.0, float(self._mat_value(mat_data, "density", PHYSICS.rho_air))))
            young[mask] = np.float32(max(1.0e4, float(self._mat_value(mat_data, "youngs_modulus", 2.0e7))))
            nu[mask] = np.float32(min(0.49, max(0.01, float(self._mat_value(mat_data, "poisson_ratio", 0.28)))))
            alpha[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "thermal_expansion_coeff", 1.2e-5))))
            sigma_y[mask] = np.float32(max(1.0e3, float(self._mat_value(mat_data, "yield_strength", 2.0e5))))
            cohesion[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "cohesion", 2.5e4))))
            friction[mask] = np.float32(min(60.0, max(0.0, float(self._mat_value(mat_data, "friction_angle_deg", 30.0)))))
            pore_coeff[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "pore_pressure_sensitivity", 0.25))))
            neo_c1[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "neo_hookean_c1", 1.0e5))))
            neo_d1[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "neo_hookean_d1", 1.0e6))))
            hardening[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "plastic_hardening", 0.02))))
            start_val = float(self._mat_value(mat_data, "degradation_temp_start", 500.0))
            start_val = max(50.0, start_val)
            end_val = float(self._mat_value(mat_data, "degradation_temp_end", 1200.0))
            end_val = max(start_val + 1.0, end_val)
            deg_start[mask] = np.float32(start_val)
            deg_end[mask] = np.float32(end_val)

            if bool(self._mat_value(mat_data, "is_brittle", m_type == "solid")):
                brittle_mask[mask] = True

        return (
            solid_mask, brittle_mask, rho, young, nu, alpha, sigma_y,
            cohesion, friction, pore_coeff, neo_c1, neo_d1, hardening, deg_start, deg_end,
        )

    def _spawn_debris_particle(self, row, col, mat_id, local_vx, local_vy):
        if not self.structural_config.debris_enabled:
            return
        if len(self.debris_particles) > 1200:
            return

        density = float(self._mat_value(self._get(mat_id), "density", 1200.0))
        radius = float(max(0.2, self.structural_config.debris_particle_radius_cells))
        area = math.pi * radius * radius
        mass = float(max(0.01, density * area * PHYSICS.dx * PHYSICS.dy * self.structural_config.debris_spawn_mass_scale))
        self.debris_particles.append({
            "x": float(col) + 0.5,
            "y": float(row) + 0.5,
            "vx": float(local_vx),
            "vy": float(local_vy),
            "omega": float(np.random.uniform(-2.0, 2.0)),
            "radius": radius,
            "mass": mass,
            "lifetime": float(self.structural_config.debris_particle_lifetime),
            "mat": int(self.material_ids.get("ash", self.material_ids.get("sand", 1))),
        })

    def _update_debris_dem(self, grid, rows, cols, counters, events):
        if not self.structural_config.debris_enabled or not self.debris_particles:
            return

        dt = float(PHYSICS.dt)
        k_n = float(max(1.0, self.structural_config.debris_contact_stiffness))
        c_n = float(max(0.0, self.structural_config.debris_contact_damping))
        wall_e = float(np.clip(self.structural_config.debris_wall_restitution, 0.0, 1.0))
        gx = 0.0
        gy = float(self.structural_config.gravity_coupling * PHYSICS.g * dt)

        # Particle-particle penalty spring-dashpot contacts
        n = len(self.debris_particles)
        for i in range(n):
            pi = self.debris_particles[i]
            for j in range(i + 1, n):
                pj = self.debris_particles[j]
                dx = pj["x"] - pi["x"]
                dy = pj["y"] - pi["y"]
                dist2 = dx * dx + dy * dy
                r_sum = pi["radius"] + pj["radius"]
                if dist2 <= 1e-8 or dist2 >= (r_sum * r_sum):
                    continue
                dist = math.sqrt(dist2)
                nx = dx / dist
                ny = dy / dist
                overlap = r_sum - dist
                rvx = pj["vx"] - pi["vx"]
                rvy = pj["vy"] - pi["vy"]
                vrel_n = rvx * nx + rvy * ny
                f_n = k_n * overlap - c_n * vrel_n
                if f_n <= 0.0:
                    continue
                im_i = 1.0 / max(1e-6, pi["mass"])
                im_j = 1.0 / max(1e-6, pj["mass"])
                imp = f_n * dt
                pi["vx"] -= imp * nx * im_i
                pi["vy"] -= imp * ny * im_i
                pj["vx"] += imp * nx * im_j
                pj["vy"] += imp * ny * im_j

        next_particles = []
        for p in self.debris_particles:
            cx = int(np.clip(round(p["x"]), 0, cols - 1))
            cy = int(np.clip(round(p["y"]), 0, rows - 1))

            # Two-way coupling: fluid accelerates debris; debris perturbs fluid momentum.
            if self.vel_x.shape == (rows, cols):
                fluid_vx = float(self.vel_x[cy, cx])
                fluid_vy = float(self.vel_y[cy, cx])
                p["vx"] += (fluid_vx - p["vx"]) * 0.18
                p["vy"] += (fluid_vy - p["vy"]) * 0.18
                coupling = 0.03 * min(1.0, p["mass"] / 50.0)
                self.vel_x[cy, cx] -= np.float32(coupling * p["vx"])
                self.vel_y[cy, cx] -= np.float32(coupling * p["vy"])

            p["vx"] += gx
            p["vy"] += gy
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["lifetime"] -= dt

            # Domain bounce
            if p["x"] < p["radius"]:
                p["x"] = p["radius"]
                p["vx"] = abs(p["vx"]) * wall_e
            elif p["x"] > (cols - 1 - p["radius"]):
                p["x"] = cols - 1 - p["radius"]
                p["vx"] = -abs(p["vx"]) * wall_e
            if p["y"] < p["radius"]:
                p["y"] = p["radius"]
                p["vy"] = abs(p["vy"]) * wall_e
            elif p["y"] > (rows - 1 - p["radius"]):
                p["y"] = rows - 1 - p["radius"]
                p["vy"] = -abs(p["vy"]) * wall_e

            cell_r = int(np.clip(round(p["y"]), 0, rows - 1))
            cell_c = int(np.clip(round(p["x"]), 0, cols - 1))
            speed = math.sqrt(p["vx"] * p["vx"] + p["vy"] * p["vy"])

            # Deposit debris back to CA field when particle dies/slows.
            if p["lifetime"] <= 0.0 or speed < 0.08:
                if grid[cell_r][cell_c] == 0:
                    grid[cell_r][cell_c] = int(p["mat"])
                    counters["changes"] += 1
                    events.append({"type": "debris_deposit", "row": cell_r, "col": cell_c, "mat": int(p["mat"])})
                continue

            next_particles.append(p)

        self.debris_particles = next_particles

    def _stage_structural(self, grid, rows, cols, counters, events):
        if not self.structural_config.enabled:
            return

        self._ensure_thermal_state(rows, cols)
        self._ensure_chemical_state(grid, rows, cols)
        self._ensure_structural_state(rows, cols)

        if rows <= 1 or cols <= 1:
            self._update_debris_dem(grid, rows, cols, counters, events)
            return

        grid_np = np.asarray(grid, dtype=np.int32)
        (solid_mask, brittle_mask, rho, young, nu, alpha, sigma_y,
         cohesion, friction, pore_coeff, neo_c1, neo_d1,
         hardening, deg_start, deg_end) = self._build_structural_material_fields(grid_np, rows, cols)

        if not np.any(solid_mask):
            self._update_debris_dem(grid, rows, cols, counters, events)
            return

        dt_sub = np.float32(PHYSICS.dt / max(1, int(self.structural_config.explicit_substeps)))
        dx_inv = np.float32(PHYSICS.dx_inv)
        dy_inv = np.float32(PHYSICS.dy_inv)
        ambient = np.float32(self.thermal_config.ambient_temp)

        damping = np.float32(np.clip(self.structural_config.damping, 0.0, 0.99))
        grav = np.float32(self.structural_config.gravity_coupling * PHYSICS.g)

        for _ in range(max(1, int(self.structural_config.explicit_substeps))):
            ux = self.disp_x
            uy = self.disp_y

            ux_l = np.empty_like(ux)
            ux_r = np.empty_like(ux)
            ux_t = np.empty_like(ux)
            ux_b = np.empty_like(ux)
            uy_l = np.empty_like(uy)
            uy_r = np.empty_like(uy)
            uy_t = np.empty_like(uy)
            uy_b = np.empty_like(uy)

            ux_l[:, 0] = ux[:, 0]
            ux_l[:, 1:] = ux[:, :-1]
            ux_r[:, -1] = ux[:, -1]
            ux_r[:, :-1] = ux[:, 1:]
            ux_t[0, :] = ux[0, :]
            ux_t[1:, :] = ux[:-1, :]
            ux_b[-1, :] = ux[-1, :]
            ux_b[:-1, :] = ux[1:, :]

            uy_l[:, 0] = uy[:, 0]
            uy_l[:, 1:] = uy[:, :-1]
            uy_r[:, -1] = uy[:, -1]
            uy_r[:, :-1] = uy[:, 1:]
            uy_t[0, :] = uy[0, :]
            uy_t[1:, :] = uy[:-1, :]
            uy_b[-1, :] = uy[-1, :]
            uy_b[:-1, :] = uy[1:, :]

            eps_xx = np.float32(0.5) * (ux_r - ux_l) * dx_inv
            eps_yy = np.float32(0.5) * (uy_b - uy_t) * dy_inv
            eps_xy = np.float32(0.25) * ((ux_b - ux_t) * dy_inv + (uy_r - uy_l) * dx_inv)

            if self.structural_config.thermal_strain_enabled:
                eps_th = alpha * (self.temperature.astype(np.float32) - ambient)
            else:
                eps_th = np.zeros_like(eps_xx)

            # Scalar plastic strain as isotropic part of inelastic strain.
            eps_xx_eff = eps_xx - eps_th - self.plastic_strain
            eps_yy_eff = eps_yy - eps_th - self.plastic_strain
            tr_eps = eps_xx_eff + eps_yy_eff

            if self.structural_config.thermal_degradation_enabled:
                temp = self.temperature.astype(np.float32)
                deg = np.clip((temp - deg_start) / np.maximum(np.float32(1.0), (deg_end - deg_start)), 0.0, 1.0)
                strength_factor = 1.0 - np.float32(0.9) * deg
            else:
                strength_factor = np.ones_like(eps_xx_eff)

            E_eff = young * strength_factor
            nu_eff = np.clip(nu, 0.01, 0.49)
            mu = E_eff / (2.0 * (1.0 + nu_eff))
            lam = (E_eff * nu_eff) / np.maximum(np.float32(1e-6), (1.0 + nu_eff) * (1.0 - 2.0 * nu_eff))

            sxx = (2.0 * mu * eps_xx_eff + lam * tr_eps).astype(np.float32)
            syy = (2.0 * mu * eps_yy_eff + lam * tr_eps).astype(np.float32)
            txy = (2.0 * mu * eps_xy).astype(np.float32)

            if self.structural_config.neo_hookean_enabled:
                clip = np.float32(max(0.05, self.structural_config.finite_strain_clip))
                ex = np.clip(eps_xx, -clip, clip)
                ey = np.clip(eps_yy, -clip, clip)
                gxy = np.clip(eps_xy, -clip, clip)
                J = np.clip((1.0 + ex) * (1.0 + ey) - gxy * gxy, 0.55, 1.8)
                volumetric = (J - 1.0)
                sxx += (2.0 * neo_c1 * ex + 2.0 * neo_d1 * volumetric).astype(np.float32)
                syy += (2.0 * neo_c1 * ey + 2.0 * neo_d1 * volumetric).astype(np.float32)
                txy += (2.0 * neo_c1 * gxy).astype(np.float32)

            # Elastoplastic projection with von-Mises equivalent stress.
            vm = np.sqrt(np.maximum(np.float32(0.0), sxx * sxx - sxx * syy + syy * syy + np.float32(3.0) * txy * txy)).astype(np.float32)
            yield_eff = sigma_y * strength_factor
            if self.structural_config.elastoplastic_enabled:
                over = vm > yield_eff
                scale = np.ones_like(vm)
                scale[over] = np.clip(yield_eff[over] / np.maximum(vm[over], np.float32(1e-6)), 0.05, 1.0)
                sxx *= scale
                syy *= scale
                txy *= scale
                self.plastic_strain[over] = np.clip(
                    self.plastic_strain[over] + (1.0 - scale[over]) * (np.float32(0.015) + hardening[over]),
                    0.0,
                    0.5,
                )

            self.sigma_xx = sxx
            self.sigma_yy = syy
            self.tau_xy = txy

            # Structural dynamics: ρ * u_tt = div(σ) + ρ g
            sxx_l = np.empty_like(sxx)
            sxx_r = np.empty_like(sxx)
            syy_t = np.empty_like(syy)
            syy_b = np.empty_like(syy)
            txy_l = np.empty_like(txy)
            txy_r = np.empty_like(txy)
            txy_t = np.empty_like(txy)
            txy_b = np.empty_like(txy)

            sxx_l[:, 0] = sxx[:, 0]
            sxx_l[:, 1:] = sxx[:, :-1]
            sxx_r[:, -1] = sxx[:, -1]
            sxx_r[:, :-1] = sxx[:, 1:]
            syy_t[0, :] = syy[0, :]
            syy_t[1:, :] = syy[:-1, :]
            syy_b[-1, :] = syy[-1, :]
            syy_b[:-1, :] = syy[1:, :]

            txy_l[:, 0] = txy[:, 0]
            txy_l[:, 1:] = txy[:, :-1]
            txy_r[:, -1] = txy[:, -1]
            txy_r[:, :-1] = txy[:, 1:]
            txy_t[0, :] = txy[0, :]
            txy_t[1:, :] = txy[:-1, :]
            txy_b[-1, :] = txy[-1, :]
            txy_b[:-1, :] = txy[1:, :]

            fx = np.float32(0.5) * ((sxx_r - sxx_l) * dx_inv + (txy_b - txy_t) * dy_inv)
            fy = np.float32(0.5) * ((txy_r - txy_l) * dx_inv + (syy_b - syy_t) * dy_inv) + rho * grav

            inv_rho = 1.0 / np.maximum(np.float32(1.0), rho)
            self.struct_vel_x += dt_sub * fx * inv_rho
            self.struct_vel_y += dt_sub * fy * inv_rho
            self.struct_vel_x *= (1.0 - damping)
            self.struct_vel_y *= (1.0 - damping)

            self.struct_vel_x[~solid_mask] = 0.0
            self.struct_vel_y[~solid_mask] = 0.0
            self.disp_x += dt_sub * self.struct_vel_x
            self.disp_y += dt_sub * self.struct_vel_y
            self.disp_x[~solid_mask] = 0.0
            self.disp_y[~solid_mask] = 0.0

        # Thermomechanical spalling driver: |∇T| + pore pressure on boundary solids.
        temp = self.temperature.astype(np.float32)
        t_l = np.empty_like(temp)
        t_r = np.empty_like(temp)
        t_t = np.empty_like(temp)
        t_b = np.empty_like(temp)
        t_l[:, 0] = temp[:, 0]
        t_l[:, 1:] = temp[:, :-1]
        t_r[:, -1] = temp[:, -1]
        t_r[:, :-1] = temp[:, 1:]
        t_t[0, :] = temp[0, :]
        t_t[1:, :] = temp[:-1, :]
        t_b[-1, :] = temp[-1, :]
        t_b[:-1, :] = temp[1:, :]
        grad_t = np.sqrt((np.float32(0.5) * (t_r - t_l) * dx_inv) ** 2 + (np.float32(0.5) * (t_b - t_t) * dy_inv) ** 2)

        moist = self.moisture.astype(np.float32) if isinstance(self.moisture, np.ndarray) and self.moisture.shape == (rows, cols) else np.zeros((rows, cols), dtype=np.float32)
        self.pore_pressure = (pore_coeff * moist * np.maximum(np.float32(0.0), temp - np.float32(100.0))).astype(np.float32)

        # Failures and conversion to debris particles.
        friction_tan = np.tan(np.deg2rad(friction.astype(np.float32)))
        vm = np.sqrt(np.maximum(np.float32(0.0), self.sigma_xx * self.sigma_xx - self.sigma_xx * self.sigma_yy + self.sigma_yy * self.sigma_yy + np.float32(3.0) * self.tau_xy * self.tau_xy))
        normal = np.float32(0.5) * (self.sigma_xx + self.sigma_yy)
        tau_max = np.sqrt(((self.sigma_xx - self.sigma_yy) * np.float32(0.5)) ** 2 + self.tau_xy * self.tau_xy)
        spall_threshold = np.float32(max(1.0, self.structural_config.spalling_temperature_gradient_threshold))

        for row in range(rows):
            for col in range(cols):
                if not solid_mask[row, col]:
                    continue

                mat = grid[row][col]
                mat_data = self._get(mat)
                is_brittle = bool(brittle_mask[row, col])
                fail = False

                if is_brittle and self.structural_config.brittle_mohr_coulomb_enabled:
                    shear_limit = float(cohesion[row, col] + max(0.0, normal[row, col]) * friction_tan[row, col])
                    tensile_cut = float(self.structural_config.brittle_tension_cutoff)
                    if float(tau_max[row, col]) > shear_limit or float(normal[row, col]) < -tensile_cut:
                        fail = True

                if not is_brittle:
                    yld = max(1.0, float(sigma_y[row, col]))
                    if float(vm[row, col]) > yld:
                        overload = (float(vm[row, col]) / yld) - 1.0
                        self.damage_field[row, col] = min(1.0, self.damage_field[row, col] + overload * self.structural_config.failure_damage_rate)
                        self.integrity[row, col] = max(0.0, self.integrity[row, col] - overload * self.structural_config.failure_damage_rate)

                if self.structural_config.spalling_enabled:
                    # Boundary-solid detection (at least one non-solid cardinal neighbor)
                    boundary = False
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = row + dr, col + dc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not solid_mask[nr, nc]:
                            boundary = True
                            break
                    if boundary:
                        spall_drive = float(grad_t[row, col] / max(1.0, spall_threshold)) + float(self.structural_config.spalling_pore_pressure_weight * self.pore_pressure[row, col] / 1.0e3)
                        if spall_drive > 1.0:
                            self.damage_field[row, col] = min(1.0, self.damage_field[row, col] + (spall_drive - 1.0) * 0.08)

                if self.damage_field[row, col] >= 1.0 or self.integrity[row, col] <= 0.02:
                    fail = True

                if not fail:
                    continue

                local_vx = float(self.struct_vel_x[row, col])
                local_vy = float(self.struct_vel_y[row, col])
                self._spawn_debris_particle(row, col, mat, local_vx, local_vy)

                # Damage mechanics: reset stress tensor and convert failed cell.
                self.sigma_xx[row, col] = 0.0
                self.sigma_yy[row, col] = 0.0
                self.tau_xy[row, col] = 0.0
                self.disp_x[row, col] = 0.0
                self.disp_y[row, col] = 0.0
                self.struct_vel_x[row, col] = 0.0
                self.struct_vel_y[row, col] = 0.0
                self.plastic_strain[row, col] = 0.0
                self.damage_field[row, col] = 0.0
                self.integrity[row, col] = 0.0
                grid[row][col] = 0
                counters["changes"] += 1
                events.append({"type": "structural_failure", "row": row, "col": col, "from": mat})

        self._update_debris_dem(grid, rows, cols, counters, events)

    def _compute_heat_capacity_and_diffusivity(self, grid, rows, cols):
        grid_np = np.asarray(grid, dtype=np.int32)
        rho = np.full((rows, cols), np.float32(PHYSICS.rho_air), dtype=np.float32)
        cp = np.full((rows, cols), np.float32(1.0), dtype=np.float32)
        k = np.full((rows, cols), np.float32(0.08), dtype=np.float32)

        for mat_id, mat_data in self.materials.items():
            mask = (grid_np == int(mat_id))
            if not np.any(mask):
                continue
            rho[mask] = np.float32(max(1.0, float(self._mat_value(mat_data, "density", PHYSICS.rho_air))))
            cp[mask] = np.float32(max(0.1, float(self._mat_value(mat_data, "thermal_capacity", 1.0))))
            k[mask] = np.float32(max(0.01, float(self._mat_value(mat_data, "thermal_conductivity", 0.08))))

        volumetric_cp = np.maximum(np.float32(1.0), rho * cp).astype(np.float32)
        alpha = (k / volumetric_cp).astype(np.float32)
        return volumetric_cp, alpha

    def _apply_adi_heat_conduction(self, grid, rows, cols):
        if rows <= 0 or cols <= 0:
            return

        temp = self.temperature.astype(np.float32, copy=True)
        _, alpha = self._compute_heat_capacity_and_diffusivity(grid, rows, cols)
        dt = np.float32(PHYSICS.dt)
        dx2 = np.float32(PHYSICS.dx * PHYSICS.dx)
        dy2 = np.float32(PHYSICS.dy * PHYSICS.dy)
        rx = np.clip(alpha * (dt / dx2), 0.0, 0.49).astype(np.float32)
        ry = np.clip(alpha * (dt / dy2), 0.0, 0.49).astype(np.float32)

        iterations = int(max(1, self.thermal_config.adi_iterations))
        for _ in range(iterations):
            t_star = temp.copy()
            for row in range(rows):
                a = np.zeros(cols, dtype=np.float32)
                b = np.ones(cols, dtype=np.float32)
                c = np.zeros(cols, dtype=np.float32)
                d = np.zeros(cols, dtype=np.float32)

                for col in range(cols):
                    rxi = np.float32(0.5) * rx[row, col]
                    ryi = np.float32(0.5) * ry[row, col]
                    t_u = temp[row - 1, col] if row > 0 else temp[row, col]
                    t_d = temp[row + 1, col] if row < (rows - 1) else temp[row, col]
                    d[col] = temp[row, col] + ryi * (t_u - 2.0 * temp[row, col] + t_d)
                    b[col] = 1.0 + 2.0 * rxi
                    if col > 0:
                        a[col] = -rxi
                    if col < (cols - 1):
                        c[col] = -rxi

                t_star[row, :] = self._solve_tridiagonal(a, b, c, d)

            t_next = t_star.copy()
            for col in range(cols):
                a = np.zeros(rows, dtype=np.float32)
                b = np.ones(rows, dtype=np.float32)
                c = np.zeros(rows, dtype=np.float32)
                d = np.zeros(rows, dtype=np.float32)

                for row in range(rows):
                    rxi = np.float32(0.5) * rx[row, col]
                    ryi = np.float32(0.5) * ry[row, col]
                    t_l = t_star[row, col - 1] if col > 0 else t_star[row, col]
                    t_r = t_star[row, col + 1] if col < (cols - 1) else t_star[row, col]
                    d[row] = t_star[row, col] + rxi * (t_l - 2.0 * t_star[row, col] + t_r)
                    b[row] = 1.0 + 2.0 * ryi
                    if row > 0:
                        a[row] = -ryi
                    if row < (rows - 1):
                        c[row] = -ryi

                t_next[:, col] = self._solve_tridiagonal(a, b, c, d)

            temp = t_next

        self.temperature = temp

    def _apply_thermal_convection(self):
        if not (isinstance(self.vel_x, np.ndarray) and isinstance(self.vel_y, np.ndarray)):
            return
        if self.vel_x.shape != self.temperature.shape or self.vel_y.shape != self.temperature.shape:
            return

        blend = np.float32(np.clip(self.thermal_config.convection_advection_blend, 0.0, 1.0))
        if blend <= 0.0:
            return

        advected = self._advect_centered_field(
            self.temperature.astype(np.float32),
            self.vel_x.astype(np.float32),
            self.vel_y.astype(np.float32),
            direction=1.0,
        )
        self.temperature = ((1.0 - blend) * self.temperature + blend * advected).astype(np.float32)

    def _apply_radiation_exchange(self, grid, rows, cols):
        if not self.thermal_config.radiation_enabled:
            return

        volumetric_cp, _ = self._compute_heat_capacity_and_diffusivity(grid, rows, cols)
        eps = np.float32(np.clip(self.thermal_config.radiation_emissivity, 0.0, 1.0))
        sigma = np.float32(max(0.0, self.thermal_config.radiation_sigma))
        strength = np.float32(max(0.0, self.thermal_config.radiation_strength))
        if sigma <= 0.0 or strength <= 0.0:
            return

        t_amb_k = np.float32(self.thermal_config.ambient_temp + 273.15)
        t_k = self.temperature.astype(np.float32) + np.float32(273.15)
        net_flux = eps * sigma * (t_k * t_k * t_k * t_k - t_amb_k * t_amb_k * t_amb_k * t_amb_k)

        attenuation = np.float32(1.0)
        if isinstance(self.smoke_density, np.ndarray) and self.smoke_density.shape == self.temperature.shape:
            attenuation = (1.0 / (1.0 + 2.0 * self.smoke_density.astype(np.float32))).astype(np.float32)

        dtemp = strength * net_flux * attenuation * np.float32(PHYSICS.dt) / (volumetric_cp + np.float32(1e-3))
        self.temperature = (self.temperature - dtemp).astype(np.float32)

    def _apply_enthalpy_porosity(self, grid, rows, cols):
        if not self.thermal_config.enthalpy_enabled:
            return

        water_id = self.material_ids.get("water", -1)
        ice_id = self.material_ids.get("ice", -1)
        if water_id < 0 and ice_id < 0:
            return

        grid_np = np.asarray(grid, dtype=np.int32)
        phase_mask = np.zeros((rows, cols), dtype=np.bool_)
        if water_id >= 0:
            phase_mask |= (grid_np == water_id)
        if ice_id >= 0:
            phase_mask |= (grid_np == ice_id)

        if not np.any(phase_mask):
            self.liquid_fraction.fill(0.0)
            return

        melt_point = np.float32(0.0)
        mushy = np.float32(max(0.1, self.thermal_config.mushy_range))
        f = ((self.temperature.astype(np.float32) - (melt_point - mushy)) / (2.0 * mushy)).astype(np.float32)
        f = np.clip(f, 0.0, 1.0).astype(np.float32)

        self.liquid_fraction.fill(0.0)
        self.liquid_fraction[phase_mask] = f[phase_mask]

        volumetric_cp, _ = self._compute_heat_capacity_and_diffusivity(grid, rows, cols)
        latent_scale = np.float32(max(0.0, self.thermal_config.latent_heat_factor))
        latent = np.float32(18.0) * latent_scale
        self.enthalpy_field = (volumetric_cp * self.temperature.astype(np.float32) + latent * self.liquid_fraction).astype(np.float32)

        if isinstance(self.vel_x, np.ndarray) and isinstance(self.vel_y, np.ndarray):
            if self.vel_x.shape == self.temperature.shape and self.vel_y.shape == self.temperature.shape:
                eps = np.float32(1e-3)
                coeff = np.float32(max(0.0, self.thermal_config.mushy_drag_strength))
                drag = coeff * ((1.0 - self.liquid_fraction) ** 2) / ((self.liquid_fraction ** 3) + eps)
                mask = phase_mask & (self.liquid_fraction < 0.999)
                damp = (1.0 + np.float32(PHYSICS.dt) * drag).astype(np.float32)
                self.vel_x[mask] = (self.vel_x[mask] / damp[mask]).astype(np.float32)
                self.vel_y[mask] = (self.vel_y[mask] / damp[mask]).astype(np.float32)

    def _diffuse_scalar_field(self, field, diffusivity, clamp_min=None, clamp_max=None):
        if field is None or not isinstance(field, np.ndarray) or field.shape != self.temperature.shape:
            return field

        lam = np.float32(diffusivity * PHYSICS.dt * PHYSICS.dx_inv * PHYSICS.dx_inv)
        lam = np.float32(np.clip(lam, 0.0, 0.24))
        if lam <= 0.0:
            return field

        lap = self._laplacian_5pt(field.astype(np.float32))
        out = (field.astype(np.float32) + lam * lap).astype(np.float32)
        if clamp_min is not None or clamp_max is not None:
            lo = -np.inf if clamp_min is None else clamp_min
            hi = np.inf if clamp_max is None else clamp_max
            out = np.clip(out, lo, hi).astype(np.float32)
        return out

    def _update_species_diffusion_phase3(self, grid, rows, cols):
        if not self.thermal_config.species_diffusion_enabled:
            return

        self.oxygen_level = self._diffuse_scalar_field(
            self.oxygen_level,
            max(0.0, self.thermal_config.oxygen_diffusivity),
            clamp_min=0.0,
            clamp_max=1.0,
        )
        self.smoke_density = self._diffuse_scalar_field(
            self.smoke_density,
            max(0.0, self.thermal_config.smoke_diffusivity),
            clamp_min=0.0,
            clamp_max=1.0,
        )

        self.steam_density = self._diffuse_scalar_field(
            self.steam_density,
            max(0.0, self.thermal_config.steam_diffusivity),
            clamp_min=0.0,
            clamp_max=1.0,
        )

        steam_id = self.material_ids.get("steam", -1)
        if steam_id >= 0:
            steam_mask = (np.asarray(grid, dtype=np.int32) == steam_id)
            self.steam_density[steam_mask] = np.clip(self.steam_density[steam_mask] + 0.03, 0.0, 1.0)

    def _update_moisture_transport_phase3(self, grid, rows, cols):
        if not self.thermal_config.porous_moisture_enabled:
            return

        self.moisture = self._diffuse_scalar_field(
            self.moisture,
            max(0.0, self.thermal_config.moisture_diffusivity),
            clamp_min=0.0,
            clamp_max=1.0,
        )

        if isinstance(self.vel_x, np.ndarray) and isinstance(self.vel_y, np.ndarray):
            if self.vel_x.shape == self.moisture.shape and self.vel_y.shape == self.moisture.shape:
                adv = self._advect_centered_field(self.moisture.astype(np.float32), self.vel_x, self.vel_y, direction=1.0)
                self.moisture = np.clip(0.75 * self.moisture + 0.25 * adv, 0.0, 1.0).astype(np.float32)

        grid_np = np.asarray(grid, dtype=np.int32)
        porosity = np.zeros((rows, cols), dtype=np.float32)
        for mat_id, mat_data in self.materials.items():
            mask = (grid_np == int(mat_id))
            if not np.any(mask):
                continue
            mat_type = str(mat_data.get("type", "air"))
            default_porosity = 0.18 if mat_type == "powder" else (0.04 if mat_type == "solid" else 0.0)
            porosity[mask] = np.float32(max(0.0, min(1.0, float(self._mat_value(mat_data, "porosity", default_porosity)))))

        water_id = self.material_ids.get("water", -1)
        if water_id >= 0:
            w = (grid_np == water_id)
            adj = np.zeros_like(w)
            adj[1:, :] |= w[:-1, :]
            adj[:-1, :] |= w[1:, :]
            adj[:, 1:] |= w[:, :-1]
            adj[:, :-1] |= w[:, 1:]
            gain = np.float32(max(0.0, self.thermal_config.porous_moisture_gain)) * porosity * adj.astype(np.float32)
            self.moisture = np.clip(self.moisture + gain, 0.0, 1.0).astype(np.float32)

        evap = np.maximum(0.0, self.temperature.astype(np.float32) - 100.0) * self.moisture * np.float32(0.0018)
        evap = np.clip(evap, 0.0, self.moisture)
        self.moisture = np.clip(self.moisture - evap, 0.0, 1.0).astype(np.float32)
        self.steam_density = np.clip(self.steam_density + 0.8 * evap, 0.0, 1.0).astype(np.float32)
        self.temperature = (self.temperature - 120.0 * evap).astype(np.float32)

    def _apply_leidenfrost_evaporation(self, grid, rows, cols):
        if not self.thermal_config.leidenfrost_enabled:
            return

        water_id = self.material_ids.get("water", -1)
        if water_id < 0:
            return

        grid_np = np.asarray(grid, dtype=np.int32)
        water_mask = (grid_np == water_id)
        if not np.any(water_mask):
            return

        t = self.temperature.astype(np.float32)
        up = np.empty_like(t)
        down = np.empty_like(t)
        left = np.empty_like(t)
        right = np.empty_like(t)
        up[0, :] = t[0, :]
        up[1:, :] = t[:-1, :]
        down[-1, :] = t[-1, :]
        down[:-1, :] = t[1:, :]
        left[:, 0] = t[:, 0]
        left[:, 1:] = t[:, :-1]
        right[:, -1] = t[:, -1]
        right[:, :-1] = t[:, 1:]

        neighbor_peak = np.maximum(np.maximum(up, down), np.maximum(left, right))
        film_mask = water_mask & (neighbor_peak >= np.float32(self.thermal_config.leidenfrost_temp))
        if not np.any(film_mask):
            return

        evap_rate = np.float32(max(0.0, self.thermal_config.leidenfrost_evap_rate))
        excess = np.maximum(0.0, neighbor_peak - np.float32(self.thermal_config.leidenfrost_temp))
        evap = np.clip(evap_rate * (1.0 + 0.01 * excess), 0.0, 0.06).astype(np.float32)
        evap *= film_mask.astype(np.float32)

        self.steam_density = np.clip(self.steam_density + 2.0 * evap, 0.0, 1.0).astype(np.float32)
        self.temperature = (self.temperature - 220.0 * evap).astype(np.float32)

    def _update_thermal_field(self, grid, rows, cols):
        if self.thermal_config.adi_enabled:
            self._apply_adi_heat_conduction(grid, rows, cols)
        else:
            next_temperature = self.temperature.copy()

            for row in range(rows):
                for col in range(cols):
                    mat = grid[row][col]
                    mat_data = self._get(mat) if mat in self.materials else self._get(0)

                    total = 0.0
                    count = 0
                    for y in range(max(0, row - 1), min(rows, row + 2)):
                        for x in range(max(0, col - 1), min(cols, col + 2)):
                            if y == row and x == col:
                                continue
                            total += self.temperature[y][x]
                            count += 1

                    neighbor_avg = total / count if count > 0 else self.temperature[row][col]
                    thermal_conductivity = self._mat_value(mat_data, "thermal_conductivity", 0.2)
                    thermal_capacity = self._mat_value(mat_data, "thermal_capacity", 1.0)
                    diffusion = self.thermal_config.heat_diffusion_rate * thermal_conductivity / max(0.25, thermal_capacity)
                    target_temp = self.temperature[row][col] + ((neighbor_avg - self.temperature[row][col]) * diffusion)

                    if self.burn_stage[row][col] < 3:
                        ambient = self.thermal_config.ambient_temp
                        excess = target_temp - ambient
                        if excess > 0.5:
                            cool_rate = max(0.005, min(0.040, thermal_conductivity * 0.05))
                            target_temp -= excess * cool_rate

                    if mat == self.material_ids["water"]:
                        ambient = self.thermal_config.ambient_temp
                        if target_temp > ambient:
                            target_temp -= min(3.0, (target_temp - ambient) * 0.18)

                    if row > 0 and self.temperature[row - 1][col] > self.temperature[row][col]:
                        target_temp += self.thermal_config.convection_bias

                    delta = max(-self.thermal_config.max_temp_delta_per_tick, min(self.thermal_config.max_temp_delta_per_tick, target_temp - self.temperature[row][col]))
                    next_temperature[row][col] = self.temperature[row][col] + delta

            self.temperature = next_temperature

        self._apply_thermal_convection()
        self._apply_radiation_exchange(grid, rows, cols)
        self._apply_enthalpy_porosity(grid, rows, cols)

    def _update_thermal_field(self, grid, rows, cols):
        next_temperature = self.temperature.copy()

        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                mat_data = self._get(mat) if mat in self.materials else self._get(0)

                total = 0.0
                count = 0
                for y in range(max(0, row - 1), min(rows, row + 2)):
                    for x in range(max(0, col - 1), min(cols, col + 2)):
                        if y == row and x == col:
                            continue
                        total += self.temperature[y][x]
                        count += 1

                neighbor_avg = total / count if count > 0 else self.temperature[row][col]
                thermal_conductivity = self._mat_value(mat_data, "thermal_conductivity", 0.2)
                thermal_capacity = self._mat_value(mat_data, "thermal_capacity", 1.0)
                diffusion = self.thermal_config.heat_diffusion_rate * thermal_conductivity / max(0.25, thermal_capacity)
                target_temp = self.temperature[row][col] + ((neighbor_avg - self.temperature[row][col]) * diffusion)

                # Ambient radiative + convective cooling: every non-burning cell
                # slowly radiates excess heat away toward ambient temperature.
                # Rate scales with thermal_conductivity so good conductors cool faster.
                if self.burn_stage[row][col] < 3:
                    ambient = self.thermal_config.ambient_temp
                    excess = target_temp - ambient
                    if excess > 0.5:
                        cool_rate = max(0.005, min(0.040, thermal_conductivity * 0.05))
                        target_temp -= excess * cool_rate

                if self.burn_stage[row][col] == 3:
                    pass  # exothermic heat now handled in _update_combustion_states
                elif self.burn_stage[row][col] == 4:
                    pass

                if mat == self.material_ids["water"]:
                    ambient = self.thermal_config.ambient_temp
                    if target_temp > ambient:
                        target_temp -= min(3.0, (target_temp - ambient) * 0.18)

                if row > 0 and self.temperature[row - 1][col] > self.temperature[row][col]:
                    target_temp += self.thermal_config.convection_bias

                delta = max(-self.thermal_config.max_temp_delta_per_tick, min(self.thermal_config.max_temp_delta_per_tick, target_temp - self.temperature[row][col]))
                next_temperature[row][col] = self.temperature[row][col] + delta

        self.temperature = next_temperature

    def _update_water_cooling(self, grid, rows, cols):
        """Water actively extracts heat from adjacent hot cells (conductive quenching).
        Each water cell pulls heat from all 4 cardinal neighbours that are hotter than it."""
        water_id = self.material_ids.get("water", -1)
        if water_id < 0:
            return
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != water_id:
                    continue
                w_temp = self.temperature[row][col]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, col + dc
                    if not self._in_bounds(nr, nc, rows, cols):
                        continue
                    n_temp = self.temperature[nr][nc]
                    diff = n_temp - w_temp
                    if diff <= 5.0:
                        continue
                    # Extraction scales with temperature difference; cap at 40°C/tick
                    extraction = min(40.0, diff * 0.15)
                    if self.thermal_config.leidenfrost_enabled:
                        if n_temp >= self.thermal_config.leidenfrost_temp:
                            extraction *= max(0.0, min(1.0, self.thermal_config.leidenfrost_transfer_factor))
                    self.temperature[nr][nc] -= extraction
                    # Water heats up (but boiling is handled by phase change)
                    self.temperature[row][col] = min(99.0, w_temp + extraction * 0.3)

    def _update_contact_heating(self, grid, rows, cols):
        """Direct-contact heat injection from very hot cells (lava, active flame) into
        their 4 cardinal neighbours.  This runs *after* diffusion so that lava reliably
        raises adjacent wood to ignition temperature before it solidifies."""
        lava_id = self.material_ids.get("lava", -1)
        for row in range(rows):
            for col in range(cols):
                src_temp = self.temperature[row][col]
                if src_temp < 300.0:
                    continue
                mat = grid[row][col]
                mat_data = self._get(mat)
                # base output from a material property (overridable), plus lava bonus
                heat_out = self._mat_value(mat_data, "contact_heat_output", 0.0)
                if mat == lava_id:
                    heat_out = max(heat_out, (src_temp - 300.0) * 0.004)
                # active flame cells radiate IR heat to all cardinal neighbours
                if self.burn_stage[row][col] == 3:
                    flame_heat_out = max(0.0, (src_temp - 200.0) * 0.008)
                    heat_out = max(heat_out, flame_heat_out)
                if heat_out <= 0.0:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, col + dc
                    if not self._in_bounds(nr, nc, rows, cols):
                        continue
                    n_temp = self.temperature[nr][nc]
                    if src_temp > n_temp:
                        bonus = min(18.0, (src_temp - n_temp) * heat_out)
                        self.temperature[nr][nc] += bonus

    def _update_ignition(self, grid, rows, cols, tick_index, rng, events):
        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    continue

                mat_data = self._get(mat)
                if not self._is_combustible(mat_data):
                    continue

                if self.burn_stage[row][col] != 0:
                    continue

                if tick_index < self.ignition_cooldown_until[row][col]:
                    continue

                oxygen = self.oxygen_level[row][col]
                min_oxygen = self._mat_value(mat_data, "min_oxygen_for_ignition", 0.2)
                if oxygen < min_oxygen:
                    continue

                ignition_temp = self._mat_value(mat_data, "ignition_temp", 400.0)
                auto_ignite_temp = self._mat_value(mat_data, "auto_ignite_temp", ignition_temp + 60)
                spark_sensitivity = self._mat_value(mat_data, "spark_sensitivity", 0.0)

                # Moisture raises effective ignition threshold and suppresses sparks
                moisture = self.moisture[row][col] if hasattr(self, "moisture") else 0.0
                eff_ignition_temp = ignition_temp + moisture * 80.0
                eff_spark_sens = spark_sensitivity * (1.0 - moisture * 0.9)

                has_flame_neighbor = self._neighbor_is_flaming(row, col, rows, cols)
                has_hot_neighbor = self._neighbor_is_hot(row, col, rows, cols, eff_ignition_temp * 0.85)

                can_ignite = self.temperature[row][col] >= eff_ignition_temp and (
                    has_flame_neighbor or has_hot_neighbor
                    or rng.random() < max(0.02, eff_spark_sens * 0.05)
                )
                if self.temperature[row][col] >= auto_ignite_temp and moisture < 0.1:
                    can_ignite = True

                if can_ignite:
                    self.burn_stage[row][col] = 1
                    self.burn_progress[row][col] = 0.0
                    events.append({"type": "ignite", "row": row, "col": col})

    def _update_combustion_states(self, grid, rows, cols, tick_index, events, counters):
        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    self.burn_stage[row][col] = 0
                    self.burn_progress[row][col] = 0.0
                    continue

                mat_data = self._get(mat)
                if not self._is_combustible(mat_data):
                    self.burn_stage[row][col] = 0
                    self.burn_progress[row][col] = 0.0
                    continue

                stage = self.burn_stage[row][col]
                if stage == 0:
                    continue

                burn_rate = self._mat_value(mat_data, "burn_rate", 0.02)
                oxygen = self.oxygen_level[row][col]
                min_oxygen = self._mat_value(mat_data, "min_oxygen_for_ignition", 0.2)
                oxygen_factor = max(0.2, oxygen)

                stage_multiplier = {1: 0.5, 2: 1.0, 3: 1.5, 4: 0.6}.get(stage, 1.0)
                self.burn_progress[row][col] += burn_rate * oxygen_factor * stage_multiplier

                # ── Exothermic heat release (oxidation) ──────────────────────
                # Stage 1: kindling only, no significant reaction yet
                # Stage 2: pyrolysis – mild gas release, small heat
                # Stage 3: full flaming – peak exothermicity  (C + O2 → CO2 + Q)
                # Stage 4: smoldering – slow surface oxidation, little heat
                hoc = self._mat_value(mat_data, "heat_of_combustion", 0.0)
                if hoc > 0.0 and stage in (2, 3, 4):
                    exo_factor = {2: 0.12, 3: 1.0, 4: 0.08}[stage]
                    Q = hoc * burn_rate * oxygen_factor * stage_multiplier * exo_factor
                    # heat released directly into the burning cell
                    thermal_cap = max(0.1, self._mat_value(mat_data, "thermal_capacity", 1.0))
                    self.temperature[row][col] += Q / thermal_cap
                    # 30 % radiates into each of the 4 cardinal neighbours
                    radiate = Q * 0.30
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = row + dr, col + dc
                        if self._in_bounds(nr, nc, rows, cols):
                            n_mat = grid[nr][nc]
                            n_cap = max(0.1, self._mat_value(self._get(n_mat), "thermal_capacity", 1.0))
                            self.temperature[nr][nc] += radiate / n_cap

                if stage == 1 and self.burn_progress[row][col] >= 0.25:
                    self.burn_stage[row][col] = 2
                    events.append({"type": "burn_stage_changed", "stage": "pyrolysis", "row": row, "col": col})
                elif stage == 2 and (self.burn_progress[row][col] >= 0.45 and oxygen >= min_oxygen):
                    self.burn_stage[row][col] = 3
                    events.append({"type": "burn_stage_changed", "stage": "flaming", "row": row, "col": col})
                elif stage == 3 and (oxygen < min_oxygen * 0.75 or self.burn_progress[row][col] >= 0.85):
                    self.burn_stage[row][col] = 4
                    events.append({"type": "burn_stage_changed", "stage": "smoldering", "row": row, "col": col})
                elif stage == 4 and self.burn_progress[row][col] >= 1.0:
                    burnout_product = self._mat_value(mat_data, "burnout_product", 0)
                    grid[row][col] = burnout_product
                    self.burn_stage[row][col] = 0
                    self.burn_progress[row][col] = 0.0
                    self.ignition_cooldown_until[row][col] = tick_index + self.thermal_config.ignition_cooldown_ticks
                    counters["changes"] += 1
                    events.append({"type": "extinguish", "row": row, "col": col})

    def _update_oxygen(self, rows, cols):
        next_oxygen = self.oxygen_level.copy()
        for row in range(rows):
            for col in range(cols):
                total = 0.0
                count = 0
                for y in range(max(0, row - 1), min(rows, row + 2)):
                    for x in range(max(0, col - 1), min(cols, col + 2)):
                        if y == row and x == col:
                            continue
                        total += self.oxygen_level[y][x]
                        count += 1

                neighbor_avg = total / count if count else self.oxygen_level[row][col]
                diffused = self.oxygen_level[row][col] + ((neighbor_avg - self.oxygen_level[row][col]) * self.thermal_config.oxygen_diffusion)

                if self.burn_stage[row][col] == 3:
                    diffused -= 0.08
                elif self.burn_stage[row][col] == 4:
                    diffused -= 0.03

                if row == 0:
                    diffused = max(diffused, 0.98)

                next_oxygen[row][col] = max(0.0, min(1.0, diffused))

        self.oxygen_level = next_oxygen

    def _update_smoke(self, rows, cols, events):
        next_smoke = self.smoke_density.copy()
        for row in range(rows):
            for col in range(cols):
                if self.burn_stage[row][col] == 3:
                    next_smoke[row][col] = min(1.0, next_smoke[row][col] + self.thermal_config.smoke_spawn_rate)
                    events.append({"type": "smoke_spawn", "row": row, "col": col})
                elif self.burn_stage[row][col] == 4:
                    next_smoke[row][col] = min(1.0, next_smoke[row][col] + (self.thermal_config.smoke_spawn_rate * 0.45))

        for row in range(1, rows):
            for col in range(cols):
                carry = next_smoke[row][col] * 0.2
                next_smoke[row][col] -= carry
                next_smoke[row - 1][col] = min(1.0, next_smoke[row - 1][col] + carry)

        for row in range(rows):
            for col in range(cols):
                next_smoke[row][col] = max(0.0, next_smoke[row][col] - 0.01)

        self.smoke_density = next_smoke

    def _spawn_fire_particles(self, grid, rows, cols, rng):
        """Spawn short-lived fire-particle cells next to actively flaming cells."""
        fire_id = self.material_ids.get("fire", -1)
        if fire_id < 0 or not hasattr(self, "fire_lifetime"):
            return
        for row in range(rows):
            for col in range(cols):
                stage = self.burn_stage[row][col]
                if stage == 3:
                    attempts = 14
                    chance = 0.88
                elif stage == 4:
                    attempts = 6
                    chance = 0.55
                elif stage == 2:
                    attempts = 4
                    chance = 0.35
                else:
                    continue
                # candidate offsets: heavily upward-biased, plus sides
                offsets = [
                    (-1,  0), (-1,  0), (-1,  0),   # directly above (weighted)
                    (-2,  0), (-2,  0), (-2,  0),   # two cells above (weighted)
                    (-3,  0), (-3,  0),              # three cells above
                    (-1, -1), (-1,  1),              # diagonal up
                    ( 0, -1), ( 0,  1),              # sides
                ]
                for _ in range(attempts):
                    dr, dc = offsets[rng.randrange(len(offsets))]
                    nr, nc = row + dr, col + dc
                    if not self._in_bounds(nr, nc, rows, cols):
                        continue
                    if grid[nr][nc] != 0:
                        continue
                    if rng.random() > chance:
                        continue
                    grid[nr][nc] = fire_id
                    self.fire_lifetime[nr][nc] = 0.85 + rng.random() * 0.40
                    self.temperature[nr][nc] = 400.0 + rng.random() * 200.0

    def _update_fire_particles(self, grid, rows, cols, rng):
        """Decay fire-particle lifetime; make them rise; convert dying fire to smoke."""
        fire_id = self.material_ids.get("fire", -1)
        smoke_id = self.material_ids.get("smoke", -1)
        if fire_id < 0 or not hasattr(self, "fire_lifetime"):
            return
        DECAY = 0.042
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != fire_id:
                    if self.fire_lifetime[row][col] > 0.0:
                        self.fire_lifetime[row][col] = 0.0
                    continue
                self.fire_lifetime[row][col] -= DECAY
                if self.fire_lifetime[row][col] <= 0.0:
                    # Convert dying fire to smoke if smoke available and space permits
                    if smoke_id >= 0 and hasattr(self, "smoke_lifetime") and rng.random() < 0.55:
                        grid[row][col] = smoke_id
                        self.smoke_lifetime[row][col] = 0.7 + rng.random() * 0.5
                        self.temperature[row][col] = 80.0 + rng.random() * 40.0
                    else:
                        grid[row][col] = 0
                    self.fire_lifetime[row][col] = 0.0
                    continue
                # Fire rises: try to move up each tick (70% chance)
                if rng.random() < 0.70:
                    for dr, dc in [(-1, 0), (-1, rng.choice([-1, 1])), (-1, 0)]:
                        nr, nc = row + dr, col + dc
                        if self._in_bounds(nr, nc, rows, cols) and grid[nr][nc] == 0:
                            grid[nr][nc] = fire_id
                            self.fire_lifetime[nr][nc] = self.fire_lifetime[row][col]
                            self.temperature[nr][nc] = self.temperature[row][col]
                            grid[row][col] = 0
                            self.fire_lifetime[row][col] = 0.0
                            break

    def _update_fire_sparks(self, grid, rows, cols, rng, events):
        for row in range(rows):
            for col in range(cols):
                if self.burn_stage[row][col] != 3:
                    continue

                if rng.random() < 0.08:
                    target_row = row - 1
                    target_col = col + rng.choice([-1, 0, 1])
                    if not self._in_bounds(target_row, target_col, rows, cols):
                        continue

                    events.append({"type": "spark_spawn", "row": target_row, "col": target_col})
                    target_mat = grid[target_row][target_col]
                    if target_mat == 0:
                        continue

                    target_data = self._get(target_mat)
                    if not self._is_combustible(target_data):
                        continue

                    if self.burn_stage[target_row][target_col] != 0:
                        continue

                    min_oxygen = self._mat_value(target_data, "min_oxygen_for_ignition", 0.2)
                    if self.oxygen_level[target_row][target_col] < min_oxygen:
                        continue

                    spark_sensitivity = self._mat_value(target_data, "spark_sensitivity", 0.0)
                    if rng.random() < max(0.05, spark_sensitivity * 0.35):
                        self.burn_stage[target_row][target_col] = 1
                        self.burn_progress[target_row][target_col] = 0.0
                        events.append({"type": "spark_ignite", "row": target_row, "col": target_col})

    def _update_moisture(self, grid, rows, cols):
        """Water adjacent to combustible cells dampens them; heat dries them out."""
        water_id = self.material_ids.get("water", -1)
        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                mat_data = self._get(mat)
                if not self._is_combustible(mat_data):
                    continue
                # Absorb moisture from adjacent water
                has_water = False
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, col + dc
                    if self._in_bounds(nr, nc, rows, cols) and grid[nr][nc] == water_id:
                        has_water = True
                        break
                if has_water:
                    self.moisture[row][col] = min(1.0, self.moisture[row][col] + 0.015)
                else:
                    # Dry out: faster at high temp
                    dry_rate = 0.003 + max(0.0, self.temperature[row][col] - 60.0) * 0.0004
                    self.moisture[row][col] = max(0.0, self.moisture[row][col] - dry_rate)
                # Burning removes moisture entirely
                if self.burn_stage[row][col] > 0:
                    self.moisture[row][col] = 0.0

    def _update_explosion(self, grid, rows, cols, rng, events, counters):
        """Gunpowder in flaming stage (3) triggers a radial heat blast."""
        gunpowder_id = self.material_ids.get("gunpowder", -1)
        fire_id = self.material_ids.get("fire", -1)
        if gunpowder_id < 0:
            return
        BLAST_RADIUS = 5
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != gunpowder_id:
                    continue
                if self.burn_stage[row][col] != 3:
                    continue
                if self.burn_progress[row][col] < 0.3:
                    continue
                # Only trigger once per cell (progress check prevents re-trigger)
                events.append({"type": "explosion", "row": row, "col": col})
                self.pending_detonations.append((row, col, max(1.0, float(self.burn_progress[row][col]))))
                self.burn_stage[row][col] = 4
                self.burn_progress[row][col] = 1.0
                for dr in range(-BLAST_RADIUS, BLAST_RADIUS + 1):
                    for dc in range(-BLAST_RADIUS, BLAST_RADIUS + 1):
                        dist = (dr * dr + dc * dc) ** 0.5
                        if dist > BLAST_RADIUS:
                            continue
                        nr, nc = row + dr, col + dc
                        if not self._in_bounds(nr, nc, rows, cols):
                            continue
                        falloff = 1.0 - dist / BLAST_RADIUS
                        heat_bonus = 420.0 * falloff
                        self.temperature[nr][nc] = min(1800.0, self.temperature[nr][nc] + heat_bonus)
                        # Scatter loose cells (powder/liquid) away from center
                        n_mat = grid[nr][nc]
                        if n_mat == 0:
                            # Spawn fire particle in void near blast
                            if fire_id >= 0 and rng.random() < 0.5 * falloff:
                                grid[nr][nc] = fire_id
                                self.fire_lifetime[nr][nc] = 0.6 + rng.random() * 0.5
                        else:
                            n_data = self._get(n_mat)
                            n_type = n_data["type"]
                            # Displace powder/liquid into outward free cell
                            if n_type in ("powder", "liquid") and dist > 0.5:
                                out_r = row + int(round(dr / max(0.001, dist) * (BLAST_RADIUS - dist + 1)))
                                out_c = col + int(round(dc / max(0.001, dist) * (BLAST_RADIUS - dist + 1)))
                                out_r = max(0, min(rows - 1, out_r))
                                out_c = max(0, min(cols - 1, out_c))
                                if grid[out_r][out_c] == 0 and rng.random() < falloff * 0.6:
                                    grid[out_r][out_c] = n_mat
                                    self.temperature[out_r][out_c] = self.temperature[nr][nc]
                                    grid[nr][nc] = 0
                                    counters["changes"] += 1

    def _update_plant_growth(self, grid, rows, cols, tick_index, rng):
        """Plants slowly spread into adjacent empty air cells that have a surface below."""
        plant_id = self.material_ids.get("plant", -1)
        if plant_id < 0:
            return
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != plant_id:
                    continue
                if self.temperature[row][col] > 55.0:
                    continue  # too hot, stunted growth
                if rng.random() > 0.004:  # ~0.4% chance per tick per cell
                    continue
                # Try spreading to a random orthogonal empty air cell
                dirs = [(-1, 0), (0, 1), (0, -1), (1, 0)]
                rng.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = row + dr, col + dc
                    if not self._in_bounds(nr, nc, rows, cols):
                        continue
                    if grid[nr][nc] != 0:
                        continue
                    # Needs a solid/powder support somewhere adjacent to target
                    has_support = False
                    for sr, sc in ((nr + 1, nc), (nr - 1, nc), (nr, nc + 1), (nr, nc - 1)):
                        if self._in_bounds(sr, sc, rows, cols) and grid[sr][sc] not in (0, plant_id):
                            has_support = True
                            break
                    if has_support:
                        grid[nr][nc] = plant_id
                        self.temperature[nr][nc] = self.temperature[row][col]
                        break

    def _spawn_smoke_particles(self, grid, rows, cols, rng):
        """Spawn rising smoke particles above smoldering/burning cells."""
        smoke_id = self.material_ids.get("smoke", -1)
        if smoke_id < 0 or not hasattr(self, "smoke_lifetime"):
            return
        for row in range(rows):
            for col in range(cols):
                stage = self.burn_stage[row][col]
                mat_data = self._get(grid[row][col])
                smoke_factor = self._mat_value(mat_data, "smoke_factor", 0.0)
                if smoke_factor <= 0.0:
                    continue
                if stage == 3:
                    attempts = max(4, int(smoke_factor * 10))
                    chance = min(0.90, 0.72 * smoke_factor + 0.18)
                elif stage == 4:
                    attempts = max(3, int(smoke_factor * 8))
                    chance = min(0.93, 0.85 * smoke_factor + 0.20)
                elif stage == 2:
                    attempts = max(1, int(smoke_factor * 3))
                    chance = min(0.50, 0.30 * smoke_factor)
                elif stage == 1:
                    attempts = 1
                    chance = min(0.20, 0.15 * smoke_factor)
                else:
                    continue
                # Spawn 2-4 cells above the flame so smoke isn't blocked by fire particles
                offsets = [
                    (-2,  0), (-2,  0), (-2,  0),
                    (-3,  0), (-3,  0),
                    (-4,  0),
                    (-2, -1), (-2,  1),
                    (-3, -1), (-3,  1),
                    (-1,  0),
                ]
                for _ in range(attempts):
                    dr, dc = offsets[rng.randrange(len(offsets))]
                    nr, nc = row + dr, col + dc
                    if not self._in_bounds(nr, nc, rows, cols):
                        continue
                    if grid[nr][nc] != 0:
                        continue
                    if rng.random() > chance:
                        continue
                    grid[nr][nc] = smoke_id
                    self.smoke_lifetime[nr][nc] = 1.2 + rng.random() * 0.8
                    self.temperature[nr][nc] = 55.0 + rng.random() * 35.0

    def _update_smoke_particles(self, grid, rows, cols, rng):
        """Decay smoke particle lifetime; make them drift upward slowly."""
        smoke_id = self.material_ids.get("smoke", -1)
        if smoke_id < 0 or not hasattr(self, "smoke_lifetime"):
            return
        DECAY = 0.010  # slow decay: ~120-200 ticks lifetime
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != smoke_id:
                    if self.smoke_lifetime[row][col] > 0.0:
                        self.smoke_lifetime[row][col] = 0.0
                    continue
                self.smoke_lifetime[row][col] -= DECAY
                if self.smoke_lifetime[row][col] <= 0.0:
                    grid[row][col] = 0
                    self.smoke_lifetime[row][col] = 0.0
                    continue
                # Only move 55% of ticks so smoke drifts slowly and stays visible
                if rng.random() > 0.55:
                    continue
                # Try to rise; favour straight up, occasionally drift sideways
                side = rng.choice([-1, 1])
                for dr, dc in [(-1, 0), (-1, side), (0, side), (-2, 0)]:
                    nr, nc = row + dr, col + dc
                    if self._in_bounds(nr, nc, rows, cols) and grid[nr][nc] == 0:
                        grid[nr][nc] = smoke_id
                        self.smoke_lifetime[nr][nc] = self.smoke_lifetime[row][col]
                        self.temperature[nr][nc] = self.temperature[row][col]
                        grid[row][col] = 0
                        self.smoke_lifetime[row][col] = 0.0
                        break

    def _stage_thermal(self, grid, rows, cols, tick_index, rng, counters, events):
        self._ensure_thermal_state(rows, cols)
        self._ensure_chemical_state(grid, rows, cols)
        self._update_moisture_transport_phase3(grid, rows, cols)
        self._update_moisture(grid, rows, cols)
        self._update_thermal_field(grid, rows, cols)
        self._apply_leidenfrost_evaporation(grid, rows, cols)
        self._update_water_cooling(grid, rows, cols)
        self._update_contact_heating(grid, rows, cols)
        self._update_ignition(grid, rows, cols, tick_index, rng, events)
        self._update_combustion_states(grid, rows, cols, tick_index, events, counters)
        self._update_explosion(grid, rows, cols, rng, events, counters)
        self._update_shock_failure(grid, rows, cols, counters, events)
        self._update_oxygen(rows, cols)
        self._update_smoke(rows, cols, events)
        self._update_species_diffusion_phase3(grid, rows, cols)
        self._spawn_fire_particles(grid, rows, cols, rng)
        self._update_fire_particles(grid, rows, cols, rng)
        self._spawn_smoke_particles(grid, rows, cols, rng)
        self._update_smoke_particles(grid, rows, cols, rng)
        self._update_fire_sparks(grid, rows, cols, rng, events)
        self._update_plant_growth(grid, rows, cols, tick_index, rng)

    def _evaluate_reaction_conditions(self, rule, row, col, n_row, n_col):
        conditions = rule.get("conditions", {})
        avg_temp = (self.temperature[row][col] + self.temperature[n_row][n_col]) * 0.5
        avg_oxygen = (self.oxygen_level[row][col] + self.oxygen_level[n_row][n_col]) * 0.5
        avg_mix = (self.mix_ratio[row][col] + self.mix_ratio[n_row][n_col]) * 0.5
        avg_contact = (self.reaction_progress[row][col] + self.reaction_progress[n_row][n_col]) * 0.5

        if avg_temp < conditions.get("min_temp", -1e9):
            return False
        if avg_temp > conditions.get("max_temp", 1e9):
            return False
        if avg_oxygen < conditions.get("min_oxygen", 0.0):
            return False
        if avg_mix < conditions.get("min_mix", 0.0):
            return False
        if avg_contact < conditions.get("min_contact_progress", 0.0):
            return False

        return True

    def _build_phase6_material_fields(self, grid_np, rows, cols):
        solid_mask = np.zeros((rows, cols), dtype=np.bool_)
        gas_mask = np.zeros((rows, cols), dtype=np.bool_)
        liquid_mask = np.zeros((rows, cols), dtype=np.bool_)
        catalyst_mask = np.zeros((rows, cols), dtype=np.bool_)
        pyro_solid_mask = np.zeros((rows, cols), dtype=np.bool_)

        A = np.full((rows, cols), np.float32(2.0e6), dtype=np.float32)
        Ea = np.full((rows, cols), np.float32(8.5e4), dtype=np.float32)
        ord_f = np.full((rows, cols), np.float32(1.0), dtype=np.float32)
        ord_o = np.full((rows, cols), np.float32(1.0), dtype=np.float32)
        stoich = np.full((rows, cols), np.float32(3.5), dtype=np.float32)
        q_release = np.full((rows, cols), np.float32(2.6e6), dtype=np.float32)
        edc_c = np.full((rows, cols), np.float32(4.0), dtype=np.float32)
        py_t0 = np.full((rows, cols), np.float32(420.0), dtype=np.float32)
        py_t1 = np.full((rows, cols), np.float32(820.0), dtype=np.float32)
        py_lv = np.full((rows, cols), np.float32(4.5e5), dtype=np.float32)
        py_yield = np.full((rows, cols), np.float32(0.35), dtype=np.float32)
        soot_nuc = np.zeros((rows, cols), dtype=np.float32)
        soot_growth = np.zeros((rows, cols), dtype=np.float32)
        soot_oxid = np.zeros((rows, cols), dtype=np.float32)
        acid_src = np.zeros((rows, cols), dtype=np.float32)
        base_src = np.zeros((rows, cols), dtype=np.float32)
        cat_act = np.zeros((rows, cols), dtype=np.float32)
        ads_f = np.zeros((rows, cols), dtype=np.float32)
        ads_o = np.zeros((rows, cols), dtype=np.float32)
        k_surf = np.zeros((rows, cols), dtype=np.float32)

        for mat_id, mat_data in self.materials.items():
            mask = (grid_np == int(mat_id))
            if not np.any(mask):
                continue
            m_type = str(mat_data.get("type", "air"))
            if m_type in ("solid", "powder"):
                solid_mask[mask] = True
            elif m_type == "gas":
                gas_mask[mask] = True
            elif m_type == "liquid":
                liquid_mask[mask] = True

            A[mask] = np.float32(max(1.0, float(self._mat_value(mat_data, "arrhenius_A", 2.0e6))))
            Ea[mask] = np.float32(max(1.0, float(self._mat_value(mat_data, "arrhenius_Ea", 8.5e4))))
            ord_f[mask] = np.float32(max(0.25, float(self._mat_value(mat_data, "arrhenius_order_fuel", 1.0))))
            ord_o[mask] = np.float32(max(0.25, float(self._mat_value(mat_data, "arrhenius_order_o2", 1.0))))
            stoich[mask] = np.float32(max(0.1, float(self._mat_value(mat_data, "stoich_o2_per_fuel", 3.5))))
            q_release[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "reaction_heat_release", 2.6e6))))
            edc_c[mask] = np.float32(max(0.1, float(self._mat_value(mat_data, "edc_coeff", 4.0))))
            py_start = max(50.0, float(self._mat_value(mat_data, "pyrolysis_temp_start", 420.0)))
            py_peak = max(py_start + 1.0, float(self._mat_value(mat_data, "pyrolysis_temp_peak", 820.0)))
            py_t0[mask] = np.float32(py_start)
            py_t1[mask] = np.float32(py_peak)
            py_lv[mask] = np.float32(max(1.0, float(self._mat_value(mat_data, "pyrolysis_latent_heat", 4.5e5))))
            py_yield[mask] = np.float32(np.clip(float(self._mat_value(mat_data, "pyrolysis_yield", 0.35)), 0.0, 1.0))
            soot_nuc[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "soot_nucleation_factor", 0.0))))
            soot_growth[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "soot_growth_factor", 0.0))))
            soot_oxid[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "soot_oxidation_factor", 0.0))))
            acid_src[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "acid_strength", 0.0))))
            base_src[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "base_strength", 0.0))))
            cat_act[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "catalytic_activity", 0.0))))
            ads_f[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "adsorption_fuel", 0.0))))
            ads_o[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "adsorption_o2", 0.0))))
            k_surf[mask] = np.float32(max(0.0, float(self._mat_value(mat_data, "surface_reaction_rate", 0.0))))

            if m_type in ("solid", "powder") and float(self._mat_value(mat_data, "pyrolysis_yield", 0.0)) > 0.0:
                pyro_solid_mask[mask] = True
            if cat_act[mask].max() > 0.0:
                catalyst_mask[mask] = True

        return {
            "solid_mask": solid_mask,
            "gas_mask": gas_mask,
            "liquid_mask": liquid_mask,
            "catalyst_mask": catalyst_mask,
            "pyro_solid_mask": pyro_solid_mask,
            "A": A,
            "Ea": Ea,
            "ord_f": ord_f,
            "ord_o": ord_o,
            "stoich": stoich,
            "q_release": q_release,
            "edc_c": edc_c,
            "py_t0": py_t0,
            "py_t1": py_t1,
            "py_lv": py_lv,
            "py_yield": py_yield,
            "soot_nuc": soot_nuc,
            "soot_growth": soot_growth,
            "soot_oxid": soot_oxid,
            "acid_src": acid_src,
            "base_src": base_src,
            "cat_act": cat_act,
            "ads_f": ads_f,
            "ads_o": ads_o,
            "k_surf": k_surf,
        }

    def _update_phase6_stiff_kinetics_edc(self, rows, cols, mats, counters, events):
        if not self.chemistry_config.stiff_kinetics_enabled:
            return

        dt = np.float32(PHYSICS.dt)
        R = np.float32(self.chemistry_config.arrhenius_R)
        T = np.clip(self.temperature.astype(np.float32), self.chemistry_config.arrhenius_temp_clamp_min, self.chemistry_config.arrhenius_temp_clamp_max)
        fuel0 = np.clip(self.fuel_vapor.astype(np.float32), 0.0, 1.0)
        o20 = np.clip(self.oxygen_level.astype(np.float32), 0.0, 1.0)
        stoich = mats["stoich"]

        A = mats["A"]
        Ea = mats["Ea"]
        ord_f = mats["ord_f"]
        ord_o = mats["ord_o"]
        k_arr = A * np.exp(-Ea / np.maximum(np.float32(1e-6), (R * T)))

        if self.vel_x.shape == (rows, cols):
            u = self.vel_x.astype(np.float32)
            v = self.vel_y.astype(np.float32)
            u_l = np.empty_like(u); u_r = np.empty_like(u); u_t = np.empty_like(u); u_b = np.empty_like(u)
            v_l = np.empty_like(v); v_r = np.empty_like(v); v_t = np.empty_like(v); v_b = np.empty_like(v)
            u_l[:, 0] = u[:, 0]; u_l[:, 1:] = u[:, :-1]
            u_r[:, -1] = u[:, -1]; u_r[:, :-1] = u[:, 1:]
            u_t[0, :] = u[0, :]; u_t[1:, :] = u[:-1, :]
            u_b[-1, :] = u[-1, :]; u_b[:-1, :] = u[1:, :]
            v_l[:, 0] = v[:, 0]; v_l[:, 1:] = v[:, :-1]
            v_r[:, -1] = v[:, -1]; v_r[:, :-1] = v[:, 1:]
            v_t[0, :] = v[0, :]; v_t[1:, :] = v[:-1, :]
            v_b[-1, :] = v[-1, :]; v_b[:-1, :] = v[1:, :]
            du_dx = np.float32(0.5) * (u_r - u_l) * np.float32(PHYSICS.dx_inv)
            dv_dy = np.float32(0.5) * (v_b - v_t) * np.float32(PHYSICS.dy_inv)
            du_dy = np.float32(0.5) * (u_b - u_t) * np.float32(PHYSICS.dy_inv)
            dv_dx = np.float32(0.5) * (v_r - v_l) * np.float32(PHYSICS.dx_inv)
            strain_mag = np.sqrt(np.maximum(np.float32(0.0), du_dx * du_dx + dv_dy * dv_dy + np.float32(0.5) * (du_dy + dv_dx) ** 2))
            self.turb_k = np.maximum(np.float32(1e-6), np.float32(0.5) * (u * u + v * v)).astype(np.float32)
            self.turb_eps = np.maximum(np.float32(self.chemistry_config.edc_min_epsilon), (np.float32(0.09) ** np.float32(0.75)) * self.turb_k * strain_mag).astype(np.float32)
        else:
            self.turb_k.fill(np.float32(1e-5))
            self.turb_eps.fill(np.float32(1e-4))

        tau_mix = np.clip(self.turb_k / np.maximum(self.turb_eps, np.float32(self.chemistry_config.edc_min_epsilon)), self.chemistry_config.edc_tau_clip_min, self.chemistry_config.edc_tau_clip_max)
        mix_factor = np.clip(np.float32(4.0) * self.mixture_fraction * (np.float32(1.0) - self.mixture_fraction), 0.0, 1.0)
        edc_lim = mats["edc_c"] * np.minimum(fuel0, o20 / np.maximum(np.float32(1e-6), stoich)) / np.maximum(np.float32(1e-6), tau_mix)
        base_poly = np.maximum(np.float32(1e-9), (fuel0 ** ord_f) * (o20 ** ord_o))
        if self.chemistry_config.edc_enabled:
            k_eff = np.minimum(k_arr, edc_lim / base_poly)
        else:
            k_eff = k_arr
        k_eff = (k_eff * mix_factor).astype(np.float32)

        xi = np.zeros_like(fuel0)
        xi_max = np.minimum(fuel0, o20 / np.maximum(np.float32(1e-6), stoich))
        active = (~mats["solid_mask"]) & (fuel0 > 1.0e-7) & (o20 > 1.0e-7) & (xi_max > 1.0e-9)

        for _ in range(max(1, int(self.chemistry_config.stiff_newton_iterations))):
            yf = np.maximum(np.float32(1e-12), fuel0 - xi)
            yo = np.maximum(np.float32(1e-12), o20 - stoich * xi)
            f = xi - dt * k_eff * (yf ** ord_f) * (yo ** ord_o)
            df = np.float32(1.0) + dt * k_eff * (
                ord_f * (yf ** np.maximum(np.float32(0.0), ord_f - np.float32(1.0))) * (yo ** ord_o)
                + stoich * ord_o * (yf ** ord_f) * (yo ** np.maximum(np.float32(0.0), ord_o - np.float32(1.0)))
            )
            step = np.zeros_like(xi)
            step[active] = f[active] / np.maximum(np.float32(1e-9), df[active])
            xi = np.clip(xi - step, 0.0, xi_max)

        if not np.any(active):
            return

        fuel_new = np.clip(fuel0 - xi, 0.0, 1.0)
        o2_new = np.clip(o20 - stoich * xi, 0.0, 1.0)
        rich = np.clip(np.float32(1.0) - (o20 / np.maximum(np.float32(1e-6), stoich * fuel0 + np.float32(1e-6))), 0.0, 1.0)
        complete = np.float32(1.0) - rich

        self.fuel_vapor = fuel_new.astype(np.float32)
        self.oxygen_level = o2_new.astype(np.float32)
        self.co2_density = np.clip(self.co2_density + xi * (np.float32(0.78) * complete + np.float32(0.3) * rich), 0.0, 2.0)
        self.co_density = np.clip(self.co_density + xi * (np.float32(0.02) * complete + np.float32(0.45) * rich), 0.0, 1.0)
        self.h2o_vapor = np.clip(self.h2o_vapor + xi * np.float32(0.5), 0.0, 2.0)

        heat = mats["q_release"] * xi * (np.float32(0.7) + np.float32(0.3) * complete)
        self.temperature = (self.temperature + (heat * np.float32(2.0e-5))).astype(np.float32)
        self.mixture_fraction = np.clip(self.fuel_vapor / np.maximum(np.float32(1e-6), self.fuel_vapor + self.oxygen_level / np.maximum(np.float32(1e-6), stoich)), 0.0, 1.0)

        reacted = float(np.sum(xi))
        if reacted > 1.0e-3:
            counters["changes"] += int(max(1, reacted * 8.0))
            events.append({"type": "stiff_combustion", "reacted": reacted})

    def _update_phase6_pyrolysis(self, grid, rows, cols, mats, counters, events):
        if not self.chemistry_config.pyrolysis_enabled:
            return

        grid_np = np.asarray(grid, dtype=np.int32)
        pyro_mask = mats["pyro_solid_mask"]
        if not np.any(pyro_mask):
            return

        amb = np.float32(self.thermal_config.ambient_temp)
        dT = np.maximum(np.float32(0.0), self.temperature - mats["py_t0"])
        progress_drive = np.clip(dT / np.maximum(np.float32(1.0), mats["py_t1"] - mats["py_t0"]), 0.0, 1.0)
        q_rad = self.chemistry_config.pyrolysis_radiative_gain * np.maximum(np.float32(0.0), self.temperature - amb)

        temp_l = np.empty_like(self.temperature); temp_r = np.empty_like(self.temperature)
        temp_t = np.empty_like(self.temperature); temp_b = np.empty_like(self.temperature)
        temp_l[:, 0] = self.temperature[:, 0]; temp_l[:, 1:] = self.temperature[:, :-1]
        temp_r[:, -1] = self.temperature[:, -1]; temp_r[:, :-1] = self.temperature[:, 1:]
        temp_t[0, :] = self.temperature[0, :]; temp_t[1:, :] = self.temperature[:-1, :]
        temp_b[-1, :] = self.temperature[-1, :]; temp_b[:-1, :] = self.temperature[1:, :]
        neighbor_avg = np.float32(0.25) * (temp_l + temp_r + temp_t + temp_b)

        q_conv = self.chemistry_config.pyrolysis_convective_gain * np.maximum(np.float32(0.0), neighbor_avg - self.temperature)
        q_cond = self.chemistry_config.pyrolysis_conductive_loss * np.maximum(np.float32(0.0), self.temperature - neighbor_avg)
        q_net = np.maximum(np.float32(0.0), q_rad + q_conv - q_cond)

        m_dot = self.chemistry_config.pyrolysis_mass_flux_scale * q_net / np.maximum(np.float32(1.0), mats["py_lv"])
        m_dot = (m_dot * mats["py_yield"] * progress_drive * pyro_mask).astype(np.float32)
        self.pyrolysis_progress = np.clip(self.pyrolysis_progress + m_dot * np.float32(8.0), 0.0, 1.0)

        smoke_id = self.material_ids.get("smoke", -1)
        for row in range(rows):
            for col in range(cols):
                if not pyro_mask[row, col]:
                    continue
                md = float(m_dot[row, col])
                if md <= 1.0e-7:
                    continue
                self.temperature[row, col] = max(float(amb), float(self.temperature[row, col] - md * mats["py_lv"][row, col] * 3.0e-4))
                injected = False
                for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    nr, nc = row + dr, col + dc
                    if not self._in_bounds(nr, nc, rows, cols):
                        continue
                    n_mat = grid_np[nr, nc]
                    n_type = self._get(int(n_mat))["type"] if int(n_mat) in self.materials else "air"
                    if n_mat != 0 and n_type not in ("gas", "air"):
                        continue
                    self.fuel_vapor[nr, nc] = np.float32(np.clip(float(self.fuel_vapor[nr, nc]) + md, 0.0, 1.0))
                    self.smoke_density[nr, nc] = np.float32(np.clip(float(self.smoke_density[nr, nc]) + md * 0.45, 0.0, 1.0))
                    if self.vel_x.shape == (rows, cols):
                        self.vel_x[nr, nc] += np.float32(self.chemistry_config.pyrolysis_blowing_velocity * dc * md)
                        self.vel_y[nr, nc] += np.float32(self.chemistry_config.pyrolysis_blowing_velocity * dr * md)
                    if grid[nr][nc] == 0 and smoke_id >= 0 and md > 0.01:
                        grid[nr][nc] = smoke_id
                        counters["changes"] += 1
                    injected = True
                if injected:
                    counters["changes"] += 1

        total_py = float(np.sum(m_dot))
        if total_py > 5.0e-4:
            events.append({"type": "pyrolysis", "mass_flux": total_py})

    def _update_phase6_soot(self, mats, events):
        if not self.chemistry_config.soot_enabled:
            return

        dt = np.float32(PHYSICS.dt)
        fuel = np.clip(self.fuel_vapor.astype(np.float32), 0.0, 1.0)
        o2 = np.clip(self.oxygen_level.astype(np.float32), 0.0, 1.0)
        rich = np.clip(fuel - (o2 / np.float32(3.5)), 0.0, 1.0)
        temp_factor = np.clip((self.temperature - np.float32(400.0)) / np.float32(1400.0), 0.0, 1.0)

        nucleation = mats["soot_nuc"] * rich * temp_factor
        growth = mats["soot_growth"] * fuel * np.maximum(np.float32(0.0), self.soot_mass_fraction)
        oxidation = (mats["soot_oxid"] + np.float32(self.chemistry_config.soot_oxidation_o2_coeff)) * o2 * self.soot_mass_fraction

        if self.chemistry_config.soot_diffusion > 0.0:
            diff = self._laplacian_5pt(self.soot_mass_fraction.astype(np.float32))
        else:
            diff = np.zeros_like(self.soot_mass_fraction, dtype=np.float32)

        dY = dt * (nucleation + growth - oxidation) + dt * np.float32(self.chemistry_config.soot_diffusion) * diff
        dN = dt * (
            np.float32(2.0) * nucleation
            - np.float32(self.chemistry_config.soot_coagulation_rate) * (self.soot_number_density ** 2)
            - np.float32(0.2) * oxidation * np.maximum(np.float32(1e-6), self.soot_number_density)
        )

        self.soot_mass_fraction = np.clip(self.soot_mass_fraction + dY, 0.0, 1.0)
        self.soot_number_density = np.clip(self.soot_number_density + dN, 0.0, 1.0)
        self.smoke_density = np.clip(self.smoke_density + np.float32(0.14) * self.soot_mass_fraction, 0.0, 1.0)
        self.temperature = (self.temperature - np.float32(self.chemistry_config.soot_radiation_coupling) * self.soot_mass_fraction * dt).astype(np.float32)

        soot_added = float(np.sum(np.maximum(np.float32(0.0), dY)))
        if soot_added > 1.0e-4:
            events.append({"type": "soot", "delta": soot_added})

    def _update_phase6_electrolyte_equilibrium(self, mats, events):
        if not self.chemistry_config.electrolyte_enabled:
            return

        dt = np.float32(PHYSICS.dt)
        src_scale = np.float32(self.chemistry_config.electrolyte_concentration_scale)
        self.h_plus = np.clip(self.h_plus + dt * mats["acid_src"] * src_scale, 1.0e-12, 1.0)
        self.oh_minus = np.clip(self.oh_minus + dt * mats["base_src"] * src_scale, 1.0e-12, 1.0)

        if self.chemistry_config.electrolyte_diffusion > 0.0:
            self.h_plus = np.clip(self.h_plus + dt * np.float32(self.chemistry_config.electrolyte_diffusion) * self._laplacian_5pt(self.h_plus), 1.0e-12, 1.0)
            self.oh_minus = np.clip(self.oh_minus + dt * np.float32(self.chemistry_config.electrolyte_diffusion) * self._laplacian_5pt(self.oh_minus), 1.0e-12, 1.0)

        h_old = self.h_plus.copy()
        oh_old = self.oh_minus.copy()

        temp_ratio = np.clip((self.temperature + np.float32(273.15)) / np.float32(298.15), 0.5, 6.0)
        Kw = np.float32(self.chemistry_config.electrolyte_kw_25c) * (temp_ratio ** np.float32(0.12))
        charge = h_old - oh_old
        disc = np.sqrt(np.maximum(np.float32(0.0), charge * charge + np.float32(4.0) * Kw))
        h_eq = np.maximum(np.float32(1.0e-12), np.float32(0.5) * (charge + disc))
        oh_eq = np.maximum(np.float32(1.0e-12), Kw / h_eq)

        neutralized = np.maximum(np.float32(0.0), np.minimum(h_old, oh_old) - np.minimum(h_eq, oh_eq))
        self.h_plus = h_eq.astype(np.float32)
        self.oh_minus = oh_eq.astype(np.float32)
        self.ph_field = np.clip(-np.log10(np.maximum(np.float32(1e-12), self.h_plus)), 0.0, 14.0).astype(np.float32)
        self.temperature = (self.temperature + neutralized * np.float32(self.chemistry_config.neutralization_enthalpy) * np.float32(2.0e-5)).astype(np.float32)

        neut = float(np.sum(neutralized))
        if neut > 1.0e-6:
            events.append({"type": "electrolyte_eq", "neutralized": neut})

    def _update_phase6_surface_kinetics(self, grid, rows, cols, mats, events):
        if not self.chemistry_config.surface_kinetics_enabled:
            return

        catalyst_mask = mats["catalyst_mask"]
        if not np.any(catalyst_mask):
            return

        dt = float(PHYSICS.dt)
        diff_flux_coeff = float(self.chemistry_config.lh_diffusivity / max(1.0e-6, self.chemistry_config.lh_reference_length))
        reacted_total = 0.0

        cat_positions = np.argwhere(catalyst_mask)
        for row, col in cat_positions:
            row = int(row); col = int(col)
            k_ads_f = float(mats["ads_f"][row, col])
            k_ads_o = float(mats["ads_o"][row, col])
            k_s = float(mats["k_surf"][row, col]) * float(max(0.0, mats["cat_act"][row, col]))
            if k_s <= 0.0:
                continue

            theta_f = float(self.catalyst_theta_fuel[row, col])
            theta_o = float(self.catalyst_theta_o2[row, col])
            neigh = []
            fuel_near = 0.0
            o2_near = 0.0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc, rows, cols):
                    continue
                n_mat = grid[nr][nc]
                n_type = self._get(n_mat)["type"] if n_mat in self.materials else "air"
                if n_mat != 0 and n_type not in ("gas", "liquid", "air"):
                    continue
                neigh.append((nr, nc))
                fuel_near += float(self.fuel_vapor[nr, nc])
                o2_near += float(self.oxygen_level[nr, nc])
            if not neigh:
                continue

            fuel_near /= float(len(neigh))
            o2_near /= float(len(neigh))
            theta_f = np.clip(theta_f + dt * (k_ads_f * fuel_near * (1.0 - theta_f) - self.chemistry_config.lh_desorption_rate * theta_f), 0.0, 1.0)
            theta_o = np.clip(theta_o + dt * (k_ads_o * o2_near * (1.0 - theta_o) - self.chemistry_config.lh_desorption_rate * theta_o), 0.0, 1.0)

            Da = k_s * self.chemistry_config.lh_reference_length / max(1.0e-8, self.chemistry_config.lh_diffusivity)
            kin_rate = k_s * theta_f * theta_o
            diff_rate = diff_flux_coeff * min(fuel_near, o2_near / 3.5)
            r = diff_rate if Da > 1.0 else kin_rate
            r = max(0.0, r)
            reacted = r * dt
            if reacted <= 0.0:
                self.catalyst_theta_fuel[row, col] = np.float32(theta_f)
                self.catalyst_theta_o2[row, col] = np.float32(theta_o)
                continue

            per_n = reacted / float(len(neigh))
            for nr, nc in neigh:
                self.fuel_vapor[nr, nc] = np.float32(np.clip(float(self.fuel_vapor[nr, nc]) - per_n, 0.0, 1.0))
                self.oxygen_level[nr, nc] = np.float32(np.clip(float(self.oxygen_level[nr, nc]) - per_n * 3.5, 0.0, 1.0))
                self.co2_density[nr, nc] = np.float32(np.clip(float(self.co2_density[nr, nc]) + per_n * 2.0, 0.0, 2.0))
                self.h2o_vapor[nr, nc] = np.float32(np.clip(float(self.h2o_vapor[nr, nc]) + per_n * 1.2, 0.0, 2.0))

            self.temperature[row, col] = np.float32(self.temperature[row, col] + reacted * self.chemistry_config.lh_heat_release_scale * 1.0e-5)
            self.catalyst_theta_fuel[row, col] = np.float32(np.clip(theta_f - reacted * 0.8, 0.0, 1.0))
            self.catalyst_theta_o2[row, col] = np.float32(np.clip(theta_o - reacted * 0.8, 0.0, 1.0))
            reacted_total += reacted

        if reacted_total > 1.0e-6:
            events.append({"type": "surface_kinetics", "reacted": reacted_total})

    def _update_corrosion(self, grid, rows, cols, counters, events):
        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    continue

                mat_data = self._get(mat)
                mat_type = mat_data["type"]
                if mat_type not in ("solid", "powder"):
                    continue

                resistance = self._mat_value(mat_data, "corrosion_resistance", 0.8)
                passivation_factor = self._mat_value(mat_data, "passivation_factor", 0.1)
                corrosion_power = 0.0
                for n_row, n_col in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
                    if not self._in_bounds(n_row, n_col, rows, cols):
                        continue
                    n_mat = grid[n_row][n_col]
                    if n_mat == 0:
                        continue
                    n_data = self._get(n_mat)
                    if n_data["type"] != "liquid":
                        continue
                    corrosion_power = max(corrosion_power, self._mat_value(n_data, "corrosion_power", 0.0))

                if corrosion_power <= 0.0:
                    self.integrity[row][col] = min(1.0, self.integrity[row][col] + (passivation_factor * 0.002))
                    continue

                damage = self.chemistry_config.corrosion_base_rate * corrosion_power
                damage *= max(0.05, 1.0 - resistance)
                damage *= max(0.1, 1.0 - (passivation_factor * 0.6))
                self.integrity[row][col] = max(0.0, self.integrity[row][col] - damage)

                if self.integrity[row][col] <= 0.05:
                    corrosion_product = self._mat_value(mat_data, "corrosion_product", 0)
                    changed = self._set_cell_material(grid, row, col, corrosion_product)
                    if changed:
                        counters["changes"] += 1
                        events.append({"type": "corrosion", "row": row, "col": col, "from": mat, "to": corrosion_product})

    def _update_dissolution(self, grid, rows, cols, counters, events):
        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    continue

                mat_data = self._get(mat)
                dissolution_rate = self._mat_value(mat_data, "dissolution_rate", 0.0)
                if dissolution_rate <= 0.0:
                    continue

                best_neighbor = None
                best_capacity = 0.0
                for n_row, n_col in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
                    if not self._in_bounds(n_row, n_col, rows, cols):
                        continue
                    n_mat = grid[n_row][n_col]
                    if n_mat == 0:
                        continue
                    n_data = self._get(n_mat)
                    if n_data["type"] != "liquid":
                        continue
                    solubility_limit = max(0.05, self._mat_value(n_data, "solubility_limit", 1.0))
                    fill = self.saturation_level[n_row][n_col] / solubility_limit
                    free_capacity = max(0.0, 1.0 - fill)
                    if free_capacity > best_capacity:
                        best_capacity = free_capacity
                        best_neighbor = (n_row, n_col, solubility_limit)

                if best_neighbor is None or best_capacity <= 0.0:
                    continue

                n_row, n_col, solubility_limit = best_neighbor
                dissolve_amount = self.chemistry_config.dissolution_base_rate * dissolution_rate * best_capacity
                self.integrity[row][col] = max(0.0, self.integrity[row][col] - (dissolve_amount * 0.75))
                self.saturation_level[n_row][n_col] = min(solubility_limit, self.saturation_level[n_row][n_col] + dissolve_amount)
                self.temperature[n_row][n_col] += dissolve_amount * 5.0

                if self.integrity[row][col] <= 0.05:
                    dissolution_product = self._mat_value(mat_data, "dissolution_product", 0)
                    changed = self._set_cell_material(grid, row, col, dissolution_product)
                    if changed:
                        counters["changes"] += 1
                        events.append({"type": "dissolution", "row": row, "col": col, "from": mat, "to": dissolution_product})

        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0 or self._get(mat)["type"] != "liquid":
                    self.saturation_level[row][col] = 0.0
                    continue
                self.saturation_level[row][col] = max(0.0, self.saturation_level[row][col] - self.chemistry_config.saturation_decay)

    def _update_reactions(self, grid, rows, cols, counters, events):
        candidates = []
        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    continue
                for n_row, n_col in ((row, col + 1), (row + 1, col)):
                    if not self._in_bounds(n_row, n_col, rows, cols):
                        continue
                    n_mat = grid[n_row][n_col]
                    if n_mat == 0:
                        continue

                    rules = self.interaction_table.get_rules(mat, n_mat)
                    if not rules:
                        continue
                    for rule in rules:
                        if self._evaluate_reaction_conditions(rule, row, col, n_row, n_col):
                            candidates.append((
                                -rule["priority"],
                                row,
                                col,
                                n_row,
                                n_col,
                                rule["_stable_index"],
                                rule,
                            ))

        candidates.sort()
        locked_cells = set()
        for _, row, col, n_row, n_col, _, rule in candidates:
            if (row, col) in locked_cells or (n_row, n_col) in locked_cells:
                continue

            progress_step = 1.0 / max(1, rule["duration_ticks"])
            self.reaction_progress[row][col] = min(1.0, self.reaction_progress[row][col] + progress_step)
            self.reaction_progress[n_row][n_col] = min(1.0, self.reaction_progress[n_row][n_col] + progress_step)

            if self.reaction_progress[row][col] < 1.0 or self.reaction_progress[n_row][n_col] < 1.0:
                continue

            products = list(rule["products"])
            if len(products) == 1:
                products.append(products[0])

            original_a = grid[row][col]
            original_b = grid[n_row][n_col]

            changed_a = self._set_cell_material(grid, row, col, products[0])
            changed_b = self._set_cell_material(grid, n_row, n_col, products[1])
            if changed_a:
                counters["changes"] += 1
            if changed_b:
                counters["changes"] += 1

            energy_delta = rule["energy_delta"]
            self.temperature[row][col] += energy_delta * 0.5
            self.temperature[n_row][n_col] += energy_delta * 0.5

            gas_release = max(0.0, rule["gas_release"])
            if gas_release > 0.0:
                self.smoke_density[row][col] = min(1.0, self.smoke_density[row][col] + gas_release)
                self.smoke_density[n_row][n_col] = min(1.0, self.smoke_density[n_row][n_col] + gas_release)

            residue = rule["residue"]
            if residue != 0:
                if grid[row][col] == 0:
                    self._set_cell_material(grid, row, col, residue)
                    counters["changes"] += 1
                elif grid[n_row][n_col] == 0:
                    self._set_cell_material(grid, n_row, n_col, residue)
                    counters["changes"] += 1

            self.reaction_progress[row][col] = 0.0
            self.reaction_progress[n_row][n_col] = 0.0
            locked_cells.add((row, col))
            locked_cells.add((n_row, n_col))
            events.append({
                "type": "reaction",
                "row": row,
                "col": col,
                "row_b": n_row,
                "col_b": n_col,
                "rule": rule.get("name", "unknown"),
                "from": (original_a, original_b),
                "to": tuple(products[:2]),
            })

        for row in range(rows):
            for col in range(cols):
                self.reaction_progress[row][col] = max(0.0, self.reaction_progress[row][col] - self.chemistry_config.reaction_progress_decay)

    def _stage_chemical(self, grid, rows, cols, counters, events):
        self._ensure_thermal_state(rows, cols)
        self._ensure_fluid_state(rows, cols)
        self._ensure_chemical_state(grid, rows, cols)
        grid_np = np.asarray(grid, dtype=np.int32)
        phase6_mats = self._build_phase6_material_fields(grid_np, rows, cols)
        self._update_phase6_pyrolysis(grid, rows, cols, phase6_mats, counters, events)
        self._update_phase6_stiff_kinetics_edc(rows, cols, phase6_mats, counters, events)
        self._update_phase6_soot(phase6_mats, events)
        self._update_phase6_electrolyte_equilibrium(phase6_mats, events)
        self._update_phase6_surface_kinetics(grid, rows, cols, phase6_mats, events)
        self._update_corrosion(grid, rows, cols, counters, events)
        self._update_dissolution(grid, rows, cols, counters, events)
        self._update_reactions(grid, rows, cols, counters, events)

    def _update_phase_change(self, grid, rows, cols, tick_index, counters, events):
        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    continue

                if tick_index < self.phase_cooldown_until[row][col]:
                    continue

                mat_data = self._get(mat)
                temperature = self.temperature[row][col]
                target = None
                transition_temp = None  # the physical threshold that triggered this change

                freeze_temp = self._mat_value(mat_data, "freeze_temp", None)
                if freeze_temp is not None and temperature <= freeze_temp - self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "freeze_target", target)
                    transition_temp = freeze_temp

                melt_temp = self._mat_value(mat_data, "melt_temp", None)
                if melt_temp is not None and temperature >= melt_temp + self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "melt_target", target)
                    transition_temp = melt_temp

                evaporate_temp = self._mat_value(mat_data, "evaporate_temp", None)
                if evaporate_temp is not None and temperature >= evaporate_temp + self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "evaporate_target", target)
                    transition_temp = evaporate_temp

                condense_temp = self._mat_value(mat_data, "condense_temp", None)
                if condense_temp is not None and temperature <= condense_temp - self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "condense_target", target)
                    transition_temp = condense_temp

                solidify_temp = self._mat_value(mat_data, "solidify_temp", None)
                if solidify_temp is not None and temperature <= solidify_temp - self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "solidify_target", target)
                    transition_temp = solidify_temp

                if target is None or target == mat:
                    self.phase_transition_target[row][col] = 0
                    self.phase_transition_progress[row][col] = max(
                        0.0,
                        self.phase_transition_progress[row][col] - self.phase_change_config.transition_progress_decay,
                    )
                    continue

                if self.phase_transition_target[row][col] != target:
                    self.phase_transition_target[row][col] = target
                    self.phase_transition_progress[row][col] = 0.0

                phase_change_rate = self._mat_value(mat_data, "phase_change_rate", 0.04)
                self.phase_transition_progress[row][col] = min(
                    1.0,
                    self.phase_transition_progress[row][col] + phase_change_rate,
                )
                if self.phase_transition_progress[row][col] < 1.0:
                    continue

                latent_heat = self._mat_value(mat_data, "latent_heat", 0.0) * self.phase_change_config.latent_heat_scale
                target_type = self._get(target)["type"] if target in self.materials else "air"
                source_type = mat_data["type"]

                # Always anchor to the physical transition temperature so that e.g.
                # superheated steam condensing doesn't produce 1200 °C water/ice.
                if transition_temp is not None:
                    # Endothermic transitions (energy absorbed from cell → slight superheat of product):
                    #   solid/powder → liquid  (melting)
                    #   liquid       → gas     (evaporation)
                    if (source_type in ("solid", "powder") and target_type == "liquid") or \
                       (source_type == "liquid" and target_type == "gas"):
                        self.temperature[row][col] = transition_temp + latent_heat
                    # Exothermic transitions (energy released into cell → slight undercool of product):
                    #   gas   → liquid/solid  (condensation / deposition)
                    #   liquid → solid        (freezing / solidification)
                    else:
                        self.temperature[row][col] = transition_temp - latent_heat
                else:
                    # Fallback for materials without an explicit transition_temp
                    if target_type in ("liquid", "gas") and source_type in ("solid", "powder", "liquid"):
                        self.temperature[row][col] = max(self.thermal_config.ambient_temp - 5.0, self.temperature[row][col] - latent_heat)
                    else:
                        self.temperature[row][col] += latent_heat

                changed = self._set_cell_material(grid, row, col, target)
                if not changed:
                    continue

                counters["changes"] += 1
                self.phase_cooldown_until[row][col] = tick_index + self.phase_change_config.min_transition_interval
                self.phase_transition_progress[row][col] = 0.0
                self.phase_transition_target[row][col] = 0
                events.append({"type": "phase_change", "row": row, "col": col, "from": mat, "to": target})

    def _stage_phase_change(self, grid, rows, cols, tick_index, counters, events):
        self._ensure_thermal_state(rows, cols)
        self._ensure_chemical_state(grid, rows, cols)
        self._update_phase_change(grid, rows, cols, tick_index, counters, events)

    def _stage_cleanup(self):
        return

    def step(self, grid, rows, cols, tick_index):
        """Run one tick of physics.  All substep wall-clock times (ms) are
        recorded in self.substep_timings for the HUD profiler (Schritt 8).
        CFL number is computed from the current PDE velocity field and logged
        in self.last_cfl (Schritt 7, Bridson SIGGRAPH07 §3.2.1)."""
        counters = {"changes": 0}
        events   = []
        t = {}  # substep timing dict

        # Ensure all numpy field arrays are sized correctly for this grid
        self._ensure_pde_state(rows, cols)

        mechanics_rng = self.random_manager.for_tick(tick_index, "mechanics")
        fluid_rng     = self.random_manager.for_tick(tick_index, "fluids")
        thermal_rng   = self.random_manager.for_tick(tick_index, "thermal")

        moved = np.zeros((rows, cols), dtype=np.bool_)

        if self.config.enable_mechanics:
            _s = time.perf_counter()
            moved = self._stage_mechanics(grid, rows, cols, moved, mechanics_rng, counters, tick_index)
            t["mechanics"] = (time.perf_counter() - _s) * 1000.0
        if self.config.enable_fluids:
            _s = time.perf_counter()
            self._stage_fluids(grid, rows, cols, moved, fluid_rng, counters)
            t["fluids"] = (time.perf_counter() - _s) * 1000.0
            if self.pde_validation_metrics:
                t["pde_fluids"] = float(self.pde_validation_metrics.get("pde_stage_ms", 0.0))
            if self.pressure_solver_stats:
                t["pressure_iters"] = float(self.pressure_solver_stats.get("iterations", 0))
                t["pressure_residual"] = float(self.pressure_solver_stats.get("residual", 0.0))
        if self.config.enable_thermal:
            _s = time.perf_counter()
            self._stage_thermal(grid, rows, cols, tick_index, thermal_rng, counters, events)
            t["thermal"] = (time.perf_counter() - _s) * 1000.0
        if self.config.enable_chemical:
            _s = time.perf_counter()
            self._stage_chemical(grid, rows, cols, counters, events)
            t["chemical"] = (time.perf_counter() - _s) * 1000.0
        if self.config.enable_phase_change:
            _s = time.perf_counter()
            self._stage_phase_change(grid, rows, cols, tick_index, counters, events)
            t["phase_change"] = (time.perf_counter() - _s) * 1000.0
        if self.structural_config.enabled:
            _s = time.perf_counter()
            self._stage_structural(grid, rows, cols, counters, events)
            t["structural"] = (time.perf_counter() - _s) * 1000.0
        if self.config.enable_cleanup:
            self._stage_cleanup()

        # CFL check (Bridson §3.2.1): CFL = max(|u|)·dt/dx
        # With semi-Lagrangian advection (Phase 2) this is purely informational;
        # values > 1 increase numerical diffusion but do not cause blow-up.
        if self.vel_x.shape == (rows, cols):
            max_vel = float(np.sqrt(np.max(self.vel_x**2 + self.vel_y**2)))
            self.last_cfl = max_vel * PHYSICS.dt * PHYSICS.dx_inv
        else:
            self.last_cfl = 0.0

        self.substep_timings = t
        return SimulationStepResult(
            changed_cells_count=counters["changes"],
            events=events,
            timings=t,
        )


