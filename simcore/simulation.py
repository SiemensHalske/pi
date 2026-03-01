from .world import *
from .world import _safe_load_json
from .physics import PowderPhysicsEngine

class Simulation:
    """SRP: Handles only the grid data and physics logic."""
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.physics_config = PhysicsConfig(seed=1337, substeps=1, enable_thermal=True, enable_chemical=True, enable_phase_change=True)
        self.physics = PowderPhysicsEngine(MATERIALS, MATERIAL_IDS, self.physics_config)
        self.tick_index = 0
        self.config_profiles = self._load_config_profiles()
        self.profile_order = list(self.config_profiles.keys()) if self.config_profiles else ["balanced"]
        self.profile_name = "balanced" if "balanced" in self.config_profiles else self.profile_order[0]
        self.apply_profile(self.profile_name)
        self.history_limit = 25
        self.undo_stack = []
        self.redo_stack = []
        self.replay_events = []
        self.replay_enabled = True
        self.replay_playing = False
        self.replay_play_index = 0

    def _copy_grid(self, source_grid):
        return [row[:] for row in source_grid]

    def _copy_field(self, field):
        """Deep-copy a physics field; handles both numpy arrays and legacy lists."""
        if isinstance(field, np.ndarray):
            return field.copy()
        return [row[:] for row in field] if field else []

    def _capture_state(self):
        return {
            "grid": self._copy_grid(self.grid),
            "tick_index": self.tick_index,
            "profile_name": self.profile_name,
            "materials": {str(mat_id): dict(mat_data) for mat_id, mat_data in MATERIALS.items()},
            "physics": {
                "lateral_bias": self._copy_field(self.physics.lateral_bias),
                "jammed_until": self._copy_field(self.physics.jammed_until),
                "fluid_pressure": self._copy_field(self.physics.fluid_pressure),
                "mix_ratio": self._copy_field(self.physics.mix_ratio),
                "temperature": self._copy_field(self.physics.temperature),
                "oxygen_level": self._copy_field(self.physics.oxygen_level),
                "burn_stage": self._copy_field(self.physics.burn_stage),
                "burn_progress": self._copy_field(self.physics.burn_progress),
                "smoke_density": self._copy_field(self.physics.smoke_density),
                "steam_density": self._copy_field(self.physics.steam_density),
                "moisture": self._copy_field(self.physics.moisture),
                "enthalpy_field": self._copy_field(self.physics.enthalpy_field),
                "liquid_fraction": self._copy_field(self.physics.liquid_fraction),
                "ignition_cooldown_until": self._copy_field(self.physics.ignition_cooldown_until),
                "integrity": self._copy_field(self.physics.integrity),
                "saturation_level": self._copy_field(self.physics.saturation_level),
                "phase_state": self._copy_field(self.physics.phase_state),
                "reaction_progress": self._copy_field(self.physics.reaction_progress),
                "phase_cooldown_until": self._copy_field(self.physics.phase_cooldown_until),
                "phase_transition_progress": self._copy_field(self.physics.phase_transition_progress),
                "phase_transition_target": self._copy_field(self.physics.phase_transition_target),
                # Phase 6 chemistry fields
                "fuel_vapor": self._copy_field(self.physics.fuel_vapor),
                "co2_density": self._copy_field(self.physics.co2_density),
                "co_density": self._copy_field(self.physics.co_density),
                "h2o_vapor": self._copy_field(self.physics.h2o_vapor),
                "mixture_fraction": self._copy_field(self.physics.mixture_fraction),
                "turb_k": self._copy_field(self.physics.turb_k),
                "turb_eps": self._copy_field(self.physics.turb_eps),
                "pyrolysis_progress": self._copy_field(self.physics.pyrolysis_progress),
                "soot_mass_fraction": self._copy_field(self.physics.soot_mass_fraction),
                "soot_number_density": self._copy_field(self.physics.soot_number_density),
                "h_plus": self._copy_field(self.physics.h_plus),
                "oh_minus": self._copy_field(self.physics.oh_minus),
                "ph_field": self._copy_field(self.physics.ph_field),
                "catalyst_theta_fuel": self._copy_field(self.physics.catalyst_theta_fuel),
                "catalyst_theta_o2": self._copy_field(self.physics.catalyst_theta_o2),
                # Phase 1 PDE fields
                "vel_x":         self._copy_field(self.physics.vel_x),
                "vel_y":         self._copy_field(self.physics.vel_y),
                "pressure_pde":  self._copy_field(self.physics.pressure_pde),
                "divergence_pde": self._copy_field(self.physics.divergence_pde),
                "density_field": self._copy_field(self.physics.density_field),
                "mac_u":         self._copy_field(self.physics.mac_u),
                "mac_v":         self._copy_field(self.physics.mac_v),
                "acoustic_pressure": self._copy_field(self.physics.acoustic_pressure),
                "acoustic_u": self._copy_field(self.physics.acoustic_u),
                "acoustic_v": self._copy_field(self.physics.acoustic_v),
                "porous_resistance": self._copy_field(self.physics.porous_resistance),
                "em_ex": self._copy_field(self.physics.em_ex),
                "em_ey": self._copy_field(self.physics.em_ey),
                "em_bz": self._copy_field(self.physics.em_bz),
                "current_x": self._copy_field(self.physics.current_x),
                "current_y": self._copy_field(self.physics.current_y),
                "joule_heating_source": self._copy_field(self.physics.joule_heating_source),
                "pressure_solver_stats": dict(self.physics.pressure_solver_stats),
                "em_solver_stats": dict(self.physics.em_solver_stats),
                "pde_validation_metrics": dict(self.physics.pde_validation_metrics),
                # Phase 5 structural fields
                "disp_x": self._copy_field(self.physics.disp_x),
                "disp_y": self._copy_field(self.physics.disp_y),
                "struct_vel_x": self._copy_field(self.physics.struct_vel_x),
                "struct_vel_y": self._copy_field(self.physics.struct_vel_y),
                "sigma_xx": self._copy_field(self.physics.sigma_xx),
                "sigma_yy": self._copy_field(self.physics.sigma_yy),
                "tau_xy": self._copy_field(self.physics.tau_xy),
                "plastic_strain": self._copy_field(self.physics.plastic_strain),
                "damage_field": self._copy_field(self.physics.damage_field),
                "pore_pressure": self._copy_field(self.physics.pore_pressure),
                "debris_particles": [dict(p) for p in self.physics.debris_particles],
            },
        }

    def _restore_state(self, state):
        self.grid = self._copy_grid(state["grid"])
        self.tick_index = state["tick_index"]

        profile_name = state.get("profile_name", self.profile_name)
        if profile_name in self.config_profiles:
            self.apply_profile(profile_name)

        if "materials" in state:
            for mat_id_str, mat_data in state["materials"].items():
                mat_id = int(mat_id_str)
                if mat_id in MATERIALS:
                    MATERIALS[mat_id].update(mat_data)

        physics_state = state["physics"]
        self.physics.lateral_bias = self._copy_field(physics_state.get("lateral_bias", []))
        self.physics.jammed_until = self._copy_field(physics_state.get("jammed_until", []))
        self.physics.fluid_pressure = self._copy_field(physics_state.get("fluid_pressure", []))
        self.physics.mix_ratio = self._copy_field(physics_state.get("mix_ratio", []))
        self.physics.temperature = self._copy_field(physics_state.get("temperature", []))
        self.physics.oxygen_level = self._copy_field(physics_state.get("oxygen_level", []))
        self.physics.burn_stage = self._copy_field(physics_state.get("burn_stage", []))
        self.physics.burn_progress = self._copy_field(physics_state.get("burn_progress", []))
        self.physics.smoke_density = self._copy_field(physics_state.get("smoke_density", []))
        self.physics.steam_density = self._copy_field(physics_state.get("steam_density", []))
        self.physics.moisture = self._copy_field(physics_state.get("moisture", []))
        self.physics.enthalpy_field = self._copy_field(physics_state.get("enthalpy_field", []))
        self.physics.liquid_fraction = self._copy_field(physics_state.get("liquid_fraction", []))
        self.physics.ignition_cooldown_until = self._copy_field(physics_state.get("ignition_cooldown_until", []))
        self.physics.integrity = self._copy_field(physics_state.get("integrity", []))
        self.physics.saturation_level = self._copy_field(physics_state.get("saturation_level", []))
        self.physics.phase_state = self._copy_field(physics_state.get("phase_state", []))
        self.physics.reaction_progress = self._copy_field(physics_state.get("reaction_progress", []))
        self.physics.phase_cooldown_until = self._copy_field(physics_state.get("phase_cooldown_until", []))
        self.physics.phase_transition_progress = self._copy_field(physics_state.get("phase_transition_progress", []))
        self.physics.phase_transition_target = self._copy_field(physics_state.get("phase_transition_target", []))
        self.physics.fuel_vapor = self._copy_field(physics_state.get("fuel_vapor", []))
        self.physics.co2_density = self._copy_field(physics_state.get("co2_density", []))
        self.physics.co_density = self._copy_field(physics_state.get("co_density", []))
        self.physics.h2o_vapor = self._copy_field(physics_state.get("h2o_vapor", []))
        self.physics.mixture_fraction = self._copy_field(physics_state.get("mixture_fraction", []))
        self.physics.turb_k = self._copy_field(physics_state.get("turb_k", []))
        self.physics.turb_eps = self._copy_field(physics_state.get("turb_eps", []))
        self.physics.pyrolysis_progress = self._copy_field(physics_state.get("pyrolysis_progress", []))
        self.physics.soot_mass_fraction = self._copy_field(physics_state.get("soot_mass_fraction", []))
        self.physics.soot_number_density = self._copy_field(physics_state.get("soot_number_density", []))
        self.physics.h_plus = self._copy_field(physics_state.get("h_plus", []))
        self.physics.oh_minus = self._copy_field(physics_state.get("oh_minus", []))
        self.physics.ph_field = self._copy_field(physics_state.get("ph_field", []))
        self.physics.catalyst_theta_fuel = self._copy_field(physics_state.get("catalyst_theta_fuel", []))
        self.physics.catalyst_theta_o2 = self._copy_field(physics_state.get("catalyst_theta_o2", []))
        # Phase 1 PDE fields
        if "vel_x" in physics_state:
            self.physics.vel_x         = self._copy_field(physics_state["vel_x"])
            self.physics.vel_y         = self._copy_field(physics_state["vel_y"])
            self.physics.pressure_pde  = self._copy_field(physics_state["pressure_pde"])
            self.physics.divergence_pde = self._copy_field(physics_state.get("divergence_pde", []))
            self.physics.density_field = self._copy_field(physics_state["density_field"])
            self.physics.mac_u         = self._copy_field(physics_state.get("mac_u", []))
            self.physics.mac_v         = self._copy_field(physics_state.get("mac_v", []))
            self.physics.acoustic_pressure = self._copy_field(physics_state.get("acoustic_pressure", []))
            self.physics.acoustic_u = self._copy_field(physics_state.get("acoustic_u", []))
            self.physics.acoustic_v = self._copy_field(physics_state.get("acoustic_v", []))
            self.physics.porous_resistance = self._copy_field(physics_state.get("porous_resistance", []))
            self.physics.em_ex = self._copy_field(physics_state.get("em_ex", []))
            self.physics.em_ey = self._copy_field(physics_state.get("em_ey", []))
            self.physics.em_bz = self._copy_field(physics_state.get("em_bz", []))
            self.physics.current_x = self._copy_field(physics_state.get("current_x", []))
            self.physics.current_y = self._copy_field(physics_state.get("current_y", []))
            self.physics.joule_heating_source = self._copy_field(physics_state.get("joule_heating_source", []))
            self.physics.pressure_solver_stats = dict(physics_state.get("pressure_solver_stats", {}))
            self.physics.em_solver_stats = dict(physics_state.get("em_solver_stats", {}))
            self.physics.pde_validation_metrics = dict(physics_state.get("pde_validation_metrics", {}))

        # Phase 5 structural fields
        self.physics.disp_x = self._copy_field(physics_state.get("disp_x", []))
        self.physics.disp_y = self._copy_field(physics_state.get("disp_y", []))
        self.physics.struct_vel_x = self._copy_field(physics_state.get("struct_vel_x", []))
        self.physics.struct_vel_y = self._copy_field(physics_state.get("struct_vel_y", []))
        self.physics.sigma_xx = self._copy_field(physics_state.get("sigma_xx", []))
        self.physics.sigma_yy = self._copy_field(physics_state.get("sigma_yy", []))
        self.physics.tau_xy = self._copy_field(physics_state.get("tau_xy", []))
        self.physics.plastic_strain = self._copy_field(physics_state.get("plastic_strain", []))
        self.physics.damage_field = self._copy_field(physics_state.get("damage_field", []))
        self.physics.pore_pressure = self._copy_field(physics_state.get("pore_pressure", []))
        self.physics.debris_particles = [dict(p) for p in physics_state.get("debris_particles", [])]

    def _load_config_profiles(self):
        payload = _safe_load_json(CONFIG_PROFILES_FILE)
        if not payload or not isinstance(payload, dict):
            return dict(DEFAULT_CONFIG_PROFILES)

        profiles = {}
        for name, profile in payload.items():
            if not isinstance(profile, dict):
                continue
            profiles[name] = profile
        return profiles if profiles else dict(DEFAULT_CONFIG_PROFILES)

    def apply_profile(self, profile_name):
        if profile_name not in self.config_profiles:
            return False

        profile = self.config_profiles[profile_name]
        for key, value in profile.get("physics", {}).items():
            if hasattr(self.physics_config, key):
                setattr(self.physics_config, key, value)
                setattr(self.physics.config, key, value)

        for key, value in profile.get("fluid", {}).items():
            if hasattr(self.physics.fluid_config, key):
                setattr(self.physics.fluid_config, key, value)

        for key, value in profile.get("thermal", {}).items():
            if hasattr(self.physics.thermal_config, key):
                setattr(self.physics.thermal_config, key, value)

        for key, value in profile.get("chemistry", {}).items():
            if hasattr(self.physics.chemistry_config, key):
                setattr(self.physics.chemistry_config, key, value)

        for key, value in profile.get("phase_change", {}).items():
            if hasattr(self.physics.phase_change_config, key):
                setattr(self.physics.phase_change_config, key, value)

        for key, value in profile.get("structural", {}).items():
            if hasattr(self.physics.structural_config, key):
                setattr(self.physics.structural_config, key, value)

        for key, value in profile.get("electromagnetics", {}).items():
            if hasattr(self.physics.fluid_config, key):
                setattr(self.physics.fluid_config, key, value)

        self.profile_name = profile_name
        return True

    def cycle_profile(self):
        if not self.profile_order:
            return self.profile_name
        current_index = self.profile_order.index(self.profile_name) if self.profile_name in self.profile_order else 0
        next_index = (current_index + 1) % len(self.profile_order)
        self.apply_profile(self.profile_order[next_index])
        return self.profile_name

    def run_physics_validation_suite(self, sample_steps=64):
        sample_steps = int(max(1, sample_steps))
        snapshot = self._capture_state()
        metrics = {
            "divergence_rms": [],
            "divergence_max": [],
            "pressure_residual": [],
            "pressure_iterations": [],
            "pressure_active_fraction": [],
            "current_rms": [],
            "magnetic_energy": [],
            "joule_mean": [],
            "pde_stage_ms": [],
            "fluid_stage_ms": [],
            "cfl": [],
        }

        try:
            for _ in range(sample_steps):
                result = self.physics.step(self.grid, self.rows, self.cols, self.tick_index)
                self.tick_index += 1
                pde_metrics = dict(self.physics.pde_validation_metrics)
                pressure_stats = dict(self.physics.pressure_solver_stats)

                metrics["divergence_rms"].append(float(pde_metrics.get("divergence_rms", 0.0)))
                metrics["divergence_max"].append(float(pde_metrics.get("divergence_max", 0.0)))
                metrics["pressure_residual"].append(float(pressure_stats.get("residual", 0.0)))
                metrics["pressure_iterations"].append(float(pressure_stats.get("iterations", 0.0)))
                metrics["pressure_active_fraction"].append(float(pressure_stats.get("active_fraction", 0.0)))
                metrics["current_rms"].append(float(pde_metrics.get("current_rms", 0.0)))
                metrics["magnetic_energy"].append(float(pde_metrics.get("magnetic_energy", 0.0)))
                metrics["joule_mean"].append(float(pde_metrics.get("joule_mean", 0.0)))
                metrics["pde_stage_ms"].append(float(pde_metrics.get("pde_stage_ms", 0.0)))
                metrics["fluid_stage_ms"].append(float(result.timings.get("fluids", 0.0)))
                metrics["cfl"].append(float(self.physics.last_cfl))
        finally:
            self._restore_state(snapshot)

        def _agg(values):
            arr = np.asarray(values, dtype=np.float32)
            return {
                "mean": float(np.mean(arr)) if arr.size else 0.0,
                "max": float(np.max(arr)) if arr.size else 0.0,
                "min": float(np.min(arr)) if arr.size else 0.0,
            }

        report = {
            "version": 1,
            "profile": self.profile_name,
            "grid": {"rows": int(self.rows), "cols": int(self.cols)},
            "samples": sample_steps,
            "metrics": {name: _agg(values) for name, values in metrics.items()},
            "solver": {
                "numba_available": bool(getattr(self.physics, "_numba_available", False)),
                "sparse_pressure_enabled": bool(self.physics.fluid_config.pde_sparse_pressure_enabled),
                "multigrid_enabled": bool(self.physics.fluid_config.pde_multigrid_enabled),
            },
            "timestamp": time.time(),
        }

        if bool(self.physics.fluid_config.validation_export_enabled):
            export_path = pathlib.Path(self.physics.fluid_config.validation_export_path)
            if not export_path.is_absolute():
                export_path = pathlib.Path.cwd() / export_path
            try:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                with export_path.open("w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2)
            except OSError:
                pass

        return report

    def _record_undo_snapshot(self):
        self.undo_stack.append(self._capture_state())
        if len(self.undo_stack) > self.history_limit:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._capture_state())
        snapshot = self.undo_stack.pop()
        self._restore_state(snapshot)
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._capture_state())
        snapshot = self.redo_stack.pop()
        self._restore_state(snapshot)
        return True

    def _record_replay_event(self, event):
        if not self.replay_enabled or self.replay_playing:
            return
        event_payload = dict(event)
        event_payload["tick"] = self.tick_index
        self.replay_events.append(event_payload)

    def clear(self, record_action=True):
        if record_action:
            self._record_undo_snapshot()
            self._record_replay_event({"type": "clear"})
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.physics = PowderPhysicsEngine(MATERIALS, MATERIAL_IDS, self.physics_config)

    def _apply_paint(self, col, row, radius, material_id, brush_shape="circle", record_action=True):
        changed = False
        if record_action:
            self._record_undo_snapshot()

        for y in range(-radius, radius):
            for x in range(-radius, radius):
                inside_brush = False
                if brush_shape == "square":
                    inside_brush = True
                else:
                    inside_brush = (x*x + y*y <= radius*radius)

                if inside_brush:
                    r, c = row + y, col + x
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        if material_id == 0:
                            if self.grid[r][c] != 0:
                                changed = True
                            self.grid[r][c] = 0
                            self.physics.apply_spawn_state(self.grid, r, c, 0, self.rows, self.cols)
                            continue

                        if self.grid[r][c] != 0:
                            continue

                        # Deterministic placement: always fill the brush footprint for empty cells.
                        self.grid[r][c] = material_id
                        self.physics.apply_spawn_state(self.grid, r, c, material_id, self.rows, self.cols)
                        changed = True

        if changed and record_action:
            self._record_replay_event({
                "type": "paint",
                "col": col,
                "row": row,
                "radius": radius,
                "material_id": material_id,
                "brush_shape": brush_shape,
            })
        elif not changed and record_action and self.undo_stack:
            self.undo_stack.pop()

        return changed

    def paint(self, col, row, radius, material_id, brush_shape="circle"):
        """Applies a brush stroke to the grid with history + replay support."""
        self._apply_paint(col, row, radius, material_id, brush_shape, record_action=True)

    def save_to_file(self, file_path="savegame.json"):
        payload = self._capture_state()
        payload["replay_events"] = list(self.replay_events)
        with open(file_path, "w", encoding="utf-8") as save_file:
            json.dump(payload, save_file)
        return file_path

    def load_from_file(self, file_path="savegame.json"):
        with open(file_path, "r", encoding="utf-8") as save_file:
            payload = json.load(save_file)
        self._restore_state(payload)
        self.replay_events = list(payload.get("replay_events", []))
        self.undo_stack.clear()
        self.redo_stack.clear()
        return file_path

    def save_replay(self, file_path="replay.json"):
        payload = {
            "seed": self.physics_config.seed,
            "events": list(self.replay_events),
        }
        with open(file_path, "w", encoding="utf-8") as replay_file:
            json.dump(payload, replay_file)
        return file_path

    def load_replay(self, file_path="replay.json"):
        with open(file_path, "r", encoding="utf-8") as replay_file:
            payload = json.load(replay_file)
        self.replay_events = list(payload.get("events", []))
        return len(self.replay_events)

    def start_replay(self):
        self.clear(record_action=False)
        self.tick_index = 0
        self.replay_play_index = 0
        self.replay_playing = True

    def stop_replay(self):
        self.replay_playing = False

    def _apply_replay_events_for_tick(self):
        while self.replay_play_index < len(self.replay_events):
            event = self.replay_events[self.replay_play_index]
            if event.get("tick", -1) != self.tick_index:
                break

            event_type = event.get("type")
            if event_type == "clear":
                self.clear(record_action=False)
            elif event_type == "paint":
                self._apply_paint(
                    event.get("col", 0),
                    event.get("row", 0),
                    event.get("radius", 1),
                    event.get("material_id", 0),
                    event.get("brush_shape", "circle"),
                    record_action=False,
                )

            self.replay_play_index += 1

        if self.replay_play_index >= len(self.replay_events):
            self.replay_playing = False

    def load_scenario(self, scenario_name):
        self.clear(record_action=False)
        if scenario_name == "basin":
            mid = self.cols // 2
            for col in range(self.cols):
                self.grid[self.rows - 3][col] = MATERIAL_IDS["wall"]
                self.physics.apply_spawn_state(self.grid, self.rows - 3, col, MATERIAL_IDS["wall"], self.rows, self.cols)
            for row in range(self.rows - 12, self.rows - 3):
                for col in range(max(0, mid - 18), min(self.cols, mid + 18)):
                    self.grid[row][col] = MATERIAL_IDS["water"]
                    self.physics.apply_spawn_state(self.grid, row, col, MATERIAL_IDS["water"], self.rows, self.cols)
        elif scenario_name == "volcano":
            mid = self.cols // 2
            for row in range(self.rows - 16, self.rows - 2):
                self.grid[row][mid - 2] = MATERIAL_IDS["wall"]
                self.grid[row][mid + 2] = MATERIAL_IDS["wall"]
                self.physics.apply_spawn_state(self.grid, row, mid - 2, MATERIAL_IDS["wall"], self.rows, self.cols)
                self.physics.apply_spawn_state(self.grid, row, mid + 2, MATERIAL_IDS["wall"], self.rows, self.cols)
            for row in range(self.rows - 12, self.rows - 3):
                self.grid[row][mid] = MATERIAL_IDS["lava"]
                self.physics.apply_spawn_state(self.grid, row, mid, MATERIAL_IDS["lava"], self.rows, self.cols)
        elif scenario_name == "steam_chamber":
            for row in range(8, min(self.rows - 8, 24)):
                for col in range(8, min(self.cols - 8, 34)):
                    if row in (8, min(self.rows - 8, 24) - 1) or col in (8, min(self.cols - 8, 34) - 1):
                        self.grid[row][col] = MATERIAL_IDS["wall"]
                        self.physics.apply_spawn_state(self.grid, row, col, MATERIAL_IDS["wall"], self.rows, self.cols)
                    else:
                        self.grid[row][col] = MATERIAL_IDS["steam"]
                        self.physics.apply_spawn_state(self.grid, row, col, MATERIAL_IDS["steam"], self.rows, self.cols)

    def run_benchmark(self, ticks=300):
        snapshot = self._capture_state()
        durations = []
        total_changes = 0
        for _ in range(max(1, ticks)):
            start = time.perf_counter()
            result = self.update_physics()
            durations.append((time.perf_counter() - start) * 1000.0)
            total_changes += result.changed_cells_count

        sorted_durations = sorted(durations)
        median = sorted_durations[len(sorted_durations) // 2]
        p95_index = min(len(sorted_durations) - 1, int(len(sorted_durations) * 0.95))
        p95 = sorted_durations[p95_index]
        avg = sum(sorted_durations) / len(sorted_durations)

        self._restore_state(snapshot)
        return {
            "ticks": len(durations),
            "median_ms": median,
            "p95_ms": p95,
            "avg_ms": avg,
            "changes_per_tick": total_changes / max(1, len(durations)),
        }

    def _collect_snapshot_metrics(self):
        material_counts = {}
        active_cells = 0
        temp_values = []
        oxygen_values = []

        for row in range(self.rows):
            for col in range(self.cols):
                mat = self.grid[row][col]
                if mat != 0:
                    active_cells += 1
                material_counts[str(mat)] = material_counts.get(str(mat), 0) + 1

                if self.physics.temperature and row < len(self.physics.temperature) and col < len(self.physics.temperature[row]):
                    temp_values.append(self.physics.temperature[row][col])
                if self.physics.oxygen_level and row < len(self.physics.oxygen_level) and col < len(self.physics.oxygen_level[row]):
                    oxygen_values.append(self.physics.oxygen_level[row][col])

        avg_temp = sum(temp_values) / max(1, len(temp_values))
        avg_oxygen = sum(oxygen_values) / max(1, len(oxygen_values))

        return {
            "active_cells": active_cells,
            "material_counts": material_counts,
            "avg_temp": round(avg_temp, 3),
            "avg_oxygen": round(avg_oxygen, 3),
        }

    def run_snapshot_regressions(self, baseline_path=SNAPSHOT_BASELINE_FILE, ticks=180):
        snapshot = self._capture_state()
        scenarios = ["basin", "volcano", "steam_chamber"]
        current_results = {}

        for scenario in scenarios:
            self.load_scenario(scenario)
            for _ in range(max(1, ticks)):
                self.update_physics()
            current_results[scenario] = self._collect_snapshot_metrics()

        baseline_payload = _safe_load_json(baseline_path)
        if not baseline_payload or "scenarios" not in baseline_payload:
            with open(baseline_path, "w", encoding="utf-8") as baseline_file:
                json.dump({"scenarios": current_results}, baseline_file, indent=2)
            self._restore_state(snapshot)
            return {"created": True, "passed": True, "failures": [], "path": baseline_path}

        failures = []
        baseline_scenarios = baseline_payload.get("scenarios", {})
        for scenario, metrics in current_results.items():
            baseline_metrics = baseline_scenarios.get(scenario)
            if not baseline_metrics:
                failures.append(f"missing baseline scenario {scenario}")
                continue

            active_delta = abs(metrics["active_cells"] - baseline_metrics.get("active_cells", metrics["active_cells"]))
            if active_delta > 450:
                failures.append(f"{scenario}: active_cells delta {active_delta}")

            temp_delta = abs(metrics["avg_temp"] - baseline_metrics.get("avg_temp", metrics["avg_temp"]))
            if temp_delta > 12.0:
                failures.append(f"{scenario}: avg_temp delta {temp_delta:.2f}")

            oxygen_delta = abs(metrics["avg_oxygen"] - baseline_metrics.get("avg_oxygen", metrics["avg_oxygen"]))
            if oxygen_delta > 0.18:
                failures.append(f"{scenario}: avg_oxygen delta {oxygen_delta:.3f}")

        self._restore_state(snapshot)
        return {
            "created": False,
            "passed": len(failures) == 0,
            "failures": failures,
            "path": baseline_path,
        }

    def update_physics(self):
        """Delegates one simulation step to the dedicated physics engine."""
        if self.replay_playing:
            self._apply_replay_events_for_tick()
        result = self.physics.step(self.grid, self.rows, self.cols, self.tick_index)
        self.tick_index += 1
        return result


