import pygame
import sys
import random
import os
import math
import time
import json
import struct
from dataclasses import dataclass

# --- Configuration ---
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
MENU_WIDTH = 200
SIM_WIDTH = WINDOW_WIDTH - MENU_WIDTH

CELL_SIZE = 8
COLS = SIM_WIDTH // CELL_SIZE
ROWS = WINDOW_HEIGHT // CELL_SIZE
FPS = 60
MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 16
TOP_BAR_HEIGHT = 28
INTERACTION_MATRIX_FILE = "interaction_matrix.json"
CONFIG_PROFILES_FILE = "config_profiles.json"
SNAPSHOT_BASELINE_FILE = "snapshot_baselines.json"


def _safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None


DEFAULT_CONFIG_PROFILES = {
    "arcade": {
        "physics": {"substeps": 1},
        "fluid": {"flow_span_base": 6, "density_swap_rate": 0.25, "mix_decay": 0.02, "boundary_damping": 0.25, "pressure_smoothing": 0.15, "gas_turbulence": 0.22},
        "thermal": {"heat_diffusion_rate": 0.12, "convection_bias": 0.08, "oxygen_diffusion": 0.2},
        "chemistry": {"corrosion_base_rate": 0.03, "dissolution_base_rate": 0.04},
        "phase_change": {"hysteresis": 1.5, "transition_progress_decay": 0.03},
    },
    "balanced": {
        "physics": {"substeps": 1},
        "fluid": {"flow_span_base": 5, "density_swap_rate": 0.3, "mix_decay": 0.015, "boundary_damping": 0.3, "pressure_smoothing": 0.2, "gas_turbulence": 0.16},
        "thermal": {"heat_diffusion_rate": 0.14, "convection_bias": 0.06, "oxygen_diffusion": 0.18},
        "chemistry": {"corrosion_base_rate": 0.025, "dissolution_base_rate": 0.03},
        "phase_change": {"hysteresis": 2.0, "transition_progress_decay": 0.04},
    },
    "realistic": {
        "physics": {"substeps": 2},
        "fluid": {"flow_span_base": 4, "density_swap_rate": 0.35, "mix_decay": 0.01, "boundary_damping": 0.4, "pressure_smoothing": 0.28, "gas_turbulence": 0.1},
        "thermal": {"heat_diffusion_rate": 0.18, "convection_bias": 0.05, "oxygen_diffusion": 0.15},
        "chemistry": {"corrosion_base_rate": 0.02, "dissolution_base_rate": 0.024},
        "phase_change": {"hysteresis": 2.4, "transition_progress_decay": 0.05},
    },
}

# --- Material Definitions (OCP & DRY) ---
# Central definition of all materials so the UI and Renderer don't need hardcoded values.
MATERIALS = {
    0: {
        "name": "Eraser",
        "color": (20, 20, 20),
        "type": "air",
        "density": 1,
        "viscosity": 0.0,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 1.0,
        "solubility_limit": 0.0,
        "latent_heat": 0.0,
        "inertia": 0.0,
        "repose_angle": 90,
        "dispersion": 0,
        "drag": 0.0,
    },
    1: {
        "name": "Sand",
        "color": (235, 200, 100),
        "type": "powder",
        "density": 1600,
        "viscosity": 0.0,
        "thermal_capacity": 0.9,
        "thermal_conductivity": 0.35,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 0.92,
        "dissolution_rate": 0.025,
        "passivation_factor": 0.12,
        "solubility_limit": 0.45,
        "dissolution_product": 3,
        "corrosion_product": 1,
        "latent_heat": 7.5,
        "inertia": 0.35,
        "repose_angle": 34,
        "dispersion": 1,
        "drag": 0.08,
    },
    2: {
        "name": "Wall",
        "color": (120, 120, 130),
        "type": "solid",
        "density": 99999,
        "viscosity": 0.0,
        "thermal_capacity": 1.2,
        "thermal_conductivity": 0.65,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 0.96,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.55,
        "solubility_limit": 0.0,
        "corrosion_product": 1,
        "latent_heat": 2.0,
        "inertia": 0.0,
        "repose_angle": 90,
        "dispersion": 0,
        "drag": 0.0,
    },
    3: {
        "name": "Water",
        "color": (50, 100, 200),
        "type": "liquid",
        "density": 1000,
        "viscosity": 0.25,
        "thermal_capacity": 4.2,
        "thermal_conductivity": 0.55,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "viscosity_temp_factor": 0.0,
        "shear_index": 1.0,
        "mix_compatibility": 0.85,
        "buoyancy_bias": 0.0,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 1.0,
        "freeze_temp": -1.0,
        "evaporate_temp": 105.0,
        "freeze_target": 5,
        "evaporate_target": 6,
        "latent_heat": 18.0,
        "inertia": 0.1,
        "repose_angle": 0,
        "dispersion": 4,
        "drag": 0.2,
    },
    4: {
        "name": "Wood",
        "color": (139, 69, 19),
        "type": "solid",
        "density": 700,
        "viscosity": 0.0,
        "thermal_capacity": 1.8,
        "thermal_conductivity": 0.18,
        "ignition_temp": 275.0,
        "auto_ignite_temp": 360.0,
        "burn_rate": 0.035,
        "smoke_factor": 0.45,
        "min_oxygen_for_ignition": 0.2,
        "spark_sensitivity": 0.6,
        "ash_yield": 0.1,
        "char_yield": 0.15,
        "heat_of_combustion": 280.0,
        "corrosion_resistance": 0.5,
        "dissolution_rate": 0.01,
        "passivation_factor": 0.08,
        "solubility_limit": 0.3,
        "corrosion_product": 7,
        "dissolution_product": 0,
        "latent_heat": 5.0,
        "inertia": 0.0,
        "repose_angle": 90,
        "dispersion": 0,
        "drag": 0.0,
    },
    5: {
        "name": "Ice",
        "color": (175, 220, 255),
        "type": "solid",
        "density": 917,
        "viscosity": 0.0,
        "thermal_capacity": 2.1,
        "thermal_conductivity": 0.35,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 0.0,
        "initial_temp": -8.0,
        "phase_change_rate": 0.012,
        "melt_temp": 2.0,
        "melt_target": 3,
        "latent_heat": 16.0,
        "inertia": 0.0,
        "repose_angle": 90,
        "dispersion": 0,
        "drag": 0.0,
    },
    6: {
        "name": "Steam",
        "color": (205, 205, 220),
        "type": "gas",
        "density": 2,
        "viscosity": 0.05,
        "thermal_capacity": 2.0,
        "thermal_conductivity": 0.08,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 0.0,
        "initial_temp": 120.0,
        "phase_change_rate": 0.01,
        "condense_temp": 92.0,
        "condense_target": 3,
        "latent_heat": 20.0,
        "inertia": 0.05,
        "repose_angle": 0,
        "dispersion": 5,
        "drag": 0.02,
    },
    7: {
        "name": "Ash",
        "color": (140, 130, 120),
        "type": "powder",
        "density": 620,
        "viscosity": 0.0,
        "thermal_capacity": 0.75,
        "thermal_conductivity": 0.12,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.02,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 0.8,
        "dissolution_rate": 0.04,
        "passivation_factor": 0.15,
        "solubility_limit": 0.25,
        "dissolution_product": 3,
        "latent_heat": 3.0,
        "inertia": 0.18,
        "repose_angle": 38,
        "dispersion": 2,
        "drag": 0.1,
    },
    8: {
        "name": "Acid",
        "color": (95, 220, 120),
        "type": "liquid",
        "density": 1120,
        "viscosity": 0.28,
        "thermal_capacity": 3.6,
        "thermal_conductivity": 0.42,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "viscosity_temp_factor": 0.0,
        "shear_index": 1.0,
        "mix_compatibility": 0.55,
        "buoyancy_bias": 0.0,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 1.0,
        "corrosion_power": 1.0,
        "latent_heat": 4.0,
        "inertia": 0.1,
        "repose_angle": 0,
        "dispersion": 4,
        "drag": 0.22,
    },
    9: {
        "name": "Lava",
        "color": (255, 95, 35),
        "type": "liquid",
        "density": 2800,
        "viscosity": 1.4,
        "thermal_capacity": 1.5,
        "thermal_conductivity": 0.7,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "viscosity_temp_factor": 0.0,
        "shear_index": 1.15,
        "mix_compatibility": 0.1,
        "buoyancy_bias": -0.1,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 1.0,
        "initial_temp": 980.0,
        "phase_change_rate": 0.008,
        "solidify_temp": 640.0,
        "solidify_target": 2,
        "latent_heat": 28.0,
        "inertia": 0.18,
        "repose_angle": 0,
        "dispersion": 2,
        "drag": 0.36,
    },
    10: {
        "name": "Fire",
        "internal": True,   # not shown in the material picker
        "color": (255, 160, 30),
        "type": "gas",
        "density": 1,
        "viscosity": 0.04,
        "thermal_capacity": 0.4,
        "thermal_conductivity": 0.2,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.06,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 0.0,
        "initial_temp": 520.0,
        "latent_heat": 0.0,
        "inertia": 0.02,
        "repose_angle": 0,
        "dispersion": 5,
        "drag": 0.01,
    }
}


MATERIAL_DEFAULTS = {
    "density": 1000,
    "viscosity": 0.0,
    "thermal_capacity": 1.0,
    "thermal_conductivity": 0.2,
    "ignition_temp": 9999,
    "auto_ignite_temp": 9999,
    "burn_rate": 0.0,
    "smoke_factor": 0.0,
    "min_oxygen_for_ignition": 1.0,
    "spark_sensitivity": 0.0,
    "ash_yield": 0.0,
    "char_yield": 0.0,
    "heat_of_combustion": 0.0,
    "initial_temp": None,
    "corrosion_resistance": 1.0,
    "dissolution_rate": 0.0,
    "passivation_factor": 0.0,
    "solubility_limit": 1.0,
    "corrosion_power": 0.0,
    "contact_heat_output": 0.0,
    "latent_heat": 0.0,
    "inertia": 0.0,
    "repose_angle": 90,
    "dispersion": 0,
    "drag": 0.0,
    "viscosity_temp_factor": 0.0,
    "shear_index": 1.0,
    "mix_compatibility": 0.0,
    "buoyancy_bias": 0.0,
    "phase_change_rate": 0.04,
    "internal": False,
}


class MaterialRegistry:
    ALLOWED_TYPES = {"air", "powder", "liquid", "solid", "gas"}
    REQUIRED_KEYS = {"name", "color", "type"}

    def __init__(self, raw_materials, defaults):
        self.raw_materials = raw_materials
        self.defaults = defaults
        self.materials = self._build_materials()
        self.ids_by_name = self._build_name_index()
        self._validate_parameter_ranges()
        self._validate_empty_material_contract()
        self._validate_targets()

    def _build_materials(self):
        normalized = {}
        for mat_id, mat_data in self.raw_materials.items():
            self._validate_base(mat_id, mat_data)
            merged = dict(self.defaults)
            merged.update(mat_data)
            normalized[mat_id] = merged
        return normalized

    def _validate_base(self, mat_id, mat_data):
        missing = self.REQUIRED_KEYS - set(mat_data.keys())
        if missing:
            raise ValueError(f"Material {mat_id} missing required keys: {sorted(missing)}")
        mat_type = mat_data["type"]
        if mat_type not in self.ALLOWED_TYPES:
            raise ValueError(f"Material {mat_id} has invalid type '{mat_type}'")

    def _build_name_index(self):
        ids = {}
        for mat_id, mat_data in self.materials.items():
            key = mat_data["name"].strip().lower()
            if key in ids:
                raise ValueError(f"Duplicate material name '{mat_data['name']}'")
            ids[key] = mat_id
        return ids

    def _validate_targets(self):
        for mat_id, mat_data in self.materials.items():
            for target_key in ("freeze_target", "evaporate_target", "condense_target", "solidify_target", "melt_target", "dissolution_product", "corrosion_product"):
                target = mat_data.get(target_key)
                if target is not None and target not in self.materials:
                    raise ValueError(f"Material {mat_id} references unknown target {target_key}={target}")

    def _validate_empty_material_contract(self):
        if 0 not in self.materials:
            raise ValueError("Material 0 must exist and represent the empty/air cell")
        if self.materials[0]["type"] != "air":
            raise ValueError("Material 0 must use type 'air'")

    def _validate_parameter_ranges(self):
        for mat_id, mat_data in self.materials.items():
            color = mat_data.get("color")
            if not isinstance(color, (list, tuple)) or len(color) != 3:
                raise ValueError(f"Material {mat_id} color must be an RGB tuple")
            if any((not isinstance(channel, int) or channel < 0 or channel > 255) for channel in color):
                raise ValueError(f"Material {mat_id} color channels must be ints in 0..255")

            if mat_data.get("density", 1) <= 0:
                raise ValueError(f"Material {mat_id} density must be > 0")
            if mat_data.get("viscosity", 0.0) < 0.0:
                raise ValueError(f"Material {mat_id} viscosity must be >= 0")
            if mat_data.get("drag", 0.0) < 0.0:
                raise ValueError(f"Material {mat_id} drag must be >= 0")
            if mat_data.get("phase_change_rate", 0.01) <= 0.0:
                raise ValueError(f"Material {mat_id} phase_change_rate must be > 0")
            if mat_data.get("dispersion", 0) < 0:
                raise ValueError(f"Material {mat_id} dispersion must be >= 0")

            freeze_temp = mat_data.get("freeze_temp")
            melt_temp = mat_data.get("melt_temp")
            if freeze_temp is not None and melt_temp is not None and freeze_temp >= melt_temp:
                raise ValueError(f"Material {mat_id} freeze_temp must be lower than melt_temp")


MATERIAL_REGISTRY = MaterialRegistry(MATERIALS, MATERIAL_DEFAULTS)
MATERIALS = MATERIAL_REGISTRY.materials
MATERIAL_IDS = MATERIAL_REGISTRY.ids_by_name


@dataclass
class PhysicsConfig:
    seed: int | None = None
    substeps: int = 1
    enable_mechanics: bool = True
    enable_fluids: bool = True
    enable_thermal: bool = False
    enable_chemical: bool = False
    enable_phase_change: bool = False
    enable_cleanup: bool = True
    use_system_load_salt: bool = False


@dataclass
class SimulationStepResult:
    changed_cells_count: int
    events: list
    timings: dict


@dataclass
class GranularModelConfig:
    inertia_decay: float = 0.65
    avalanche_threshold: float = 34.0
    theta_start_offset: float = 4.0
    theta_stop_offset: float = 2.0
    jam_probability_scale: float = 0.22
    min_jam_ticks: int = 3
    max_jam_ticks: int = 10


@dataclass
class FluidModelConfig:
    flow_span_base: int = 5
    pressure_weight: float = 0.45
    density_swap_rate: float = 0.3
    mix_decay: float = 0.015
    enable_non_newtonian: bool = True
    boundary_damping: float = 0.3
    pressure_smoothing: float = 0.2
    gas_turbulence: float = 0.16


@dataclass
class ThermalCombustionConfig:
    heat_diffusion_rate: float = 0.14
    convection_bias: float = 0.06
    oxygen_diffusion: float = 0.18
    smoke_spawn_rate: float = 0.08
    max_temp_delta_per_tick: float = 55.0
    ambient_temp: float = 20.0
    ignition_cooldown_ticks: int = 20


@dataclass
class ChemistryConfig:
    corrosion_base_rate: float = 0.025
    dissolution_base_rate: float = 0.03
    saturation_decay: float = 0.002
    reaction_progress_decay: float = 0.015


@dataclass
class PhaseChangeConfig:
    hysteresis: float = 2.0
    latent_heat_scale: float = 0.6
    min_transition_interval: int = 8
    transition_progress_decay: float = 0.04


class MaterialInteractionTable:
    REQUIRED_KEYS = {"pair", "priority", "products", "energy_delta", "gas_release", "residue", "duration_ticks"}

    def __init__(self, rules):
        self.rules = rules
        self._index = {}
        for rule_index, rule in enumerate(self.rules):
            a, b = self._normalize_pair(rule["pair"][0], rule["pair"][1])
            stable_rule = dict(rule)
            stable_rule["_stable_index"] = rule_index
            self._index.setdefault((a, b), []).append(stable_rule)

        for pair in self._index:
            self._index[pair].sort(key=lambda entry: (-entry["priority"], entry["_stable_index"]))

    def _normalize_pair(self, mat_a, mat_b):
        return (mat_a, mat_b) if mat_a <= mat_b else (mat_b, mat_a)

    def validate(self, materials):
        for rule in self.rules:
            missing = self.REQUIRED_KEYS - set(rule.keys())
            if missing:
                raise ValueError(f"Interaction rule missing keys: {sorted(missing)}")

            mat_a, mat_b = rule["pair"]
            if mat_a not in materials or mat_b not in materials:
                raise ValueError(f"Interaction rule references unknown material pair: {(mat_a, mat_b)}")

            if not isinstance(rule["products"], (list, tuple)) or len(rule["products"]) not in (1, 2):
                raise ValueError("Interaction rule 'products' must contain one or two entries")

            if rule["duration_ticks"] <= 0:
                raise ValueError("Interaction rule duration_ticks must be > 0")

    def get_rules(self, mat_a, mat_b):
        pair = self._normalize_pair(mat_a, mat_b)
        return self._index.get(pair, [])


class RandomManager:
    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.global_seed = config.seed if config.seed is not None else int.from_bytes(os.urandom(8), "big")

    def _stream_salt(self, stream_name: str):
        return sum((index + 1) * ord(char) for index, char in enumerate(stream_name))

    def _system_salt(self):
        if not self.config.use_system_load_salt:
            return 0

        load_value = int(os.getloadavg()[0] * 1000)
        mem_available_kb = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        mem_available_kb = int(line.split()[1])
                        break
        except OSError:
            mem_available_kb = 0

        return (load_value << 16) ^ (mem_available_kb & 0xFFFF)

    def for_tick(self, tick_index: int, stream_name: str = "default"):
        seed = self.global_seed ^ tick_index ^ self._stream_salt(stream_name) ^ self._system_salt()
        return random.Random(seed)


class PowderPhysicsEngine:
    """Dedicated physics engine for granular and fluid cellular behavior."""
    def __init__(self, materials, material_ids, config: PhysicsConfig | None = None):
        self.materials = materials
        self.material_ids = material_ids
        self.config = config or PhysicsConfig()
        self.random_manager = RandomManager(self.config)
        self.granular_config = GranularModelConfig()
        self.fluid_config = FluidModelConfig()
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
        self.integrity = []
        self.saturation_level = []
        self.phase_state = []
        self.reaction_progress = []
        self.phase_cooldown_until = []
        self.phase_transition_progress = []
        self.phase_transition_target = []

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
        return [
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
                "products": [self.material_ids["sand"], acid],
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
                "pair": (self.material_ids["sand"], acid),
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
                "pair": (self.material_ids["ice"], acid),
                "priority": 80,
                "conditions": {},
                "products": [water, acid],
                "energy_delta": 5.0,
                "gas_release": 0.04,
                "residue": 0,
                "duration_ticks": 6,
            },
        ]

    def _in_bounds(self, row, col, rows, cols):
        return 0 <= row < rows and 0 <= col < cols

    def _get(self, mat_id):
        return self.materials[mat_id]

    def _mat_value(self, mat_data, key, default):
        return mat_data[key] if key in mat_data else default

    def _ensure_granular_state(self, rows, cols):
        if len(self.lateral_bias) != rows or (rows > 0 and len(self.lateral_bias[0]) != cols):
            self.lateral_bias = [[0 for _ in range(cols)] for _ in range(rows)]
            self.jammed_until = [[0 for _ in range(cols)] for _ in range(rows)]

    def _ensure_fluid_state(self, rows, cols):
        if len(self.fluid_pressure) != rows or (rows > 0 and len(self.fluid_pressure[0]) != cols):
            self.fluid_pressure = [[0.0 for _ in range(cols)] for _ in range(rows)]
            self.mix_ratio = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def _ensure_thermal_state(self, rows, cols):
        if len(self.temperature) != rows or (rows > 0 and len(self.temperature[0]) != cols):
            self.temperature = [[self.thermal_config.ambient_temp for _ in range(cols)] for _ in range(rows)]
            self.oxygen_level = [[1.0 for _ in range(cols)] for _ in range(rows)]
            self.burn_stage = [[0 for _ in range(cols)] for _ in range(rows)]
            self.burn_progress = [[0.0 for _ in range(cols)] for _ in range(rows)]
            self.smoke_density = [[0.0 for _ in range(cols)] for _ in range(rows)]
            self.ignition_cooldown_until = [[0 for _ in range(cols)] for _ in range(rows)]
            self.fire_lifetime = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def _ensure_chemical_state(self, grid, rows, cols):
        if len(self.integrity) != rows or (rows > 0 and len(self.integrity[0]) != cols):
            self.integrity = [[1.0 for _ in range(cols)] for _ in range(rows)]
            self.saturation_level = [[0.0 for _ in range(cols)] for _ in range(rows)]
            self.phase_state = [[0 for _ in range(cols)] for _ in range(rows)]
            self.reaction_progress = [[0.0 for _ in range(cols)] for _ in range(rows)]
            self.phase_cooldown_until = [[0 for _ in range(cols)] for _ in range(rows)]
            self.phase_transition_progress = [[0.0 for _ in range(cols)] for _ in range(rows)]
            self.phase_transition_target = [[0 for _ in range(cols)] for _ in range(rows)]

        for row in range(rows):
            for col in range(cols):
                mat = grid[row][col]
                if mat == 0:
                    self.integrity[row][col] = 1.0
                    self.reaction_progress[row][col] = 0.0
                    self.phase_transition_progress[row][col] = 0.0
                    self.phase_transition_target[row][col] = 0

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

    def _reset_empty_state(self, row, col):
        if self._state_grid_ready(self.temperature, len(self.temperature), len(self.temperature[0]) if self.temperature else 0):
            self.temperature[row][col] = self.thermal_config.ambient_temp
        if self._state_grid_ready(self.burn_stage, len(self.burn_stage), len(self.burn_stage[0]) if self.burn_stage else 0):
            self.burn_stage[row][col] = 0
        if self._state_grid_ready(self.burn_progress, len(self.burn_progress), len(self.burn_progress[0]) if self.burn_progress else 0):
            self.burn_progress[row][col] = 0.0
        if self._state_grid_ready(self.ignition_cooldown_until, len(self.ignition_cooldown_until), len(self.ignition_cooldown_until[0]) if self.ignition_cooldown_until else 0):
            self.ignition_cooldown_until[row][col] = 0
        if self._state_grid_ready(self.integrity, len(self.integrity), len(self.integrity[0]) if self.integrity else 0):
            self.integrity[row][col] = 1.0
        if self._state_grid_ready(self.saturation_level, len(self.saturation_level), len(self.saturation_level[0]) if self.saturation_level else 0):
            self.saturation_level[row][col] = 0.0
        if self._state_grid_ready(self.phase_state, len(self.phase_state), len(self.phase_state[0]) if self.phase_state else 0):
            self.phase_state[row][col] = 0
        if self._state_grid_ready(self.reaction_progress, len(self.reaction_progress), len(self.reaction_progress[0]) if self.reaction_progress else 0):
            self.reaction_progress[row][col] = 0.0
        if self._state_grid_ready(self.phase_cooldown_until, len(self.phase_cooldown_until), len(self.phase_cooldown_until[0]) if self.phase_cooldown_until else 0):
            self.phase_cooldown_until[row][col] = 0
        if self._state_grid_ready(self.phase_transition_progress, len(self.phase_transition_progress), len(self.phase_transition_progress[0]) if self.phase_transition_progress else 0):
            self.phase_transition_progress[row][col] = 0.0
        if self._state_grid_ready(self.phase_transition_target, len(self.phase_transition_target), len(self.phase_transition_target[0]) if self.phase_transition_target else 0):
            self.phase_transition_target[row][col] = 0
        if self._state_grid_ready(self.mix_ratio, len(self.mix_ratio), len(self.mix_ratio[0]) if self.mix_ratio else 0):
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
        self._update_hydrostatic_balance(grid, rows, cols)
        self._update_density_sorting(grid, rows, cols, moved, rng, counters)
        self._update_fluid_flow(grid, rows, cols, moved, rng, counters)
        self._update_mixing(grid, rows, cols, rng)

    def _update_thermal_field(self, grid, rows, cols):
        next_temperature = [row[:] for row in self.temperature]

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

                if self.burn_stage[row][col] == 3:
                    pass  # exothermic heat now handled in _update_combustion_states
                elif self.burn_stage[row][col] == 4:
                    pass

                if mat == self.material_ids["water"]:
                    ambient = self.thermal_config.ambient_temp
                    if target_temp > ambient:
                        target_temp -= min(1.2, (target_temp - ambient) * 0.08)

                if row > 0 and self.temperature[row - 1][col] > self.temperature[row][col]:
                    target_temp += self.thermal_config.convection_bias

                delta = max(-self.thermal_config.max_temp_delta_per_tick, min(self.thermal_config.max_temp_delta_per_tick, target_temp - self.temperature[row][col]))
                next_temperature[row][col] = self.temperature[row][col] + delta

        self.temperature = next_temperature

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
                # active flame cells radiate based on their own temperature (handled below)
                if self.burn_stage[row][col] == 3:
                    heat_out = max(heat_out, 0.0)  # no extra bonus; exothermic calc handles self+neighbors
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
                has_flame_neighbor = self._neighbor_is_flaming(row, col, rows, cols)
                has_hot_neighbor = self._neighbor_is_hot(row, col, rows, cols, ignition_temp * 0.85)

                can_ignite = self.temperature[row][col] >= ignition_temp and (
                    has_flame_neighbor or has_hot_neighbor
                    or rng.random() < max(0.02, spark_sensitivity * 0.05)
                )
                if self.temperature[row][col] >= auto_ignite_temp:
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
                    grid[row][col] = 0
                    self.burn_stage[row][col] = 0
                    self.burn_progress[row][col] = 0.0
                    self.ignition_cooldown_until[row][col] = tick_index + self.thermal_config.ignition_cooldown_ticks
                    counters["changes"] += 1
                    events.append({"type": "extinguish", "row": row, "col": col})

    def _update_oxygen(self, rows, cols):
        next_oxygen = [row[:] for row in self.oxygen_level]
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
        next_smoke = [row[:] for row in self.smoke_density]
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
                if self.burn_stage[row][col] != 3:
                    continue
                # try a couple of candidate positions above/beside the flame
                for _ in range(3):
                    dr = rng.choice([-1, -1, 0])
                    dc = rng.choice([-1, 0, 1])
                    nr, nc = row + dr, col + dc
                    if not self._in_bounds(nr, nc, rows, cols):
                        continue
                    if grid[nr][nc] != 0:
                        continue
                    if rng.random() > 0.30:
                        continue
                    grid[nr][nc] = fire_id
                    self.fire_lifetime[nr][nc] = 0.75 + rng.random() * 0.25
                    self.temperature[nr][nc] = 420.0 + rng.random() * 160.0

    def _update_fire_particles(self, grid, rows, cols):
        """Decay fire-particle lifetime; remove dead ones from the grid."""
        fire_id = self.material_ids.get("fire", -1)
        if fire_id < 0 or not hasattr(self, "fire_lifetime"):
            return
        DECAY = 0.06
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != fire_id:
                    if self.fire_lifetime[row][col] > 0.0:
                        self.fire_lifetime[row][col] = 0.0
                    continue
                self.fire_lifetime[row][col] -= DECAY
                if self.fire_lifetime[row][col] <= 0.0:
                    grid[row][col] = 0
                    self.fire_lifetime[row][col] = 0.0

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

    def _stage_thermal(self, grid, rows, cols, tick_index, rng, counters, events):
        self._ensure_thermal_state(rows, cols)
        self._update_thermal_field(grid, rows, cols)
        self._update_contact_heating(grid, rows, cols)
        self._update_ignition(grid, rows, cols, tick_index, rng, events)
        self._update_combustion_states(grid, rows, cols, tick_index, events, counters)
        self._update_oxygen(rows, cols)
        self._update_smoke(rows, cols, events)
        self._spawn_fire_particles(grid, rows, cols, rng)
        self._update_fire_particles(grid, rows, cols)
        self._update_fire_sparks(grid, rows, cols, rng, events)

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

                freeze_temp = self._mat_value(mat_data, "freeze_temp", None)
                if freeze_temp is not None and temperature <= freeze_temp - self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "freeze_target", target)

                melt_temp = self._mat_value(mat_data, "melt_temp", None)
                if melt_temp is not None and temperature >= melt_temp + self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "melt_target", target)

                evaporate_temp = self._mat_value(mat_data, "evaporate_temp", None)
                if evaporate_temp is not None and temperature >= evaporate_temp + self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "evaporate_target", target)

                condense_temp = self._mat_value(mat_data, "condense_temp", None)
                if condense_temp is not None and temperature <= condense_temp - self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "condense_target", target)

                solidify_temp = self._mat_value(mat_data, "solidify_temp", None)
                if solidify_temp is not None and temperature <= solidify_temp - self.phase_change_config.hysteresis:
                    target = self._mat_value(mat_data, "solidify_target", target)

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
        counters = {"changes": 0}
        events = []

        mechanics_rng = self.random_manager.for_tick(tick_index, "mechanics")
        fluid_rng = self.random_manager.for_tick(tick_index, "fluids")
        thermal_rng = self.random_manager.for_tick(tick_index, "thermal")

        moved = [[False for _ in range(cols)] for _ in range(rows)]

        if self.config.enable_mechanics:
            moved = self._stage_mechanics(grid, rows, cols, moved, mechanics_rng, counters, tick_index)
        if self.config.enable_fluids:
            self._stage_fluids(grid, rows, cols, moved, fluid_rng, counters)
        if self.config.enable_thermal:
            self._stage_thermal(grid, rows, cols, tick_index, thermal_rng, counters, events)
        if self.config.enable_chemical:
            self._stage_chemical(grid, rows, cols, counters, events)
        if self.config.enable_phase_change:
            self._stage_phase_change(grid, rows, cols, tick_index, counters, events)
        if self.config.enable_cleanup:
            self._stage_cleanup()

        return SimulationStepResult(changed_cells_count=counters["changes"], events=events, timings={})


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
                "ignition_cooldown_until": self._copy_field(self.physics.ignition_cooldown_until),
                "integrity": self._copy_field(self.physics.integrity),
                "saturation_level": self._copy_field(self.physics.saturation_level),
                "phase_state": self._copy_field(self.physics.phase_state),
                "reaction_progress": self._copy_field(self.physics.reaction_progress),
                "phase_cooldown_until": self._copy_field(self.physics.phase_cooldown_until),
                "phase_transition_progress": self._copy_field(self.physics.phase_transition_progress),
                "phase_transition_target": self._copy_field(self.physics.phase_transition_target),
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
        self.physics.ignition_cooldown_until = self._copy_field(physics_state.get("ignition_cooldown_until", []))
        self.physics.integrity = self._copy_field(physics_state.get("integrity", []))
        self.physics.saturation_level = self._copy_field(physics_state.get("saturation_level", []))
        self.physics.phase_state = self._copy_field(physics_state.get("phase_state", []))
        self.physics.reaction_progress = self._copy_field(physics_state.get("reaction_progress", []))
        self.physics.phase_cooldown_until = self._copy_field(physics_state.get("phase_cooldown_until", []))
        self.physics.phase_transition_progress = self._copy_field(physics_state.get("phase_transition_progress", []))
        self.physics.phase_transition_target = self._copy_field(physics_state.get("phase_transition_target", []))

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

        self.profile_name = profile_name
        return True

    def cycle_profile(self):
        if not self.profile_order:
            return self.profile_name
        current_index = self.profile_order.index(self.profile_name) if self.profile_name in self.profile_order else 0
        next_index = (current_index + 1) % len(self.profile_order)
        self.apply_profile(self.profile_order[next_index])
        return self.profile_name

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

        paint_rng = self.physics.random_manager.for_tick(self.tick_index, "paint")
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

                        if paint_rng.random() > 0.2 or material_id in [MATERIAL_IDS["wall"], MATERIAL_IDS["wood"]]:
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


class MenuUI:
    """Redesigned side panel: material grid, brush tools, favorites, tooltip."""

    HEADER_H = 42
    PAD = 8
    MAT_CELL_H = 32
    MAT_COLS = 2
    SECTION_H = 20
    SEP_GAP = 6
    BTN_H = 26
    FAV_H = 34

    def __init__(self, x_offset, width, height):
        self.x = x_offset
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x_offset, 0, width, height)
        self.favorite_materials = []
        self._font_sm = pygame.font.SysFont("Arial", 12)
        self._font_label = pygame.font.SysFont("Arial", 10, bold=True)
        self._font_title = pygame.font.SysFont("Arial", 14, bold=True)
        self._font_sub = pygame.font.SysFont("Arial", 10)
        self._font_size = pygame.font.SysFont("Arial", 15, bold=True)
        self._font_tip_bold = pygame.font.SysFont("Arial", 12, bold=True)
        self._build_layout()

    def resize(self, x_offset, width, height):
        self.x = x_offset
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x_offset, 0, width, height)
        self._build_layout()

    def _build_layout(self):
        x, w, P = self.x, self.width, self.PAD
        y = self.HEADER_H + self.SEP_GAP

        # Materials — skip internal (non-player) materials
        self._mat_y = y
        y += self.SECTION_H
        col_w = (w - P * 3) // 2
        self.buttons = {}
        mat_ids = [m for m in MATERIALS.keys() if not MATERIALS[m].get("internal", False)]
        for i, mat_id in enumerate(mat_ids):
            ci = i % self.MAT_COLS
            ri = i // self.MAT_COLS
            bx = x + P + ci * (col_w + P)
            by = y + ri * (self.MAT_CELL_H + 3)
            self.buttons[mat_id] = pygame.Rect(bx, by, col_w, self.MAT_CELL_H)
        num_rows = (len(mat_ids) + self.MAT_COLS - 1) // self.MAT_COLS
        y += num_rows * (self.MAT_CELL_H + 3) + self.SEP_GAP

        # Brush
        self._brush_y = y
        y += self.SECTION_H
        shape_w = (w - P * 3) // 2
        self.circle_btn = pygame.Rect(x + P, y, shape_w, self.BTN_H)
        self.square_btn = pygame.Rect(x + P * 2 + shape_w, y, shape_w, self.BTN_H)
        y += self.BTN_H + 4
        mw = pw = 24
        val_w = w - P * 2 - mw - pw - 8
        self.tool_size_minus_btn = pygame.Rect(x + P, y, mw, self.BTN_H)
        self.tool_size_value_rect = pygame.Rect(x + P + mw + 4, y, val_w, self.BTN_H)
        self.tool_size_plus_btn = pygame.Rect(x + P + mw + 4 + val_w + 4, y, pw, self.BTN_H)
        y += self.BTN_H + self.SEP_GAP * 2

        # Favorites
        self._fav_y = y
        y += self.SECTION_H
        slot_w = max(10, (w - P * 2 - 3 * 5) // 4)
        self.favorite_slots = []
        for i in range(4):
            self.favorite_slots.append(pygame.Rect(x + P + i * (slot_w + 5), y, slot_w, self.FAV_H))
        y += self.FAV_H + self.SEP_GAP

        # Tooltip
        self._tip_y = y
        self._tip_h = max(68, self.height - y - 4)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _rrect(self, surface, color, rect, r=4, bw=0, bc=None):
        pygame.draw.rect(surface, color, rect, border_radius=r)
        if bw:
            pygame.draw.rect(surface, bc or color, rect, bw, border_radius=r)

    def _section_hdr(self, surface, text, y):
        t = self._font_label.render(text, True, (115, 138, 178))
        surface.blit(t, (self.x + self.PAD, y + (self.SECTION_H - t.get_height()) // 2))
        pygame.draw.line(surface, (48, 52, 66),
                         (self.x + self.PAD, y + self.SECTION_H - 1),
                         (self.x + self.width - self.PAD, y + self.SECTION_H - 1), 1)

    # ── public API ───────────────────────────────────────────────────────────
    def handle_click(self, mouse_pos):
        """Returns action tuple or None."""
        if self.rect.collidepoint(mouse_pos):
            for i, slot_rect in enumerate(self.favorite_slots):
                if slot_rect.collidepoint(mouse_pos) and i < len(self.favorite_materials):
                    return ("favorite", self.favorite_materials[i])
            for mat_id, btn_rect in self.buttons.items():
                if btn_rect.collidepoint(mouse_pos):
                    return ("material", mat_id)
            if self.circle_btn.collidepoint(mouse_pos):
                return ("brush_shape", "circle")
            if self.square_btn.collidepoint(mouse_pos):
                return ("brush_shape", "square")
            if self.tool_size_minus_btn.collidepoint(mouse_pos):
                return ("brush_delta", -1)
            if self.tool_size_plus_btn.collidepoint(mouse_pos):
                return ("brush_delta", 1)
        return None

    def draw(self, surface, font, active_mat_id, brush_size, brush_shape, favorites):
        pygame.draw.rect(surface, (24, 25, 32), self.rect)
        pygame.draw.line(surface, (60, 64, 82), (self.rect.left, 0), (self.rect.left, self.rect.bottom), 2)
        self.favorite_materials = [m for m in favorites if m in MATERIALS][:4]
        mouse_pos = pygame.mouse.get_pos()

        # ── Header ──────────────────────────────────────────────────────────
        hdr = pygame.Rect(self.x, 0, self.width, self.HEADER_H)
        pygame.draw.rect(surface, (16, 17, 24), hdr)
        pygame.draw.line(surface, (52, 56, 74),
                         (self.x, self.HEADER_H), (self.x + self.width, self.HEADER_H), 1)
        cx = self.x + self.width // 2
        t1 = self._font_title.render("SANDBOX", True, (140, 170, 255))
        t2 = self._font_sub.render("Falling Sand Engine", True, (80, 92, 120))
        surface.blit(t1, t1.get_rect(centerx=cx, y=7))
        surface.blit(t2, t2.get_rect(centerx=cx, y=25))

        # ── Materials ───────────────────────────────────────────────────────
        self._section_hdr(surface, "MATERIALS", self._mat_y)
        hovered_mat_id = None
        for mat_id, btn_rect in self.buttons.items():
            mat_data = MATERIALS[mat_id]
            is_active = mat_id == active_mat_id
            is_hov = btn_rect.collidepoint(mouse_pos)
            if is_hov:
                hovered_mat_id = mat_id
            bg = (44, 68, 100) if is_active else ((38, 40, 54) if is_hov else (30, 31, 40))
            bc = (88, 145, 235) if is_active else ((64, 68, 88) if is_hov else (40, 43, 56))
            self._rrect(surface, bg, btn_rect, r=4, bw=1, bc=bc)
            sw = pygame.Rect(btn_rect.x + 4, btn_rect.y + (btn_rect.height - 14) // 2, 14, 14)
            pygame.draw.rect(surface, mat_data["color"], sw, border_radius=2)
            pygame.draw.rect(surface, (90, 95, 112), sw, 1, border_radius=2)
            clr = (220, 232, 255) if is_active else (155, 158, 172)
            nt = self._font_sm.render(mat_data["name"], True, clr)
            surface.blit(nt, (sw.right + 4, btn_rect.y + (btn_rect.height - nt.get_height()) // 2))

        # ── Brush ────────────────────────────────────────────────────────────
        self._section_hdr(surface, "BRUSH SETTINGS", self._brush_y)
        for btn, sn, sym in [
            (self.circle_btn, "circle", "● Circle"),
            (self.square_btn, "square", "■ Square"),
        ]:
            is_a = brush_shape == sn
            self._rrect(surface, (44, 68, 100) if is_a else (30, 31, 40), btn, r=4,
                        bw=1, bc=(88, 145, 235) if is_a else (52, 55, 70))
            lt = self._font_sm.render(sym, True, (215, 232, 255) if is_a else (125, 128, 145))
            surface.blit(lt, lt.get_rect(center=btn.center))
        for btn, sym in [(self.tool_size_minus_btn, "−"), (self.tool_size_plus_btn, "+")]:
            self._rrect(surface, (34, 36, 48), btn, r=4, bw=1, bc=(62, 65, 82))
            bt = self._font_sm.render(sym, True, (185, 192, 210))
            surface.blit(bt, bt.get_rect(center=btn.center))
        self._rrect(surface, (28, 30, 42), self.tool_size_value_rect, r=4, bw=1, bc=(60, 64, 82))
        st = self._font_size.render(str(brush_size), True, (190, 215, 255))
        surface.blit(st, st.get_rect(center=self.tool_size_value_rect.center))

        # ── Favorites ────────────────────────────────────────────────────────
        self._section_hdr(surface, "FAVORITES  (F · Alt+1-4)", self._fav_y)
        for i, slot in enumerate(self.favorite_slots):
            filled = i < len(self.favorite_materials)
            self._rrect(surface, (28, 30, 40), slot, r=5,
                        bw=1, bc=(78, 105, 150) if filled else (50, 53, 68))
            if filled:
                mid = self.favorite_materials[i]
                pygame.draw.rect(surface, MATERIALS[mid]["color"], slot.inflate(-8, -8), border_radius=3)
            nt = self._font_label.render(str(i + 1), True,
                                         (195, 210, 240) if filled else (55, 58, 72))
            surface.blit(nt, (slot.x + 3, slot.y + 2))

        # ── Tooltip ──────────────────────────────────────────────────────────
        if hovered_mat_id is not None:
            mat_data = MATERIALS[hovered_mat_id]
            t_rect = pygame.Rect(self.x + 6, self._tip_y, self.width - 12, self._tip_h)
            self._rrect(surface, (20, 24, 36), t_rect, r=6, bw=1, bc=(65, 92, 140))
            lines = [
                (mat_data["name"], self._font_tip_bold, (180, 208, 255)),
                (f"Type: {mat_data['type']}", self._font_sm, (130, 136, 155)),
                (f"Density: {mat_data['density']}", self._font_sm, (130, 136, 155)),
                (f"Viscosity: {mat_data.get('viscosity', '—')}", self._font_sm, (130, 136, 155)),
            ]
            for j, (text, tf, col) in enumerate(lines):
                ts = tf.render(text, True, col)
                surface.blit(ts, (t_rect.x + 8, t_rect.y + 6 + j * 16))


class MenuBar:
    """Full-width dropdown menu bar drawn on top of everything."""

    H = 28          # bar height (same as TOP_BAR_HEIGHT)
    _ITEM_PAD = 12  # horizontal padding inside each top-level button
    DROP_W = 220    # dropdown panel width
    _ITEM_H = 24    # height of one dropdown row
    _SEP_H  = 9     # height of a separator row

    def __init__(self):
        self._font   = pygame.font.SysFont("Arial", 13)
        self._font_b = pygame.font.SysFont("Arial", 13, bold=True)
        self.open_idx = -1          # index of currently open top-level menu
        self._total_w = 1024
        self._top_rects: list[pygame.Rect] = []

        # Structure:  (top_label, [ entry | None ])
        # entry = (label, action_key, check_fn | None)
        # None  = separator
        self.menus = [
            ("File", [
                ("Save",               "save",          None),
                ("Load",               "load",          None),
                None,
                ("Save Replay",        "save_replay",   None),
                ("Load Replay",        "load_replay",   None),
                None,
                ("Benchmark",          "benchmark",     None),
                ("Snapshot Test",      "snapshot",      None),
                None,
                ("Quit",               "quit",          None),
            ]),
            ("Simulation", [
                ("Clear",              "clear",         None),
                ("Undo",               "undo",          None),
                ("Redo",               "redo",          None),
                None,
                ("Preset: Basin",      "preset_basin",  None),
                ("Preset: Volcano",    "preset_volcano",None),
                ("Preset: Steam",      "preset_steam",  None),
                None,
                ("Cycle Profile",      "cycle_profile", None),
                ("Reload Interactions","reload_interactions", None),
            ]),
            ("View", [
                ("Temperature Overlay","toggle_temp",   lambda e: e.show_temp_overlay),
                ("O\u2082 Overlay",    "toggle_oxygen", lambda e: e.show_oxygen_overlay),
                ("Smoke Overlay",      "toggle_smoke",  lambda e: e.show_smoke_overlay),
                ("Phase Overlay",      "toggle_phase",  lambda e: e.show_phase_overlay),
                None,
                ("Editor Mode",        "toggle_editor", lambda e: e.editor_mode),
                ("Sound",              "toggle_sound",  lambda e: e.sound_enabled),
                None,
                ("Show Help (H)",      "toggle_help",   lambda e: e.show_help),
            ]),
            ("Help", [
                ("Keyboard Shortcuts", "toggle_help",   None),
            ]),
        ]
        self._build_top_rects(self._total_w)

    # ── layout ────────────────────────────────────────────────────────────
    def _build_top_rects(self, total_w: int):
        self._total_w = total_w
        x = 6
        self._top_rects = []
        for label, _ in self.menus:
            w = self._font_b.size(label)[0] + self._ITEM_PAD * 2
            self._top_rects.append(pygame.Rect(x, 0, w, self.H))
            x += w

    def _drop_entries(self, idx: int):
        """Return list of (rect, entry_or_None).  entry = (label, action, check_fn)."""
        items = self.menus[idx][1]
        x = self._top_rects[idx].x
        if x + self.DROP_W > self._total_w:
            x = max(0, self._total_w - self.DROP_W)
        y = self.H
        result = []
        for item in items:
            h = self._SEP_H if item is None else self._ITEM_H
            result.append((pygame.Rect(x, y, self.DROP_W, h), item))
            y += h
        return result, x, y   # y = bottom of panel

    # ── input ─────────────────────────────────────────────────────────────
    def handle_event(self, event) -> str | None:
        """Returns action string, '__consumed__' (event eaten, no action), or None."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if my < self.H:
                for i, rect in enumerate(self._top_rects):
                    if rect.collidepoint(mx, my):
                        self.open_idx = -1 if self.open_idx == i else i
                        return "__consumed__"
                self.open_idx = -1
                return None
            if self.open_idx >= 0:
                entries, _, _ = self._drop_entries(self.open_idx)
                for rect, item in entries:
                    if item is not None and rect.collidepoint(mx, my):
                        self.open_idx = -1
                        return item[1]
                self.open_idx = -1
                return "__consumed__"   # closed menu; swallow click
        elif event.type == pygame.MOUSEMOTION:
            if self.open_idx >= 0:
                mx, my = event.pos
                if my < self.H:
                    for i, rect in enumerate(self._top_rects):
                        if rect.collidepoint(mx, my) and i != self.open_idx:
                            self.open_idx = i
                            break
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.open_idx >= 0:
                self.open_idx = -1
                return "__consumed__"
        return None

    def blocks_input(self, pos) -> bool:
        """True when pos falls inside the bar or the open dropdown."""
        mx, my = pos
        if my < self.H:
            return True
        if self.open_idx >= 0:
            entries, dx, bot = self._drop_entries(self.open_idx)
            if pygame.Rect(dx, self.H, self.DROP_W, bot - self.H).collidepoint(mx, my):
                return True
        return False

    # ── drawing ───────────────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface, engine):
        w = surface.get_width()
        if w != self._total_w:
            self._build_top_rects(w)
        mx, my = pygame.mouse.get_pos()

        # Bar background
        pygame.draw.rect(surface, (14, 15, 22), pygame.Rect(0, 0, w, self.H))
        pygame.draw.line(surface, (42, 46, 66), (0, self.H - 1), (w, self.H - 1), 1)

        # Top-level labels
        for i, (label, _) in enumerate(self.menus):
            rect = self._top_rects[i]
            is_open = self.open_idx == i
            is_hov  = rect.collidepoint(mx, my) and my < self.H
            if is_open:
                pygame.draw.rect(surface, (34, 54, 96), rect)
            elif is_hov:
                pygame.draw.rect(surface, (26, 28, 42), rect, border_radius=3)
            t = self._font_b.render(
                label, True,
                (210, 226, 255) if (is_open or is_hov) else (148, 155, 185)
            )
            surface.blit(t, t.get_rect(centery=rect.centery, x=rect.x + self._ITEM_PAD))

        # Dropdown panel
        if self.open_idx >= 0:
            entries, dx, bot = self._drop_entries(self.open_idx)
            panel = pygame.Rect(dx, self.H, self.DROP_W, bot - self.H)

            # Drop shadow
            shad = pygame.Surface((panel.w + 5, panel.h + 5), pygame.SRCALPHA)
            shad.fill((0, 0, 0, 60))
            surface.blit(shad, (panel.x + 3, panel.y + 3))

            pygame.draw.rect(surface, (18, 20, 32), panel, border_radius=6)
            pygame.draw.rect(surface, (50, 55, 80), panel, 1, border_radius=6)

            for rect, item in entries:
                if item is None:
                    pygame.draw.line(
                        surface, (40, 44, 64),
                        (rect.x + 10, rect.centery),
                        (rect.right - 10, rect.centery), 1
                    )
                    continue
                label, action, check_fn = item
                is_hov = rect.collidepoint(mx, my)
                if is_hov:
                    pygame.draw.rect(
                        surface, (36, 60, 108),
                        rect.inflate(-4, -2), border_radius=3
                    )
                checked = bool(check_fn(engine)) if check_fn else False
                if checked:
                    ct = self._font.render("\u2713", True, (118, 205, 110))
                    surface.blit(ct, (rect.x + 7, rect.y + (rect.h - ct.get_height()) // 2))
                lt = self._font.render(
                    label, True,
                    (208, 222, 255) if is_hov else (152, 160, 192)
                )
                surface.blit(lt, (rect.x + 24, rect.y + (rect.h - lt.get_height()) // 2))


class Engine:
    """Orchestrator: Connects Pygame events, the Simulation, and the UI."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Structured Falling Sand Engine")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.hud_font = pygame.font.SysFont("Arial", 16, bold=False)
        self.running = True
        
        # Modules
        self.sim = Simulation(COLS, ROWS)
        self.menu = MenuUI(SIM_WIDTH, MENU_WIDTH, WINDOW_HEIGHT)
        self.menu_bar = MenuBar()
        
        self.current_mat = MATERIAL_IDS["sand"]
        self.brush_size = 4
        self.last_step_ms = 0.0
        self.last_changed_cells = 0
        self.show_help = False
        self.show_temp_overlay = False
        self.show_oxygen_overlay = False
        self.show_smoke_overlay = False
        self.show_phase_overlay = False
        self.brush_shape = "circle"
        self.editor_mode = False
        self.last_action_message = ""
        self.last_action_message_until = 0.0
        self.benchmark_message = ""
        self.favorites = [MATERIAL_IDS["sand"], MATERIAL_IDS["water"], MATERIAL_IDS["wall"], MATERIAL_IDS["lava"]]
        self.sound_enabled = False
        self.sounds = {}
        self._sound_btn_rect = None   # built each frame by draw_top_bar
        self._init_sound_effects()
        self.number_hotkeys = {
            pygame.K_0: 0,
            pygame.K_1: 1,
            pygame.K_2: 2,
            pygame.K_3: 3,
            pygame.K_4: 4,
            pygame.K_5: 5,
            pygame.K_6: 6,
            pygame.K_7: 7,
            pygame.K_8: 8,
            pygame.K_9: 9,
        }

    def _init_sound_effects(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1)
            self.sounds = {
                "ignite": self._generate_tone_sound(760, 70, 0.35),
                "extinguish": self._generate_tone_sound(280, 90, 0.25),
                "reaction": self._generate_tone_sound(520, 80, 0.3),
                "phase_change": self._generate_tone_sound(430, 90, 0.28),
                "spark": self._generate_tone_sound(980, 35, 0.25),
            }
            self.sound_enabled = True
        except pygame.error:
            self.sound_enabled = False
            self.sounds = {}

    def _generate_tone_sound(self, frequency_hz, duration_ms, volume):
        mixer_info = pygame.mixer.get_init()
        sample_rate = mixer_info[0] if mixer_info else 22050
        channel_count = mixer_info[2] if mixer_info else 1
        sample_count = int(sample_rate * (duration_ms / 1000.0))
        frames = bytearray()
        for index in range(sample_count):
            t = index / sample_rate
            sample = int(32767 * volume * math.sin((2 * math.pi * frequency_hz) * t))
            if channel_count <= 1:
                frames.extend(struct.pack("<h", sample))
            else:
                frames.extend(struct.pack("<hh", sample, sample))
        return pygame.mixer.Sound(buffer=bytes(frames))

    def _on_resize(self, new_w, new_h):
        global WINDOW_WIDTH, WINDOW_HEIGHT, SIM_WIDTH
        WINDOW_WIDTH = max(600, new_w)
        WINDOW_HEIGHT = max(400, new_h)
        SIM_WIDTH = WINDOW_WIDTH - MENU_WIDTH
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        self.menu.resize(SIM_WIDTH, MENU_WIDTH, WINDOW_HEIGHT)
        self.menu_bar._build_top_rects(WINDOW_WIDTH)

    def _play_event_sounds(self, events):
        if not self.sound_enabled:
            return
        for event in events:
            event_type = event.get("type")
            if event_type == "ignite" and "ignite" in self.sounds:
                self.sounds["ignite"].play()
            elif event_type == "extinguish" and "extinguish" in self.sounds:
                self.sounds["extinguish"].play()
            elif event_type == "reaction" and "reaction" in self.sounds:
                self.sounds["reaction"].play()
            elif event_type == "phase_change" and "phase_change" in self.sounds:
                self.sounds["phase_change"].play()
            elif event_type in ("spark_spawn", "spark_ignite") and "spark" in self.sounds:
                self.sounds["spark"].play()

    def _toggle_favorite(self):
        if self.current_mat == 0:
            return
        if self.current_mat in self.favorites:
            self.favorites = [mat for mat in self.favorites if mat != self.current_mat]
            self._update_action_message(f"Favorite removed: {MATERIALS[self.current_mat]['name']}")
            return

        self.favorites.append(self.current_mat)
        self.favorites = self.favorites[:4]
        self._update_action_message(f"Favorite added: {MATERIALS[self.current_mat]['name']}")

    def _update_action_message(self, text):
        self.last_action_message = text
        self.last_action_message_until = time.perf_counter() + 4.0

    def _dispatch_menu_action(self, action: str):
        """Execute a menu action string returned by MenuBar."""
        if action == "save":
            path = self.sim.save_to_file("savegame.json")
            self._update_action_message(f"Saved: {path}")
        elif action == "load":
            path = self.sim.load_from_file("savegame.json")
            self._update_action_message(f"Loaded: {path}")
        elif action == "save_replay":
            path = self.sim.save_replay("replay.json")
            self._update_action_message(f"Replay saved: {path}")
        elif action == "load_replay":
            count = self.sim.load_replay("replay.json")
            self.sim.start_replay()
            self._update_action_message(f"Replay started ({count} events)")
        elif action == "benchmark":
            bm = self.sim.run_benchmark(240)
            self.benchmark_message = (
                f"Bench {bm['ticks']}t avg {bm['avg_ms']:.2f}ms "
                f"med {bm['median_ms']:.2f} p95 {bm['p95_ms']:.2f}"
            )
            self._update_action_message("Benchmark done")
        elif action == "snapshot":
            report = self.sim.run_snapshot_regressions()
            if report["created"]:
                self._update_action_message("Snapshot baseline created")
            elif report["passed"]:
                self._update_action_message("Snapshot regressions passed")
            else:
                self._update_action_message(f"Snapshot failed: {len(report['failures'])}")
        elif action == "quit":
            self.running = False
        elif action == "clear":
            self.sim.clear()
            self._update_action_message("Cleared world")
        elif action == "undo":
            if self.sim.undo():
                self._update_action_message("Undo")
        elif action == "redo":
            if self.sim.redo():
                self._update_action_message("Redo")
        elif action == "preset_basin":
            self.sim.load_scenario("basin")
            self._update_action_message("Preset: basin")
        elif action == "preset_volcano":
            self.sim.load_scenario("volcano")
            self._update_action_message("Preset: volcano")
        elif action == "preset_steam":
            self.sim.load_scenario("steam_chamber")
            self._update_action_message("Preset: steam_chamber")
        elif action == "cycle_profile":
            name = self.sim.cycle_profile()
            self._update_action_message(f"Profile: {name}")
        elif action == "reload_interactions":
            self.sim.physics.reload_interaction_table()
            self._update_action_message("Interactions reloaded")
        elif action == "toggle_temp":
            self.show_temp_overlay = not self.show_temp_overlay
        elif action == "toggle_oxygen":
            self.show_oxygen_overlay = not self.show_oxygen_overlay
        elif action == "toggle_smoke":
            self.show_smoke_overlay = not self.show_smoke_overlay
        elif action == "toggle_phase":
            self.show_phase_overlay = not self.show_phase_overlay
        elif action == "toggle_editor":
            self.editor_mode = not self.editor_mode
            self._update_action_message(f"Editor {'ON' if self.editor_mode else 'OFF'}")
        elif action == "toggle_sound":
            self.sound_enabled = not self.sound_enabled
            self._update_action_message(f"Sound {'ON' if self.sound_enabled else 'OFF'}")
        elif action == "toggle_help":
            self.show_help = not self.show_help

    def _edit_material_field(self, field_name, delta, minimum=0.0):
        if self.current_mat == 0:
            return
        mat_data = MATERIALS[self.current_mat]
        current_value = float(mat_data.get(field_name, 0.0))
        next_value = max(minimum, current_value + delta)
        mat_data[field_name] = round(next_value, 4)
        self._update_action_message(f"Edit {mat_data['name']} {field_name}={mat_data[field_name]}")

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            # Menu bar has first priority on every event
            _mb = self.menu_bar.handle_event(event)
            if _mb and _mb != "__consumed__":
                self._dispatch_menu_action(_mb)
            # Swallow mouse events that fall inside the bar / open dropdown
            if event.type == pygame.MOUSEBUTTONDOWN and self.menu_bar.blocks_input(event.pos):
                continue

            if event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if event.key == pygame.K_ESCAPE:
                    # Close open menu first; only quit when nothing is open
                    if self.menu_bar.open_idx < 0:
                        self.running = False
                elif event.key == pygame.K_c:
                    self.sim.clear()
                    self._update_action_message("Cleared world")
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_f:
                    self._toggle_favorite()
                elif event.key == pygame.K_e:
                    self.editor_mode = not self.editor_mode
                    self._update_action_message(f"Material editor {'ON' if self.editor_mode else 'OFF'}")
                elif event.key == pygame.K_t:
                    self.show_temp_overlay = not self.show_temp_overlay
                elif event.key == pygame.K_o:
                    self.show_oxygen_overlay = not self.show_oxygen_overlay
                elif event.key == pygame.K_m:
                    self.show_smoke_overlay = not self.show_smoke_overlay
                elif event.key == pygame.K_p:
                    self.show_phase_overlay = not self.show_phase_overlay
                elif event.key == pygame.K_b:
                    self.brush_shape = "square" if self.brush_shape == "circle" else "circle"
                    self._update_action_message(f"Brush shape: {self.brush_shape}")
                elif event.key == pygame.K_s:
                    self.sound_enabled = not self.sound_enabled
                    self._update_action_message(f"Sound {'ON' if self.sound_enabled else 'OFF'}")
                elif event.key == pygame.K_z and (mods & pygame.KMOD_CTRL):
                    if self.sim.undo():
                        self._update_action_message("Undo")
                elif event.key == pygame.K_y and (mods & pygame.KMOD_CTRL):
                    if self.sim.redo():
                        self._update_action_message("Redo")
                elif event.key == pygame.K_F5:
                    path = self.sim.save_to_file("savegame.json")
                    self._update_action_message(f"Saved {path}")
                elif event.key == pygame.K_F9:
                    path = self.sim.load_from_file("savegame.json")
                    self._update_action_message(f"Loaded {path}")
                elif event.key == pygame.K_F6:
                    path = self.sim.save_replay("replay.json")
                    self._update_action_message(f"Replay saved {path}")
                elif event.key == pygame.K_F7:
                    count = self.sim.load_replay("replay.json")
                    self.sim.start_replay()
                    self._update_action_message(f"Replay start ({count} events)")
                elif event.key == pygame.K_F1:
                    self.sim.load_scenario("basin")
                    self._update_action_message("Scenario: basin")
                elif event.key == pygame.K_F2:
                    self.sim.load_scenario("volcano")
                    self._update_action_message("Scenario: volcano")
                elif event.key == pygame.K_F3:
                    self.sim.load_scenario("steam_chamber")
                    self._update_action_message("Scenario: steam_chamber")
                elif event.key == pygame.K_F10:
                    benchmark = self.sim.run_benchmark(240)
                    self.benchmark_message = (
                        f"Bench {benchmark['ticks']}t avg {benchmark['avg_ms']:.2f}ms "
                        f"med {benchmark['median_ms']:.2f} p95 {benchmark['p95_ms']:.2f}"
                    )
                    self._update_action_message("Benchmark done")
                elif event.key == pygame.K_F11:
                    profile_name = self.sim.cycle_profile()
                    self._update_action_message(f"Profile: {profile_name}")
                elif event.key == pygame.K_F12:
                    self.sim.physics.reload_interaction_table()
                    self._update_action_message("Interaction matrix reloaded")
                elif event.key == pygame.K_F8:
                    report = self.sim.run_snapshot_regressions()
                    if report["created"]:
                        self._update_action_message(f"Snapshot baseline created: {report['path']}")
                    elif report["passed"]:
                        self._update_action_message("Snapshot regressions passed")
                    else:
                        self._update_action_message(f"Snapshot regressions failed: {len(report['failures'])}")
                elif self.editor_mode and event.key == pygame.K_LEFTBRACKET:
                    self._edit_material_field("density", -50.0, minimum=1.0)
                elif self.editor_mode and event.key == pygame.K_RIGHTBRACKET:
                    self._edit_material_field("density", 50.0, minimum=1.0)
                elif self.editor_mode and event.key == pygame.K_SEMICOLON:
                    self._edit_material_field("viscosity", -0.02, minimum=0.0)
                elif self.editor_mode and event.key == pygame.K_QUOTE:
                    self._edit_material_field("viscosity", 0.02, minimum=0.0)
                elif self.editor_mode and event.key == pygame.K_COMMA:
                    self._edit_material_field("phase_change_rate", -0.002, minimum=0.001)
                elif self.editor_mode and event.key == pygame.K_PERIOD:
                    self._edit_material_field("phase_change_rate", 0.002, minimum=0.001)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.brush_size = max(MIN_BRUSH_SIZE, self.brush_size - 1)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.brush_size = min(MAX_BRUSH_SIZE, self.brush_size + 1)
                elif event.key in self.number_hotkeys:
                    mat_id = self.number_hotkeys[event.key]
                    if mods & pygame.KMOD_ALT:
                        favorite_index = mat_id - 1 if mat_id > 0 else 3
                        if 0 <= favorite_index < len(self.favorites):
                            self.current_mat = self.favorites[favorite_index]
                    elif mat_id in MATERIALS:
                        self.current_mat = mat_id
            elif event.type == pygame.VIDEORESIZE:
                self._on_resize(event.w, event.h)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    # Sound-toggle button in the top bar
                    if self._sound_btn_rect and self._sound_btn_rect.collidepoint(event.pos):
                        self.sound_enabled = not self.sound_enabled
                        self._update_action_message(f"Sound {'ON' if self.sound_enabled else 'OFF'}")
                        continue
                    # Check if click was in UI
                    action = self.menu.handle_click(event.pos)
                    if action is not None:
                        action_type, value = action
                        if action_type == "material":
                            self.current_mat = value
                        elif action_type == "favorite":
                            self.current_mat = value
                        elif action_type == "brush_delta":
                            self.brush_size = max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, self.brush_size + value))
                        elif action_type == "brush_shape":
                            self.brush_shape = value

        # Continuous drawing (handling drag)
        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]: # Left or Right click
            mx, my = pygame.mouse.get_pos()
            
            # Only draw if mouse is inside the simulation area (and menu bar not open)
            if mx < SIM_WIDTH and my >= TOP_BAR_HEIGHT and not self.menu_bar.blocks_input((mx, my)):
                mat_to_draw = self.current_mat if buttons[0] else 0 # Right click forces Eraser
                col = mx // CELL_SIZE
                row = my // CELL_SIZE
                self.sim.paint(col, row, self.brush_size, mat_to_draw, self.brush_shape)

    def draw_top_bar(self):
        bar_rect = pygame.Rect(0, 0, SIM_WIDTH, TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (28, 28, 34), bar_rect)
        pygame.draw.line(self.screen, (85, 85, 95), (0, TOP_BAR_HEIGHT - 1), (SIM_WIDTH, TOP_BAR_HEIGHT - 1), 1)

        statuses = [
            ("[T] Temp",  self.show_temp_overlay),
            ("[O] O2",    self.show_oxygen_overlay),
            ("[M] Smoke", self.show_smoke_overlay),
            ("[P] Phase", self.show_phase_overlay),
            ("[E] Edit",  self.editor_mode),
        ]
        x = 10
        for label, active in statuses:
            txt = f"{label} {'ON' if active else 'OFF'}"
            col = (110, 200, 110) if active else (140, 140, 155)
            surf = self.hud_font.render(txt, True, col)
            self.screen.blit(surf, (x, 5))
            x += surf.get_width() + 14
            sep = self.hud_font.render("|", True, (55, 58, 72))
            self.screen.blit(sep, (x - 8, 5))

        # Sound button on the right side of the bar
        sound_lbl = "\u266b Sound ON" if self.sound_enabled else "\u266a Sound OFF"
        sound_col = (110, 200, 110) if self.sound_enabled else (180, 80, 80)
        s_surf = self.hud_font.render(sound_lbl, True, sound_col)
        s_x = SIM_WIDTH - s_surf.get_width() - 10
        # clickable rect stored for handle_input
        self._sound_btn_rect = pygame.Rect(s_x - 4, 2, s_surf.get_width() + 8, TOP_BAR_HEIGHT - 4)
        pygame.draw.rect(self.screen, (32, 34, 46), self._sound_btn_rect, border_radius=3)
        pygame.draw.rect(self.screen, sound_col, self._sound_btn_rect, 1, border_radius=3)
        self.screen.blit(s_surf, (s_x, 5))

    def draw_overlays(self):
        if not (self.show_temp_overlay or self.show_oxygen_overlay or self.show_smoke_overlay or self.show_phase_overlay):
            return

        physics = self.sim.physics
        rows = self.sim.rows
        cols = self.sim.cols

        temp_ready = len(physics.temperature) == rows and (rows == 0 or len(physics.temperature[0]) == cols)
        oxygen_ready = len(physics.oxygen_level) == rows and (rows == 0 or len(physics.oxygen_level[0]) == cols)
        smoke_ready = len(physics.smoke_density) == rows and (rows == 0 or len(physics.smoke_density[0]) == cols)
        phase_ready = len(physics.phase_transition_progress) == rows and (rows == 0 or len(physics.phase_transition_progress[0]) == cols)

        ambient = physics.thermal_config.ambient_temp

        for row in range(rows):
            pixel_y = row * CELL_SIZE
            if pixel_y < TOP_BAR_HEIGHT:
                continue
            for col in range(cols):
                x = col * CELL_SIZE
                y = row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                if self.show_temp_overlay and temp_ready:
                    temp = physics.temperature[row][col]
                    delta = max(-60.0, min(300.0, temp - ambient))
                    if delta >= 0:
                        intensity = min(255, int((delta / 300.0) * 255))
                        overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        overlay.fill((255, 80, 40, max(0, min(170, 25 + intensity // 2))))
                        self.screen.blit(overlay, rect)
                    else:
                        cold_intensity = min(255, int((abs(delta) / 60.0) * 255))
                        overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        overlay.fill((60, 130, 255, max(0, min(170, 25 + cold_intensity // 2))))
                        self.screen.blit(overlay, rect)

                if self.show_oxygen_overlay and oxygen_ready:
                    oxygen = max(0.0, min(1.0, physics.oxygen_level[row][col]))
                    red = int((1.0 - oxygen) * 220)
                    green = int(oxygen * 220)
                    overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    overlay.fill((red, green, 40, 80))
                    self.screen.blit(overlay, rect)

                if self.show_smoke_overlay and smoke_ready:
                    smoke = max(0.0, min(1.0, physics.smoke_density[row][col]))
                    alpha = int(smoke * 185)
                    if alpha > 0:
                        overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        overlay.fill((160, 160, 170, alpha))
                        self.screen.blit(overlay, rect)

                if self.show_phase_overlay and phase_ready:
                    phase_progress = max(0.0, min(1.0, physics.phase_transition_progress[row][col]))
                    alpha = int(phase_progress * 190)
                    if alpha > 0:
                        overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        overlay.fill((190, 90, 255, alpha))
                        self.screen.blit(overlay, rect)

    def draw_hud(self):
        active_name = MATERIALS[self.current_mat]["name"]
        fps_value = self.clock.get_fps()
        ambient = self.sim.physics.thermal_config.ambient_temp
        lines = [
            f"Mat [{self.current_mat}] {active_name}",
            f"Brush {self.brush_size} ({self.brush_shape}) | FPS {fps_value:.1f} | Step {self.last_step_ms:.2f}ms",
        ]

        if self.last_action_message and time.perf_counter() <= self.last_action_message_until:
            lines.append(self.last_action_message)
        if self.benchmark_message and self.show_help:
            lines.append(self.benchmark_message)
        if self.show_help:
            lines.append(f"Ambient {ambient:.1f} C | Profile {self.sim.profile_name} | Sound {'ON' if self.sound_enabled else 'OFF'}")

        if self.editor_mode and self.current_mat != 0:
            material = MATERIALS[self.current_mat]
            lines.append(
                f"Edit [{material['name']}] dens {material.get('density', 0)} vis {material.get('viscosity', 0):.3f}"
            )
            lines.append(
                f"phase_rate {material.get('phase_change_rate', 0):.3f}  [ ] dens ; ' vis , . rate"
            )

        for index, text in enumerate(lines):
            surface = self.hud_font.render(text, True, (235, 235, 235))
            self.screen.blit(surface, (10, 34 + (index * 18)))

        if self.show_help:
            help_lines = [
                "H toggle help",
                "1-9/0 choose material",
                "B brush shape, +/- size, C clear, S sound",
                "T temp, O oxygen, M smoke, P phase",
                "Ctrl+Z/Y undo redo | F5/F9 save load",
                "F6/F7 replay save/start | F1-F3 presets | F10 bench",
                "F11 cycle profile | F12 reload interactions",
                "F favorites toggle | Alt+1..4 quick favorites",
                "F8 snapshot regressions",
                "E editor mode",
            ]
            for index, text in enumerate(help_lines):
                surface = self.hud_font.render(text, True, (210, 210, 210))
                self.screen.blit(surface, (10, 88 + (index * 16)))

    def draw_simulation(self):
        # Fill sim area background (Air color)
        pygame.draw.rect(self.screen, MATERIALS[0]["color"], (0, 0, SIM_WIDTH, WINDOW_HEIGHT))

        _fire_id = MATERIAL_IDS.get("fire", -1)
        _fire_lt = self.sim.physics.fire_lifetime if hasattr(self.sim.physics, "fire_lifetime") else None
        _ticks = pygame.time.get_ticks()

        # Render active grid cells
        for row in range(self.sim.rows):
            for col in range(self.sim.cols):
                mat = self.sim.grid[row][col]
                if mat == 0:
                    continue
                x = col * CELL_SIZE
                y = row * CELL_SIZE
                if mat == _fire_id and _fire_lt is not None:
                    lt = _fire_lt[row][col]
                    # flicker: shift colour every ~50ms based on position
                    flicker = (_ticks // 50 + row * 3 + col * 7) % 5
                    if lt > 0.65 or flicker == 0:
                        color = (255, min(255, 220 + flicker * 7), max(0, int(80 * lt)))
                    elif lt > 0.35:
                        color = (255, int(80 + 120 * lt), 0)
                    else:
                        color = (max(160, int(255 * lt * 2.5)), int(40 * lt), 0)
                    pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(self.screen, MATERIALS[mat]["color"], (x, y, CELL_SIZE, CELL_SIZE))

        self.draw_overlays()

    def run(self):
        while self.running:
            self.handle_input()
            step_start = time.perf_counter()
            step_result = self.sim.update_physics()
            self.last_step_ms = (time.perf_counter() - step_start) * 1000.0
            self.last_changed_cells = step_result.changed_cells_count
            
            self.draw_simulation()
            self.menu.draw(self.screen, self.font, self.current_mat, self.brush_size, self.brush_shape, self.favorites)
            self.draw_hud()
            self.menu_bar.draw(self.screen, self)   # drawn last so it's always on top
            self._play_event_sounds(step_result.events)
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = Engine()
    app.run()