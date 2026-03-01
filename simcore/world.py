import pygame
import numpy as np
import sys
import traceback
import random
import os
import math
import time
import json
import struct
import io
import ast
import types
import copy
import itertools
import collections
import re
import pathlib
import importlib.util
import subprocess
import threading
from enum        import IntEnum
from dataclasses import dataclass
from simcore.config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    MENU_WIDTH,
    SIM_WIDTH,
    CELL_SIZE,
    COLS,
    ROWS,
    FPS,
    MIN_BRUSH_SIZE,
    MAX_BRUSH_SIZE,
    TOP_BAR_HEIGHT,
    INTERACTION_MATRIX_FILE,
    CONSOLE_HEIGHT,
    CONSOLE_MAX_LINES,
    CONSOLE_FONT_SIZE,
    CONSOLE_INPUT_COLOR,
    CONSOLE_OUTPUT_COLOR,
    CONSOLE_ERROR_COLOR,
    CONSOLE_SYSTEM_COLOR,
    CONSOLE_PROMPT,
    CONSOLE_CONT_PROMPT,
    CONFIG_PROFILES_FILE,
    SNAPSHOT_BASELINE_FILE,
)
from simcore.logging import GameLogger, LogLevel, log

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
        "auto_ignite_temp": 280.0,
        "burn_rate": 0.004,
        "smoke_factor": 0.70,
        "ash_yield": 0.35,
        "char_yield": 0.15,
        "heat_of_combustion": 1000.0,
        "burnout_product": 7,
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
        "thermal_conductivity": 0.06,
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
        "initial_temp": 1050.0,
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
        "internal": True,
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
    },
    11: {
        "name": "Stone",
        "color": (148, 150, 155),
        "type": "solid",
        "density": 2600,
        "viscosity": 0.0,
        "thermal_capacity": 0.84,
        "thermal_conductivity": 0.45,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 0.85,
        "dissolution_rate": 0.005,
        "passivation_factor": 0.3,
        "solubility_limit": 0.0,
        "initial_temp": None,
        "phase_change_rate": 0.005,
        "melt_temp": 1400.0,
        "melt_target": 9,
        "latent_heat": 35.0,
        "inertia": 0.0,
        "repose_angle": 90,
        "dispersion": 0,
        "drag": 0.0,
    },
    12: {
        "name": "Oil",
        "color": (90, 60, 15),
        "type": "liquid",
        "density": 850,
        "viscosity": 0.6,
        "thermal_capacity": 1.67,
        "thermal_conductivity": 0.15,
        "ignition_temp": 210.0,
        "auto_ignite_temp": 260.0,
        "burn_rate": 0.008,
        "smoke_factor": 0.9,
        "min_oxygen_for_ignition": 0.15,
        "spark_sensitivity": 0.4,
        "heat_of_combustion": 1400.0,
        "burnout_product": 0,
        "corrosion_resistance": 0.95,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 0.0,
        "initial_temp": None,
        "mix_compatibility": 0.05,
        "buoyancy_bias": 0.15,
        "inertia": 0.12,
        "repose_angle": 0,
        "dispersion": 3,
        "drag": 0.22,
    },
    13: {
        "name": "Gunpowder",
        "color": (58, 52, 46),
        "type": "powder",
        "density": 1000,
        "viscosity": 0.0,
        "thermal_capacity": 0.9,
        "thermal_conductivity": 0.08,
        "ignition_temp": 240.0,
        "auto_ignite_temp": 260.0,
        "burn_rate": 0.85,
        "smoke_factor": 0.6,
        "min_oxygen_for_ignition": 0.05,
        "spark_sensitivity": 0.98,
        "heat_of_combustion": 2200.0,
        "burnout_product": 7,
        "corrosion_resistance": 0.3,
        "dissolution_rate": 0.1,
        "passivation_factor": 0.0,
        "solubility_limit": 0.5,
        "initial_temp": None,
        "inertia": 0.12,
        "repose_angle": 34,
        "dispersion": 2,
        "drag": 0.1,
    },
    14: {
        "name": "Plant",
        "color": (50, 170, 55),
        "type": "solid",
        "density": 400,
        "viscosity": 0.0,
        "thermal_capacity": 1.5,
        "thermal_conductivity": 0.12,
        "ignition_temp": 240.0,
        "auto_ignite_temp": 300.0,
        "burn_rate": 0.006,
        "smoke_factor": 0.55,
        "min_oxygen_for_ignition": 0.12,
        "spark_sensitivity": 0.65,
        "heat_of_combustion": 600.0,
        "burnout_product": 7,
        "corrosion_resistance": 0.2,
        "dissolution_rate": 0.06,
        "passivation_factor": 0.0,
        "solubility_limit": 0.0,
        "initial_temp": None,
        "inertia": 0.0,
        "repose_angle": 90,
        "dispersion": 0,
        "drag": 0.0,
    },
    15: {
        "name": "Smoke",
        "internal": True,
        "color": (110, 110, 120),
        "type": "gas",
        "density": 3,
        "viscosity": 0.03,
        "thermal_capacity": 1.0,
        "thermal_conductivity": 0.05,
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
        "initial_temp": None,
        "latent_heat": 0.0,
        "inertia": 0.01,
        "repose_angle": 0,
        "dispersion": 4,
        "drag": 0.01,
    },
    16: {
        # Magma — superheated silicate melt, brighter and far hotter than lava.
        # Flows faster, solidifies into Basalt (17) instead of generic Wall.
        "name": "Magma",
        "color": (255, 200, 40),
        "type": "liquid",
        "density": 2700,
        "viscosity": 0.6,               # more fluid than lava (1.4)
        "thermal_capacity": 1.4,
        "thermal_conductivity": 0.08,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "viscosity_temp_factor": 0.0,
        "shear_index": 1.1,
        "mix_compatibility": 0.05,
        "buoyancy_bias": -0.1,
        "corrosion_resistance": 1.0,
        "dissolution_rate": 0.0,
        "passivation_factor": 0.0,
        "solubility_limit": 1.0,
        "initial_temp": 2200.0,         # much hotter than lava (1050 °C)
        "phase_change_rate": 0.006,
        "solidify_temp": 900.0,         # solidifies at higher temp than lava (640 °C)
        "solidify_target": 17,          # → Basalt
        "latent_heat": 32.0,
        "inertia": 0.14,
        "repose_angle": 0,
        "dispersion": 3,
        "drag": 0.28,
    },
    17: {
        # Basalt — dark volcanic rock formed when Magma cools.
        # Harder and denser than Stone; visually distinct (near-black with blue tint).
        "name": "Basalt",
        "color": (45, 45, 60),
        "type": "solid",
        "density": 3000,
        "viscosity": 0.0,
        "thermal_capacity": 0.84,
        "thermal_conductivity": 0.55,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
        "burn_rate": 0.0,
        "smoke_factor": 0.0,
        "min_oxygen_for_ignition": 1.0,
        "spark_sensitivity": 0.0,
        "corrosion_resistance": 0.95,
        "dissolution_rate": 0.002,
        "passivation_factor": 0.4,
        "solubility_limit": 0.0,
        "initial_temp": None,
        "phase_change_rate": 0.004,
        "melt_temp": 1800.0,            # melts back to Magma at extreme heat
        "melt_target": 16,
        "latent_heat": 40.0,
        "inertia": 0.0,
        "repose_angle": 90,
        "dispersion": 0,
        "drag": 0.0,
    },
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
    "burnout_product": 0,
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
    "youngs_modulus": 2.0e7,
    "poisson_ratio": 0.28,
    "yield_strength": 2.0e5,
    "cohesion": 2.5e4,
    "friction_angle_deg": 30.0,
    "thermal_expansion_coeff": 1.2e-5,
    "degradation_temp_start": 500.0,
    "degradation_temp_end": 1200.0,
    "plastic_hardening": 0.02,
    "neo_hookean_c1": 1.0e5,
    "neo_hookean_d1": 1.0e6,
    "is_brittle": False,
    "pore_pressure_sensitivity": 0.25,
    "spall_temp_gradient_threshold": 650.0,
    "debris_restitution": 0.25,
    "debris_friction": 0.35,
    "arrhenius_A": 2.0e6,
    "arrhenius_Ea": 8.5e4,
    "arrhenius_order_fuel": 1.0,
    "arrhenius_order_o2": 1.0,
    "stoich_o2_per_fuel": 3.5,
    "reaction_heat_release": 2.6e6,
    "edc_coeff": 4.0,
    "pyrolysis_temp_start": 420.0,
    "pyrolysis_temp_peak": 820.0,
    "pyrolysis_latent_heat": 4.5e5,
    "pyrolysis_yield": 0.35,
    "pyrolysis_gas_material": "smoke",
    "soot_nucleation_factor": 0.0,
    "soot_growth_factor": 0.0,
    "soot_oxidation_factor": 0.0,
    "acid_strength": 0.0,
    "base_strength": 0.0,
    "catalytic_activity": 0.0,
    "adsorption_fuel": 0.0,
    "adsorption_o2": 0.0,
    "surface_reaction_rate": 0.0,
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


# ──────────────────────────────────────────────────────────────────────────────
# P H Y S I C S   C O N S T A N T S   (Phase 1 — Schritt 7)
# Based on: Bridson, "Fluid Simulation" SIGGRAPH 2007, §2.3 Time Steps.
# Grid spacing 0.05 m/cell → 128×96 cell grid ≈ 6.4 m × 4.8 m room (plausible
# for a firefighter scenario). dt = 1/60 s matches pygame FPS.
# CFL limit: max(|u|)·dt/dx < 1.0 (semi-Lagrangian is unconditionally stable
# per Stam 1999 but we still clamp to avoid excessive numerical diffusion).
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PhysicsConstants:
    dx: float = 0.05          # metres per cell (horizontal)
    dy: float = 0.05          # metres per cell (vertical)
    dt: float = 1.0 / 60.0   # seconds per tick (matches FPS)
    g:  float = 9.81          # m/s²  gravitational acceleration
    rho_air:  float = 1.225   # kg/m³ air at 20 °C, 1 atm (ISA standard)
    rho_water: float = 997.0  # kg/m³ water at 25 °C
    c_sound:  float = 343.0   # m/s   speed of sound in air at 20 °C
    mu_air:   float = 1.81e-5 # Pa·s  dynamic viscosity of air at 20 °C
    CFL_limit: float = 1.0    # Courant number ceiling (semi-Lagrangian is
                               # unconditionally stable, so >1 is allowed but
                               # increases numerical diffusion — keep ≤ 1.0)

    @property
    def dx_inv(self) -> float:  return 1.0 / self.dx
    @property
    def dy_inv(self) -> float:  return 1.0 / self.dy


# Boundary condition types (Bridson §1.6, §4.5)
class BoundaryConditionType(IntEnum):
    NO_SLIP   = 0   # solid wall: u·n = 0, u·t = 0
    FREE_SLIP = 1   # smooth wall: u·n = 0, u·t unchanged
    OPEN      = 2   # atmospheric: p = 0 at boundary (outflow)
    INLET     = 3   # prescribed velocity inflow
    OUTLET    = 4   # zero-gradient (Neumann) outflow


# Global singleton — accessed by PowderPhysicsEngine and future PDE solvers.
PHYSICS = PhysicsConstants()


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
    pde_enabled: bool = True
    pde_jacobi_iterations: int = 32
    pde_viscosity_iterations: int = 10
    pde_kinematic_viscosity: float = 2.5e-4
    pde_vorticity_eps: float = 0.35
    pde_buoyancy_scale: float = 1.0
    thermal_expansion_beta: float = 3.4e-3
    pde_scalar_advection: bool = True
    pde_use_bfecc: bool = True
    pde_numba_enabled: bool = True
    pde_sparse_pressure_enabled: bool = True
    pde_pressure_active_divergence_threshold: float = 1.0e-4
    pde_pressure_active_dilation: int = 1
    pde_pressure_iterations_min: int = 8
    pde_pressure_iterations_max: int = 64
    pde_pressure_residual_tolerance: float = 1.0e-3
    pde_pressure_budget_ms: float = 2.0
    pde_pressure_budget_adapt: bool = True
    pde_multigrid_enabled: bool = False
    pde_multigrid_levels: int = 2
    pde_multigrid_presmooth: int = 2
    pde_multigrid_postsmooth: int = 2
    pde_boundary_type: BoundaryConditionType = BoundaryConditionType.NO_SLIP
    acoustic_enabled: bool = True
    acoustic_wave_speed: float = 25.0
    acoustic_cfl_target: float = 0.6
    acoustic_substeps_min: int = 2
    acoustic_substeps_max: int = 16
    acoustic_pml_thickness: int = 8
    acoustic_pml_strength: float = 24.0
    acoustic_pml_power: float = 2.0
    acoustic_velocity_coupling: float = 0.12
    acoustic_pressure_coupling: float = 0.08
    detonation_enabled: bool = True
    detonation_energy_scale: float = 9.5e4
    detonation_radius_scale: float = 1.25
    detonation_divergence_scale: float = 6.0
    detonation_impulse_scale: float = 4.5
    shock_failure_enabled: bool = True
    shock_yield_threshold: float = 1.8e4
    shock_damage_scale: float = 6.0e-5
    shock_pressure_gradient_weight: float = 0.15
    shock_shear_weight: float = 1.0
    spallation_probability_scale: float = 0.35
    free_surface_enabled: bool = True
    wind_forcing_enabled: bool = False
    wind_reference_speed: float = 2.5
    wind_reference_height: float = 2.0
    wind_roughness_length: float = 0.03
    wind_displacement_height: float = 0.0
    wind_kappa: float = 0.41
    porous_drag_enabled: bool = True
    porous_particle_diameter: float = 0.012
    porous_porosity: float = 0.45
    porous_drag_multiplier: float = 1.0
    validation_export_enabled: bool = True
    validation_export_path: str = "physics_validation.json"


@dataclass
class StructuralModelConfig:
    enabled: bool = True
    explicit_substeps: int = 2
    damping: float = 0.12
    gravity_coupling: float = 1.0
    thermal_strain_enabled: bool = True
    thermal_degradation_enabled: bool = True
    elastoplastic_enabled: bool = True
    neo_hookean_enabled: bool = True
    finite_strain_clip: float = 0.35
    brittle_mohr_coulomb_enabled: bool = True
    failure_damage_rate: float = 0.08
    brittle_tension_cutoff: float = 8.0e4
    debris_enabled: bool = True
    debris_spawn_mass_scale: float = 0.9
    debris_particle_radius_cells: float = 0.45
    debris_particle_lifetime: float = 3.5
    debris_contact_stiffness: float = 1800.0
    debris_contact_damping: float = 24.0
    debris_wall_restitution: float = 0.25
    spalling_enabled: bool = True
    spalling_temperature_gradient_threshold: float = 700.0
    spalling_pore_pressure_weight: float = 0.5


@dataclass
class ThermalCombustionConfig:
    heat_diffusion_rate: float = 0.14
    convection_bias: float = 0.06
    oxygen_diffusion: float = 0.18
    smoke_spawn_rate: float = 0.08
    max_temp_delta_per_tick: float = 55.0
    ambient_temp: float = 20.0
    ignition_cooldown_ticks: int = 20
    adi_enabled: bool = True
    adi_iterations: int = 1
    radiation_enabled: bool = True
    radiation_emissivity: float = 0.85
    radiation_strength: float = 0.035
    radiation_sigma: float = 5.670374419e-8
    convection_advection_blend: float = 0.55
    species_diffusion_enabled: bool = True
    oxygen_diffusivity: float = 0.18
    smoke_diffusivity: float = 0.10
    steam_diffusivity: float = 0.14
    moisture_diffusivity: float = 0.06
    porous_moisture_enabled: bool = True
    porous_moisture_gain: float = 0.015
    enthalpy_enabled: bool = True
    mushy_range: float = 2.5
    latent_heat_factor: float = 1.0
    mushy_drag_strength: float = 5.0
    leidenfrost_enabled: bool = True
    leidenfrost_temp: float = 193.0
    leidenfrost_transfer_factor: float = 0.22
    leidenfrost_evap_rate: float = 0.02


@dataclass
class ChemistryConfig:
    corrosion_base_rate: float = 0.025
    dissolution_base_rate: float = 0.03
    saturation_decay: float = 0.002
    reaction_progress_decay: float = 0.015
    stiff_kinetics_enabled: bool = True
    stiff_newton_iterations: int = 4
    arrhenius_R: float = 8.314462618
    arrhenius_temp_clamp_min: float = 250.0
    arrhenius_temp_clamp_max: float = 2600.0
    edc_enabled: bool = True
    edc_mixing_length: float = 0.08
    edc_min_epsilon: float = 1.0e-5
    edc_tau_clip_min: float = 1.0e-4
    edc_tau_clip_max: float = 0.4
    pyrolysis_enabled: bool = True
    pyrolysis_convective_gain: float = 0.22
    pyrolysis_radiative_gain: float = 0.08
    pyrolysis_conductive_loss: float = 0.18
    pyrolysis_mass_flux_scale: float = 2.5e-4
    pyrolysis_blowing_velocity: float = 1.8
    soot_enabled: bool = True
    soot_diffusion: float = 0.05
    soot_coagulation_rate: float = 0.08
    soot_oxidation_o2_coeff: float = 0.24
    soot_oxidation_oh_coeff: float = 0.02
    soot_radiation_coupling: float = 0.45
    electrolyte_enabled: bool = True
    electrolyte_diffusion: float = 0.09
    electrolyte_kw_25c: float = 1.0e-14
    neutralization_enthalpy: float = 5.8e4
    electrolyte_concentration_scale: float = 1.0e-3
    surface_kinetics_enabled: bool = True
    lh_desorption_rate: float = 0.06
    lh_diffusivity: float = 0.08
    lh_reference_length: float = 0.05
    lh_heat_release_scale: float = 2.2e4


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

    def add_rule(self, rule: dict):
        """Add a new interaction rule at runtime (used by ScriptAPI.interaction)."""
        rule = dict(rule)
        rule.setdefault("_stable_index", len(self.rules))
        self.rules.append(rule)
        a, b = self._normalize_pair(rule["pair"][0], rule["pair"][1])
        self._index.setdefault((a, b), []).append(rule)
        self._index[(a, b)].sort(key=lambda e: (-e["priority"], e["_stable_index"]))

    def remove_rule(self, pair):
        """Remove all rules for the given material-ID pair (tuple of two ints)."""
        if len(pair) != 2:
            raise ValueError("pair must be a sequence of two material IDs")
        a, b = self._normalize_pair(int(pair[0]), int(pair[1]))
        removed = len(self._index.pop((a, b), []))
        self.rules = [r for r in self.rules
                      if self._normalize_pair(r["pair"][0], r["pair"][1]) != (a, b)]
        return removed


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


