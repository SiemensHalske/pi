"""plasma.py  —  registers a hot 'Plasma' material."""
plasma_id = api.register_material({
    "name": "Plasma",
    "color": (180, 60, 255),
    "type": "gas",
    "density": 2,
    "thermal_conductivity": 0.8,
    "thermal_capacity": 0.3,
    "initial_temp": 4000.0,
    "viscosity": 0.02,
    "burn_rate": 0.0,
    "smoke_factor": 0.0,
    "ignition_temp": 9999,
    "auto_ignite_temp": 9999,
    "min_oxygen_for_ignition": 1.0,
    "spark_sensitivity": 0.0,
    "latent_heat": 0.0,
    "inertia": 0.01,
    "repose_angle": 0,
    "dispersion": 6,
    "drag": 0.01,
})
api.notify(f"Plasma → id {plasma_id}", 4)
