"""lightning.py  —  actual lightning bolts with a plasma flash.

Registers a temporary 'Plasma' material for the bright bolt visual.
Each strike:
  - zig-zags from the top row to a random depth
  - draws a glowing plasma trail (bright white-blue)
  - superheats everything it touches
  - ignites flammable materials at the impact zone
  - erases the bolt after ~6 ticks, leaving scorch behind

Press L to trigger an immediate strike.
"""

# ── register Plasma material for the bolt visual (once) ──────────────────────
if api.material_id("plasma") is None:
    _plasma_id = api.register_material({
        "name":    "Plasma",
        "color":   (200, 230, 255),   # cold white-blue
        "type":    "gas",
        "density": 1,
        "dispersion": 0,
        "drag":    0.0,
        "inertia": 0.0,
        "repose_angle": 0,
        "smoke_factor": 0.0,
        "burn_rate":    0.0,
        "ignition_temp": 9999,
        "auto_ignite_temp": 9999,
    })
else:
    _plasma_id = api.material_id("plasma")

# Always ensure smoke is off (handles reloads where material already exists)
api.reload_material(_plasma_id, {"smoke_factor": 0.0, "burn_rate": 0.0})

_INTERVAL   = 240   # ticks between auto-strikes (~4 s)
_BOLT_HEAT  = 3500  # °C deposited along the bolt
_ZONE_HEAT  = 5000  # °C at the impact zone
_FLASH_LIFE = 6     # ticks the plasma bolt stays visible

def _strike(api):
    rows, cols = api.rows, api.cols
    fire_id    = api.material_id("fire")
    plasma_id  = _plasma_id

    # pick a target column and depth
    cx    = random.randint(3, cols - 4)
    depth = random.randint(rows // 2, rows - 3)

    # build zig-zag bolt path
    bolt, c = [], cx
    for r in range(depth):
        bolt.append((r, c))
        c = max(0, min(cols - 1, c + random.randint(-2, 2)))

    # draw plasma bolt + superheat everything touched
    for r, bc in bolt:
        current = api.get(r, bc)
        if current == 0:
            api.set(r, bc, plasma_id)
        api.heat(r, bc, _BOLT_HEAT)

    # impact zone — wide superheat + ignite flammables
    land_r, land_c = bolt[-1]
    for r in range(max(0, land_r - 2), min(rows, land_r + 3)):
        for c in range(max(0, land_c - 4), min(cols, land_c + 5)):
            api.heat(r, c, _ZONE_HEAT)
            mat = api.get(r, c)
            if mat != 0 and mat != plasma_id:
                t = api.temp(r, c)
                mat_info = api.materials().get(mat, {})
                ignition = mat_info.get("ignition_temp", 9999)
                if t >= ignition * 0.85:
                    api.set(r, c, fire_id)

    api.notify("⚡", 0.8)

    # erase plasma bolt after _FLASH_LIFE ticks
    bolt_copy = list(bolt)
    def _erase(a):
        for r, bc in bolt_copy:
            if a.get(r, bc) == plasma_id:
                a.set(r, bc, 0)
    api.after(_FLASH_LIFE, _erase)

def _tick(tick, api):
    if tick % _INTERVAL == 0:
        _strike(api)

api.on_tick(_tick)
api.on_key("l", lambda a: _strike(a))
api.notify("⚡ lightning.py  (L = strike now)", 3)

