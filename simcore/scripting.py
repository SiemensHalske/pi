from .world import *


# ---------------------------------------------------------------------------
# Scripting Engine  (Phase 1 – Steps 1-10 of the scripting plan)
# ---------------------------------------------------------------------------

_API_REFERENCE = """\
=== PowderSim Script API ===
GRID
  api.get(r,c)                 material id
  api.set(r,c, id)             place material
  api.fill(r1,c1,r2,c2, id)   fill rectangle
  api.circle(r,c,radius, id)  fill circle
  api.replace(src,dst)         replace every cell of type src
  api.clear()                  erase whole grid
  api.count(id)                count cells of that type
  api.find(id)                 list of (row,col) up to 2000
  api.query(r,c)               rich cell info dict

PHYSICS STATE
  api.temp(r,c)                °C
  api.set_temp(r,c, t)
  api.fill_temp(r1,c1,r2,c2,t)
  api.heat(r,c, delta)
  api.burn_stage(r,c)          0-4
  api.oxygen(r,c)              0-1
  api.set_oxygen(r,c, v)

MATERIALS
  api.material_id(name)        int or None
  api.material_name(id)        str
  api.materials()              dict copy
  api.register_material(spec)  add new live material (id >= 100)
  api.reload_material(id,patch) live-patch properties
  api.interaction(spec)        add reaction rule
  api.interactions()           list current rules

SIMULATION
  api.pause() / api.resume() / api.is_paused()
  api.step(n=1)                advance n ticks
  api.fps()                    current FPS
  api.tick()                   current tick counter
  api.set_profile(name)        e.g. "realistic"
  api.profile()                current profile name

HOOKS
  api.on_tick(fn)              fn(tick, api) every physics tick
  api.on_event(type, fn)       fn(event, api) for named physics events
  api.on_key(key, fn)          fn(api) on pygame key name (e.g. "k")
  api.after(n, fn)             fn(api) deferred n ticks
  api.every(n, fn, count=None) repeating callback, optional repeat count
  api.cancel(fn)               cancel any hook/callback by reference
  api.clear_hooks()            unregister all hooks of current exec context

OUTPUT
  api.print(*a)                append to console
  api.notify(msg, secs=3)      HUD message
  api.log(msg)                 scripts/script.log
  api.ls()                     list scripts/ dir
  api.load(name)               load a .py file from scripts/
  api.unload(name)             unload a loaded script
  api.loaded()                 list of loaded script names
  api.autoload(name)           add to .autoload
  api.no_autoload(name)        remove from .autoload
  api.save_data(key, val)      persist JSON data
  api.load_data(key)           load persisted data (or None)
  api.query(r,c)               rich cell info dict

CONSOLE COMMANDS
  .clear        erase console output
  .load <file>  load a script from scripts/  (e.g. .load lightning.py)
  .unload <name> unload a running script
  .reload       reload all loaded scripts from disk
  .ls           list scripts/
  .help         print this reference
  .bench N      run N ticks and time them
"""


class ScriptSecurityError(Exception):
    pass


class ScriptAPI:
    """Public surface exposed to every script. Never hand out self._app directly."""

    def __init__(self, app, engine: "ScriptEngine"):
        self._app = app
        self._engine = engine

    # ------------------------------------------------------------------ grid
    @property
    def _grid(self):
        return self._app.sim.grid

    @property
    def _physics(self):
        return self._app.sim.physics

    def _in(self, r, c):
        return 0 <= r < self._app.sim.rows and 0 <= c < self._app.sim.cols

    @property
    def rows(self):
        return self._app.sim.rows

    @property
    def cols(self):
        return self._app.sim.cols

    def get(self, row, col):
        """Return the material ID at (row, col), or 0 if out-of-bounds."""
        if self._in(row, col):
            return self._grid[row][col]
        return 0

    def set(self, row, col, mat_id):
        """Place *mat_id* at (row, col) and fire any registered cell-change hooks.
        Silently ignores out-of-bounds calls; raises ValueError for unknown mat_id."""
        if not self._in(row, col):
            return
        if mat_id not in MATERIALS:
            raise ValueError(f"Unknown material id {mat_id}")
        old_mat = self._grid[row][col]   # capture BEFORE writing
        self._grid[row][col] = mat_id
        for fn, _ns in self._engine.hooks["cell"].get((row, col), []):
            try:
                fn(old_mat, mat_id, self)
            except Exception:
                self._engine._push_exc(f"on_cell_change:({row},{col})")

    def fill(self, r1, c1, r2, c2, mat_id):
        """Fill the rectangle [r1,c1]-[r2,c2] (inclusive) with *mat_id*.
        Coordinates are clamped to grid bounds. Does NOT fire cell-change hooks."""
        r1, r2 = max(0, min(r1, r2)), min(self._app.sim.rows - 1, max(r1, r2))
        c1, c2 = max(0, min(c1, c2)), min(self._app.sim.cols - 1, max(c1, c2))
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self._grid[r][c] = mat_id

    def circle(self, row, col, radius, mat_id):
        """Fill a circle of *radius* cells centred on (row, col) with *mat_id*."""
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    self.set(row + dr, col + dc, mat_id)

    def replace(self, src_id, dst_id):
        """Replace every cell containing *src_id* with *dst_id* across the whole grid."""
        for r in range(self._app.sim.rows):
            for c in range(self._app.sim.cols):
                if self._grid[r][c] == src_id:
                    self._grid[r][c] = dst_id

    def clear(self):
        """Erase the entire grid (set every cell to 0/Eraser)."""
        self.fill(0, 0, self._app.sim.rows - 1, self._app.sim.cols - 1, 0)

    def count(self, mat_id):
        """Return the number of cells containing *mat_id*."""
        return sum(self._grid[r][c] == mat_id
                   for r in range(self._app.sim.rows)
                   for c in range(self._app.sim.cols))

    def find(self, mat_id):
        """Return a list of (row, col) for every cell matching *mat_id* (int or name str).
        Capped at 2000 results; a console message is printed when truncated."""
        # Accept name strings as well as int ids
        if isinstance(mat_id, str):
            mat_id = self.material_id(mat_id)
            if mat_id is None:
                return []
        results = []
        for r in range(self._app.sim.rows):
            for c in range(self._app.sim.cols):
                if self._grid[r][c] == mat_id:
                    results.append((r, c))
                    if len(results) >= 2000:
                        self.print(f"find: truncated at 2000 results")
                        return results
        return results

    def query(self, row, col):
        """Return a dict with all available physics state for cell (row, col).
        Keys: mat, name, temp, burn_stage, burn_progress, oxygen, smoke_density,
        and optionally moisture, fire_lifetime, smoke_lifetime.
        Returns {} for out-of-bounds coordinates."""
        if not self._in(row, col):
            return {}
        p = self._physics
        mat = self._grid[row][col]
        d = {
            "mat": mat,
            "name": MATERIALS.get(mat, {}).get("name", "?"),
            "temp": p.temperature[row][col] if len(p.temperature) > row else 0.0,
            "burn_stage": p.burn_stage[row][col] if len(p.burn_stage) > row else 0,
            "burn_progress": p.burn_progress[row][col] if len(p.burn_progress) > row else 0.0,
            "oxygen": p.oxygen_level[row][col] if len(p.oxygen_level) > row else 0.0,
            "smoke_density": p.smoke_density[row][col] if len(p.smoke_density) > row else 0.0,
        }
        if hasattr(p, "moisture") and len(p.moisture) > row:
            d["moisture"] = p.moisture[row][col]
        if hasattr(p, "fire_lifetime") and len(p.fire_lifetime) > row:
            d["fire_lifetime"] = p.fire_lifetime[row][col]
        if hasattr(p, "smoke_lifetime") and len(p.smoke_lifetime) > row:
            d["smoke_lifetime"] = p.smoke_lifetime[row][col]
        return d

    def cells_in_rect(self, r1, c1, r2, c2):
        """Yield (row, col, mat_id) for every cell in [r1,c1]-[r2,c2].
        Coordinates are clamped to grid bounds.  Useful for scanning regions
        without building a temporary list."""
        r1, r2 = max(0, min(r1, r2)), min(self._app.sim.rows - 1, max(r1, r2))
        c1, c2 = max(0, min(c1, c2)), min(self._app.sim.cols - 1, max(c1, c2))
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                yield r, c, self._grid[r][c]

    # ------------------------------------------------------- physics state
    def temp(self, row, col):
        """Return temperature in °C at (row, col); returns 0.0 if out-of-bounds."""
        p = self._physics
        if self._in(row, col) and len(p.temperature) > row:
            return p.temperature[row][col]
        return 0.0

    def set_temp(self, row, col, t):
        """Set temperature at (row, col) clamped to [-273, 5000] °C."""
        p = self._physics
        if self._in(row, col) and len(p.temperature) > row:
            p.temperature[row][col] = max(-273.0, min(5000.0, float(t)))

    def fill_temp(self, r1, c1, r2, c2, t):
        """Set temperature to *t* for every cell in [r1,c1]-[r2,c2]."""
        r1, r2 = max(0, min(r1, r2)), min(self._app.sim.rows - 1, max(r1, r2))
        c1, c2 = max(0, min(c1, c2)), min(self._app.sim.cols - 1, max(c1, c2))
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.set_temp(r, c, t)

    def heat(self, row, col, delta):
        """Add *delta* degrees to the temperature of cell (row, col)."""
        self.set_temp(row, col, self.temp(row, col) + delta)

    def burn_stage(self, row, col):
        """Return the burn stage (0–4) of cell (row, col)."""
        p = self._physics
        if self._in(row, col) and len(p.burn_stage) > row:
            return p.burn_stage[row][col]
        return 0

    def oxygen(self, row, col):
        """Return the oxygen level (0.0–1.0) at (row, col)."""
        p = self._physics
        if self._in(row, col) and len(p.oxygen_level) > row:
            return p.oxygen_level[row][col]
        return 0.0

    def set_oxygen(self, row, col, v):
        """Set the oxygen level at (row, col), clamped to [0, 1]."""
        p = self._physics
        if self._in(row, col) and len(p.oxygen_level) > row:
            p.oxygen_level[row][col] = max(0.0, min(1.0, float(v)))

    # ----------------------------------------------------- material registry
    def material_id(self, name):
        return MATERIAL_IDS.get(name.strip().lower())

    def material_name(self, mid):
        return MATERIALS.get(mid, {}).get("name", "?")

    def materials(self):
        return {k: dict(v) for k, v in MATERIALS.items()}

    def register_material(self, spec: dict) -> int:
        for key in ("name", "color", "type"):
            if key not in spec:
                raise ValueError(f"register_material: missing required key '{key}'")
        name_key = spec["name"].strip().lower()
        if name_key in MATERIAL_IDS:
            raise ValueError(f"Material '{spec['name']}' already exists (id {MATERIAL_IDS[name_key]})")
        # Runtime materials use ids 100+
        new_id = max((k for k in MATERIALS if k >= 100), default=99) + 1
        merged = dict(MATERIAL_DEFAULTS)
        merged.update(spec)
        MATERIALS[new_id] = merged
        MATERIAL_IDS[name_key] = new_id
        self._engine._push("system", f"[MAT+] '{spec['name']}' → id {new_id}")
        return new_id

    def reload_material(self, id_or_name, patch: dict):
        mid = id_or_name if isinstance(id_or_name, int) else self.material_id(str(id_or_name))
        if mid is None or mid not in MATERIALS:
            raise ValueError(f"Unknown material: {id_or_name}")
        MATERIALS[mid].update(patch)
        self._engine._push("system", f"[MAT~] id {mid} patched: {list(patch.keys())}")

    def unregister_material(self, id_or_name):
        """Remove a runtime material (id >= 100). Clears all its cells from the grid."""
        mid = id_or_name if isinstance(id_or_name, int) else self.material_id(str(id_or_name))
        if mid is None or mid not in MATERIALS:
            raise ValueError(f"Unknown material: {id_or_name}")
        if mid < 100:
            raise ValueError(f"Cannot unregister built-in material id {mid} (only runtime ids ≥ 100)")
        name = MATERIALS[mid].get("name", str(mid))
        # Wipe all cells of that type
        grid = self._app.sim.grid
        for r in range(self._app.sim.rows):
            for c in range(self._app.sim.cols):
                if grid[r][c] == mid:
                    grid[r][c] = 0
        # Remove from registries
        MATERIAL_IDS.pop(name.strip().lower(), None)
        del MATERIALS[mid]
        # Remove any custom renderer for this material
        self._engine._custom_renderers.pop(mid, None)
        # Rebuild menu so the button disappears
        self._app.menu._build_layout()
        self._engine._push("system", f"[MAT-] '{name}' (id {mid}) unregistered")

    def interaction(self, spec: dict):
        self._physics.interaction_table.add_rule(spec)
        self._engine._push("system", f"[RULE+] {spec.get('pair', '?')}")

    def remove_interaction(self, pair):
        """Remove all interaction rules for a pair (names or ids).
        pair can be a list/tuple of two material names or two material ids."""
        resolved = []
        for item in pair:
            if isinstance(item, str):
                mid = self.material_id(item)
                if mid is None:
                    raise ValueError(f"Unknown material: {item!r}")
                resolved.append(mid)
            else:
                resolved.append(int(item))
        removed = self._physics.interaction_table.remove_rule(resolved)
        self._engine._push("system", f"[RULE-] pair {pair!r} → {removed} rule(s) removed")
        return removed

    def interactions(self):
        return list(self._physics.interaction_table.rules)

    def set_renderer(self, mat_id, fn):
        """Register a custom draw fn for mat_id.
        fn(screen, sx, sy, cell_px, row, col, api) → None
        Pass fn=None to unregister."""
        if fn is None:
            self._engine._custom_renderers.pop(mat_id, None)
        else:
            self._engine._custom_renderers[mat_id] = fn

    def remove_renderer(self, mat_id):
        """Unregister any custom renderer for mat_id."""
        self._engine._custom_renderers.pop(mat_id, None)

    def add_physics_stage(self, fn, priority: int = 50):
        """Inject a custom physics stage called each tick.
        fn(grid, rows, cols, tick, rng, api) → None
        Max 4 active custom stages; raises ValueError when full."""
        if len(self._engine._custom_stages) >= 4:
            raise ValueError("Maximum of 4 custom physics stages already registered")
        self._engine._custom_stages.append((priority, fn))
        self._engine._custom_stages.sort(key=lambda e: e[0])
        self._engine._push("system", f"[STAGE+] priority={priority} total={len(self._engine._custom_stages)}")

    def remove_physics_stage(self, fn):
        """Unregister a custom physics stage by function reference."""
        before = len(self._engine._custom_stages)
        self._engine._custom_stages = [(p, f) for p, f in self._engine._custom_stages if f is not fn]
        removed = before - len(self._engine._custom_stages)
        if removed:
            self._engine._push("system", f"[STAGE-] removed {removed} stage(s)")

    # --------------------------------------------------- simulation control
    def pause(self):
        """Pause the simulation (physics ticks stop)."""
        self._app.paused = True

    def resume(self):
        """Resume the simulation after a pause."""
        self._app.paused = False

    def is_paused(self):
        """Return True if the simulation is currently paused."""
        return self._app.paused

    def step(self, n=1):
        """Advance the simulation by *n* physics ticks (min 1, max 1000).
        Useful for running the sim forward from a paused state."""
        for _ in range(max(1, min(1000, int(n)))):
            self._app.sim.update_physics()

    def fps(self):
        """Return the current measured frames-per-second as a float."""
        return self._app.clock.get_fps()

    def tick(self):
        """Return the current physics tick counter (increments every update)."""
        return self._app.sim.tick_index

    def set_profile(self, name):
        """Switch to a named physics profile (e.g. 'realistic', 'fast')."""
        self._app.sim.apply_profile(name)
        self._app.sim.profile_name = name

    def profile(self):
        """Return the name of the currently active physics profile."""
        return self._app.sim.profile_name

    def fps_limit(self, n):
        """Override the engine frame-rate cap to *n* FPS for this session.
        Pass 0 or None to restore the default FPS constant."""
        self._app._fps_limit_override = max(1, int(n)) if n else None

    # ---------------------------------------------------------------- hooks
    def on_tick(self, fn):
        """Register fn(tick, api) to be called every physics tick."""
        self._engine.hooks["tick"].append(fn)

    def on_event(self, event_type: str, fn):
        """Register fn(event, api) for physics events matching *event_type*.
        Use '*' to receive all events."""
        self._engine.hooks["event"].append((event_type, fn))

    def on_key(self, key_name: str, fn):
        """Register fn(api) to fire when the pygame key named *key_name* is pressed."""
        self._engine.hooks["key"].append((key_name.lower(), fn))

    def on_cell_change(self, row, col, fn):
        """Register fn(old_mat_id, new_mat_id, api) to fire when cell (row, col) changes."""
        # Store as (fn, None) so all cell-hook consumers can unpack uniformly
        self._engine.hooks["cell"].setdefault((row, col), []).append((fn, None))

    def remove_hook(self, fn):
        """Unregister a hook or callback by function reference (any category)."""
        self._engine.hooks["tick"] = [f for f in self._engine.hooks["tick"] if f is not fn]
        self._engine.hooks["event"] = [(t, f) for t, f in self._engine.hooks["event"] if f is not fn]
        self._engine.hooks["key"] = [(k, f) for k, f in self._engine.hooks["key"] if f is not fn]
        for key in list(self._engine.hooks["cell"]):
            self._engine.hooks["cell"][key] = [(f, ns) for f, ns in self._engine.hooks["cell"][key] if f is not fn]

    def clear_hooks(self):
        """Unregister ALL hooks and pending callbacks from this script context."""
        self._engine.hooks["tick"].clear()
        self._engine.hooks["event"].clear()
        self._engine.hooks["key"].clear()
        self._engine.hooks["cell"].clear()
        self._engine._deferred.clear()
        self._engine._repeating.clear()

    def after(self, ticks: int, fn):
        """Schedule fn(api) to run once after *ticks* physics ticks (minimum 0 = next tick)."""
        self._engine._deferred.append((self._app.sim.tick_index + max(0, int(ticks)), fn))

    def every(self, ticks: int, fn, count=None):
        """Repeat fn(api) every *ticks* ticks. *count* limits repetitions (None = infinite).
        count=0 is a no-op. ticks is clamped to a minimum of 1."""
        if count is not None and int(count) <= 0:
            return  # nothing to schedule
        interval = max(1, int(ticks))
        self._engine._repeating.append([self._app.sim.tick_index + interval, interval, fn, count])

    def cancel(self, fn):
        """Cancel any pending after()/every() callback by function reference."""
        self._engine._deferred = [(t, f) for t, f in self._engine._deferred if f is not fn]
        self._engine._repeating = [e for e in self._engine._repeating if e[2] is not fn]

    # -------------------------------------------------------------- output
    def print(self, *args):
        """Append a line to the scripting console output buffer."""
        self._engine._push("output", " ".join(str(a) for a in args))

    def notify(self, msg, duration=3.0):
        """Show a HUD notification message for *duration* seconds (displayed in the top bar)."""
        self._app._update_action_message(str(msg))

    def screenshot(self, path=None):
        """Save a PNG screenshot of the current frame.
        *path* defaults to screenshots/shot_YYYYMMDD_HHMMSS.png.
        Returns the file path string on success, or None on failure."""
        shots_dir = pathlib.Path(path).parent if path else pathlib.Path("screenshots")
        shots_dir.mkdir(parents=True, exist_ok=True)
        fname = str(path) if path else str(shots_dir / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png")
        try:
            pygame.image.save(self._app.screen, fname)
            self._engine._push("system", f"[screenshot] {fname}")
            return fname
        except Exception as e:
            self._engine._push("error", f"[screenshot] {e}")
            return None

    def log(self, msg):
        """Append a timestamped line to scripts/script.log."""
        self._engine._log_event(str(msg))

    def ls(self):
        """Print a list of all .py files in the scripts/ directory to the console."""
        sd = pathlib.Path("scripts")
        files = sorted(p.name for p in sd.glob("*.py")) if sd.exists() else []
        self._engine._push("system", "scripts/: " + (", ".join(files) if files else "(empty)"))

    def load(self, name: str):
        """Load a script by filename from the scripts/ directory (hot-start)."""
        self._engine.load_script(pathlib.Path("scripts") / name)

    def unload(self, name: str):
        """Unload a currently loaded script by name, calling its on_unload() if present."""
        self._engine.unload_script(name)

    def loaded(self):
        """Return a list of names of all currently loaded scripts (excludes console namespace)."""
        return [k for k in self._engine.loaded_scripts if k != "__console__"]

    def autoload(self, name: str):
        """Add *name* to scripts/.autoload so it loads automatically on next startup."""
        p = pathlib.Path("scripts") / ".autoload"
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        if name not in lines:
            lines.append(name)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def no_autoload(self, name: str):
        """Remove *name* from scripts/.autoload."""
        p = pathlib.Path("scripts") / ".autoload"
        if not p.exists():
            return
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l != name]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_data(self, key: str, value):
        """Persist *value* (must be JSON-serializable) under *key* in scripts/.scriptdata.json.
        Raises ValueError if the value cannot be serialized."""
        # Validate serializability before touching disk
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"save_data: value for '{key}' is not JSON-serializable: {e}") from e
        p = pathlib.Path("scripts") / ".scriptdata.json"
        data = {}
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            pass
        data[key] = value
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_data(self, key: str):
        """Return the value stored under *key* in scripts/.scriptdata.json, or None if absent."""
        p = pathlib.Path("scripts") / ".scriptdata.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            return data.get(key)
        except Exception:
            return None


# ---------------------------------------------------------------------------
_EXAMPLE_LAVA_RAIN = '''\
"""lava_rain.py  —  rains lava from the top row every 30 ticks."""
import random as _r

def _rain(tick, api):
    if tick % 30 == 0:
        c = _r.randint(0, api.cols - 1)
        api.set(0, c, api.material_id("lava"))

api.on_tick(_rain)
api.notify("Lava rain active!", 3)
'''

_EXAMPLE_PLASMA = '''\
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
'''

_EXAMPLE_EXTINGUISH = '''\
"""auto_extinguish.py  —  quenches any cell that reaches stage 3 burn."""

def _quench(event, api):
    r, c = event.get("row", -1), event.get("col", -1)
    if r >= 0:
        api.set_temp(r, c, 20.0)

api.on_event("ignition", _quench)
api.notify("Auto-extinguish active", 3)
'''

_EXAMPLE_HEATMAP = '''\
"""heatmap_overlay.py  —  custom per-cell color renderer using temperature.
Each cell glows from black (cold) to bright orange/white (hot)."""

_lava_id = api.material_id("lava")
_fire_id = api.material_id("fire")
_SKIP_IDS = {_lava_id, _fire_id}

def _heat_color(temp):
    t = max(0.0, min(1.0, temp / 800.0))
    if t < 0.33:
        return (int(t * 3 * 180), 0, 0)
    elif t < 0.66:
        s = (t - 0.33) / 0.33
        return (180 + int(s * 75), int(s * 120), 0)
    else:
        s = (t - 0.66) / 0.34
        return (255, 120 + int(s * 135), int(s * 200))

def _renderer(screen, sx, sy, cell_px, row, col, api):
    import pygame as _pg
    color = _heat_color(api.temp(row, col))
    if any(c > 5 for c in color):
        _pg.draw.rect(screen, color, (sx, sy, cell_px, cell_px))

for _mid in api.materials():
    if _mid not in _SKIP_IDS and not api.materials()[_mid].get("internal"):
        api.set_renderer(_mid, _renderer)

def on_unload():
    for _mid in list(api.materials()):
        api.remove_renderer(_mid)
    api.print("[heatmap] renderers removed")

api.notify("Heatmap overlay active", 3)
'''


class ScriptEngine:
    """Manages the in-game Python REPL, script files, and hook dispatch."""

    def __init__(self, app: "Engine"):
        self._app = app
        self.api = ScriptAPI(app, self)
        self.console_lines: list[tuple[str, str]] = []   # (text, kind)
        self.history: list[str] = []
        self.history_pos: int = -1
        self.input_buf: str = ""
        self.visible: bool = False
        self.hooks: dict = {
            "tick": [],          # list[Callable]
            "event": [],         # list[(type_str, Callable)]
            "key": [],           # list[(key_str, Callable)]
            "cell": {},          # dict[(r,c) → list[Callable]]
        }
        self._deferred: list = []    # [(fire_at_tick, fn)]
        self._repeating: list = []   # [[fire_at, interval, fn, count_left]]
        self.loaded_scripts: dict[str, dict] = {}   # name → globals dict
        self.script_mtimes: dict[str, float] = {}
        self._sandbox_globals: dict = self._make_sandbox()
        self._multiline_mode: bool = False
        self._pending_lines: list[str] = []
        self._console_scroll: int = 0
        self._console_font: pygame.font.Font | None = None
        self._console_font_size: int = CONSOLE_FONT_SIZE  # adjustable via Ctrl+=/Ctrl+-
        self._console_font_size_built: int = 0               # size the current font was built at
        self._hook_ms: float = 0.0
        self._budget_warned: bool = False      # True while budget is being exceeded
        self._budget_warn_at: int = -9999      # tick when last budget warning was pushed
        self._load_prompt: bool = False   # True when console is waiting for a filename
        self._custom_renderers: dict = {}  # mat_id → fn(screen,sx,sy,cell_px,row,col,api)
        self._custom_stages: list = []     # [(priority, fn)], max 4, sorted asc by priority
        self._editor_proc = None           # subprocess.Popen for Ctrl+E editor
        self._editor_scratch: pathlib.Path | None = None  # path being edited

        # Bootstrap scripts/ directory
        sd = pathlib.Path("scripts")
        sd.mkdir(exist_ok=True)
        for fname, src in [
            ("lava_rain.py", _EXAMPLE_LAVA_RAIN),
            ("plasma.py", _EXAMPLE_PLASMA),
            ("auto_extinguish.py", _EXAMPLE_EXTINGUISH),
            ("heatmap_overlay.py", _EXAMPLE_HEATMAP),
        ]:
            p = sd / fname
            if not p.exists():
                p.write_text(src, encoding="utf-8")

        # Auto-load
        al = sd / ".autoload"
        if al.exists():
            for name in al.read_text(encoding="utf-8").splitlines():
                name = name.strip()
                if name:
                    try:
                        self.load_script(sd / name)
                    except Exception as e:
                        self._push("error", f"[autoload] {name}: {e}")
                        log.error("scripting", f"Autoload failed: {name}", error=str(e))

    # ---------------------------------------------------------------- sandbox
    def _make_sandbox(self) -> dict:
        safe_builtins_names = [
            "print", "len", "range", "int", "float", "str", "list", "dict", "set",
            "tuple", "abs", "min", "max", "round", "sum", "sorted", "enumerate", "zip",
            "map", "filter", "isinstance", "type", "repr", "bool", "chr", "ord", "hex",
            "bin", "oct", "divmod", "pow", "any", "all", "hasattr", "getattr", "setattr",
            "delattr", "callable", "id", "hash", "iter", "next", "reversed", "slice",
            "NotImplemented", "True", "False", "None",
            "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
            "AttributeError", "RuntimeError", "StopIteration", "NameError",
            "ZeroDivisionError", "OverflowError", "ArithmeticError",
        ]
        import builtins as _builtins_mod
        safe_builtins = {k: getattr(_builtins_mod, k) for k in safe_builtins_names if hasattr(_builtins_mod, k)}

        # Override print to redirect to console
        def _safe_print(*args, **kwargs):
            self._push("output", " ".join(str(a) for a in args))
        safe_builtins["print"] = _safe_print

        # help() — prints API reference or falls back to pydoc for objects
        def _console_help(obj=None):
            if obj is None:
                self._push("system", _API_REFERENCE)
            else:
                _buf = io.StringIO()
                import pydoc as _pydoc
                _pydoc.Helper(output=_buf).help(obj)
                for _ln in _buf.getvalue().splitlines():
                    self._push("output", _ln)
        safe_builtins["help"] = _console_help

        g = {
            "__builtins__": safe_builtins,
            "api": self.api,
            "MATERIALS": MATERIALS,
            "MATERIAL_IDS": MATERIAL_IDS,
            "math": math,
            "random": random,
            "json": json,
            "re": re,
            "copy": copy,
            "itertools": itertools,
            "collections": collections,
            "time": time,
        }
        # Provide a sandboxed __import__ so scripts can use normal `import` syntax
        # for the modules already injected above.  Everything else is blocked.
        _allowed = {k: g[k] for k in ("math", "random", "json", "re",
                                       "copy", "itertools", "collections", "time")}
        def _safe_import(name, *args, **kwargs):
            root = name.split(".")[0]
            if root in _allowed:
                return _allowed[root]
            raise ImportError(f"import '{name}' is not allowed in scripts")
        safe_builtins["__import__"] = _safe_import
        return g

    # ---------------------------------------------------- AST safety check
    def _ast_check(self, src: str, name: str):
        BLOCKED_IMPORTS = {"os", "sys", "subprocess", "socket", "ctypes",
                           "importlib", "builtins", "_thread", "threading",
                           "multiprocessing", "shutil", "pathlib", "pickle"}
        BLOCKED_CALLS = {"__import__", "eval", "exec", "open", "compile",
                         "breakpoint", "input"}
        BLOCKED_ATTRS = {"__class__", "__bases__", "__subclasses__", "__globals__",
                         "__dict__", "__code__", "__closure__"}
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise SyntaxError(str(e)) from e
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""])
                for m in mods:
                    root = m.split(".")[0]
                    if root in BLOCKED_IMPORTS:
                        raise ScriptSecurityError(
                            f"line {node.lineno}: import '{m}' is not allowed")
            elif isinstance(node, ast.Call):
                fn = node.func
                fname = (fn.id if isinstance(fn, ast.Name) else
                         fn.attr if isinstance(fn, ast.Attribute) else None)
                if fname in BLOCKED_CALLS:
                    raise ScriptSecurityError(
                        f"line {node.lineno}: call to '{fname}' is not allowed")
            elif isinstance(node, ast.Attribute):
                if node.attr in BLOCKED_ATTRS:
                    raise ScriptSecurityError(
                        f"line {node.lineno}: access to '{node.attr}' is not allowed")

    # -------------------------------------------------------- script loading
    def load_script(self, path):
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        src = path.read_text(encoding="utf-8")
        self._ast_check(src, path.name)
        globs = dict(self._sandbox_globals)
        globs["__name__"] = path.stem
        try:
            exec(compile(src, str(path), "exec"), globs)  # noqa: S102
        except Exception as e:
            raise RuntimeError(f"{path.name}: {e}") from e
        self.loaded_scripts[path.name] = globs
        self.script_mtimes[path.name] = path.stat().st_mtime
        self._push("system", f"[LOAD] {path.name}")
        self._log_event(f"LOAD {path.name}")
        log.success("scripting", f"Loaded {path.name}")

    def unload_script(self, name: str):
        globs = self.loaded_scripts.get(name)
        if globs is None:
            return
        # Call on_unload if defined
        if "on_unload" in globs and callable(globs["on_unload"]):
            try:
                globs["on_unload"]()
            except Exception:
                pass
        # Remove all hooks belonging to this script
        def _from_globs(fn):
            return getattr(fn, "__globals__", None) is globs
        self.hooks["tick"] = [f for f in self.hooks["tick"] if not _from_globs(f)]
        self.hooks["event"] = [(t, f) for t, f in self.hooks["event"] if not _from_globs(f)]
        self.hooks["key"] = [(k, f) for k, f in self.hooks["key"] if not _from_globs(f)]
        for key in list(self.hooks["cell"]):
            self.hooks["cell"][key] = [e for e in self.hooks["cell"][key] if not _from_globs(e[0] if isinstance(e, tuple) else e)]
        self._deferred = [(t, f) for t, f in self._deferred if not _from_globs(f)]
        self._repeating = [e for e in self._repeating if not _from_globs(e[2])]
        del self.loaded_scripts[name]
        del self.script_mtimes[name]
        self._push("system", f"[UNLOAD] {name}")
        self._log_event(f"UNLOAD {name}")
        log.info("scripting", f"Unloaded {name}")

    def _launch_editor(self):
        """Open scripts/scratch.py in $EDITOR (or nano). Auto-reloads on exit."""
        scratch = pathlib.Path("scripts") / "scratch.py"
        if not scratch.exists():
            scratch.write_text(
                '"""scratch.py \u2014 experimental scratch pad.\n'
                'Edit here, Ctrl+E to reopen, auto-reloads when editor closes.\n'
                '"""\n\n# Type your script here\napi.notify("scratch.py loaded", 3)\n',
                encoding="utf-8")
        editor = os.environ.get("EDITOR", "nano")
        try:
            self._editor_proc = subprocess.Popen([editor, str(scratch)])
            self._editor_scratch = scratch
            self._push("system", f"[editor] {scratch.name} opened in {editor} \u2014 reloads on save+exit")
        except Exception as e:
            self._push("error", f"[editor] failed to launch '{editor}': {e}")

    # ------------------------------------------------------------------ logging
    def _log_event(self, msg: str) -> None:
        """Append a timestamped line to scripts/script.log."""
        try:
            with open(pathlib.Path("scripts") / "script.log", "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except OSError:
            pass

    def reload_all_scripts(self):
        """Force-reload every currently loaded script from disk."""
        for name in list(self.script_mtimes):
            p = pathlib.Path("scripts") / name
            try:
                self.unload_script(name)
                self.load_script(p)
            except Exception as e:
                self._push("error", f"[reload] {name}: {e}")
        count = len(self.script_mtimes)
        self._push("system", f"[F12] Reloaded {count} script(s)" if count else "[F12] No scripts loaded")

    def dispatch_physics_stages(self, sim):
        """Call each custom physics stage registered via api.add_physics_stage()."""
        if not self._custom_stages:
            return
        rng = sim.physics.random_manager.for_tick(sim.tick_index, "script_stage")
        bad = []
        for priority, fn in self._custom_stages:
            try:
                fn(sim.grid, sim.rows, sim.cols, sim.tick_index, rng, self.api)
            except Exception:
                self._push_exc("physics_stage")
                bad.append(fn)
        if bad:
            self._custom_stages = [(p, f) for p, f in self._custom_stages if f not in bad]
            self._push("system", f"[STAGE] {len(bad)} stage(s) unregistered due to errors")

    def poll_reload(self):
        """Call once per frame to hot-reload changed scripts."""
        # Check if external editor closed — if so, reload the file it was editing
        if self._editor_proc is not None and self._editor_proc.poll() is not None:
            p = self._editor_scratch
            self._editor_proc = None
            if p and p.exists():
                name = p.name
                if name in self.script_mtimes:
                    self.unload_script(name)
                try:
                    self.load_script(p)
                except Exception as e:
                    self._push("error", f"[editor] reload {name}: {e}")
        for name in list(self.script_mtimes):
            p = pathlib.Path("scripts") / name
            try:
                if p.exists() and p.stat().st_mtime > self.script_mtimes[name]:
                    self.unload_script(name)
                    self.load_script(p)
                    self._app._update_action_message(f"Reloaded: {name}")
            except Exception as e:
                self._push("error", f"[reload] {name}: {e}")

    # -------------------------------------------------------- hook dispatch
    _HOOK_BUDGET_MS = 8.0          # max ms per frame for all hook work
    _BUDGET_WARN_INTERVAL = 120    # minimum ticks between repeated budget warnings

    def _budget_elapsed(self, t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def dispatch_tick(self, tick: int):
        t0 = time.perf_counter()
        budget = self._HOOK_BUDGET_MS
        over_budget = False

        # ---- Deferred one-shots ----------------------------------------
        fired = []
        for i, (fire_at, fn) in enumerate(self._deferred):
            if tick < fire_at:
                continue
            if self._budget_elapsed(t0) >= budget:
                over_budget = True
                break
            fired.append(i)
            try:
                fn(self.api)
            except Exception:
                self._push_exc("deferred")
        for i in reversed(fired):
            self._deferred.pop(i)

        # ---- Repeating callbacks ----------------------------------------
        for entry in self._repeating:
            if self._budget_elapsed(t0) >= budget:
                over_budget = True
                break
            fire_at, interval, fn, count = entry
            if tick >= fire_at:
                try:
                    fn(self.api)
                except Exception:
                    self._push_exc("repeating")
                entry[0] = tick + interval
                if count is not None:
                    entry[3] = count - 1
        self._repeating = [e for e in self._repeating if e[3] is None or e[3] > 0]

        # ---- Per-tick hooks --------------------------------------------
        for fn in list(self.hooks["tick"]):
            if self._budget_elapsed(t0) >= budget:
                over_budget = True
                break
            try:
                fn(tick, self.api)
            except Exception:
                self._push_exc("on_tick")

        # ---- Record elapsed and throttled budget warning ---------------
        self._hook_ms = self._budget_elapsed(t0)
        if over_budget or self._hook_ms > budget:
            if tick - self._budget_warn_at >= self._BUDGET_WARN_INTERVAL:
                self._push("system",
                           f"[PERF] Hook budget exceeded: {self._hook_ms:.1f}ms — "
                           f"some hooks were skipped this tick")
                self._budget_warn_at = tick
            self._budget_warned = True
        else:
            self._budget_warned = False

    def dispatch_events(self, events: list):
        for event in events:
            etype = event.get("type", "")
            for registered_type, fn in list(self.hooks["event"]):
                if registered_type == etype or registered_type == "*":
                    try:
                        fn(event, self.api)
                    except Exception:
                        self._push_exc(f"on_event:{etype}")

    def dispatch_key(self, key_name: str):
        for k, fn in list(self.hooks["key"]):
            if k == key_name:
                try:
                    fn(self.api)
                except Exception:
                    self._push_exc(f"on_key:{key_name}")

    # ---------------------------------------------------- console output
    def _push(self, kind: str, text: str):
        for line in text.splitlines() or [""]:
            self.console_lines.append((line, kind))
        if len(self.console_lines) > CONSOLE_MAX_LINES:
            self.console_lines = self.console_lines[-CONSOLE_MAX_LINES:]
        if self._console_scroll > 0:
            self._console_scroll = min(self._console_scroll,
                                       max(0, len(self.console_lines) - 1))
        if kind == "error":
            log.error("scripting", text)

    def _push_exc(self, context: str = "") -> None:
        """Push the last line of the current exception to the console and log
        the full traceback (with context) to the file + terminal."""
        tb_full  = traceback.format_exc()
        tb_short = tb_full.splitlines()[-1]
        prefix   = f"{context}: " if context else ""
        self._push("error", prefix + tb_short)
        # Full traceback goes to the structured logger (file + Rich terminal)
        log.error("scripting", f"{prefix}full traceback:\n{tb_full}")

    # -------------------------------------------------------- REPL execute
    def _exec_line(self, line: str):
        # Built-in dot-commands
        cmd = line.strip()
        if cmd == ".clear":
            self.console_lines.clear()
            self._console_scroll = 0
            return
        if cmd == ".save":
            shots_dir = pathlib.Path("screenshots")
            shots_dir.mkdir(exist_ok=True)
            fname = shots_dir / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            try:
                pygame.image.save(self._app.screen, str(fname))
                self._push("system", f"[save] {fname}")
            except Exception as e:
                self._push("error", f"[save] {e}")
            return
        if cmd == ".reload":
            for name in list(self.script_mtimes):
                p = pathlib.Path("scripts") / name
                try:
                    self.unload_script(name)
                    self.load_script(p)
                except Exception as e:
                    self._push("error", str(e))
            return
        if cmd.startswith(".load "):
            name = cmd[6:].strip()
            if name:
                path = pathlib.Path("scripts") / name
                if not path.suffix:
                    path = path.with_suffix(".py")
                try:
                    self.load_script(path)
                except Exception as e:
                    self._push("error", f"[load] {e}")
            else:
                self._push("system", "usage: .load <filename>")
            return
        if cmd.startswith(".unload "):
            name = cmd[8:].strip()
            if name:
                self.unload_script(name if name.endswith(".py") else name + ".py")
            else:
                self._push("system", "usage: .unload <name>")
            return
        if cmd == ".ls":
            self.api.ls()
            return
        if cmd == ".help":
            self._push("system", _API_REFERENCE)
            return
        if cmd.startswith(".bench "):
            try:
                n = int(cmd.split()[1])
            except (IndexError, ValueError):
                n = 60
            t0 = time.perf_counter()
            for _ in range(n):
                self._app.sim.update_physics()
            ms = (time.perf_counter() - t0) * 1000.0
            self._push("output", f"bench {n} ticks: {ms:.2f}ms total, {ms/n:.2f}ms avg")
            return
        if cmd.startswith("."):
            self._push("error", f"Unknown command: {cmd.split()[0]}  (type .help for commands)")
            return

        # Python exec
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            try:
                code = compile(line, "<console>", "single")
            except SyntaxError:
                code = compile(line, "<console>", "exec")
            globs = dict(self._sandbox_globals)
            globs.update(self.loaded_scripts.get("__console__", {}))
            exec(code, globs)  # noqa: S102
            # Persist new names back into console namespace
            self.loaded_scripts.setdefault("__console__", {}).update(
                {k: v for k, v in globs.items() if k not in self._sandbox_globals}
            )
        except Exception:
            sys.stdout = old_stdout
            for ln in traceback.format_exc().splitlines():
                self._push("error", ln)
            return
        finally:
            sys.stdout = old_stdout
        out = buf.getvalue()
        if out:
            self._push("output", out.rstrip("\n"))

    def submit_input(self):
        line = self.input_buf
        self.input_buf = ""
        self.history_pos = -1

        # Load-prompt mode: treat the whole input as a filename
        if self._load_prompt:
            self._load_prompt = False
            name = line.strip()
            if name:
                path = pathlib.Path("scripts") / name
                if not path.suffix:
                    path = path.with_suffix(".py")
                try:
                    self.load_script(path)
                except Exception as e:
                    self._push("error", f"[load] {e}")
                    log.error("scripting", f"Load failed: {e}")
            else:
                self._push("system", "[load] cancelled")
            return

        if not line.strip():
            if self._multiline_mode:
                # Empty line flushes block
                full = "\n".join(self._pending_lines)
                self._pending_lines.clear()
                self._multiline_mode = False
                self._push("input", CONSOLE_PROMPT + "(block)")
                self._exec_line(full)
            return

        self._push("input", (CONSOLE_CONT_PROMPT if self._multiline_mode else CONSOLE_PROMPT) + line)

        if self.history and self.history[-1] == line:
            pass
        else:
            self.history.append(line)
            if len(self.history) > 100:
                self.history.pop(0)

        stripped = line.rstrip()
        if self._multiline_mode:
            self._pending_lines.append(stripped)
            if stripped.endswith(":") or stripped.endswith("\\"):
                pass  # keep accumulating
            return

        # Enter multi-line mode if line ends with :
        if stripped.endswith(":"):
            self._multiline_mode = True
            self._pending_lines = [stripped]
            return

        self._exec_line(stripped)
        self._console_scroll = 0

    def handle_key(self, event) -> bool:
        """Process a pygame KEYDOWN event. Returns True if consumed."""
        if not self.visible:
            return False
        key = event.key
        mods = pygame.key.get_mods()
        ctrl = bool(mods & pygame.KMOD_CTRL)

        if key == pygame.K_ESCAPE:
            if self._load_prompt:
                self._load_prompt = False
                self.input_buf = ""
                self._push("system", "[load] cancelled")
            elif self._multiline_mode:
                self._multiline_mode = False
                self._pending_lines.clear()
                self._push("system", "[multiline cancelled]")
            else:
                self.visible = False
            return True
        if key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            self.submit_input()
            return True
        if key == pygame.K_BACKSPACE:
            if ctrl:
                # Delete last word
                parts = self.input_buf.rstrip().rsplit(" ", 1)
                self.input_buf = parts[0] + " " if len(parts) > 1 else ""
            else:
                self.input_buf = self.input_buf[:-1]
            return True
        if key == pygame.K_DELETE:
            self.input_buf = ""
            return True
        if key == pygame.K_c and ctrl:
            self.input_buf = ""
            return True
        if key == pygame.K_l and ctrl:
            self.console_lines.clear()
            self._console_scroll = 0
            return True
        if key == pygame.K_e and ctrl:
            self._launch_editor()
            return True
        # Ctrl+= or Ctrl++: increase console font size
        if key in (pygame.K_EQUALS, pygame.K_PLUS) and ctrl:
            self._console_font_size = min(24, self._console_font_size + 1)
            self._console_font = None  # force rebuild on next draw
            return True
        # Ctrl+-: decrease console font size
        if key == pygame.K_MINUS and ctrl:
            self._console_font_size = max(8, self._console_font_size - 1)
            self._console_font = None
            return True
        if key == pygame.K_v and ctrl:
            try:
                text = pygame.scrap.get(pygame.SCRAP_TEXT)
                if text:
                    decoded = text.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "")
                    first_line = decoded.split("\n")[0].rstrip("\x00")
                    self.input_buf += first_line
            except Exception:
                pass
            return True
        if key == pygame.K_UP:
            if self.history:
                self.history_pos = max(0, min(len(self.history) - 1,
                                              self.history_pos + 1 if self.history_pos >= 0
                                              else len(self.history) - 1))
                self.input_buf = self.history[-(self.history_pos + 1)]
            return True
        if key == pygame.K_DOWN:
            if self.history_pos > 0:
                self.history_pos -= 1
                self.input_buf = self.history[-(self.history_pos + 1)]
            else:
                self.history_pos = -1
                self.input_buf = ""
            return True
        if key == pygame.K_PAGEUP:
            self._console_scroll = min(len(self.console_lines),
                                       self._console_scroll + 8)
            return True
        if key == pygame.K_PAGEDOWN:
            self._console_scroll = max(0, self._console_scroll - 8)
            return True
        if key == pygame.K_TAB:
            self._autocomplete()
            return True
        # Printable characters
        char = event.unicode
        if char and char.isprintable():
            self.input_buf += char
            return True
        return True   # consume ALL keys when console is open

    def _autocomplete(self):
        buf = self.input_buf
        prefix = re.split(r"[\s(,\[]", buf)[-1]
        if "." in prefix:
            obj_name, _, attr_prefix = prefix.rpartition(".")
            try:
                globs = dict(self._sandbox_globals)
                globs.update(self.loaded_scripts.get("__console__", {}))
                obj = eval(obj_name, globs)  # noqa: S307
                matches = [a for a in dir(obj) if a.startswith(attr_prefix) and not a.startswith("__")]
            except Exception:
                matches = []
            base = obj_name + "."
        else:
            globs = dict(self._sandbox_globals)
            globs.update(self.loaded_scripts.get("__console__", {}))
            matches = [k for k in globs if k.startswith(prefix)]
            base = ""
        if len(matches) == 1:
            self.input_buf = buf[: len(buf) - len(prefix)] + base + matches[0]
        elif len(matches) > 1:
            self._push("system", "  ".join(matches))

    # ------------------------------------------------------ draw console
    def draw(self, screen: pygame.Surface):
        if not self.visible:
            return
        if self._console_font is None or self._console_font_size_built != self._console_font_size:
            self._console_font = pygame.font.SysFont("Monospace", self._console_font_size)
            self._console_font_size_built = self._console_font_size
        font = self._console_font
        lh = font.get_linesize()
        y0 = WINDOW_HEIGHT - CONSOLE_HEIGHT
        w = SIM_WIDTH

        # Background
        bg = pygame.Surface((w, CONSOLE_HEIGHT), pygame.SRCALPHA)
        bg.fill((10, 10, 18, 215))
        screen.blit(bg, (0, y0))

        # Title bar
        loaded_names = [k for k in self.loaded_scripts if k != "__console__"]
        loaded_str = ", ".join(loaded_names) or "none"
        perf_tag = "  ⚡PERF" if self._budget_warned else ""
        title = f"  PY CONSOLE  ·  loaded: [{loaded_str}]{perf_tag}  ·  ESC close  ·  .help for API  "
        title_col = (255, 80, 80) if self._budget_warned else CONSOLE_SYSTEM_COLOR
        bar_col = (50, 10, 10) if self._budget_warned else (20, 20, 40)
        ts = font.render(title, True, title_col)
        pygame.draw.rect(screen, bar_col, (0, y0, w, lh + 4))
        screen.blit(ts, (4, y0 + 2))
        y0 += lh + 6

        # Output area
        visible_rows = (CONSOLE_HEIGHT - lh * 2 - 10) // lh
        total = len(self.console_lines)
        scroll = max(0, min(self._console_scroll, max(0, total - visible_rows)))
        start = max(0, total - visible_rows - scroll)
        end = total - scroll
        color_map = {
            "input":  CONSOLE_INPUT_COLOR,
            "output": CONSOLE_OUTPUT_COLOR,
            "error":  CONSOLE_ERROR_COLOR,
            "system": CONSOLE_SYSTEM_COLOR,
        }
        for i, (text, kind) in enumerate(self.console_lines[start:end]):
            col = color_map.get(kind, CONSOLE_OUTPUT_COLOR)
            surf = font.render(text[:120], True, col)
            screen.blit(surf, (4, y0 + i * lh))

        # Scroll position indicator (thin bar on right edge)
        if total > visible_rows:
            bar_h = CONSOLE_HEIGHT - lh * 2 - 10
            bar_x = w - 5
            pygame.draw.rect(screen, (40, 40, 60), (bar_x, y0, 4, bar_h))
            thumb_ratio = visible_rows / total
            thumb_h = max(12, int(bar_h * thumb_ratio))
            # scroll=0 means bottom; scroll=max means top
            max_scroll = max(1, total - visible_rows)
            thumb_frac = 1.0 - (scroll / max_scroll)
            thumb_y = y0 + int((bar_h - thumb_h) * thumb_frac)
            pygame.draw.rect(screen, (100, 120, 180), (bar_x, thumb_y, 4, thumb_h))

        # Input line
        if self._load_prompt:
            prompt = "Load: "
        elif self._multiline_mode:
            prompt = CONSOLE_CONT_PROMPT
        else:
            prompt = CONSOLE_PROMPT
        cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else " "
        input_text = prompt + self.input_buf + cursor
        iy = WINDOW_HEIGHT - lh - 6
        pygame.draw.line(screen, (50, 50, 80), (0, iy - 2), (w, iy - 2))
        is_ = font.render(input_text[:120], True, CONSOLE_INPUT_COLOR)
        screen.blit(is_, (4, iy))


