# 50-Step Plan: In-Game Python Scripting Engine

Target file: `/home/bortex/pi/game.py`  
Engine: Python 3.13 / pygame 2.6 — no new hard dependencies.  
Toggle key: **`** (backtick) opens/closes the console overlay.  
Scripts live in `./scripts/` and are hot-reloaded on save.

---

## Phase 1 — Core Infrastructure (Steps 1–10)

### Step 1 — Add `import` block for scripting
Add `import code`, `import io`, `import threading`, `import importlib.util`, `import inspect`,
`import builtins`, `import pathlib`, `import ast` to the top of `game.py`.  
All scripting subsystems live in plain Python; no third-party libs required.

### Step 2 — Create `ScriptAPI` class
Define a `ScriptAPI` class that wraps the live `GameApp`, `PowderSimulation`, and
`PowderPhysicsEngine` references.  This is the single object every script receives — scripts
never get a direct reference to the app.

```
class ScriptAPI:
    def __init__(self, app): self._app = app
    @property
    def sim(self): return self._app.sim
    @property
    def physics(self): return self._app.sim.physics
    @property
    def grid(self): return self._app.sim.grid   # live reference
    @property
    def rows(self): return self._app.sim.rows
    @property
    def cols(self): return self._app.sim.cols
```

### Step 3 — Grid read/write helpers on `ScriptAPI`
```
api.get(row, col)                → material id (int)
api.set(row, col, mat_id)        → sets cell, triggers undo snapshot
api.fill(r1,c1,r2,c2, mat_id)    → fill rectangle
api.circle(row, col, r, mat_id)  → fill circle
api.replace(src_id, dst_id)      → replace all occurrences
api.clear()                      → api.fill(0,0,rows,cols, 0)
api.count(mat_id)                → count cells of type
```
All setters clamp to grid bounds and validate mat_id against `MATERIALS`.

### Step 4 — Temperature / physics state helpers on `ScriptAPI`
```
api.temp(row, col)               → float °C
api.set_temp(row, col, t)        → clamp to [−273, 5000]
api.fill_temp(r1,c1,r2,c2, t)
api.heat(row, col, delta)        → relative change
api.burn_stage(row, col)         → int 0-4
api.oxygen(row, col)             → float 0-1
api.set_oxygen(row, col, v)
```

### Step 5 — Material registry helpers on `ScriptAPI`
```
api.material_id(name)            → int or None
api.material_name(id)            → str
api.materials()                  → dict copy of MATERIALS
api.register_material(spec)      → add new material at runtime (step 16)
api.reload_material(id, patch)   → update properties live
```

### Step 6 — Simulation control helpers on `ScriptAPI`
```
api.pause()  /  api.resume()  /  api.is_paused()
api.step(n=1)                    → advance N physics ticks
api.fps()                        → current FPS float
api.tick()                       → current physics tick counter
api.set_profile(name)            → e.g. "realistic"
api.profile()                    → current profile name
```

### Step 7 — Event / hook system on `ScriptAPI`
```
api.on_tick(fn)                  → fn(tick, api) called every physics tick
api.on_event(event_type, fn)     → fn(event, api) for physics events
api.on_key(key_name, fn)         → fn(api) when key pressed (pygame key name)
api.on_cell_change(row, col, fn) → fn(old_mat, new_mat, api)
api.remove_hook(fn)              → unregister any hook by reference
api.clear_hooks()                → unregister all hooks from current script
```
Hooks are stored in `ScriptEngine.hooks` dict keyed by hook type.

### Step 8 — Notification / output helpers on `ScriptAPI`
```
api.print(*args)                 → append to console output buffer
api.notify(msg, duration=3.0)   → HUD action message
api.screenshot(path=None)        → save PNG of current frame
api.log(msg)                     → append to scripts/script.log with timestamp
```
`api.print` captures stdout for the console display; print() is also redirected.

### Step 9 — `ScriptEngine` class skeleton
```python
class ScriptEngine:
    def __init__(self, app):
        self.api = ScriptAPI(app)
        self.console_lines: list[str] = []   # output ring buffer (200 lines)
        self.history: list[str] = []         # command history (100 entries)
        self.history_pos: int = -1
        self.input_buf: str = ""
        self.visible: bool = False
        self.hooks: dict = {"tick": [], "event": [], "key": [], "cell": []}
        self.loaded_scripts: dict[str, types.ModuleType] = {}
        self.script_mtimes: dict[str, float] = {}
        self._sandbox_globals: dict = self._make_sandbox()
```

### Step 10 — Sandboxed execution environment
`_make_sandbox()` returns a globals dict that:
- Includes safe builtins: `print, len, range, int, float, str, list, dict, set, tuple,
  abs, min, max, round, sum, sorted, enumerate, zip, map, filter, isinstance, type,
  repr, bool, chr, ord, hex, bin, oct, divmod, pow, any, all, hasattr, getattr,
  setattr, math, random, json, time, re, copy, itertools, collections`
- Injects `api` = the live `ScriptAPI` instance
- Injects `MATERIALS`, `MATERIAL_IDS` as read references
- **Blocks** `__import__`, `open`, `os`, `sys`, `subprocess`, `eval`, `exec`,
  `compile`, `globals`, `locals`, `vars` by omission or override  
- `__builtins__` is replaced with the filtered dict, not the real module

Safe imports that scripts may use: `math`, `random`, `json`, `re`, `copy`,
`itertools`, `collections` — these are pre-validated and injected directly.

---

## Phase 2 — Console UI (Steps 11–20)

### Step 11 — Console overlay surface
In `GameApp.__init__` add:
```python
self.script_engine = ScriptEngine(self)
self._console_surf = None   # lazily allocated in draw_console()
```
Also add `pygame.K_BACKQUOTE` → toggle `self.script_engine.visible` to `handle_input`.

### Step 12 — Console layout constants
```
CONSOLE_HEIGHT = 280          # px, bottom portion of sim area
CONSOLE_BG     = (10,10,14,210)  # dark with alpha
CONSOLE_FONT_SIZE = 13
CONSOLE_INPUT_COLOR  = (180, 230, 180)
CONSOLE_OUTPUT_COLOR = (220, 220, 220)
CONSOLE_ERROR_COLOR  = (255, 100, 80)
CONSOLE_PROMPT = ">>> "
CONSOLE_MAX_LINES = 200
```

### Step 13 — `draw_console()` method
Renders onto a persistent SRCALPHA surface re-blitted each frame when visible:
1. Dark semi-transparent background rect over bottom of sim area
2. Title bar: `"PY CONSOLE  —  ` + last loaded script name + `  [F2 reload] [ESC close]"`
3. Scrollable output area: last N lines fitting in the window, colored by line type
   (`>>>` = input echo in green, errors in red, output in white)
4. Input line at bottom: `>>> ` + `self.script_engine.input_buf` + blinking cursor `|`
5. Cursor blinks using `(pygame.time.get_ticks() // 500) % 2`

### Step 14 — Console keyboard input handling
Inside `handle_input`, when console is visible, intercept ALL `KEYDOWN` events before
the normal input path:
- Printable chars → append to `input_buf`
- `BACKSPACE` → pop last char  
- `DELETE` → clear whole input line  
- `RETURN/KP_ENTER` → submit line (step 15)  
- `UP/DOWN` arrows → walk `history` list, fill `input_buf`  
- `TAB` → autocomplete (step 22)  
- `PAGEUP/PAGEDOWN` → scroll output buffer  
- `ESC` → close console (set `visible = False`)  
- `CTRL+C` → clear input line  
- `CTRL+L` → clear output buffer  
All other events (scroll, pause, etc.) are consumed/ignored while console is open.

### Step 15 — REPL execution
When Enter is pressed:
1. Strip input, skip empty lines
2. Append `">>> " + line` to `console_lines`
3. Append to `history` (dedup last entry), reset `history_pos`
4. Clear `input_buf`
5. Redirect `sys.stdout` to a `StringIO` capture
6. Try `compile(line, "<console>", "single")` — if that raises `SyntaxError` try
   `compile(line, "<console>", "exec")` (multi-line pasted block support)
7. `exec(code, sandbox_globals)` inside try/except
8. Restore `sys.stdout`, flush captured output into `console_lines`
9. On exception: `traceback.format_exc()` → split by `\n` → append as error lines

### Step 16 — Multi-line input mode
If the user types a line ending in `:` (def, class, for, if, with, try…) the console
enters **multi-line mode**:
- Prompt changes to `... ` 
- Lines accumulate in a `_pending_block: list[str]`
- An empty line or double-Enter flushes the block for execution
- `ESC` cancels the block

Implemented via a `_multiline_mode: bool` and `_pending_lines: list[str]` on `ScriptEngine`.

### Step 17 — Console output ring buffer
`console_lines` is capped at `CONSOLE_MAX_LINES = 200`.  
Each entry is a `(text: str, kind: str)` tuple where kind ∈ `{"input","output","error","system"}`.  
`draw_console` uses kind to pick color.  Scroll offset `_console_scroll: int` tracks how many
lines from the bottom the viewport is offset (0 = latest line visible).

### Step 18 — Console scroll
`PAGEUP` → `_console_scroll += 8` (capped at len)  
`PAGEDOWN` / new output → `_console_scroll = max(0, _console_scroll - 8)`  
Mouse wheel inside console rect also scrolls while console is visible.

### Step 19 — `GameApp.draw_simulation` integration
Call `self.draw_console()` at the very end of `draw_simulation()` (after pause banner,
before `draw_overlays()`) when `self.script_engine.visible`.

### Step 20 — Console activation message
When console opens, push a system line:
```
[PY] Console ready. Type help() for API reference.  Loaded scripts: N
```
When it closes, restore normal input flow silently.

---

## Phase 3 — Script File System (Steps 21–28)

### Step 21 — `scripts/` directory setup
On `ScriptEngine.__init__`:
```python
self.scripts_dir = pathlib.Path("scripts")
self.scripts_dir.mkdir(exist_ok=True)
(self.scripts_dir / "example.py").write_text(EXAMPLE_SCRIPT, encoding="utf-8")
```
`EXAMPLE_SCRIPT` is a constant string containing a well-commented starter script
demonstrating `api.set`, `api.on_tick`, `api.notify`.

### Step 22 — `load_script(path)` method
```python
def load_script(self, path: str | pathlib.Path):
    path = pathlib.Path(path)
    src = path.read_text(encoding="utf-8")
    # AST safety check (step 23)
    self._ast_check(src, path.name)
    globs = dict(self._sandbox_globals)   # fresh copy per script
    exec(compile(src, str(path), "exec"), globs)
    self.loaded_scripts[path.name] = globs
    self.script_mtimes[path.name] = path.stat().st_mtime
    self._push("system", f"[LOAD] {path.name}")
```

### Step 23 — AST safety check
`_ast_check(src, name)` uses `ast.parse(src)` and walks the tree looking for:
- `ast.Import` / `ast.ImportFrom` nodes for disallowed modules
  (`os`, `sys`, `subprocess`, `socket`, `ctypes`, `importlib`, `builtins`)
- `ast.Call` with func name `__import__`, `eval`, `exec`, `open`, `compile`
- `ast.Attribute` accessing `__class__`, `__bases__`, `__subclasses__`, `__globals__`

If any are found, raise `ScriptSecurityError` with the node's line number and
description.  This is a best-effort check — it does **not** replace the sandbox but
provides early feedback.

### Step 24 — Hot-reload watcher
`ScriptEngine.poll_reload()` is called once per frame (from `run()` loop, outside
the physics tick to keep it cheap):
```python
def poll_reload(self):
    for name, mtime in list(self.script_mtimes.items()):
        p = self.scripts_dir / name
        if p.exists() and p.stat().st_mtime > mtime:
            self.unload_script(name)
            self.load_script(p)
            self.api.notify(f"Reloaded: {name}", 2.0)
```
Errors during reload are caught and pushed to console only (no crash).

### Step 25 — `unload_script(name)` method
1. Remove all hooks registered by that script's globals namespace (match by
   checking hook function's `__globals__ is script_globals`)
2. Call `on_unload()` if defined in the script's globals
3. Remove from `loaded_scripts` and `script_mtimes`

### Step 26 — F-key bindings for script management
- `F12` already reloads interaction matrix — reassign to `CTRL+F12`
- `F12` (plain) → reload all scripts in `scripts/`
- `CTRL+F12` → open file picker line in console: `"Load script: "` prompt that
  accepts a filename relative to `scripts/`
Add help text: `"F12 reload scripts | Ctrl+F12 load script"`.

### Step 27 — `scripts/` directory listing command
`api.ls()` → pushes list of `.py` files in `scripts/` to console output.  
`api.load(name)` → `self._engine.load_script(self._engine.scripts_dir / name)`.  
`api.unload(name)` → `self._engine.unload_script(name)`.  
`api.loaded()` → list of currently loaded script names.

### Step 28 — Persistent script auto-load
The game reads `scripts/.autoload` (plain text, one filename per line) on startup.
Scripts listed there are loaded automatically after the engine initialises.
`api.autoload(name)` / `api.no_autoload(name)` add/remove from this file.

---

## Phase 4 — Hook Execution (Steps 29–35)

### Step 29 — Tick hook dispatch
In `PowderSimulation.update_physics()`, after all physics stages complete, call:
```python
if self._script_engine:
    self._script_engine.dispatch_tick(tick_index)
```
`dispatch_tick` iterates `hooks["tick"]`, calls each `fn(tick, api)` inside try/except
— exceptions go to console, never propagate.  Budget: skip dispatch if last call took
> 4ms (measured with `time.perf_counter`), emit a warning once.

### Step 30 — Event hook dispatch
Physics already emits `events` list from `_stage_thermal` and the reaction system.
Pass this list to `dispatch_events(events)` which iterates `hooks["event"]`, calls
`fn(event, api)` for events whose `type` matches the subscription filter.

### Step 31 — Key hook dispatch
In `handle_input` KEYDOWN section, after all built-in handlers, iterate
`hooks["key"]` and call matching handlers:
```python
for key_name, fn in self.script_engine.hooks["key"]:
    if event.key == getattr(pygame, f"K_{key_name.upper()}", None):
        fn(self.script_engine.api)
```

### Step 32 — Cell-change hook
`ScriptAPI.set()` (step 3) checks `hooks["cell"]` after writing and fires any
handlers whose `(row, col)` matches the set cell.  For performance, cell hooks are
stored in a `dict[(row,col)] → list[fn]` for O(1) lookup.

### Step 33 — `api.after(ticks, fn)` — deferred callback
Scripts can schedule a one-shot callback N ticks in the future:
```python
api.after(60, lambda api: api.notify("1 second elapsed"))
```
Stored as `_deferred: list[(fire_at_tick, fn)]` on `ScriptEngine`; dispatched from
`dispatch_tick` before the regular tick hooks.

### Step 34 — `api.every(ticks, fn, count=None)` — repeating callback
Like `api.after` but re-schedules itself. Optional `count` limits repetitions.
`api.cancel(fn)` removes any pending deferred or repeating callback by reference.

### Step 35 — Hook safety budget
`ScriptEngine` tracks cumulative hook CPU time per frame in `_hook_ms: float`.
If it exceeds `8ms` in a single frame, remaining hooks are deferred to the next frame
and a yellow warning is shown in the console: `"[PERF] Hook budget exceeded: Xms"`.

---

## Phase 5 — Runtime Material System (Steps 36–42)

### Step 36 — `api.register_material(spec)` implementation
```python
def register_material(self, spec: dict) -> int:
```
1. Validate required keys: `name`, `color`, `type`
2. Check name uniqueness (case-insensitive) in `MATERIALS`
3. Find next free ID starting at 100 (runtime materials use 100–999)
4. Merge spec with `MATERIAL_DEFAULTS`
5. Insert into `MATERIALS` and `MATERIAL_IDS`
6. Re-run `MaterialRegistry._validate_parameter_ranges` on new entry
7. Add material to the menu panel (calls `MenuPanel.rebuild()`)
8. Return new ID

### Step 37 — `api.reload_material(id_or_name, patch)` implementation
```python
api.reload_material("Wood", {"burn_rate": 0.01, "smoke_factor": 0.9})
```
1. Resolve id
2. Deep-merge patch into `MATERIALS[id]`
3. Re-validate
4. Log change to console

### Step 38 — `api.unregister_material(id_or_name)`
1. Replace all grid cells of that type with `0`
2. Remove from `MATERIALS`, `MATERIAL_IDS`
3. Remove from menu panel
4. Only allowed for runtime materials (id >= 100)

### Step 39 — `api.interaction(spec)` — register a custom reaction rule
```python
api.interaction({
    "pair": ["water", "lava"],
    "priority": 95,
    "products": ["steam", "stone"],
    "energy_delta": -200,
    "gas_release": 0.4,
    "residue": 0.0,
    "duration_ticks": 1,
    "conditions": {"min_temp": 80}
})
```
Calls `physics.reload_interaction_table()` internally after adding to the in-memory
table so the rule is active immediately without restarting.

### Step 40 — `api.remove_interaction(pair)` 
Removes a named reaction pair from the live interaction table.  
`api.interactions()` → returns copy of the full table as a list of dicts.

### Step 41 — Script-defined custom renderers
Scripts can register a per-material draw hook:
```python
api.set_renderer(mat_id, fn)
# fn(screen, sx, sy, cell_px, row, col, api) → None
```
Called from `draw_simulation` instead of the default `pygame.draw.rect`.  Allows
animated, multi-cell, or shader-like rendering per material.  Renderer exceptions are
caught and unregistered with a console warning.

### Step 42 — Script-defined custom physics stage
```python
api.add_physics_stage(fn, priority=50)
# fn(grid, rows, cols, tick, rng, api) → None
```
Injected into `_stage_thermal` dispatch after existing stages.  `priority` controls
ordering (lower = earlier).  Limited to 4 active custom stages to prevent abuse.

---

## Phase 6 — Quality of Life & Polish (Steps 43–50)

### Step 43 — `help()` command in console
`help()` (no args) prints a compact API reference table to console output.  
`help(obj)` falls back to Python's built-in `help` piped through `StringIO`.  
Pre-built reference is a constant string covering all `api.*` methods grouped by
category (Grid, Physics, Materials, Hooks, Simulation, Output).

### Step 44 — Tab autocomplete
On `TAB` press, call `_autocomplete(input_buf)`:
1. Split on last `.` and space
2. If prefix is `api.`, complete against `dir(api)`
3. If bare word, complete against sandbox globals + builtins
4. Single match → complete in place
5. Multiple matches → print list as system line, keep input unchanged

### Step 45 — Console command shortcuts
Built-in commands (not Python, just shortcuts):
```
.clear   → api.clear()
.save    → api.screenshot()
.reload  → reload all scripts
.ls      → api.ls()
.help    → api.print(API_REFERENCE)
.bench N → run N physics ticks and print timing
```
These are parsed before `exec` so they work even with a broken sandbox.

### Step 46 — Script editor shortcut
`CTRL+E` opens `scripts/scratch.py` in the user's `$EDITOR` (or nano) via
`subprocess.Popen` **in the main process** (not the sandbox).  When the editor exits,
`scratch.py` is automatically reloaded.  This is the one place `subprocess` is used,
but it runs entirely outside the scripting sandbox.

### Step 47 — `api.query(row, col)` — rich cell inspector
Returns a dict with all available state for a cell:
```python
{
  "mat": 4, "name": "Wood",
  "temp": 312.4, "burn_stage": 2, "burn_progress": 0.41,
  "oxygen": 0.72, "moisture": 0.08, "smoke_density": 0.1,
  "fire_lifetime": 0.0, "smoke_lifetime": 0.0,
}
```
Pretty-printed to console when called interactively.

### Step 48 — `api.find(mat_id_or_name)` — locate all cells
Returns a `list[(row, col)]` of every cell matching the material.  Capped at 2000
results with a "… (truncated)" notice.  Useful for scripted search operations.

### Step 49 — Serialization: save/load script state
Scripts can persist state across sessions:
```python
api.save_data("my_script", {"counter": 42})
data = api.load_data("my_script")   # None if not found
```
Data is stored in `scripts/.scriptdata.json` (top-level keys = script name).
Values must be JSON-serializable (checked before write).

### Step 50 — Integration test & example scripts bundle
Ship a `scripts/` bundle with four ready-to-run example scripts:

| File | What it demonstrates |
|---|---|
| `example_lava_rain.py` | `api.on_tick` + `api.set` — rains lava from top row every 30 ticks |
| `example_material_plasma.py` | `api.register_material` — defines "Plasma" with custom color and temp |
| `example_auto_extinguish.py` | `api.on_event("ignition")` + `api.set_temp` — auto-quench fires |
| `example_heatmap.py` | `api.set_renderer` — custom per-cell overlay using temp gradient |

Each script is self-contained, heavily commented, and exercises a different part of the
API surface so new users can learn by reading and modifying them.

---

## Final Cleanup Phase

- Add docstrings to all `ScriptAPI` methods and key `ScriptEngine` methods
- Add inline comments to all complex logic in `ScriptEngine` and hook dispatch
- Add error handling and edge case checks (e.g. max custom stages, material ID limits)
- Polish console UI: colors, fonts, layout, scroll behavior
- Add logging to `scripts/script.log` for script load/unload, errors, and key actions
- Test on all platforms, fix any path or encoding issues with script loading/saving
- Add unit tests for `ScriptAPI` methods and `ScriptEngine` internals where possible (e.g. AST checker, material registration, deferred callbacks).

## Architecture Summary

```
GameApp
 ├── ScriptEngine
 │    ├── ScriptAPI  (public surface exposed to scripts)
 │    ├── _sandbox_globals  (safe builtins + api)
 │    ├── hooks: {tick, event, key, cell}
 │    ├── _deferred / _repeating callbacks
 │    ├── loaded_scripts: {name → globals_dict}
 │    └── console_lines: [(text, kind)]  ← drawn by draw_console()
 └── PowderSimulation
      └── PowderPhysicsEngine
           └── dispatch_tick / dispatch_events called here
```

## Implementation Order (recommended)
1. Steps 1–10 (API + engine data structures) — no UI yet, unit-testable  
2. Steps 11–20 (console overlay) — can test REPL interactively  
3. Steps 29–35 (hooks) — wire into existing physics  
4. Steps 21–28 (file system) — scripts/ dir + hot reload  
5. Steps 36–42 (material system) — runtime materials  
6. Steps 43–50 (polish + examples) — quality of life  

Total estimated new code: ~900–1200 lines added to `game.py`.
