from .world import *
from .simulation import Simulation
from .ui import MenuUI, MenuBar
from .scripting import ScriptEngine

class Engine:
    """Orchestrator: Connects Pygame events, the Simulation, and the UI."""
    def __init__(self):
        pygame.init()
        try:
            pygame.scrap.init()
        except Exception:
            pass
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
        self.show_thermal_imaging = False
        self.paused = False
        self.single_step_pending = False
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_dragging = False
        self._pan_last = (0, 0)
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
        # Scripting engine (initialised last so it can reference fully-built app)
        self._fps_limit_override: int | None = None  # set by api.fps_limit(n)
        self.script_engine = ScriptEngine(self)
        log.success("engine", "Simulation engine ready",
                    rows=self.sim.rows, cols=self.sim.cols,
                    fps=FPS, profile=self.sim.profile_name)

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
            self.sound_enabled = False
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
        elif action == "toggle_thermal":
            self.show_thermal_imaging = not self.show_thermal_imaging
            self._update_action_message(f"Thermal imaging {'ON' if self.show_thermal_imaging else 'OFF'}")
        elif action == "toggle_editor":
            self.editor_mode = not self.editor_mode
            self._update_action_message(f"Editor {'ON' if self.editor_mode else 'OFF'}")
        elif action == "toggle_sound":
            self.sound_enabled = not self.sound_enabled
            self._update_action_message(f"Sound {'ON' if self.sound_enabled else 'OFF'}")
        elif action == "toggle_help":
            self.show_help = not self.show_help

    def _screen_to_grid(self, mx, my):
        """Convert screen pixel coords to (row, col) grid indices honouring zoom/pan."""
        ecs = CELL_SIZE * self.zoom
        col = int((mx + self.pan_x) / ecs)
        row = int((my - TOP_BAR_HEIGHT + self.pan_y) / ecs)
        return row, col

    def _clamp_pan(self):
        """Prevent panning outside the grid bounds."""
        ecs = CELL_SIZE * self.zoom
        sim_h = WINDOW_HEIGHT - TOP_BAR_HEIGHT
        max_px = max(0.0, self.sim.cols * ecs - SIM_WIDTH)
        max_py = max(0.0, self.sim.rows * ecs - sim_h)
        self.pan_x = max(0.0, min(max_px, self.pan_x))
        self.pan_y = max(0.0, min(max_py, self.pan_y))

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

            # Backtick toggles scripting console
            if event.type == pygame.KEYDOWN and event.key == pygame.K_BACKQUOTE:
                self.script_engine.visible = not self.script_engine.visible
                if self.script_engine.visible:
                    self.script_engine._push("system",
                        "[PY] Console ready. Type .help for API reference. "
                        f"Loaded: {len(self.script_engine.loaded_scripts)} script(s)")
                continue

            # Route ALL keydown events through scripting console when open
            if event.type == pygame.KEYDOWN and self.script_engine.visible:
                key_name = pygame.key.name(event.key).lower().replace(" ", "_")
                self.script_engine.dispatch_key(key_name)
                if self.script_engine.handle_key(event):
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
                elif event.key == pygame.K_i:
                    self.show_thermal_imaging = not self.show_thermal_imaging
                    self._update_action_message(f"Thermal imaging {'ON' if self.show_thermal_imaging else 'OFF'}")
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self._update_action_message("PAUSED" if self.paused else "Resumed")
                elif not self.editor_mode and event.key in (pygame.K_PERIOD, pygame.K_RIGHT) and self.paused:
                    self.single_step_pending = True
                    self._update_action_message("Step +1")
                elif event.key == pygame.K_r and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self.zoom = 1.0
                    self.pan_x = 0.0
                    self.pan_y = 0.0
                    self._update_action_message("Zoom reset")
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
                elif event.key == pygame.K_e and (mods & pygame.KMOD_CTRL):
                    # Ctrl+E: open scratch.py in editor, auto-open console to show status
                    self.script_engine.visible = True
                    self.script_engine._launch_editor()
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
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL:
                        # Ctrl+F12 → reload interaction matrix (old F12 behaviour)
                        self.sim.physics.reload_interaction_table()
                        self._update_action_message("Interaction matrix reloaded")
                    elif self.script_engine.visible:
                        # F12 with console open → load-script prompt
                        self.script_engine._load_prompt = True
                        self.script_engine.input_buf = ""
                        self.script_engine._push("system", "[load] Enter filename (e.g. my_script.py) — ESC to cancel")
                    else:
                        # F12 plain → reload all scripts + open console to show results
                        self.script_engine.visible = True
                        self.script_engine.reload_all_scripts()
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
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                # Console gets priority when visible and mouse is over it
                if self.script_engine.visible and my >= WINDOW_HEIGHT - CONSOLE_HEIGHT:
                    if event.y > 0:
                        self.script_engine._console_scroll = min(
                            len(self.script_engine.console_lines),
                            self.script_engine._console_scroll + 3)
                    else:
                        self.script_engine._console_scroll = max(
                            0, self.script_engine._console_scroll - 3)
                elif mx < SIM_WIDTH and my >= TOP_BAR_HEIGHT and not self.script_engine.visible:
                    ecs = CELL_SIZE * self.zoom
                    # world coords under cursor
                    wx = (mx + self.pan_x) / ecs
                    wy = (my - TOP_BAR_HEIGHT + self.pan_y) / ecs
                    factor = 1.18 if event.y > 0 else (1.0 / 1.18)
                    self.zoom = max(0.5, min(8.0, self.zoom * factor))
                    new_ecs = CELL_SIZE * self.zoom
                    self.pan_x = wx * new_ecs - mx
                    self.pan_y = wy * new_ecs - (my - TOP_BAR_HEIGHT)
                    self._clamp_pan()
            elif event.type == pygame.MOUSEMOTION:
                if self._pan_dragging:
                    dx = event.pos[0] - self._pan_last[0]
                    dy = event.pos[1] - self._pan_last[1]
                    self.pan_x -= dx
                    self.pan_y -= dy
                    self._clamp_pan()
                    self._pan_last = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self._pan_dragging = False
            elif event.type == pygame.VIDEORESIZE:
                self._on_resize(event.w, event.h)
                self._clamp_pan()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Block all sim mouse interaction when console is open
                if self.script_engine.visible:
                    continue
                if event.button == 2 and event.pos[0] < SIM_WIDTH:
                    self._pan_dragging = True
                    self._pan_last = event.pos
                    continue
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
        if (buttons[0] or buttons[2]) and not self.script_engine.visible: # Left or Right click
            mx, my = pygame.mouse.get_pos()
            
            # Only draw if mouse is inside the simulation area (and menu bar not open)
            if mx < SIM_WIDTH and my >= TOP_BAR_HEIGHT and not self.menu_bar.blocks_input((mx, my)):
                mat_to_draw = self.current_mat if buttons[0] else 0 # Right click forces Eraser
                row, col = self._screen_to_grid(mx, my)
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
            ("[I] Therm", self.show_thermal_imaging),
            ("[E] Edit",  self.editor_mode),
            (f"[Space] {'⏸ PAUSED' if self.paused else 'Running'}", self.paused),
            (f"[Scroll] {self.zoom:.1f}x", self.zoom != 1.0),
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

    def _draw_thermal_legend(self):
        """Draw a thin false-colour temperature legend bar at the bottom of the sim area."""
        bar_h = 14
        bar_w = min(320, SIM_WIDTH - 20)
        bar_x = 10
        bar_y = WINDOW_HEIGHT - bar_h - 4
        max_t = 1200.0
        for i in range(bar_w):
            t = max_t * i / bar_w
            color = self._thermal_color(t)
            pygame.draw.line(self.screen, color, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_h))
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h), 1)
        font = self.hud_font
        for label, frac in (("0°", 0.0), ("300°", 0.25), ("600°", 0.5), ("900°", 0.75), ("1200°C", 1.0)):
            lx = bar_x + int(frac * bar_w)
            surf = font.render(label, True, (230, 230, 230))
            self.screen.blit(surf, (lx - surf.get_width() // 2, bar_y - 14))

    def draw_overlays(self):
        if self.show_thermal_imaging:
            self._draw_thermal_legend()
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

        ecs = CELL_SIZE * self.zoom
        cell_px = max(1, int(ecs))
        sim_h = WINDOW_HEIGHT - TOP_BAR_HEIGHT
        row_start = max(0, int(self.pan_y / ecs))
        row_end   = min(rows, int((self.pan_y + sim_h) / ecs) + 2)
        col_start = max(0, int(self.pan_x / ecs))
        col_end   = min(cols, int((self.pan_x + SIM_WIDTH) / ecs) + 2)

        # Pre-allocate one reusable SRCALPHA surface to avoid per-cell alloc
        _ov_surf = pygame.Surface((cell_px, cell_px), pygame.SRCALPHA)

        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                sx = int(col * ecs - self.pan_x)
                sy = int(row * ecs - self.pan_y) + TOP_BAR_HEIGHT
                rect = pygame.Rect(sx, sy, cell_px, cell_px)

                if self.show_temp_overlay and temp_ready:
                    temp = physics.temperature[row][col]
                    delta = max(-60.0, min(300.0, temp - ambient))
                    if delta >= 0:
                        intensity = min(255, int((delta / 300.0) * 255))
                        _ov_surf.fill((255, 80, 40, max(0, min(170, 25 + intensity // 2))))
                        self.screen.blit(_ov_surf, rect)
                    else:
                        cold_intensity = min(255, int((abs(delta) / 60.0) * 255))
                        _ov_surf.fill((60, 130, 255, max(0, min(170, 25 + cold_intensity // 2))))
                        self.screen.blit(_ov_surf, rect)

                if self.show_oxygen_overlay and oxygen_ready:
                    oxygen = max(0.0, min(1.0, physics.oxygen_level[row][col]))
                    red = int((1.0 - oxygen) * 220)
                    green = int(oxygen * 220)
                    _ov_surf.fill((red, green, 40, 80))
                    self.screen.blit(_ov_surf, rect)

                if self.show_smoke_overlay and smoke_ready:
                    smoke = max(0.0, min(1.0, physics.smoke_density[row][col]))
                    alpha = int(smoke * 185)
                    if alpha > 0:
                        _ov_surf.fill((160, 160, 170, alpha))
                        self.screen.blit(_ov_surf, rect)

                if self.show_phase_overlay and phase_ready:
                    phase_progress = max(0.0, min(1.0, physics.phase_transition_progress[row][col]))
                    alpha = int(phase_progress * 190)
                    if alpha > 0:
                        _ov_surf.fill((190, 90, 255, alpha))
                        self.screen.blit(_ov_surf, rect)

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

        # Mouseover cell info
        mx, my = pygame.mouse.get_pos()
        if mx < SIM_WIDTH and my >= TOP_BAR_HEIGHT:
            row_h, col_h = self._screen_to_grid(mx, my)
            physics = self.sim.physics
            grid = self.sim.grid
            if 0 <= row_h < self.sim.rows and 0 <= col_h < self.sim.cols:
                mat_h = grid[row_h][col_h]
                mat_name = MATERIALS[mat_h]["name"] if mat_h in MATERIALS else "?"
                t_h = physics.temperature[row_h][col_h] if len(physics.temperature) > row_h else 0.0
                bs_h = physics.burn_stage[row_h][col_h] if len(physics.burn_stage) > row_h else 0
                o2_h = physics.oxygen_level[row_h][col_h] if len(physics.oxygen_level) > row_h else 0.0
                hover_text = f"[{row_h},{col_h}] {mat_name}  T:{t_h:.1f}°C  O2:{o2_h:.2f}  stage:{bs_h}"
                lines.append(hover_text)

        for index, text in enumerate(lines):
            surface = self.hud_font.render(text, True, (235, 235, 235))
            self.screen.blit(surface, (10, 34 + (index * 18)))

        if self.show_help:
            help_lines = [
                "H toggle help",
                "1-9/0 choose material",
                "B brush shape, +/- size, C clear, S sound",
                "T temp, O oxygen, M smoke, P phase, I thermal imaging",
                "Space pause | . or → single-step | Scroll zoom | Middle-drag pan | R reset zoom",
                "Ctrl+Z/Y undo redo | F5/F9 save load",
                "F6/F7 replay save/start | F1-F3 presets | F10 bench",
                "F11 cycle profile | Ctrl+F12 reload interactions",
                "F12 reload scripts | Ctrl+F12 reload interactions",
                "F favorites toggle | Alt+1..4 quick favorites",
                "F8 snapshot regressions",
                "E editor mode",
                "` (backtick) open/close Python scripting console",
                "F12 (console open) prompt to load a script from scripts/",
            ]
            for index, text in enumerate(help_lines):
                surface = self.hud_font.render(text, True, (210, 210, 210))
                self.screen.blit(surface, (10, 88 + (index * 16)))

    @staticmethod
    def _thermal_color(temp):
        """Map a temperature (°C) to a false-color 'iron' palette.
        Black → purple → red → orange → yellow → white across 0-1200 °C."""
        t = max(0.0, min(1.0, temp / 1200.0))
        # key stops: (t, R, G, B)
        stops = [
            (0.00,   0,   0,   0),
            (0.12,  60,   0, 100),
            (0.25, 160,   0, 120),
            (0.40, 255,  30,   0),
            (0.58, 255, 140,   0),
            (0.75, 255, 230,  40),
            (0.88, 255, 255, 160),
            (1.00, 255, 255, 255),
        ]
        for i in range(len(stops) - 1):
            t0, r0, g0, b0 = stops[i]
            t1, r1, g1, b1 = stops[i + 1]
            if t <= t1:
                f = (t - t0) / (t1 - t0)
                return (int(r0 + f * (r1 - r0)), int(g0 + f * (g1 - g0)), int(b0 + f * (b1 - b0)))
        return (255, 255, 255)

    def draw_simulation(self):
        # Fill sim area background
        bg = (0, 0, 0) if self.show_thermal_imaging else MATERIALS[0]["color"]
        pygame.draw.rect(self.screen, bg, (0, 0, SIM_WIDTH, WINDOW_HEIGHT))

        _fire_id = MATERIAL_IDS.get("fire", -1)
        _fire_lt = self.sim.physics.fire_lifetime if hasattr(self.sim.physics, "fire_lifetime") else None
        _smoke_id = MATERIAL_IDS.get("smoke", -1)
        _smoke_lt = self.sim.physics.smoke_lifetime if hasattr(self.sim.physics, "smoke_lifetime") else None
        _ticks = pygame.time.get_ticks()
        _thermal = self.show_thermal_imaging
        _temp_field = self.sim.physics.temperature if _thermal and len(self.sim.physics.temperature) == self.sim.rows else None

        ecs = CELL_SIZE * self.zoom
        cell_px = max(1, int(ecs))
        sim_h = WINDOW_HEIGHT - TOP_BAR_HEIGHT
        rows, cols = self.sim.rows, self.sim.cols
        row_start = max(0, int(self.pan_y / ecs))
        row_end   = min(rows, int((self.pan_y + sim_h) / ecs) + 2)
        col_start = max(0, int(self.pan_x / ecs))
        col_end   = min(cols, int((self.pan_x + SIM_WIDTH) / ecs) + 2)

        # Render active grid cells
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                mat = self.sim.grid[row][col]
                sx = int(col * ecs - self.pan_x)
                sy = int(row * ecs - self.pan_y) + TOP_BAR_HEIGHT

                try:
                    if _thermal and _temp_field is not None:
                        color = self._thermal_color(_temp_field[row][col])
                        if any(c > 5 for c in color) or mat != 0:
                            pygame.draw.rect(self.screen, color, (sx, sy, cell_px, cell_px))
                        continue

                    if mat == 0:
                        continue
                    if mat == _fire_id and _fire_lt is not None:
                        lt = max(0.0, _fire_lt[row][col])
                        flicker = (_ticks // 50 + row * 3 + col * 7) % 5
                        if lt > 0.65 or flicker == 0:
                            color = (255, min(255, 220 + flicker * 7), max(0, int(80 * lt)))
                        elif lt > 0.35:
                            color = (255, max(0, int(80 + 120 * lt)), 0)
                        else:
                            color = (max(0, max(160, int(255 * lt * 2.5))), max(0, int(40 * lt)), 0)
                        pygame.draw.rect(self.screen, color, (sx, sy, cell_px, cell_px))
                    elif mat == _smoke_id and _smoke_lt is not None:
                        lt = max(0.0, _smoke_lt[row][col])
                        if lt > 0.0:
                            v = min(255, max(55, int(130 * min(lt, 1.0))))
                            color = (v, v, min(255, v + 15))
                            pygame.draw.rect(self.screen, color, (sx, sy, cell_px, cell_px))
                    else:
                        cr = self.script_engine._custom_renderers
                        if mat in cr:
                            try:
                                cr[mat](self.screen, sx, sy, cell_px, row, col, self.script_engine.api)
                            except Exception as _rend_err:
                                cr.pop(mat, None)
                                self.script_engine._push("error",
                                    f"[renderer] mat {mat} unregistered: {_rend_err}")
                        else:
                            mat_color = MATERIALS.get(mat, MATERIALS[0])["color"]
                            pygame.draw.rect(self.screen, mat_color, (sx, sy, cell_px, cell_px))
                except Exception as _draw_err:
                    print(f"[draw] row={row} col={col} mat={mat} color={color if 'color' in dir() else '?'} lt={lt if 'lt' in dir() else '?'} err={_draw_err}", flush=True)
                    traceback.print_exc()

        # Brush cursor preview
        mx, my = pygame.mouse.get_pos()
        if mx < SIM_WIDTH and my >= TOP_BAR_HEIGHT and not self.show_thermal_imaging:
            row_c, col_c = self._screen_to_grid(mx, my)
            if 0 <= row_c < self.sim.rows and 0 <= col_c < self.sim.cols:
                r = self.brush_size
                if self.brush_shape == "circle":
                    cx = int(col_c * ecs - self.pan_x + ecs / 2)
                    cy = int(row_c * ecs - self.pan_y + TOP_BAR_HEIGHT + ecs / 2)
                    radius = max(1, int(r * ecs))
                    pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius, 1)
                else:
                    bx = int((col_c - r) * ecs - self.pan_x)
                    by = int((row_c - r) * ecs - self.pan_y) + TOP_BAR_HEIGHT
                    bw = max(1, int((2 * r + 1) * ecs))
                    pygame.draw.rect(self.screen, (255, 255, 255), (bx, by, bw, bw), 1)

        # Pause overlay
        if self.paused:
            pause_surf = self.font.render("  ⏸ PAUSED  ", True, (255, 220, 60))
            px = (SIM_WIDTH - pause_surf.get_width()) // 2
            py = TOP_BAR_HEIGHT + 8
            bg_rect = pygame.Rect(px - 6, py - 4, pause_surf.get_width() + 12, pause_surf.get_height() + 8)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect, border_radius=6)
            self.screen.blit(pause_surf, (px, py))

        self.draw_overlays()
        # Scripting console overlay (always on top of sim, below menu)
        self.script_engine.draw(self.screen)

    def run(self):
        while self.running:
            self.handle_input()
            self.script_engine.poll_reload()
            step_start = time.perf_counter()
            if not self.paused or self.single_step_pending:
                step_result = self.sim.update_physics()
                self.last_step_ms = (time.perf_counter() - step_start) * 1000.0
                self.last_changed_cells = step_result.changed_cells_count
                self.single_step_pending = False
                # Hook dispatch
                self.script_engine.dispatch_tick(self.sim.tick_index)
                self.script_engine.dispatch_events(step_result.events)
                self.script_engine.dispatch_physics_stages(self.sim)
            else:
                step_result = None
            
            self.draw_simulation()
            self.menu.draw(self.screen, self.font, self.current_mat, self.brush_size, self.brush_shape, self.favorites)
            self.draw_hud()
            self.menu_bar.draw(self.screen, self)   # drawn last so it's always on top
            if step_result is not None:
                self._play_event_sounds(step_result.events)
            
            pygame.display.flip()
            self.clock.tick(self._fps_limit_override or FPS)
            
        pygame.quit()
        log.info("engine", "Session ended cleanly")
        log.close()
        sys.exit()

