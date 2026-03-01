from .world import *

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
                ("Thermal Imaging",    "toggle_thermal",lambda e: e.show_thermal_imaging),
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

