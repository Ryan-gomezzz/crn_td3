# =============================================================================
# visualization.py — Real-time Pygame GUI for the CRN TD3 simulation
#
# Four main classes:
#   PlotSurface   — self-contained scrolling line-plot rendered with pygame.draw
#   NetworkPanel  — animated 4-node CRN diagram (left panel)
#   InsightsPanel — AI training analysis sidebar (far-right panel)
#   PygameRenderer — top-level compositor; owns the pygame window
# =============================================================================

from __future__ import annotations
import math
from collections import deque
import numpy as np
import pygame
import pygame.gfxdraw

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    LEFT_PANEL_WIDTH, RIGHT_PANEL_WIDTH, INSIGHTS_WIDTH,
    TOP_BAR_HEIGHT, BOTTOM_BAR_HEIGHT,
    NODE_POSITIONS, NODE_COLORS, NODE_RADIUS,
    SINR_THRESHOLD, P_MAX,
    BG_COLOR, PANEL_BG, PANEL_BORDER, PLOT_BG,
    WHITE, LIGHT_GRAY, GRAY, DARK_GRAY,
    GREEN, RED, BLUE, ORANGE, PURPLE, YELLOW, CYAN,
    LINK_PRIMARY_OK, LINK_PRIMARY_BAD, LINK_SU,
    LINK_INTERF_SP, LINK_INTERF_PS,
    FPS_CAP,
)


# =============================================================================
# PlotSurface — pure-pygame scrolling line plot
# =============================================================================

class PlotSurface:
    """
    A self-contained scrolling line plot rendered entirely with pygame.draw.
    Supports multiple data series, axis labels, title, and an optional
    horizontal threshold line.

    Usage:
        plot = PlotSurface(width=400, height=180, title="Reward", y_label="r",
                           series_colors=[(150,150,150),(241,196,15)],
                           series_names=["raw","avg100"])
        plot.push(raw_val, avg_val)          # one value per series per call
        surface = plot.render()
        screen.blit(surface, (x, y))
    """

    PAD_LEFT   = 42
    PAD_RIGHT  = 10
    PAD_TOP    = 28
    PAD_BOTTOM = 22

    def __init__(
        self,
        width:         int,
        height:        int,
        title:         str,
        y_label:       str  = "",
        max_points:    int  = 400,
        bg_color:      tuple = None,
        series_colors: list  = None,
        series_names:  list  = None,
        y_min:         float | None = None,
        y_max:         float | None = None,
        threshold:     float | None = None,
        threshold_color: tuple = (241, 196, 15),
    ):
        self.width   = width
        self.height  = height
        self.title   = title
        self.y_label = y_label
        self.max_points      = max_points
        self.bg_color        = bg_color or PLOT_BG
        self.series_colors   = series_colors or [CYAN]
        self.series_names    = series_names  or [""]
        self.y_min_fixed     = y_min
        self.y_max_fixed     = y_max
        self.threshold       = threshold
        self.threshold_color = threshold_color

        n = len(self.series_colors)
        self._data: list[deque] = [deque(maxlen=max_points) for _ in range(n)]
        self._surface = pygame.Surface((width, height))

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, *values: float) -> None:
        """Append one value per series (must match number of series)."""
        for i, v in enumerate(values):
            if i < len(self._data):
                self._data[i].append(float(v))

    def clear(self) -> None:
        for d in self._data:
            d.clear()

    def render(self) -> pygame.Surface:
        """Redraw the entire plot and return the surface."""
        surf = self._surface
        surf.fill(self.bg_color)

        plot_w = self.width  - self.PAD_LEFT - self.PAD_RIGHT
        plot_h = self.height - self.PAD_TOP  - self.PAD_BOTTOM
        plot_x = self.PAD_LEFT
        plot_y = self.PAD_TOP

        # ── Border rect ──────────────────────────────────────────────────────
        pygame.draw.rect(surf, PANEL_BORDER,
                         (plot_x, plot_y, plot_w, plot_h), 1)

        # ── Compute y range ──────────────────────────────────────────────────
        all_vals = [v for d in self._data for v in d]
        if len(all_vals) < 2:
            # Nothing meaningful to draw yet; just show title
            self._draw_title(surf)
            return surf

        y_min = self.y_min_fixed if self.y_min_fixed is not None else min(all_vals)
        y_max = self.y_max_fixed if self.y_max_fixed is not None else max(all_vals)
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0

        # ── Grid lines (4 horizontal) ────────────────────────────────────────
        for i in range(1, 4):
            gy = plot_y + int(plot_h * i / 4)
            pygame.draw.line(surf, DARK_GRAY, (plot_x, gy), (plot_x + plot_w, gy), 1)
            # y tick label
            tick_val = y_max - (y_max - y_min) * i / 4
            self._draw_small_text(surf, f"{tick_val:.2f}", plot_x - 2, gy, align="right")

        # ── Y-axis extremes ──────────────────────────────────────────────────
        self._draw_small_text(surf, f"{y_max:.2f}", plot_x - 2, plot_y, align="right")
        self._draw_small_text(surf, f"{y_min:.2f}", plot_x - 2, plot_y + plot_h, align="right")

        # ── Threshold line ───────────────────────────────────────────────────
        if self.threshold is not None:
            ty = self._to_py(self.threshold, y_min, y_max, plot_y, plot_h)
            if plot_y <= ty <= plot_y + plot_h:
                self._draw_dashed_h_line(surf, self.threshold_color,
                                         plot_x, ty, plot_w, dash=6, gap=4)
                self._draw_small_text(surf, f"thr={self.threshold:.1f}",
                                      plot_x + plot_w - 2, ty - 1, align="right",
                                      color=self.threshold_color)

        # ── Data series ──────────────────────────────────────────────────────
        for series_idx, (d, color) in enumerate(zip(self._data, self.series_colors)):
            pts = list(d)
            if len(pts) < 2:
                continue
            pixel_pts = []
            for i, v in enumerate(pts):
                px = plot_x + int(i / (self.max_points - 1) * plot_w)
                py = self._to_py(v, y_min, y_max, plot_y, plot_h)
                pixel_pts.append((px, py))
            pygame.draw.lines(surf, color, False, pixel_pts, 2)

        # ── Legend ───────────────────────────────────────────────────────────
        if any(n for n in self.series_names):
            lx = plot_x + 4
            for i, (name, color) in enumerate(zip(self.series_names, self.series_colors)):
                if name:
                    pygame.draw.rect(surf, color, (lx, plot_y + 4 + i * 12, 8, 3))
                    self._draw_small_text(surf, name, lx + 10, plot_y + 2 + i * 12,
                                          color=color, align="left")

        self._draw_title(surf)
        self._draw_y_label(surf, plot_y, plot_h)
        return surf

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_py(self, v: float, y_min: float, y_max: float,
               plot_y: int, plot_h: int) -> int:
        """Convert data value to pixel y (inverted: top = y_max)."""
        norm = (v - y_min) / (y_max - y_min)
        return int(plot_y + plot_h * (1.0 - norm))

    def _init_fonts(self) -> None:
        if not hasattr(self, '_font_small'):
            self._font_small = pygame.font.SysFont("consolas", 10)
            self._font_title = pygame.font.SysFont("consolas", 12, bold=True)

    def _draw_title(self, surf: pygame.Surface) -> None:
        self._init_fonts()
        txt = self._font_title.render(self.title, True, LIGHT_GRAY)
        surf.blit(txt, (self.PAD_LEFT + 2, 6))

    def _draw_small_text(self, surf: pygame.Surface, text: str,
                         x: int, y: int, color: tuple = GRAY,
                         align: str = "right") -> None:
        self._init_fonts()
        txt = self._font_small.render(text, True, color)
        if align == "right":
            surf.blit(txt, (x - txt.get_width(), y - txt.get_height() // 2))
        else:
            surf.blit(txt, (x, y))

    def _draw_y_label(self, surf: pygame.Surface, plot_y: int, plot_h: int) -> None:
        if not self.y_label:
            return
        self._init_fonts()
        txt = self._font_small.render(self.y_label, True, GRAY)
        rotated = pygame.transform.rotate(txt, 90)
        cx = 8
        cy = plot_y + plot_h // 2
        surf.blit(rotated, (cx - rotated.get_width() // 2,
                             cy - rotated.get_height() // 2))

    def _draw_dashed_h_line(self, surf: pygame.Surface, color: tuple,
                             x: int, y: int, length: int,
                             dash: int = 6, gap: int = 4) -> None:
        pos = x
        while pos < x + length:
            end = min(pos + dash, x + length)
            pygame.draw.line(surf, color, (pos, y), (end, y), 1)
            pos += dash + gap


# =============================================================================
# NetworkPanel — animated 4-node CRN diagram
# =============================================================================

class NetworkPanel:
    """
    Renders the left panel showing the 4-node CRN topology:
      - PT (Primary Transmitter)   — top-left
      - PR (Primary Receiver)      — top-right
      - ST (Secondary Transmitter) — bottom-left
      - SR (Secondary Receiver)    — bottom-right

    Communication links are drawn as colored lines:
      - PT→PR: solid green (SINR_p ≥ threshold) or red (violation)
      - ST→SR: solid blue, width ∝ SU throughput
      - ST→PR: dashed red (interference to PU)
      - PT→SR: dashed orange (interference to SU)

    Signal pulses animate along each link driven by pulse_phase.
    """

    def __init__(self, width: int = LEFT_PANEL_WIDTH,
                 height: int = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT):
        self.width  = width
        self.height = height

        # Convert node positions to absolute coords within this panel
        self.positions = dict(NODE_POSITIONS)

        pygame.font.init()
        self._font_label  = pygame.font.SysFont("consolas", 13, bold=True)
        self._font_value  = pygame.font.SysFont("consolas", 11)
        self._font_node   = pygame.font.SysFont("consolas", 12, bold=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(
        self,
        surface:          pygame.Surface,
        offset:           tuple,            # (x, y) panel origin on screen
        state:            np.ndarray,       # 7D state vector
        p_s:              float,
        sinr_p:           float,
        sinr_s:           float,
        r_s:              float,
        pulse_phase:      float,            # 0.0 → 1.0, drives animation
        show_interference: bool = True,
    ) -> None:
        ox, oy = offset

        # ── Background ────────────────────────────────────────────────────────
        panel_rect = pygame.Rect(ox, oy, self.width, self.height)
        pygame.draw.rect(surface, PANEL_BG, panel_rect)
        pygame.draw.rect(surface, PANEL_BORDER, panel_rect, 1)

        # Dot grid (decorative)
        for gx in range(ox + 20, ox + self.width - 10, 30):
            for gy in range(oy + 20, oy + self.height - 10, 30):
                pygame.draw.circle(surface, DARK_GRAY, (gx, gy), 1)

        pt = self._abs(ox, oy, "PT")
        pr = self._abs(ox, oy, "PR")
        st = self._abs(ox, oy, "ST")
        sr = self._abs(ox, oy, "SR")

        # ── Interference links (drawn first, below primary links) ─────────────
        if show_interference:
            # ST→PR dashed red
            self._draw_dashed_line(surface, LINK_INTERF_SP, st, pr,
                                   dash=8, gap=5, width=2)
            # PT→SR dashed orange
            self._draw_dashed_line(surface, LINK_INTERF_PS, pt, sr,
                                   dash=8, gap=5, width=2)
            # Pulse on interference links (slower, half phase)
            self._draw_pulse(surface, st, pr, pulse_phase,      LINK_INTERF_SP, 4)
            self._draw_pulse(surface, pt, sr, (pulse_phase + 0.5) % 1.0, LINK_INTERF_PS, 4)

        # ── Primary link PT→PR ────────────────────────────────────────────────
        link_color = LINK_PRIMARY_OK if sinr_p >= SINR_THRESHOLD else LINK_PRIMARY_BAD
        pygame.draw.line(surface, link_color, pt, pr, 3)
        self._draw_pulse(surface, pt, pr, pulse_phase, link_color, 6)

        # ── SU link ST→SR (thickness proportional to throughput) ──────────────
        su_width = max(2, min(7, int(2 + r_s)))
        pygame.draw.line(surface, LINK_SU, st, sr, su_width)
        self._draw_pulse(surface, st, sr, (pulse_phase + 0.25) % 1.0, LINK_SU, 5)

        # ── Nodes ─────────────────────────────────────────────────────────────
        pu_ok  = sinr_p >= SINR_THRESHOLD
        su_ok  = sinr_s > 0.1

        self._draw_node(surface, pt, NODE_COLORS["PT"], "PT",
                        halo=GREEN if pu_ok else RED)
        self._draw_node(surface, pr, NODE_COLORS["PR"], "PR",
                        halo=GREEN if pu_ok else RED)
        self._draw_node(surface, st, NODE_COLORS["ST"], "ST",
                        halo=GREEN if su_ok else ORANGE)
        self._draw_node(surface, sr, NODE_COLORS["SR"], "SR",
                        halo=GREEN if su_ok else ORANGE)

        # ── Numeric readouts ──────────────────────────────────────────────────
        self._draw_readouts(surface, ox, oy, p_s, sinr_p, sinr_s, r_s,
                            pt, pr, st, sr)

        # ── Power bar for ST ──────────────────────────────────────────────────
        self._draw_power_bar(surface, ox + 20, oy + self.height - 130, p_s)

        # ── Panel title ───────────────────────────────────────────────────────
        title_surf = self._font_label.render("CRN Network — Real-time", True, LIGHT_GRAY)
        surface.blit(title_surf, (ox + 10, oy + 8))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _abs(self, ox: int, oy: int, node: str) -> tuple:
        x, y = self.positions[node]
        return (ox + x, oy + y)

    def _draw_node(self, surface: pygame.Surface, pos: tuple,
                   color: tuple, label: str,
                   halo: tuple = None, radius: int = NODE_RADIUS) -> None:
        x, y = pos

        # Outer halo (glow effect using concentric transparent circles)
        if halo:
            for r in range(radius + 14, radius, -3):
                alpha = max(0, 60 - (r - radius) * 6)
                glow_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*halo, alpha), (r + 1, r + 1), r)
                surface.blit(glow_surf, (x - r - 1, y - r - 1))

        # Shadow
        pygame.draw.circle(surface, (5, 5, 15), (x + 3, y + 3), radius)
        # Filled circle
        pygame.draw.circle(surface, color, (x, y), radius)
        # White border ring
        pygame.draw.circle(surface, WHITE, (x, y), radius, 2)

        # Label text centered inside node
        txt = self._font_node.render(label, True, WHITE)
        surface.blit(txt, (x - txt.get_width() // 2, y - txt.get_height() // 2))

    def _draw_dashed_line(self, surface: pygame.Surface, color: tuple,
                          start: tuple, end: tuple,
                          dash: int = 8, gap: int = 5, width: int = 2) -> None:
        """Draw a dashed line from start to end using the unit direction vector."""
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length  = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length   # unit vector

        t = 0.0
        drawing = True
        while t < length:
            seg_len = dash if drawing else gap
            t_next  = min(t + seg_len, length)
            if drawing:
                p1 = (int(sx + ux * t),     int(sy + uy * t))
                p2 = (int(sx + ux * t_next), int(sy + uy * t_next))
                pygame.draw.line(surface, color, p1, p2, width)
            t       = t_next
            drawing = not drawing

    def _draw_pulse(self, surface: pygame.Surface, start: tuple, end: tuple,
                    phase: float, color: tuple, radius: int = 5) -> None:
        """Draw a small filled circle traveling along the link at the given phase."""
        sx, sy = start
        ex, ey = end
        px = int(sx + (ex - sx) * phase)
        py = int(sy + (ey - sy) * phase)
        pygame.draw.circle(surface, WHITE, (px, py), radius + 1)
        pygame.draw.circle(surface, color, (px, py), radius)

    def _draw_power_bar(self, surface: pygame.Surface,
                        x: int, y: int, p_s: float,
                        bar_w: int = 18, bar_h: int = 100) -> None:
        """Vertical power bar for ST. Color: green→yellow→red."""
        fill_ratio = p_s / P_MAX
        fill_h     = int(bar_h * fill_ratio)

        # Background (empty bar)
        pygame.draw.rect(surface, DARK_GRAY, (x, y, bar_w, bar_h))
        pygame.draw.rect(surface, PANEL_BORDER, (x, y, bar_w, bar_h), 1)

        # Fill color gradient
        if fill_ratio < 0.4:
            bar_color = GREEN
        elif fill_ratio < 0.7:
            bar_color = YELLOW
        else:
            bar_color = RED

        if fill_h > 0:
            pygame.draw.rect(surface, bar_color,
                             (x, y + bar_h - fill_h, bar_w, fill_h))

        # Label
        label = self._font_value.render("P_s", True, LIGHT_GRAY)
        surface.blit(label, (x + bar_w + 3, y + bar_h // 2 - 5))
        pct = self._font_value.render(f"{p_s:.2f}W", True, bar_color)
        surface.blit(pct, (x, y + bar_h + 3))

    def _draw_readouts(self, surface: pygame.Surface,
                       ox: int, oy: int,
                       p_s: float, sinr_p: float, sinr_s: float, r_s: float,
                       pt: tuple, pr: tuple, st: tuple, sr: tuple) -> None:
        """Draw numeric value labels near nodes."""
        color_ok  = GREEN
        color_bad = RED

        def label(text, pos, dx, dy, color=LIGHT_GRAY):
            txt = self._font_value.render(text, True, color)
            surface.blit(txt, (pos[0] + dx, pos[1] + dy))

        # Near PT (top-left)
        label(f"P_p=1.00W", pt, -NODE_RADIUS, -NODE_RADIUS - 28)

        # Near PR — show SINR_p
        sinr_p_col = color_ok if sinr_p >= SINR_THRESHOLD else color_bad
        label(f"SINR_p={sinr_p:.2f}", pr, NODE_RADIUS // 2, -NODE_RADIUS - 28, sinr_p_col)

        # Near ST — show P_s
        label(f"P_s={p_s:.3f}W", st, -NODE_RADIUS - 10, NODE_RADIUS + 8)

        # Near SR — show SINR_s and R_s
        label(f"SINR_s={sinr_s:.2f}", sr, NODE_RADIUS // 2, NODE_RADIUS + 8)
        label(f"R_s={r_s:.3f} b/s/Hz", sr, NODE_RADIUS // 2, NODE_RADIUS + 22, CYAN)


# =============================================================================
# InsightsPanel — AI training analysis sidebar
# =============================================================================

class InsightsPanel:
    """
    Far-right sidebar showing real-time analytical summary of training.
    Renders: training stage badge, progress bar, PU protection rate,
    SU throughput stats, reward trend, power trend, policy insight text,
    and a violation counter — all as pure Pygame drawing.

    Expected stats dict keys:
        episode, total_episodes, constraint_rate (0–1), avg_reward,
        best_reward, avg_power, avg_r_s, peak_r_s, reward_trend (str),
        power_trend (str), buffer_size, is_training, violations_last100,
        policy_insight (str)
    """

    def __init__(self, width: int = 240,
                 height: int = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT):
        self.width  = width
        self.height = height
        pygame.font.init()
        self._f_title  = pygame.font.SysFont("consolas", 13, bold=True)
        self._f_badge  = pygame.font.SysFont("consolas", 14, bold=True)
        self._f_label  = pygame.font.SysFont("consolas", 11, bold=True)
        self._f_value  = pygame.font.SysFont("consolas", 12)
        self._f_small  = pygame.font.SysFont("consolas", 10)
        self._f_insight= pygame.font.SysFont("consolas", 10)

    # ── render ─────────────────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface, ox: int, oy: int,
               stats: dict) -> None:
        """Draw the full insights panel at offset (ox, oy)."""
        w, h = self.width, self.height
        panel_rect = pygame.Rect(ox, oy, w, h)
        pygame.draw.rect(surface, PANEL_BG, panel_rect)
        pygame.draw.rect(surface, PANEL_BORDER, panel_rect, 1)

        y = oy + 8
        cx = ox + w // 2

        # ── Title ─────────────────────────────────────────────────────────────
        self._text(surface, "AI  INSIGHTS", cx, y, self._f_title, CYAN, center=True)
        y += 18
        self._hline(surface, ox, y, w)
        y += 8

        # ── Training Stage Badge ──────────────────────────────────────────────
        self._text(surface, "TRAINING STAGE", cx, y, self._f_label, GRAY, center=True)
        y += 14

        stage       = stats.get("stage", "Exploring")
        stage_color = {"Exploring": ORANGE, "Learning": BLUE,
                       "Converging": CYAN, "Converged": GREEN}.get(stage, YELLOW)

        badge_rect = pygame.Rect(ox + 10, y, w - 20, 26)
        pygame.draw.rect(surface, tuple(max(0, c // 4) for c in stage_color), badge_rect, border_radius=5)
        pygame.draw.rect(surface, stage_color, badge_rect, 2, border_radius=5)
        self._text(surface, stage.upper(), cx, y + 13, self._f_badge, stage_color, center=True)
        y += 34

        # ── Training Progress Bar ──────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        self._text(surface, "PROGRESS", ox + 8, y, self._f_label, GRAY)
        y += 14

        episode = stats.get("episode", 0)
        total   = stats.get("total_episodes", 3000)
        pct     = min(1.0, episode / max(1, total))
        self._bar(surface, ox + 8, y, w - 16, 12, pct, BLUE, DARK_GRAY)
        y += 14
        self._text(surface, f"Ep {episode} / {total}  ({pct*100:.0f}%)",
                   cx, y, self._f_small, LIGHT_GRAY, center=True)
        y += 16

        # ── PU Protection Rate ────────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        self._text(surface, "PU PROTECTION", ox + 8, y, self._f_label, GRAY)
        y += 14

        cr = stats.get("constraint_rate", 0.0)
        cr_color = GREEN if cr >= 0.85 else (YELLOW if cr >= 0.6 else RED)
        self._bar(surface, ox + 8, y, w - 16, 12, cr, cr_color, DARK_GRAY)
        y += 14
        self._text(surface, f"{cr*100:.1f}%  of steps safe", cx, y, self._f_small,
                   cr_color, center=True)
        y += 12
        self._text(surface, f"(SINR_p \u2265 {SINR_THRESHOLD:.1f} threshold)",
                   cx, y, self._f_small, GRAY, center=True)
        y += 16

        # ── SU Throughput Stats ───────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        self._text(surface, "SU THROUGHPUT", ox + 8, y, self._f_label, GRAY)
        y += 14

        avg_rs  = stats.get("avg_r_s", 0.0)
        peak_rs = stats.get("peak_r_s", 0.0)
        self._kv(surface, ox + 8, y, "Avg this ep:", f"{avg_rs:.3f} b/s/Hz", BLUE)
        y += 14
        self._kv(surface, ox + 8, y, "Session peak:", f"{peak_rs:.3f} b/s/Hz", CYAN)
        y += 18

        # ── Reward Trend ──────────────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        self._text(surface, "REWARD TREND", ox + 8, y, self._f_label, GRAY)
        y += 14

        trend       = stats.get("reward_trend", "→ Stable")
        trend_color = GREEN if "↑" in trend else (RED if "↓" in trend else YELLOW)
        self._text(surface, trend, cx, y, self._f_badge, trend_color, center=True)
        y += 16

        best_r = stats.get("best_reward", 0.0)
        avg_r  = stats.get("avg_reward", 0.0)
        self._kv(surface, ox + 8, y, "Avg100:", f"{avg_r:+.2f}", LIGHT_GRAY)
        y += 13
        self._kv(surface, ox + 8, y, "Best ep:", f"{best_r:+.2f}", YELLOW)
        y += 18

        # ── Power Trend ───────────────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        self._text(surface, "AVG POWER (ST)", ox + 8, y, self._f_label, GRAY)
        y += 14

        avg_pw  = stats.get("avg_power", 0.5)
        pw_trend= stats.get("power_trend", "→")
        pw_color= GREEN if "↓" in pw_trend else (RED if "↑" in pw_trend else YELLOW)
        pw_pct  = avg_pw / P_MAX
        self._bar(surface, ox + 8, y, w - 16, 10, pw_pct,
                  GREEN if pw_pct < 0.4 else (YELLOW if pw_pct < 0.7 else RED), DARK_GRAY)
        y += 12
        self._text(surface, f"{pw_trend}  {avg_pw:.3f} W",
                   cx, y, self._f_value, pw_color, center=True)
        y += 18

        # ── Violation Counter ─────────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        self._text(surface, "VIOLATIONS (last 100)", ox + 8, y, self._f_label, GRAY)
        y += 14

        viols    = stats.get("violations_last100", 0)
        viol_pct = viols / 100.0
        viol_color = RED if viol_pct > 0.3 else (YELLOW if viol_pct > 0.1 else GREEN)
        self._bar(surface, ox + 8, y, w - 16, 10, viol_pct, viol_color, DARK_GRAY)
        y += 12
        self._text(surface, f"{viols}/100 steps  ({viol_pct*100:.0f}%)",
                   cx, y, self._f_small, viol_color, center=True)
        y += 16

        # ── Policy Insight ────────────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        self._text(surface, "POLICY INSIGHT", ox + 8, y, self._f_label, CYAN)
        y += 14

        insight = stats.get("policy_insight", "Initialising...")
        y = self._wrapped_text(surface, insight, ox + 8, y, w - 16,
                                self._f_insight, LIGHT_GRAY)
        y += 8

        # ── Buffer Info ───────────────────────────────────────────────────────
        self._hline(surface, ox, y, w); y += 6
        buf  = stats.get("buffer_size", 0)
        trained = stats.get("is_training", False)
        buf_color = GREEN if trained else ORANGE
        self._text(surface, "REPLAY BUFFER", ox + 8, y, self._f_label, GRAY)
        y += 14
        self._bar(surface, ox + 8, y, w - 16, 8,
                  min(1.0, buf / 100_000), buf_color, DARK_GRAY)
        y += 10
        status_txt = "Training  ✓" if trained else "Filling..."
        self._text(surface, f"{buf:,}  {status_txt}",
                   cx, y, self._f_small, buf_color, center=True)

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _text(self, surf, text, x, y, font, color, center=False):
        s = font.render(text, True, color)
        if center:
            surf.blit(s, (x - s.get_width() // 2, y - s.get_height() // 2))
        else:
            surf.blit(s, (x, y))

    def _kv(self, surf, x, y, key, value, val_color):
        """Key: value pair on one line."""
        ks = self._f_small.render(key + " ", True, GRAY)
        vs = self._f_value.render(value, True, val_color)
        surf.blit(ks, (x, y))
        surf.blit(vs, (x + ks.get_width(), y - 1))

    def _bar(self, surf, x, y, w, h, fraction, fill_color, bg_color):
        """Horizontal progress/fill bar."""
        pygame.draw.rect(surf, bg_color,  (x, y, w, h), border_radius=3)
        pygame.draw.rect(surf, PANEL_BORDER, (x, y, w, h), 1, border_radius=3)
        fill_w = max(0, int(w * fraction))
        if fill_w > 0:
            pygame.draw.rect(surf, fill_color, (x, y, fill_w, h), border_radius=3)

    def _hline(self, surf, ox, y, w):
        pygame.draw.line(surf, PANEL_BORDER, (ox + 4, y), (ox + w - 4, y), 1)

    def _wrapped_text(self, surf, text, x, y, max_w, font, color) -> int:
        """Word-wrap text within max_w pixels. Returns final y position."""
        words = text.split()
        line  = ""
        for word in words:
            test = line + word + " "
            if font.size(test)[0] > max_w and line:
                s = font.render(line.strip(), True, color)
                surf.blit(s, (x, y))
                y    += font.get_height() + 1
                line  = word + " "
            else:
                line = test
        if line.strip():
            s = font.render(line.strip(), True, color)
            surf.blit(s, (x, y))
            y += font.get_height() + 1
        return y


# =============================================================================
# PygameRenderer — top-level GUI compositor
# =============================================================================

class PygameRenderer:
    """
    Owns the pygame window and all sub-panels.
    Call init() once, then update(training_state) every frame.
    """

    # Speed options: label → steps_per_render_frame
    SPEED_OPTIONS = [("1×", 1), ("5×", 5), ("10×", 10), ("Max", 9999)]

    def __init__(self):
        self._initialized    = False
        self._screen         = None
        self._clock          = None
        self._network_panel  = None
        self._reward_plot    = None
        self._throughput_plot = None
        self._sinr_plot       = None

        self._pulse_phase    = 0.0
        self._speed_idx      = 0          # index into SPEED_OPTIONS
        self._paused         = False
        self._show_interf    = True

        # Button rects (set in _build_bottom_bar)
        self._btn_pause_rect   = None
        self._btn_reset_rect   = None
        self._btn_interf_rect  = None
        self._speed_btn_rects  = []

        # Events to report back to main loop
        self._events_out = {
            "quit": False, "pause_toggle": False, "reset": False,
            "speed_change": None, "toggle_interference": False,
        }

        # Fonts (initialized in init())
        self._font_top    = None
        self._font_status = None
        self._font_small  = None

    # ── Initialization ─────────────────────────────────────────────────────────

    def init(self) -> None:
        pygame.init()
        pygame.display.set_caption("CRN TD3 — Cognitive Radio Network Simulation | RIT Bangalore")
        self._screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self._clock  = pygame.time.Clock()

        self._font_top    = pygame.font.SysFont("consolas", 13, bold=True)
        self._font_status = pygame.font.SysFont("consolas", 12)
        self._font_small  = pygame.font.SysFont("consolas", 11)
        self._font_btn    = pygame.font.SysFont("consolas", 12, bold=True)

        # Network panel (left side)
        panel_h = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT
        self._network_panel = NetworkPanel(LEFT_PANEL_WIDTH, panel_h)

        # Right panel plots (each 1/3 of right panel height)
        plot_w = RIGHT_PANEL_WIDTH - 8
        plot_h = (WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT) // 3

        self._reward_plot = PlotSurface(
            width=plot_w, height=plot_h,
            title="Episode Reward",
            y_label="reward",
            max_points=400,
            series_colors=[GRAY, YELLOW],
            series_names=["raw", "avg100"],
        )
        self._throughput_plot = PlotSurface(
            width=plot_w, height=plot_h,
            title="SU Throughput  R_s  [bits/s/Hz]",
            y_label="R_s",
            max_points=400,
            series_colors=[BLUE],
            series_names=["R_s"],
            y_min=0.0,
        )
        self._sinr_plot = PlotSurface(
            width=plot_w, height=plot_h,
            title="PU SINR  (yellow = protection threshold)",
            y_label="SINR_p",
            max_points=400,
            series_colors=[CYAN],
            series_names=["SINR_p"],
            y_min=0.0,
            threshold=SINR_THRESHOLD,
            threshold_color=YELLOW,
        )

        # AI Insights panel (far right)
        self._insights_panel = InsightsPanel(
            width=INSIGHTS_WIDTH,
            height=WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT,
        )

        self._initialized = True

    # ── Event handling ─────────────────────────────────────────────────────────

    def handle_events(self) -> dict:
        """
        Process pygame event queue and return a dict of actions for main loop.
        Keys: quit, pause_toggle, reset, speed_change (int or None),
              toggle_interference.
        """
        out = {
            "quit": False, "pause_toggle": False, "reset": False,
            "speed_change": None, "toggle_interference": False,
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                out["quit"] = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    out["pause_toggle"] = True
                    self._paused = not self._paused
                elif event.key == pygame.K_r:
                    out["reset"] = True
                elif event.key == pygame.K_i:
                    out["toggle_interference"] = True
                    self._show_interf = not self._show_interf

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Pause button
                if self._btn_pause_rect and self._btn_pause_rect.collidepoint(mx, my):
                    out["pause_toggle"] = True
                    self._paused = not self._paused

                # Reset button
                elif self._btn_reset_rect and self._btn_reset_rect.collidepoint(mx, my):
                    out["reset"] = True

                # Interference toggle
                elif self._btn_interf_rect and self._btn_interf_rect.collidepoint(mx, my):
                    out["toggle_interference"] = True
                    self._show_interf = not self._show_interf

                # Speed buttons
                else:
                    for i, rect in enumerate(self._speed_btn_rects):
                        if rect.collidepoint(mx, my):
                            self._speed_idx = i
                            out["speed_change"] = self.SPEED_OPTIONS[i][1]
                            break

        return out

    # ── Main render call ───────────────────────────────────────────────────────

    def update(self, ts: dict) -> None:
        """
        Full-frame render. Call every iteration of the main loop.

        ts (training_state) keys:
            episode, step, reward, avg100, su_rate, pu_sinr, p_s,
            sinr_p, sinr_s, r_s, exploration_noise, is_training,
            env_state (np.ndarray or None), buffer_size, paused
        """
        if not self._initialized:
            return

        screen = self._screen
        screen.fill(BG_COLOR)

        # Advance pulse animation
        speed_val = self.SPEED_OPTIONS[self._speed_idx][1]
        phase_inc = 0.008 if speed_val < 10 else 0.0   # no animation at max speed
        self._pulse_phase = (self._pulse_phase + phase_inc) % 1.0

        # ── Top status bar ────────────────────────────────────────────────────
        top_rect = pygame.Rect(0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT)
        pygame.draw.rect(screen, PANEL_BG, top_rect)
        pygame.draw.line(screen, PANEL_BORDER, (0, TOP_BAR_HEIGHT), (WINDOW_WIDTH, TOP_BAR_HEIGHT), 1)
        self._draw_top_bar(screen, ts)

        # ── Left panel — network visualization ───────────────────────────────
        panel_offset = (0, TOP_BAR_HEIGHT)
        env_state = ts.get("env_state")
        if env_state is None:
            env_state = np.zeros(7, dtype=np.float32)

        self._network_panel.render(
            surface=screen,
            offset=panel_offset,
            state=env_state,
            p_s=ts.get("p_s", 0.0),
            sinr_p=ts.get("sinr_p", 0.0),
            sinr_s=ts.get("sinr_s", 0.0),
            r_s=ts.get("r_s", 0.0),
            pulse_phase=self._pulse_phase,
            show_interference=self._show_interf,
        )

        # ── Right panel — live plots ──────────────────────────────────────────
        rx = LEFT_PANEL_WIDTH + 4
        ry = TOP_BAR_HEIGHT
        plot_h = (WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT) // 3

        # Push data every frame (plots handle deque size internally)
        ep_done = ts.get("_episode_done", False)
        if ep_done:
            self._reward_plot.push(ts.get("reward", 0.0), ts.get("avg100", 0.0))

        self._throughput_plot.push(ts.get("r_s", 0.0))
        self._sinr_plot.push(ts.get("sinr_p", 0.0))

        # Render and blit each plot
        for i, plot in enumerate([self._reward_plot, self._throughput_plot, self._sinr_plot]):
            surf = plot.render()
            screen.blit(surf, (rx, ry + i * plot_h))
            # Separator line
            if i < 2:
                pygame.draw.line(screen, PANEL_BORDER,
                                 (rx, ry + (i + 1) * plot_h),
                                 (rx + RIGHT_PANEL_WIDTH - 8, ry + (i + 1) * plot_h), 1)

        # Vertical separator: left | plots
        pygame.draw.line(screen, PANEL_BORDER,
                         (LEFT_PANEL_WIDTH, TOP_BAR_HEIGHT),
                         (LEFT_PANEL_WIDTH, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT), 1)

        # Vertical separator: plots | insights
        ix = LEFT_PANEL_WIDTH + RIGHT_PANEL_WIDTH
        pygame.draw.line(screen, PANEL_BORDER,
                         (ix, TOP_BAR_HEIGHT),
                         (ix, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT), 1)

        # ── AI Insights panel (far right) ─────────────────────────────────────
        self._insights_panel.render(
            surface=screen,
            ox=ix,
            oy=TOP_BAR_HEIGHT,
            stats=ts.get("insights", {}),
        )

        # ── Bottom control bar ────────────────────────────────────────────────
        bot_y = WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT
        bot_rect = pygame.Rect(0, bot_y, WINDOW_WIDTH, BOTTOM_BAR_HEIGHT)
        pygame.draw.rect(screen, PANEL_BG, bot_rect)
        pygame.draw.line(screen, PANEL_BORDER, (0, bot_y), (WINDOW_WIDTH, bot_y), 1)
        self._draw_bottom_bar(screen, bot_y, ts)

        pygame.display.flip()

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _draw_top_bar(self, screen: pygame.Surface, ts: dict) -> None:
        episode  = ts.get("episode", 0)
        step     = ts.get("step", 0)
        reward   = ts.get("reward", 0.0)
        avg100   = ts.get("avg100", 0.0)
        noise    = ts.get("exploration_noise", 0.0)
        buf_size = ts.get("buffer_size", 0)
        from utils import training_status
        from config import MIN_SAMPLES, TRAINING_EPISODES
        status = training_status(episode, buf_size, avg100)

        # Status badge color
        status_color = {
            "Exploring": ORANGE,
            "Learning":  BLUE,
            "Converging": CYAN,
            "Training":  YELLOW,
        }.get(status, WHITE)

        y = (TOP_BAR_HEIGHT - 16) // 2
        x = 10

        def item(label: str, value: str, color: tuple = WHITE) -> None:
            nonlocal x
            lbl = self._font_small.render(label + ":", True, GRAY)
            val = self._font_top.render(value, True, color)
            screen.blit(lbl, (x, y + 2))
            x += lbl.get_width() + 2
            screen.blit(val, (x, y))
            x += val.get_width() + 18

        item("Episode", f"{episode:4d}/{ts.get('total_episodes', 3000)}")
        item("Step",    f"{step:3d}/{ts.get('steps_ep', 200)}")
        item("Reward",  f"{reward:+.4f}", GREEN if reward >= 0 else RED)
        item("Avg100",  f"{avg100:+.4f}", GREEN if avg100 >= 0 else ORANGE)
        item("Noise",   f"{noise:.4f}", LIGHT_GRAY)
        item("Buffer",  f"{buf_size:6d}", LIGHT_GRAY)

        # Status badge on right
        badge_txt = self._font_top.render(f"[ {status} ]", True, status_color)
        screen.blit(badge_txt, (WINDOW_WIDTH - badge_txt.get_width() - 12, y))

        # Paused indicator
        if ts.get("paused", False):
            p_txt = self._font_top.render("⏸ PAUSED", True, YELLOW)
            screen.blit(p_txt, (WINDOW_WIDTH - badge_txt.get_width() - p_txt.get_width() - 24, y))

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _draw_bottom_bar(self, screen: pygame.Surface, bot_y: int, ts: dict) -> None:
        x = 10
        cy = bot_y + BOTTOM_BAR_HEIGHT // 2
        bh = 28
        bw_speed = 42

        # Speed buttons
        speed_label = self._font_small.render("Speed:", True, GRAY)
        screen.blit(speed_label, (x, cy - speed_label.get_height() // 2))
        x += speed_label.get_width() + 6

        self._speed_btn_rects = []
        for i, (label, _) in enumerate(self.SPEED_OPTIONS):
            rect = pygame.Rect(x, cy - bh // 2, bw_speed, bh)
            active = (i == self._speed_idx)
            self._draw_button(screen, rect, label, active=active)
            self._speed_btn_rects.append(rect)
            x += bw_speed + 4
        x += 10

        # Pause / Resume
        pause_label = "Resume" if self._paused else "Pause"
        p_rect = pygame.Rect(x, cy - bh // 2, 70, bh)
        self._draw_button(screen, p_rect, pause_label, active=self._paused, color=YELLOW)
        self._btn_pause_rect = p_rect
        x += 80

        # Interference toggle
        interf_label = "Interf: ON" if self._show_interf else "Interf: OFF"
        interf_color = ORANGE if self._show_interf else GRAY
        i_rect = pygame.Rect(x, cy - bh // 2, 90, bh)
        self._draw_button(screen, i_rect, interf_label, active=self._show_interf, color=interf_color)
        self._btn_interf_rect = i_rect
        x += 100

        # Reset
        r_rect = pygame.Rect(x, cy - bh // 2, 60, bh)
        self._draw_button(screen, r_rect, "Reset", active=False, color=RED)
        self._btn_reset_rect = r_rect
        x += 70

        # Keyboard shortcuts hint
        hint = self._font_small.render(
            "  Space=Pause  R=Reset  I=Interference", True, DARK_GRAY
        )
        screen.blit(hint, (x, cy - hint.get_height() // 2))

    def _draw_button(self, screen: pygame.Surface, rect: pygame.Rect,
                     label: str, active: bool = False,
                     color: tuple = BLUE) -> None:
        """Draw a rounded-rectangle button."""
        bg = tuple(min(255, int(c * 0.4)) for c in color) if not active else \
             tuple(min(255, int(c * 0.7)) for c in color)
        border_c = color

        pygame.draw.rect(screen, bg,       rect, border_radius=5)
        pygame.draw.rect(screen, border_c, rect, 1, border_radius=5)

        txt = self._font_btn.render(label, True, WHITE if active else LIGHT_GRAY)
        screen.blit(txt, (
            rect.centerx - txt.get_width()  // 2,
            rect.centery - txt.get_height() // 2,
        ))

    # ── Plot data management ──────────────────────────────────────────────────

    def clear_plots(self) -> None:
        """Reset all live plots (called on Reset)."""
        if self._reward_plot:
            self._reward_plot.clear()
        if self._throughput_plot:
            self._throughput_plot.clear()
        if self._sinr_plot:
            self._sinr_plot.clear()

    def push_episode_reward(self, raw: float, avg: float) -> None:
        """Push per-episode reward data (call once per episode end)."""
        self._reward_plot.push(raw, avg)

    def push_step_data(self, r_s: float, sinr_p: float) -> None:
        """Push per-step throughput and SINR data."""
        self._throughput_plot.push(r_s)
        self._sinr_plot.push(sinr_p)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def quit(self) -> None:
        pygame.quit()

    @property
    def current_speed(self) -> int:
        return self.SPEED_OPTIONS[self._speed_idx][1]

    @property
    def paused(self) -> bool:
        return self._paused
