# =============================================================================
# visualization.py — Real-time Pygame GUI for the CRN TD3 simulation
#
# Redesigned for presentation quality: modern dark-dashboard aesthetic.
#   - PlotSurface   : scrolling line plot with shaded area fills
#   - NetworkPanel  : animated 4-node topology with glow nodes & circuit grid
#   - InsightsPanel : card-based AI training analysis sidebar
#   - PygameRenderer: top-level compositor owning the window
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
    FPS_CAP,
)

# =============================================================================
# Design system — local colour/style constants
# =============================================================================

DS = {
    "bg":         (8,  11, 22),      # Window fill
    "panel":      (12, 16, 32),      # Panel fill
    "card":       (17, 22, 42),      # Card / plot fill
    "card_hi":    (22, 29, 55),      # Slightly lighter card
    "border":     (32, 43, 74),      # Subtle border
    "border_hi":  (52, 70, 118),     # Active / accent border
    "sep":        (24, 32, 58),      # Separator lines

    "text_hi":    (230, 237, 255),   # Primary text
    "text_sec":   (120, 140, 182),   # Secondary text
    "text_muted": (55,  70, 105),    # Muted / hint text

    "blue":       (59,  130, 246),   # Accent blue
    "teal":       (20,  184, 166),   # Accent teal
    "green":      (34,  197, 94),    # Accent green
    "red":        (239, 68,  68),    # Accent red
    "amber":      (245, 158, 11),    # Accent amber
    "orange":     (249, 115, 22),    # Accent orange
    "purple":     (168, 85,  247),   # Accent purple
    "cyan":       (6,   182, 212),   # Accent cyan

    "node_pt":    (52,  130, 219),
    "node_pr":    (34,  197, 94),
    "node_st":    (239, 68,  68),
    "node_sr":    (168, 85,  247),
}


# ── Font loader ────────────────────────────────────────────────────────────────

def _sysfont(names: list[str], size: int, bold: bool = False) -> pygame.font.Font:
    """Load first available system font from the list."""
    for name in names:
        f = pygame.font.SysFont(name, size, bold=bold)
        if f is not None:
            return f
    return pygame.font.SysFont(None, size, bold=bold)

_SANS  = ["segoeui", "calibri", "tahoma", "arial", "helveticaneue", "helvetica", "freesans"]
_MONO  = ["cascadiacodepl", "jetbrainsmono", "consolasligaturized", "consolas", "couriernew"]


# =============================================================================
# Helper: draw a rounded filled rect with optional gradient header stripe
# =============================================================================

def _card(surf: pygame.Surface, rect: pygame.Rect,
          bg: tuple = None, border: tuple = None,
          radius: int = 8) -> None:
    bg     = bg     or DS["card"]
    border = border or DS["border"]
    pygame.draw.rect(surf, bg,     rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, 1, border_radius=radius)


def _accent_bar(surf: pygame.Surface, x: int, y: int, w: int, h: int,
                col: tuple, fraction: float, radius: int = 4) -> None:
    """Rounded progress bar — background then filled portion."""
    pygame.draw.rect(surf, DS["card"], (x, y, w, h), border_radius=radius)
    pygame.draw.rect(surf, DS["border"], (x, y, w, h), 1, border_radius=radius)
    fw = max(0, int(w * min(1.0, fraction)))
    if fw > 2:
        pygame.draw.rect(surf, col, (x, y, fw, h), border_radius=radius)


def _glow_circle(surf: pygame.Surface, cx: int, cy: int,
                 radius: int, color: tuple, layers: int = 5) -> None:
    """Draw a soft glow around a circle using alpha surfaces."""
    for i in range(layers, 0, -1):
        r    = radius + i * 4
        alpha = max(0, 55 - i * 10)
        g_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(g_surf, (*color, alpha), (r + 1, r + 1), r)
        surf.blit(g_surf, (cx - r - 1, cy - r - 1))


# =============================================================================
# PlotSurface — modern scrolling line plot with shaded area fill
# =============================================================================

class PlotSurface:
    """
    Self-contained scrolling line plot with:
      - Semi-transparent area fill under each series
      - Subtle dotted grid
      - Rounded card background
      - Clean sans-serif labels
    """

    PAD_L = 48
    PAD_R = 12
    PAD_T = 36
    PAD_B = 24

    def __init__(
        self,
        width:          int,
        height:         int,
        title:          str,
        y_label:        str   = "",
        max_points:     int   = 500,
        series_colors:  list  = None,
        series_names:   list  = None,
        y_min:          float | None = None,
        y_max:          float | None = None,
        threshold:      float | None = None,
        threshold_color: tuple = None,
        header_color:   tuple = None,
    ):
        self.width          = width
        self.height         = height
        self.title          = title
        self.y_label        = y_label
        self.max_points     = max_points
        self.series_colors  = series_colors or [DS["cyan"]]
        self.series_names   = series_names  or [""]
        self.y_min_fixed    = y_min
        self.y_max_fixed    = y_max
        self.threshold      = threshold
        self.threshold_color = threshold_color or DS["amber"]
        self.header_color   = header_color or DS["blue"]

        n = len(self.series_colors)
        self._data: list[deque] = [deque(maxlen=max_points) for _ in range(n)]
        self._surface = pygame.Surface((width, height))
        self._fonts_ready = False

    # ── Font init (called lazily after pygame.init) ─────────────────────────

    def _init_fonts(self) -> None:
        if self._fonts_ready:
            return
        self._f_title = _sysfont(_SANS, 13, bold=True)
        self._f_tick  = _sysfont(_SANS, 10)
        self._f_label = _sysfont(_SANS, 10)
        self._fonts_ready = True

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, *values: float) -> None:
        for i, v in enumerate(values):
            if i < len(self._data):
                self._data[i].append(float(v))

    def clear(self) -> None:
        for d in self._data:
            d.clear()

    def render(self) -> pygame.Surface:
        self._init_fonts()
        surf = self._surface
        surf.fill(DS["bg"])

        pw = self.width  - self.PAD_L - self.PAD_R
        ph = self.height - self.PAD_T - self.PAD_B
        px = self.PAD_L
        py = self.PAD_T

        # ── Card background ────────────────────────────────────────────────────
        card_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(surf, DS["card"], card_rect, border_radius=6)
        pygame.draw.rect(surf, DS["border"], card_rect, 1, border_radius=6)

        # Accent left stripe
        stripe_rect = pygame.Rect(0, 0, 3, self.height)
        pygame.draw.rect(surf, self.header_color, stripe_rect,
                         border_radius=3)

        # Title
        t_surf = self._f_title.render(self.title, True, DS["text_hi"])
        surf.blit(t_surf, (px, 10))

        # Live value of last data point (rightmost series)
        if self._data and self._data[-1]:
            last_v = self._data[-1][-1]
            lv_col = self.series_colors[-1]
            lv_s   = self._f_title.render(f"{last_v:+.3f}", True, lv_col)
            surf.blit(lv_s, (self.width - self.PAD_R - lv_s.get_width(), 10))

        # ── Plot area background ───────────────────────────────────────────────
        plot_rect = pygame.Rect(px - 1, py - 1, pw + 2, ph + 2)
        pygame.draw.rect(surf, DS["panel"], plot_rect)

        # ── Compute y range ────────────────────────────────────────────────────
        all_vals = [v for d in self._data for v in d]
        if len(all_vals) < 2:
            return surf

        y_min = self.y_min_fixed if self.y_min_fixed is not None else min(all_vals)
        y_max = self.y_max_fixed if self.y_max_fixed is not None else max(all_vals)
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0

        def to_py(v: float) -> int:
            norm = (v - y_min) / (y_max - y_min)
            return int(py + ph * (1.0 - norm))

        # ── Dotted grid (5 horizontal) ─────────────────────────────────────────
        n_grid = 5
        for i in range(n_grid + 1):
            gy  = py + int(ph * i / n_grid)
            gv  = y_max - (y_max - y_min) * i / n_grid
            # Dashed horizontal line
            for gx in range(px, px + pw, 8):
                pygame.draw.line(surf, DS["sep"], (gx, gy), (min(gx + 4, px + pw), gy), 1)
            # Y-tick label
            tick_s = self._f_tick.render(f"{gv:.2f}", True, DS["text_muted"])
            surf.blit(tick_s, (px - tick_s.get_width() - 4,
                                gy - tick_s.get_height() // 2))

        # ── Threshold line ─────────────────────────────────────────────────────
        if self.threshold is not None:
            ty = to_py(self.threshold)
            if py <= ty <= py + ph:
                # Dashed threshold
                for tx in range(px, px + pw, 10):
                    pygame.draw.line(surf, self.threshold_color,
                                     (tx, ty), (min(tx + 6, px + pw), ty), 1)
                thr_s = self._f_tick.render(
                    f"thr {self.threshold:.1f}", True, self.threshold_color)
                surf.blit(thr_s, (px + pw - thr_s.get_width() - 2, ty - thr_s.get_height() - 2))

        # ── Data series ────────────────────────────────────────────────────────
        for d, color in zip(self._data, self.series_colors):
            pts = list(d)
            if len(pts) < 2:
                continue
            n_pts = len(pts)
            pixel_pts = []
            for i, v in enumerate(pts):
                ppx = px + int(i / (self.max_points - 1) * pw)
                ppy = to_py(v)
                pixel_pts.append((ppx, ppy))

            # ── Area fill (SRCALPHA polygon) ──────────────────────────────────
            base_y = to_py(max(y_min, 0.0))  # Fill down to zero or y_min
            base_y = max(py, min(py + ph, base_y))

            fill_pts = [(pixel_pts[0][0], base_y)]
            fill_pts.extend(pixel_pts)
            fill_pts.append((pixel_pts[-1][0], base_y))

            fill_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(fill_surf, (*color, 35), fill_pts)
            surf.blit(fill_surf, (0, 0))

            # ── Line ──────────────────────────────────────────────────────────
            pygame.draw.lines(surf, color, False, pixel_pts, 2)

        # ── Legend ────────────────────────────────────────────────────────────
        lx = px + 4
        for name, color in zip(self.series_names, self.series_colors):
            if name:
                pygame.draw.rect(surf, color, (lx, py + 4, 14, 3), border_radius=1)
                ns = self._f_label.render(name, True, color)
                surf.blit(ns, (lx + 17, py + 1))
                lx += 17 + ns.get_width() + 10

        # ── Y-axis label (rotated) ────────────────────────────────────────────
        if self.y_label:
            yl_s    = self._f_label.render(self.y_label, True, DS["text_muted"])
            rotated = pygame.transform.rotate(yl_s, 90)
            surf.blit(rotated, (6, py + ph // 2 - rotated.get_height() // 2))

        return surf


# =============================================================================
# NetworkPanel — animated 4-node CRN diagram
# =============================================================================

class NetworkPanel:
    """
    Left panel: animated topology with circuit-board grid background,
    glowing nodes, pulse animations, and a channel-quality HUD.
    """

    _NODE_R = 34   # Node radius

    _NODE_COL = {
        "PT": DS["node_pt"],
        "PR": DS["node_pr"],
        "ST": DS["node_st"],
        "SR": DS["node_sr"],
    }

    def __init__(self, width: int = LEFT_PANEL_WIDTH,
                 height: int = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT):
        self.width  = width
        self.height = height
        self.positions = dict(NODE_POSITIONS)
        self._fonts_ready = False

    def _init_fonts(self) -> None:
        if self._fonts_ready:
            return
        self._f_node  = _sysfont(_SANS, 13, bold=True)
        self._f_label = _sysfont(_SANS, 11, bold=True)
        self._f_value = _sysfont(_SANS, 11)
        self._f_small = _sysfont(_SANS, 10)
        self._f_title = _sysfont(_SANS, 13, bold=True)
        self._fonts_ready = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(
        self,
        surface:          pygame.Surface,
        offset:           tuple,
        state:            np.ndarray,
        p_s:              float,
        sinr_p:           float,
        sinr_s:           float,
        r_s:              float,
        pulse_phase:      float,
        show_interference: bool = True,
    ) -> None:
        self._init_fonts()
        ox, oy = offset

        # ── Panel background ───────────────────────────────────────────────────
        panel_rect = pygame.Rect(ox, oy, self.width, self.height)
        pygame.draw.rect(surface, DS["panel"], panel_rect)

        # Circuit-board dot grid
        for gx in range(ox + 18, ox + self.width - 8, 28):
            for gy in range(oy + 18, oy + self.height - 8, 28):
                pygame.draw.circle(surface, DS["sep"], (gx, gy), 1)

        # Panel border
        pygame.draw.rect(surface, DS["border"], panel_rect, 1)

        pt = self._abs(ox, oy, "PT")
        pr = self._abs(ox, oy, "PR")
        st = self._abs(ox, oy, "ST")
        sr = self._abs(ox, oy, "SR")

        pu_ok = sinr_p >= SINR_THRESHOLD
        link_col = DS["green"] if pu_ok else DS["red"]

        # ── Interference links (underneath) ───────────────────────────────────
        if show_interference:
            self._dashed_line(surface, DS["red"],    st, pr, dash=9, gap=5, width=2)
            self._dashed_line(surface, DS["orange"], pt, sr, dash=9, gap=5, width=2)
            self._pulse(surface, st, pr, pulse_phase,              DS["red"],    4)
            self._pulse(surface, pt, sr, (pulse_phase + 0.5) % 1.0, DS["orange"], 4)

        # ── Primary link PT→PR ─────────────────────────────────────────────────
        # Glow line
        self._glow_line(surface, link_col, pt, pr, glow_w=6, line_w=3)
        self._pulse(surface, pt, pr, pulse_phase, link_col, 6)

        # ── SU link ST→SR ──────────────────────────────────────────────────────
        su_w = max(2, min(7, int(2 + r_s)))
        self._glow_line(surface, DS["blue"], st, sr, glow_w=su_w + 4, line_w=su_w)
        self._pulse(surface, st, sr, (pulse_phase + 0.25) % 1.0, DS["blue"], 5)

        # ── Nodes ──────────────────────────────────────────────────────────────
        halo_pu = DS["green"] if pu_ok else DS["red"]
        halo_su = DS["green"] if sinr_s > 0.1 else DS["orange"]

        self._node(surface, pt, DS["node_pt"], "PT", halo_pu)
        self._node(surface, pr, DS["node_pr"], "PR", halo_pu)
        self._node(surface, st, DS["node_st"], "ST", halo_su)
        self._node(surface, sr, DS["node_sr"], "SR", halo_su)

        # ── Value HUD (metric cards near nodes) ────────────────────────────────
        self._hud_card(surface, pt[0] - 80, pt[1] - 68,
                       "P_p", "1.00 W", DS["text_sec"])
        sinr_col = DS["green"] if pu_ok else DS["red"]
        self._hud_card(surface, pr[0] - 20, pr[1] - 68,
                       "SINR_p", f"{sinr_p:.2f}", sinr_col)
        self._hud_card(surface, st[0] - 80, st[1] + 44,
                       "P_s", f"{p_s:.3f} W", DS["blue"])
        self._hud_card(surface, sr[0] - 20, sr[1] + 44,
                       "R_s", f"{r_s:.3f} b/s/Hz", DS["cyan"])
        self._hud_card(surface, sr[0] - 20, sr[1] + 76,
                       "SINR_s", f"{sinr_s:.2f}", DS["teal"])

        # ── Channel quality bars (bottom area) ────────────────────────────────
        if state is not None and len(state) >= 4:
            self._channel_bars(surface, ox + 12, oy + self.height - 105,
                               state[:4])

        # ── Power bar (ST) ────────────────────────────────────────────────────
        self._power_bar(surface, ox + self.width - 42, oy + 140, p_s)

        # ── Panel title ────────────────────────────────────────────────────────
        title_s = self._f_title.render("NETWORK  TOPOLOGY", True, DS["teal"])
        surface.blit(title_s, (ox + 10, oy + 8))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _abs(self, ox: int, oy: int, node: str) -> tuple:
        x, y = self.positions[node]
        return (ox + x, oy + y)

    def _node(self, surface: pygame.Surface, pos: tuple, color: tuple,
              label: str, halo: tuple) -> None:
        x, y = pos
        r = self._NODE_R

        # Glow
        _glow_circle(surface, x, y, r, halo, layers=4)

        # Drop shadow
        pygame.draw.circle(surface, (4, 5, 12), (x + 3, y + 4), r)

        # Filled body
        pygame.draw.circle(surface, color, (x, y), r)

        # Inner highlight ring (lighter top-left)
        hi = tuple(min(255, c + 60) for c in color)
        pygame.draw.circle(surface, hi, (x, y), r, 3)

        # White border
        pygame.draw.circle(surface, (200, 215, 245), (x, y), r, 2)

        # Label
        txt = self._f_node.render(label, True, (240, 245, 255))
        surface.blit(txt, (x - txt.get_width() // 2, y - txt.get_height() // 2))

    def _hud_card(self, surface: pygame.Surface,
                  x: int, y: int, key: str, value: str, val_col: tuple,
                  w: int = 130, h: int = 28) -> None:
        """Small labelled value card."""
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, DS["card"], rect, border_radius=5)
        pygame.draw.rect(surface, DS["border"], rect, 1, border_radius=5)

        key_s = self._f_small.render(key, True, DS["text_muted"])
        val_s = self._f_label.render(value, True, val_col)
        surface.blit(key_s, (x + 6, y + 3))
        surface.blit(val_s, (x + w - val_s.get_width() - 6, y + h - val_s.get_height() - 2))

    def _dashed_line(self, surface: pygame.Surface, color: tuple,
                     start: tuple, end: tuple,
                     dash: int = 9, gap: int = 5, width: int = 2) -> None:
        sx, sy = start; ex, ey = end
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        t, on = 0.0, True
        while t < length:
            seg = dash if on else gap
            tn  = min(t + seg, length)
            if on:
                p1 = (int(sx + ux * t),  int(sy + uy * t))
                p2 = (int(sx + ux * tn), int(sy + uy * tn))
                pygame.draw.line(surface, color, p1, p2, width)
            t, on = tn, not on

    def _glow_line(self, surface: pygame.Surface, color: tuple,
                   start: tuple, end: tuple,
                   glow_w: int = 6, line_w: int = 3) -> None:
        """Draw a line with a soft halo underneath."""
        glow_col = (*color, 40)
        g_surf   = pygame.Surface((self.width + 20, self.height + 20), pygame.SRCALPHA)
        ox_off   = surface.get_clip().x if surface.get_clip() else 0
        pygame.draw.line(g_surf, glow_col,
                         (start[0], start[1]),
                         (end[0],   end[1]), glow_w + 2)
        surface.blit(g_surf, (0, 0))
        pygame.draw.line(surface, color, start, end, line_w)

    def _pulse(self, surface: pygame.Surface, start: tuple, end: tuple,
               phase: float, color: tuple, radius: int = 5) -> None:
        sx, sy = start; ex, ey = end
        px = int(sx + (ex - sx) * phase)
        py = int(sy + (ey - sy) * phase)
        pygame.draw.circle(surface, (255, 255, 255), (px, py), radius + 2)
        pygame.draw.circle(surface, color,           (px, py), radius)

    def _power_bar(self, surface: pygame.Surface,
                   x: int, y: int, p_s: float,
                   bar_w: int = 18, bar_h: int = 200) -> None:
        fill_ratio = p_s / P_MAX
        fill_h     = int(bar_h * fill_ratio)

        # Background
        pygame.draw.rect(surface, DS["card"],   (x, y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(surface, DS["border"], (x, y, bar_w, bar_h), 1, border_radius=4)

        # Colour by power level
        if fill_ratio < 0.35:
            col = DS["green"]
        elif fill_ratio < 0.65:
            col = DS["amber"]
        else:
            col = DS["red"]

        if fill_h > 0:
            top_y = y + bar_h - fill_h
            pygame.draw.rect(surface, col, (x + 2, top_y, bar_w - 4, fill_h),
                             border_radius=3)

        # Ticks every 25%
        for frac in [0.25, 0.5, 0.75]:
            ty = y + int(bar_h * (1 - frac))
            pygame.draw.line(surface, DS["border_hi"], (x - 3, ty), (x + bar_w + 3, ty), 1)

        # Labels
        lbl_s = self._f_small.render("Tx", True, DS["text_muted"])
        surface.blit(lbl_s, (x + bar_w // 2 - lbl_s.get_width() // 2, y - 16))
        pct_s = self._f_small.render(f"{fill_ratio*100:.0f}%", True, col)
        surface.blit(pct_s, (x + bar_w // 2 - pct_s.get_width() // 2, y + bar_h + 4))

    def _channel_bars(self, surface: pygame.Surface,
                      x: int, y: int, gains: np.ndarray) -> None:
        """Draw 4 mini channel-gain bars (h_pp, h_sp, h_ss, h_ps)."""
        labels = ["h_pp", "h_sp", "h_ss", "h_ps"]
        colors = [DS["teal"], DS["red"], DS["blue"], DS["orange"]]
        bw, bh, gap = 66, 8, 6

        title_s = self._f_small.render("CHANNEL GAINS", True, DS["text_muted"])
        surface.blit(title_s, (x, y - 16))

        for i, (lbl, col, gain) in enumerate(zip(labels, colors, gains)):
            bx = x + i * (bw + gap)
            # Background bar
            pygame.draw.rect(surface, DS["card"],   (bx, y, bw, bh), border_radius=3)
            pygame.draw.rect(surface, DS["border"], (bx, y, bw, bh), 1, border_radius=3)
            # Fill (cap at 3 for display)
            fw = max(0, int(bw * min(1.0, gain / 3.0)))
            if fw > 0:
                pygame.draw.rect(surface, col, (bx, y, fw, bh), border_radius=3)
            # Label
            lbl_s = self._f_small.render(lbl, True, DS["text_muted"])
            val_s = self._f_small.render(f"{gain:.2f}", True, col)
            surface.blit(lbl_s, (bx, y + bh + 2))
            surface.blit(val_s, (bx + bw - val_s.get_width(), y + bh + 2))


# =============================================================================
# InsightsPanel — card-based AI training analysis sidebar
# =============================================================================

class InsightsPanel:
    """
    Far-right sidebar with card-based sections:
      Training Stage badge | Progress | PU Protection | SU Throughput
      Reward Trend | Avg Power | Violations | Policy Insight | Buffer
    """

    ACCENT  = DS["teal"]
    PAD     = 10

    def __init__(self, width: int = 300,
                 height: int = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT):
        self.width  = width
        self.height = height
        self._fonts_ready = False

    def _init_fonts(self) -> None:
        if self._fonts_ready:
            return
        self._f_title   = _sysfont(_SANS, 15, bold=True)
        self._f_section = _sysfont(_SANS, 10, bold=True)
        self._f_badge   = _sysfont(_SANS, 14, bold=True)
        self._f_big     = _sysfont(_SANS, 22, bold=True)
        self._f_value   = _sysfont(_SANS, 12, bold=True)
        self._f_label   = _sysfont(_SANS, 11)
        self._f_small   = _sysfont(_SANS, 10)
        self._fonts_ready = True

    # ── render ─────────────────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface, ox: int, oy: int,
               stats: dict) -> None:
        self._init_fonts()
        w, h  = self.width, self.height
        pad   = self.PAD
        cx    = ox + w // 2

        # Panel background
        pygame.draw.rect(surface, DS["panel"], pygame.Rect(ox, oy, w, h))
        pygame.draw.rect(surface, DS["border"], pygame.Rect(ox, oy, w, h), 1)

        y = oy + 10

        # ── Title ───────────────────────────────────────────────────────────────
        self._text_c(surface, "AI  INSIGHTS", cx, y + 8, self._f_title, self.ACCENT)
        y += 24
        pygame.draw.line(surface, self.ACCENT, (ox + pad, y), (ox + w - pad, y), 2)
        y += 10

        # ── Training Stage ──────────────────────────────────────────────────────
        stage       = stats.get("stage", "Exploring")
        stage_color = {
            "Exploring":  DS["orange"],
            "Learning":   DS["blue"],
            "Converging": DS["teal"],
            "Converged":  DS["green"],
        }.get(stage, DS["amber"])

        y = self._card_block(surface, ox, y, w, pad, "TRAINING  STAGE")
        brect = pygame.Rect(ox + pad, y, w - pad * 2, 30)
        bg    = tuple(max(0, min(255, int(c * 0.15))) for c in stage_color)
        pygame.draw.rect(surface, bg,          brect, border_radius=6)
        pygame.draw.rect(surface, stage_color, brect, 2, border_radius=6)
        self._text_c(surface, stage.upper(), cx, y + 15, self._f_badge, stage_color)
        y += 40

        # ── Progress ────────────────────────────────────────────────────────────
        episode = stats.get("episode", 0)
        total   = stats.get("total_episodes", 3000)
        pct     = min(1.0, episode / max(1, total))

        y = self._card_block(surface, ox, y, w, pad, "PROGRESS")
        _accent_bar(surface, ox + pad, y, w - pad * 2, 12, DS["blue"], pct, radius=6)
        y += 14
        self._text_c(surface, f"Ep {episode:,} / {total:,}  ({pct*100:.1f}%)",
                     cx, y + 5, self._f_small, DS["text_sec"])
        y += 18

        # ── PU Protection ───────────────────────────────────────────────────────
        cr       = stats.get("constraint_rate", 0.0)
        cr_col   = DS["green"] if cr >= 0.85 else (DS["amber"] if cr >= 0.60 else DS["red"])

        y = self._card_block(surface, ox, y, w, pad, "PU  PROTECTION  RATE")
        # Big percentage display
        big_s = self._f_big.render(f"{cr*100:.1f}%", True, cr_col)
        surface.blit(big_s, (cx - big_s.get_width() // 2, y))
        y += big_s.get_height() + 4
        _accent_bar(surface, ox + pad, y, w - pad * 2, 8, cr_col, cr, radius=4)
        y += 10
        self._text_c(surface, f"SINR_p >= {SINR_THRESHOLD:.1f}  requirement",
                     cx, y + 4, self._f_small, DS["text_muted"])
        y += 16

        # ── SU Throughput ───────────────────────────────────────────────────────
        avg_rs  = stats.get("avg_r_s",  0.0)
        peak_rs = stats.get("peak_r_s", 0.0)

        y = self._card_block(surface, ox, y, w, pad, "SU  THROUGHPUT")
        big_r = self._f_big.render(f"{avg_rs:.3f}", True, DS["blue"])
        surface.blit(big_r, (cx - big_r.get_width() // 2, y))
        unit_s = self._f_small.render("bits/s/Hz  (current avg)", True, DS["text_muted"])
        surface.blit(unit_s, (cx - unit_s.get_width() // 2, y + big_r.get_height() + 1))
        y += big_r.get_height() + 14
        self._stat_row(surface, ox + pad, y, w - pad * 2,
                       "Session peak", f"{peak_rs:.3f}", self.ACCENT)
        y += 18

        # ── Reward ─────────────────────────────────────────────────────────────
        trend_raw   = stats.get("reward_trend", "[=]  Stable")
        avg_r       = stats.get("avg_reward",  0.0)
        best_r      = stats.get("best_reward", 0.0)
        trend_col   = DS["green"] if "UP" in trend_raw.upper() \
                      else (DS["red"] if "DN" in trend_raw.upper() else DS["amber"])

        y = self._card_block(surface, ox, y, w, pad, "REWARD  TREND")
        self._text_c(surface, trend_raw, cx, y + 8, self._f_badge, trend_col)
        y += 22
        self._stat_row(surface, ox + pad, y, w - pad * 2,
                       "Avg100", f"{avg_r:+.2f}", DS["text_sec"])
        y += 15
        self._stat_row(surface, ox + pad, y, w - pad * 2,
                       "Best ep", f"{best_r:+.2f}", DS["amber"])
        y += 18

        # ── Avg Power ──────────────────────────────────────────────────────────
        avg_pw  = stats.get("avg_power", 0.5)
        pw_pct  = avg_pw / P_MAX
        pw_col  = DS["green"] if pw_pct < 0.35 else (DS["amber"] if pw_pct < 0.65 else DS["red"])
        pw_tr   = stats.get("power_trend", "[=] Stable")

        y = self._card_block(surface, ox, y, w, pad, "AVG  TRANSMIT  POWER")
        _accent_bar(surface, ox + pad, y, w - pad * 2, 10, pw_col, pw_pct, radius=5)
        y += 12
        self._stat_row(surface, ox + pad, y, w - pad * 2,
                       pw_tr, f"{avg_pw:.3f} W", pw_col)
        y += 18

        # ── Violations ─────────────────────────────────────────────────────────
        viols     = stats.get("violations_last100", 0)
        viol_pct  = viols / 100.0
        viol_col  = DS["red"] if viol_pct > 0.3 else (DS["amber"] if viol_pct > 0.1 else DS["green"])

        y = self._card_block(surface, ox, y, w, pad, "PU  VIOLATIONS  /100")
        _accent_bar(surface, ox + pad, y, w - pad * 2, 10, viol_col, viol_pct, radius=5)
        y += 12
        self._text_c(surface, f"{viols} violations  ({viol_pct*100:.0f}%)",
                     cx, y + 4, self._f_small, viol_col)
        y += 16

        # ── Policy Insight ──────────────────────────────────────────────────────
        insight = stats.get("policy_insight", "Initialising...")
        y = self._card_block(surface, ox, y, w, pad, "POLICY  INSIGHT")
        y = self._wrapped_text(surface, insight, ox + pad, y,
                               w - pad * 2, self._f_small, DS["text_sec"])
        y += 8

        # ── Replay Buffer ───────────────────────────────────────────────────────
        buf       = stats.get("buffer_size", 0)
        trained   = stats.get("is_training", False)
        buf_col   = DS["green"] if trained else DS["orange"]
        buf_label = "[ACTIVE]" if trained else "[FILLING]"

        y = self._card_block(surface, ox, y, w, pad, "REPLAY  BUFFER")
        _accent_bar(surface, ox + pad, y, w - pad * 2, 10,
                    buf_col, min(1.0, buf / 100_000), radius=5)
        y += 12
        self._text_c(surface, f"{buf:,} / 100,000   {buf_label}",
                     cx, y + 4, self._f_small, buf_col)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _card_block(self, surf: pygame.Surface,
                    ox: int, y: int, w: int, pad: int, title: str) -> int:
        """Draw section label with a thin accent line. Returns new y."""
        pygame.draw.line(surf, DS["sep"], (ox + pad, y + 6), (ox + w - pad, y + 6), 1)
        t = self._f_section.render(title, True, self.ACCENT)
        surf.blit(t, (ox + pad, y))
        return y + 16

    def _text_c(self, surf, text, cx, cy, font, color) -> None:
        s = font.render(str(text), True, color)
        surf.blit(s, (cx - s.get_width() // 2, cy - s.get_height() // 2))

    def _stat_row(self, surf, x, y, w, key, value, val_color) -> None:
        key_s = self._f_label.render(key, True, DS["text_muted"])
        val_s = self._f_value.render(value, True, val_color)
        surf.blit(key_s, (x, y))
        surf.blit(val_s, (x + w - val_s.get_width(), y - 1))

    def _wrapped_text(self, surf, text, x, y, max_w, font, color) -> int:
        words = text.split()
        line  = ""
        for word in words:
            test = line + word + " "
            if font.size(test)[0] > max_w and line:
                s = font.render(line.strip(), True, color)
                surf.blit(s, (x, y))
                y   += font.get_height() + 2
                line = word + " "
            else:
                line = test
        if line.strip():
            s = font.render(line.strip(), True, color)
            surf.blit(s, (x, y))
            y += font.get_height() + 2
        return y


# =============================================================================
# PygameRenderer — top-level GUI compositor
# =============================================================================

class PygameRenderer:
    """Owns the pygame window and all sub-panels."""

    SPEED_OPTIONS = [("1x", 1), ("5x", 5), ("10x", 10), ("Max", 9999)]

    def __init__(self):
        self._initialized     = False
        self._screen          = None
        self._network_panel   = None
        self._reward_plot     = None
        self._throughput_plot = None
        self._sinr_plot       = None
        self._insights_panel  = None

        self._pulse_phase  = 0.0
        self._speed_idx    = 0
        self._paused       = False
        self._show_interf  = True

        self._btn_pause_rect  = None
        self._btn_reset_rect  = None
        self._btn_interf_rect = None
        self._speed_btn_rects = []
        self._fonts_ready = False

    # ── Init ───────────────────────────────────────────────────────────────────

    def init(self) -> None:
        pygame.init()
        pygame.display.set_caption(
            "CRN TD3 — Cognitive Radio Network  |  RIT Bangalore")
        self._screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        self._f_top   = _sysfont(_SANS, 13, bold=True)
        self._f_small = _sysfont(_SANS, 11)
        self._f_btn   = _sysfont(_SANS, 12, bold=True)
        self._f_hint  = _sysfont(_SANS, 10)

        panel_h = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT
        self._network_panel = NetworkPanel(LEFT_PANEL_WIDTH, panel_h)

        plot_w = RIGHT_PANEL_WIDTH - 8
        plot_h = panel_h // 3

        self._reward_plot = PlotSurface(
            width=plot_w, height=plot_h,
            title="Episode Reward",
            y_label="reward",
            max_points=500,
            series_colors=[DS["text_muted"], DS["amber"]],
            series_names=["raw", "avg100"],
            header_color=DS["amber"],
        )
        self._throughput_plot = PlotSurface(
            width=plot_w, height=plot_h,
            title="SU Throughput  R_s  [bits/s/Hz]",
            y_label="R_s",
            max_points=500,
            series_colors=[DS["blue"]],
            series_names=["R_s"],
            y_min=0.0,
            header_color=DS["blue"],
        )
        self._sinr_plot = PlotSurface(
            width=plot_w, height=plot_h,
            title="PU SINR  (dashed = protection threshold)",
            y_label="SINR_p",
            max_points=500,
            series_colors=[DS["cyan"]],
            series_names=["SINR_p"],
            y_min=0.0,
            threshold=SINR_THRESHOLD,
            threshold_color=DS["amber"],
            header_color=DS["cyan"],
        )

        self._insights_panel = InsightsPanel(
            width=INSIGHTS_WIDTH,
            height=panel_h,
        )
        self._initialized = True

    # ── Events ─────────────────────────────────────────────────────────────────

    def handle_events(self) -> dict:
        out = {"quit": False, "pause_toggle": False, "reset": False,
               "speed_change": None, "toggle_interference": False}

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
                if self._btn_pause_rect and self._btn_pause_rect.collidepoint(mx, my):
                    out["pause_toggle"] = True
                    self._paused = not self._paused
                elif self._btn_reset_rect and self._btn_reset_rect.collidepoint(mx, my):
                    out["reset"] = True
                elif self._btn_interf_rect and self._btn_interf_rect.collidepoint(mx, my):
                    out["toggle_interference"] = True
                    self._show_interf = not self._show_interf
                else:
                    for i, rect in enumerate(self._speed_btn_rects):
                        if rect.collidepoint(mx, my):
                            self._speed_idx = i
                            out["speed_change"] = self.SPEED_OPTIONS[i][1]
                            break

        return out

    # ── Frame render ───────────────────────────────────────────────────────────

    def update(self, ts: dict) -> None:
        if not self._initialized:
            return

        screen = self._screen
        screen.fill(DS["bg"])

        # Pulse phase
        speed_val = self.SPEED_OPTIONS[self._speed_idx][1]
        phase_inc = 0.009 if speed_val < 10 else 0.0
        self._pulse_phase = (self._pulse_phase + phase_inc) % 1.0

        # ── Top bar ────────────────────────────────────────────────────────────
        top_rect = pygame.Rect(0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT)
        pygame.draw.rect(screen, DS["panel"], top_rect)
        # Accent bottom line on top bar
        pygame.draw.line(screen, self.ACCENT if True else DS["border"],
                         (0, TOP_BAR_HEIGHT - 1), (WINDOW_WIDTH, TOP_BAR_HEIGHT - 1), 2)
        self._draw_top_bar(screen, ts)

        # ── Network panel ──────────────────────────────────────────────────────
        env_state = ts.get("env_state")
        if env_state is None:
            env_state = np.zeros(7, dtype=np.float32)
        self._network_panel.render(
            surface=screen,
            offset=(0, TOP_BAR_HEIGHT),
            state=env_state,
            p_s=ts.get("p_s", 0.0),
            sinr_p=ts.get("sinr_p", 0.0),
            sinr_s=ts.get("sinr_s", 0.0),
            r_s=ts.get("r_s", 0.0),
            pulse_phase=self._pulse_phase,
            show_interference=self._show_interf,
        )

        # ── Plots ──────────────────────────────────────────────────────────────
        rx     = LEFT_PANEL_WIDTH + 4
        ry     = TOP_BAR_HEIGHT + 2
        plot_h = (WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT) // 3

        for i, plot in enumerate([self._reward_plot,
                                   self._throughput_plot,
                                   self._sinr_plot]):
            surf = plot.render()
            screen.blit(surf, (rx, ry + i * plot_h))
            if i < 2:
                pygame.draw.line(screen, DS["border"],
                                 (rx, ry + (i + 1) * plot_h),
                                 (rx + RIGHT_PANEL_WIDTH - 8, ry + (i + 1) * plot_h), 1)

        # Vertical separators
        for vx in [LEFT_PANEL_WIDTH, LEFT_PANEL_WIDTH + RIGHT_PANEL_WIDTH]:
            pygame.draw.line(screen, DS["border"],
                             (vx, TOP_BAR_HEIGHT),
                             (vx, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT), 1)

        # ── Insights panel ─────────────────────────────────────────────────────
        ix = LEFT_PANEL_WIDTH + RIGHT_PANEL_WIDTH
        self._insights_panel.render(
            surface=screen,
            ox=ix,
            oy=TOP_BAR_HEIGHT,
            stats=ts.get("insights", {}),
        )

        # ── Bottom bar ─────────────────────────────────────────────────────────
        bot_y = WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT
        pygame.draw.rect(screen, DS["panel"],
                         pygame.Rect(0, bot_y, WINDOW_WIDTH, BOTTOM_BAR_HEIGHT))
        pygame.draw.line(screen, DS["border_hi"],
                         (0, bot_y), (WINDOW_WIDTH, bot_y), 1)
        self._draw_bottom_bar(screen, bot_y, ts)

        pygame.display.flip()

    # Keep the accent reference accessible
    @property
    def ACCENT(self):
        return DS["teal"]

    # ── Top bar ────────────────────────────────────────────────────────────────

    def _draw_top_bar(self, screen: pygame.Surface, ts: dict) -> None:
        episode  = ts.get("episode", 0)
        step     = ts.get("step", 0)
        reward   = ts.get("reward", 0.0)
        avg100   = ts.get("avg100", 0.0)
        noise    = ts.get("exploration_noise", 0.0)
        buf_size = ts.get("buffer_size", 0)
        from utils import training_status
        status = training_status(episode, buf_size, avg100)

        status_col = {
            "Exploring":  DS["orange"],
            "Learning":   DS["blue"],
            "Converging": DS["teal"],
            "Training":   DS["amber"],
        }.get(status, DS["text_hi"])

        bar_cy = TOP_BAR_HEIGHT // 2
        x = 14

        def item(label: str, value: str, val_col: tuple = DS["text_hi"]) -> None:
            nonlocal x
            if x > 14:
                pipe = self._f_small.render("|", True, DS["sep"])
                screen.blit(pipe, (x, bar_cy - pipe.get_height() // 2))
                x += pipe.get_width() + 8
            lbl_s = self._f_hint.render(label, True, DS["text_muted"])
            val_s = self._f_top.render(value, True, val_col)
            screen.blit(lbl_s, (x, bar_cy - lbl_s.get_height() // 2 + 1))
            x += lbl_s.get_width() + 4
            screen.blit(val_s, (x, bar_cy - val_s.get_height() // 2))
            x += val_s.get_width() + 8

        item("Ep",     f"{episode}/{ts.get('total_episodes', 3000)}")
        item("Step",   f"{step}/{ts.get('steps_ep', 200)}")
        item("Reward", f"{reward:+.4f}", DS["green"] if reward >= 0 else DS["red"])
        item("Avg100", f"{avg100:+.4f}", DS["green"] if avg100 >= 0 else DS["orange"])
        item("Noise",  f"{noise:.4f}", DS["text_sec"])
        item("Buffer", f"{buf_size:,}", DS["text_sec"])

        if ts.get("paused", False):
            p_s = self._f_top.render("PAUSED", True, DS["amber"])
            bx  = WINDOW_WIDTH - p_s.get_width() - 120
            screen.blit(p_s, (bx, bar_cy - p_s.get_height() // 2))

        badge_s = self._f_top.render(f"[ {status.upper()} ]", True, status_col)
        screen.blit(badge_s, (WINDOW_WIDTH - badge_s.get_width() - 14,
                               bar_cy - badge_s.get_height() // 2))

    # ── Bottom bar ─────────────────────────────────────────────────────────────

    def _draw_bottom_bar(self, screen: pygame.Surface, bot_y: int, ts: dict) -> None:
        cy  = bot_y + BOTTOM_BAR_HEIGHT // 2
        bh  = 28
        bw  = 46
        x   = 14

        # "SPEED" label
        spd = self._f_hint.render("SPEED", True, DS["text_muted"])
        screen.blit(spd, (x, cy - spd.get_height() // 2))
        x += spd.get_width() + 8

        # Speed buttons
        self._speed_btn_rects = []
        for i, (lbl, _) in enumerate(self.SPEED_OPTIONS):
            rect   = pygame.Rect(x, cy - bh // 2, bw, bh)
            active = (i == self._speed_idx)
            self._btn(screen, rect, lbl, active=active, col=DS["blue"])
            self._speed_btn_rects.append(rect)
            x += bw + 3
        x += 16

        # Divider
        pygame.draw.line(screen, DS["border_hi"],
                         (x, bot_y + 6), (x, bot_y + bh + 6))
        x += 12

        # Pause
        p_lbl = "Resume" if self._paused else "Pause"
        p_rect = pygame.Rect(x, cy - bh // 2, 74, bh)
        self._btn(screen, p_rect, p_lbl, active=self._paused, col=DS["amber"])
        self._btn_pause_rect = p_rect
        x += 84

        # Interference
        i_lbl  = "Interf ON" if self._show_interf else "Interf OFF"
        i_rect = pygame.Rect(x, cy - bh // 2, 88, bh)
        self._btn(screen, i_rect, i_lbl, active=self._show_interf, col=DS["orange"])
        self._btn_interf_rect = i_rect
        x += 98

        # Reset
        r_rect = pygame.Rect(x, cy - bh // 2, 62, bh)
        self._btn(screen, r_rect, "Reset", active=False, col=DS["red"])
        self._btn_reset_rect = r_rect
        x += 78

        # Divider + hint text
        pygame.draw.line(screen, DS["border_hi"],
                         (x, bot_y + 6), (x, bot_y + bh + 6))
        x += 12
        hint = self._f_hint.render(
            "Space = Pause     R = Reset     I = Toggle interference", True, DS["text_muted"])
        screen.blit(hint, (x, cy - hint.get_height() // 2))

    def _btn(self, screen: pygame.Surface, rect: pygame.Rect,
             label: str, active: bool = False, col: tuple = None) -> None:
        col = col or DS["blue"]
        bg  = tuple(min(255, int(c * 0.45)) for c in col) if not active else \
              tuple(min(255, int(c * 0.65)) for c in col)
        pygame.draw.rect(screen, bg,  rect, border_radius=5)
        pygame.draw.rect(screen, col, rect, 1, border_radius=5)
        txt = self._f_btn.render(label, True,
                                  DS["text_hi"] if active else DS["text_sec"])
        screen.blit(txt, (rect.centerx - txt.get_width()  // 2,
                          rect.centery - txt.get_height() // 2))

    # ── Plot data ──────────────────────────────────────────────────────────────

    def clear_plots(self) -> None:
        for p in [self._reward_plot, self._throughput_plot, self._sinr_plot]:
            if p:
                p.clear()

    def push_episode_reward(self, raw: float, avg: float) -> None:
        self._reward_plot.push(raw, avg)

    def push_step_data(self, r_s: float, sinr_p: float) -> None:
        self._throughput_plot.push(r_s)
        self._sinr_plot.push(sinr_p)

    def quit(self) -> None:
        pygame.quit()

    @property
    def current_speed(self) -> int:
        return self.SPEED_OPTIONS[self._speed_idx][1]

    @property
    def paused(self) -> bool:
        return self._paused
