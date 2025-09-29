# vz_mk1.py — Hot/Cold map with per-zone shading (no boxes) and 2-line labels
# Label format inside each zone:
#   FGM/FGA
#   FG%
#
# Zone geometry and classifier are UNCHANGED from your prior version.

import os
import json
import math
import numpy as np
import dash
from dash import html, dcc, Output, Input
import plotly.graph_objects as go

# =========================
# Config
# =========================
DATA_PATH = os.environ.get("BBALL_DATA", "data/possessions.json")

# Court geometry (must match entry app exactly)
COURT_W = 50.0
HALF_H = 47.0

RIM_X, RIM_Y, RIM_R = 25.0, 4.25, 0.75
BACKBOARD_Y, RESTRICTED_R = 3.0, 4.0

LANE_W, FT_CY, FT_R = 16.0, 19.0, 6.0
LANE_X0, LANE_X1 = RIM_X - LANE_W/2.0, RIM_X + LANE_W/2.0

THREE_R, SIDELINE_INSET = 22.15, 3.0
LEFT_POST_X, RIGHT_POST_X = SIDELINE_INSET, COURT_W - SIDELINE_INSET

# Global cache
CACHED_DATA = []

def result_from_shorthand(s: str):
    import re
    if not s: return None
    m = re.search(r'(?:\+\+|\+|-)(?!.*(?:\+\+|\+|-))', s)
    if not m: return None
    return "Make" if m.group(0) in ("+","++") else ("Miss" if m.group(0)=="-" else None)

def safe_load_data():
    global CACHED_DATA
    try:
        if not os.path.exists(DATA_PATH):
            return CACHED_DATA
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]
        shots = []
        for row in (data or []):
            try:
                x = float(row.get("x", 0))
                y = float(row.get("y", 0))
                result = row.get("result") or result_from_shorthand(row.get("possession", ""))
                if 0 <= x <= COURT_W and 0 <= y <= HALF_H and result in ("Make", "Miss"):
                    shots.append({"x": x, "y": y, "result": result})
            except:
                continue
        CACHED_DATA = shots
        return shots
    except:
        return CACHED_DATA

# -------------------------
# Zone classifier (UNCHANGED)
# -------------------------
def point_in_zone(x, y, zone_id):
    """Zone classifier with clear, non-overlapping definitions."""
    dist_from_rim = math.hypot(x - RIM_X, y - RIM_Y)
    ft_dist = math.hypot(x - RIM_X, y - FT_CY)

    # Mini 3pt arc radius
    mini_R = FT_CY - RIM_Y  # 13.75

    # Elbow diagonals
    x0L, y0L = (RIM_X - RESTRICTED_R), RIM_Y
    x1L, y1L = LANE_X0, FT_CY
    slope_L = (y1L - y0L) / (x1L - x0L)
    def y_elbow_L(xq): return y0L + slope_L * (xq - x0L)

    x0R, y0R = (RIM_X + RESTRICTED_R), RIM_Y
    x1R, y1R = LANE_X1, FT_CY
    slope_R = (y1R - y0R) / (x1R - x0R)
    def y_elbow_R(xq): return y0R + slope_R * (xq - x0R)

    # Zone 1: restricted circle + vertical posts to baseline
    if zone_id == 1:
        in_circle = dist_from_rim <= RESTRICTED_R
        in_posts  = (RIM_X - RESTRICTED_R <= x <= RIM_X + RESTRICTED_R) and (0.0 <= y <= RIM_Y)
        return in_circle or in_posts

    # Zone 2: LEFT close shot - between RA and mini-arc, below left elbow line
    elif zone_id == 2:
        mini_left_x = RIM_X + mini_R * math.sin(-math.pi/2.2)
        if not ((x < RIM_X) and (x >= mini_left_x) and (x <= RIM_X - RESTRICTED_R) and (y >= 0.0) and (y <= y_elbow_L(x))):
            return False
        if dist_from_rim <= RESTRICTED_R:
            return False
        if y < 2.0:
            return True
        return dist_from_rim <= mini_R

    # Zone 3: Under basket
    elif zone_id == 3:
        in_circle = dist_from_rim <= RESTRICTED_R
        in_posts = (RIM_X - RESTRICTED_R <= x <= RIM_X + RESTRICTED_R) and (0.0 <= y <= RIM_Y)
        if in_circle or in_posts:
            return False
        return (
            (dist_from_rim > RESTRICTED_R) and
            (dist_from_rim <= mini_R) and
            (y >= y_elbow_L(x)) and
            (y >= y_elbow_R(x))
        )

    # Zone 4: RIGHT close shot
    elif zone_id == 4:
        mini_right_x = RIM_X + mini_R * math.sin(math.pi/2.2)
        if not ((x > RIM_X) and (x <= mini_right_x) and (x >= RIM_X + RESTRICTED_R) and (y >= 0.0) and (y <= y_elbow_R(x))):
            return False
        if dist_from_rim <= RESTRICTED_R:
            return False
        if y < 2.0:
            return True
        return dist_from_rim <= mini_R

    # Zone 5: Left corner/baseline
    elif zone_id == 5:
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        left_start_x = RIM_X + mini_radius * math.sin(-max_angle)
        left_start_y = RIM_Y + mini_radius * math.cos(-max_angle)
        diagonal_slope = -0.8
        def y_diagonal_line(xq):
            return left_start_y + diagonal_slope * (xq - left_start_x)
        mini_left_x = left_start_x
        left_3pt_x = LEFT_POST_X
        return (
            (y >= 0.0) and
            (x >= left_3pt_x) and
            (x <= mini_left_x) and
            (y <= y_diagonal_line(x)) and
            (dist_from_rim < THREE_R)
        )

    # Zone 6: Left wing
    elif zone_id == 6:
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        left_start_x = RIM_X + mini_radius * math.sin(-max_angle)
        left_start_y = RIM_Y + mini_radius * math.cos(-max_angle)
        diagonal_slope = -0.8
        def y_diagonal_line(xq):
            return left_start_y + diagonal_slope * (xq - left_start_x)
        return (
            (dist_from_rim >= mini_R) and
            (dist_from_rim < THREE_R) and
            (y < y_elbow_L(x)) and
            (y >= y_diagonal_line(x)) and
            (x < RIM_X)
        )

    # Zone 7: Top of key
    elif zone_id == 7:
        return (
            (dist_from_rim >= mini_R) and
            (dist_from_rim < THREE_R) and
            (y >= y_elbow_L(x)) and
            (y >= y_elbow_R(x))
        )

    # Zone 8: Right wing
    elif zone_id == 8:
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        right_start_x = RIM_X + mini_radius * math.sin(max_angle)
        right_start_y = RIM_Y + mini_radius * math.cos(max_angle)
        diagonal_slope = 0.8
        def y_diagonal_line_right(xq):
            return right_start_y + diagonal_slope * (xq - right_start_x)
        return (
            (dist_from_rim >= mini_R) and
            (dist_from_rim < THREE_R) and
            (y < y_elbow_R(x)) and
            (y >= y_diagonal_line_right(x)) and
            (x > RIM_X)
        )

    # Zone 9: Right corner/baseline
    elif zone_id == 9:
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        right_start_x = RIM_X + mini_radius * math.sin(+max_angle)
        right_start_y = RIM_Y + mini_radius * math.cos(+max_angle)
        diagonal_slope = 0.8
        def y_diagonal_line_right(xq):
            return right_start_y + diagonal_slope * (xq - right_start_x)
        mini_right_x = right_start_x
        right_3pt_x = RIGHT_POST_X
        return (
            (y >= 0.0) and
            (x <= right_3pt_x) and
            (x >= mini_right_x) and
            (y <= y_diagonal_line_right(x)) and
            (dist_from_rim < THREE_R)
        )

    # Zone 10: Left corner 3
    elif zone_id == 10:
        if x <= 0 or y <= 0:
            return False
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        left_start_x = RIM_X + mini_radius * math.sin(-max_angle)
        left_start_y = RIM_Y + mini_radius * math.cos(-max_angle)
        diagonal_slope = -0.8
        def find_three_point_intersection(start_x, start_y, slope):
            x_range = np.linspace(start_x, -2, 1000)
            for xv in x_range:
                yv = start_y + slope * (xv - start_x)
                if abs(math.hypot(xv - RIM_X, yv - RIM_Y) - THREE_R) < 0.1:
                    return xv, yv
            xv = start_x - 8
            yv = start_y + slope * (xv - start_x)
            return xv, yv
        left_ix, left_iy = find_three_point_intersection(left_start_x, left_start_y, diagonal_slope)
        def y_diagonal_top(xq):
            if xq <= left_ix:
                return left_iy
            else:
                return left_start_y + diagonal_slope * (xq - left_start_x)
        eps = 1e-6
        if y >= y_diagonal_top(x) - eps:
            return False
        if (x <= LEFT_POST_X - eps) or (math.hypot(x - RIM_X, y - RIM_Y) >= THREE_R + eps):
            return True
        return False

    # Zone 11: Left wing 3
    elif zone_id == 11:
        if dist_from_rim < THREE_R: return False
        if x < 0: return False
        left_start_x = RIM_X - RESTRICTED_R
        left_elbow_x = LANE_X0
        start_y = RIM_Y
        elbow_y = FT_CY
        left_slope = (elbow_y - start_y) / (left_elbow_x - left_start_x)
        def y_elbow_line_extended(xq):
            return start_y + left_slope * (xq - left_start_x)
        if y > y_elbow_line_extended(x): return False
        if y > HALF_H: return False
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        left_start_x_diag = RIM_X + mini_radius * math.sin(-max_angle)
        left_start_y_diag = RIM_Y + mini_radius * math.cos(-max_angle)
        diagonal_slope = -0.8
        def find_three_point_intersection(start_x, start_y, slope):
            x_range = np.linspace(start_x, -2, 1000)
            for xv in x_range:
                yv = start_y + slope * (xv - start_x)
                dist_to_rim = math.hypot(xv - RIM_X, yv - RIM_Y)
                if abs(dist_to_rim - THREE_R) < 0.1:
                    return xv, yv
            xv = start_x - 8
            yv = start_y + slope * (xv - start_x)
            return xv, yv
        left_ix, left_iy = find_three_point_intersection(left_start_x_diag, left_start_y_diag, diagonal_slope)
        def y_diagonal_line(xq):
            if xq <= left_ix:
                return left_iy
            else:
                return left_start_y_diag + diagonal_slope * (xq - left_start_x_diag)
        if y <= y_diagonal_line(x): return False
        return True

    # Zone 12: Top 3
    elif zone_id == 12:
        if dist_from_rim < THREE_R: return False
        if y > HALF_H: return False
        left_start_x = RIM_X - RESTRICTED_R
        left_elbow_x = LANE_X0
        start_y = RIM_Y
        elbow_y = FT_CY
        left_slope = (elbow_y - start_y) / (left_elbow_x - left_start_x)
        def y_left_elbow_line_extended(xq):
            return start_y + left_slope * (xq - left_start_x)
        if y < y_left_elbow_line_extended(x): return False
        right_start_x = RIM_X + RESTRICTED_R
        right_elbow_x = LANE_X1
        right_slope = (elbow_y - start_y) / (right_elbow_x - right_start_x)
        def y_right_elbow_line_extended(xq):
            return start_y + right_slope * (xq - right_start_x)
        if y < y_right_elbow_line_extended(x): return False
        return True

    # Zone 13: Right wing 3
    elif zone_id == 13:
        if dist_from_rim < THREE_R: return False
        if x > COURT_W: return False
        right_start_x = RIM_X + RESTRICTED_R
        right_elbow_x = LANE_X1
        start_y = RIM_Y
        elbow_y = FT_CY
        right_slope = (elbow_y - start_y) / (right_elbow_x - right_start_x)
        def y_elbow_line_extended(xq):
            return start_y + right_slope * (xq - right_start_x)
        if y > y_elbow_line_extended(x): return False
        if y > HALF_H: return False
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        right_start_x_diag = RIM_X + mini_radius * math.sin(max_angle)
        right_start_y_diag = RIM_Y + mini_radius * math.cos(max_angle)
        diagonal_slope = 0.8
        def find_three_point_intersection(start_x, start_y, slope):
            x_range = np.linspace(start_x, COURT_W + 2, 1000)
            for xv in x_range:
                yv = start_y + slope * (xv - start_x)
                dist_to_rim = math.hypot(xv - RIM_X, yv - RIM_Y)
                if abs(dist_to_rim - THREE_R) < 0.1:
                    return xv, yv
            xv = start_x + 8
            yv = start_y + slope * (xv - start_x)
            return xv, yv
        right_ix, right_iy = find_three_point_intersection(right_start_x_diag, right_start_y_diag, diagonal_slope)
        def y_diagonal_line(xq):
            if xq >= right_ix:
                return right_iy
            else:
                return right_start_y_diag + diagonal_slope * (xq - right_start_x_diag)
        if y <= y_diagonal_line(x): return False
        return True

    # Zone 14: Right corner 3
    elif zone_id == 14:
        if x >= COURT_W or y <= 0:
            return False
        mini_radius = FT_CY - RIM_Y
        max_angle = math.pi / 2.2
        right_start_x = RIM_X + mini_radius * math.sin(+max_angle)
        right_start_y = RIM_Y + mini_radius * math.cos(+max_angle)
        diagonal_slope = 0.8
        def find_three_point_intersection(start_x, start_y, slope):
            x_range = np.linspace(start_x, COURT_W + 2, 1000)
            for xv in x_range:
                yv = start_y + slope * (xv - start_x)
                if abs(math.hypot(xv - RIM_X, yv - RIM_Y) - THREE_R) < 0.1:
                    return xv, yv
            xv = start_x + 8
            yv = start_y + slope * (xv - start_x)
            return xv, yv
        right_ix, right_iy = find_three_point_intersection(right_start_x, right_start_y, diagonal_slope)
        def y_diagonal_top(xq):
            if xq >= right_ix:
                return right_iy
            else:
                return right_start_y + diagonal_slope * (xq - right_start_x)
        eps = 1e-6
        if y >= y_diagonal_top(x) - eps:
            return False
        if (x >= RIGHT_POST_X + eps) or (math.hypot(x - RIM_X, y - RIM_Y) >= THREE_R + eps):
            return True
        return False

    return False

# ---- stats
def calculate_zone_stats(shots):
    zone_stats = {}
    for zone_id in range(1, 15):
        makes = attempts = 0
        for shot in shots:
            if point_in_zone(shot["x"], shot["y"], zone_id):
                attempts += 1
                if shot["result"] == "Make":
                    makes += 1
        pct = (makes / attempts) * 100 if attempts else 0.0
        zone_stats[zone_id] = {"makes": makes, "attempts": attempts, "percentage": round(pct, 1)}
    return zone_stats

# ---- helper to shade a specific zone (Heatmap mask)
def add_zone_fill(fig, zone_id, rgba="rgba(255,255,0,0.35)", step=0.25):
    xs = np.arange(0.0, COURT_W + step, step)
    ys = np.arange(0.0, HALF_H + step, step)
    Z = []
    for yv in ys:
        row = []
        for xv in xs:
            row.append(1.0 if (zone_id and point_in_zone(xv, yv, zone_id)) else np.nan)
        Z.append(row)

    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z,
        colorscale=[[0.0, rgba], [1.0, rgba]],
        showscale=False, hoverinfo="skip",
        opacity=1.0, zmin=1.0, zmax=1.0
    ))

# ------------ drawing helpers (UNCHANGED lines)
def court_lines_traces():
    traces = []
    traces.append(go.Scatter(x=[0, COURT_W, COURT_W, 0, 0], y=[0, 0, HALF_H, HALF_H, 0],
                             mode='lines', line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=[LANE_X0, LANE_X1, LANE_X1, LANE_X0, LANE_X0],
                             y=[0, 0, FT_CY, FT_CY, 0], mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    theta = np.linspace(0, 2*np.pi, 100)
    traces.append(go.Scatter(x=RIM_X + FT_R*np.cos(theta), y=FT_CY + FT_R*np.sin(theta),
                             mode='lines', line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    def t_for_x(x_target, r):
        val = (x_target - RIM_X) / r
        val = max(-1.0, min(1.0, val))
        return math.asin(val)
    tL = t_for_x(LEFT_POST_X, THREE_R)
    tR = t_for_x(RIGHT_POST_X, THREE_R)
    yL = RIM_Y + THREE_R * math.cos(tL)
    yR = RIM_Y + THREE_R * math.cos(tR)
    traces.append(go.Scatter(x=[LEFT_POST_X, LEFT_POST_X], y=[0, yL], mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=[RIGHT_POST_X, RIGHT_POST_X], y=[0, yR], mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    ts = np.linspace(tL, tR, 100)
    traces.append(go.Scatter(x=RIM_X + THREE_R*np.sin(ts), y=RIM_Y + THREE_R*np.cos(ts),
                             mode='lines', line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    rim_t = np.linspace(0, 2*np.pi, 50)
    traces.append(go.Scatter(x=RIM_X + RIM_R*np.cos(rim_t), y=RIM_Y + RIM_R*np.sin(rim_t),
                             mode='lines', line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=[RIM_X-3.0, RIM_X+3.0], y=[BACKBOARD_Y, BACKBOARD_Y],
                             mode='lines', line=dict(width=3, color='black'),
                             showlegend=False, hoverinfo='skip'))
    ra_t = np.linspace(0, np.pi, 50)
    traces.append(go.Scatter(x=RIM_X + RESTRICTED_R*np.cos(ra_t), y=RIM_Y + RESTRICTED_R*np.sin(ra_t),
                             mode='lines', line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=[0, COURT_W], y=[HALF_H, HALF_H], mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    return traces

def first_zone_line_traces():
    x_left = RIM_X - RESTRICTED_R
    x_right = RIM_X + RESTRICTED_R
    y_top = RIM_Y
    style = dict(width=2, color="black")
    return [
        go.Scatter(x=[x_left, x_left], y=[0.0, y_top], mode="lines",
                   line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[x_right, x_right], y=[0.0, y_top], mode="lines",
                   line=style, hoverinfo="skip", showlegend=False)
    ]

def elbow_lines():
    left_start_x = RIM_X - RESTRICTED_R
    right_start_x = RIM_X + RESTRICTED_R
    start_y = RIM_Y
    left_elbow_x = LANE_X0
    right_elbow_x = LANE_X1
    elbow_y = FT_CY
    end_y = HALF_H
    left_slope = (elbow_y - start_y) / (left_elbow_x - left_start_x)
    left_end_x = left_start_x + (end_y - start_y) / left_slope
    right_slope = (elbow_y - start_y) / (right_elbow_x - right_start_x)
    right_end_x = right_start_x + (end_y - start_y) / right_slope
    style = dict(width=2, color="black")
    return [
        go.Scatter(x=[left_start_x, left_end_x], y=[start_y, end_y], mode="lines",
                   line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[right_start_x, right_end_x], y=[start_y, end_y], mode="lines",
                   line=style, hoverinfo="skip", showlegend=False),
    ]

def mini_three_point_line():
    mini_radius = FT_CY - RIM_Y
    max_angle = math.pi / 2.2
    tL = -max_angle
    tR =  max_angle
    arc_left_x  = RIM_X + mini_radius * math.sin(tL)
    arc_right_x = RIM_X + mini_radius * math.sin(tR)
    arc_left_y  = RIM_Y + mini_radius * math.cos(tL)
    arc_right_y = RIM_Y + mini_radius * math.cos(tR)
    ts = np.linspace(tL, tR, 100)
    arc_x = RIM_X + mini_radius * np.sin(ts)
    arc_y = RIM_Y + mini_radius * np.cos(ts)
    style = dict(width=2, color="black")
    return [
        go.Scatter(x=[arc_left_x, arc_left_x], y=[0.0, arc_left_y], mode="lines",
                   line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=arc_x, y=arc_y, mode="lines", line=style,
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=[arc_right_x, arc_right_x], y=[0.0, arc_right_y], mode="lines",
                   line=style, hoverinfo="skip", showlegend=False),
    ]

def diagonal_zone_lines():
    mini_radius = FT_CY - RIM_Y
    max_angle = math.pi / 2.2
    left_start_x  = RIM_X + mini_radius * math.sin(-max_angle)
    left_start_y  = RIM_Y + mini_radius * math.cos(-max_angle)
    right_start_x = RIM_X + mini_radius * math.sin(+max_angle)
    right_start_y = RIM_Y + mini_radius * math.cos(+max_angle)
    diagonal_slope = -0.8
    def find_three_point_intersection(start_x, start_y, slope):
        x_range = np.linspace(start_x, 0 if slope < 0 else COURT_W, 500)
        for xv in x_range:
            yv = start_y + slope * (xv - start_x)
            if abs(math.hypot(xv - RIM_X, yv - RIM_Y) - THREE_R) < 0.1:
                return xv, yv
        xv = start_x - 8 if slope < 0 else start_x + 8
        yv = start_y + slope * (xv - start_x)
        return xv, yv
    left_ix, left_iy   = find_three_point_intersection(left_start_x,  left_start_y,  diagonal_slope)
    right_ix, right_iy = find_three_point_intersection(right_start_x, right_start_y, -diagonal_slope)
    style = dict(width=2, color="black")
    return [
        go.Scatter(x=[left_start_x, left_ix],   y=[left_start_y, left_iy],   mode="lines", line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[left_ix, 0],              y=[left_iy, left_iy],        mode="lines", line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[right_start_x, right_ix], y=[right_start_y, right_iy], mode="lines", line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[right_ix, COURT_W],       y=[right_iy, right_iy],      mode="lines", line=style, hoverinfo="skip", showlegend=False),
    ]

def base_layout(fig_w=520, fig_h=720):
    return dict(
        paper_bgcolor="white", plot_bgcolor="white",
        width=fig_w, height=fig_h,
        margin=dict(l=10, r=10, t=0, b=0),
        xaxis=dict(range=[0, COURT_W], showgrid=False, zeroline=False, ticks="",
                   showticklabels=False, mirror=True, fixedrange=True),
        yaxis=dict(range=[0, HALF_H], showgrid=False, zeroline=False, ticks="",
                   showticklabels=False, scaleanchor="x", scaleratio=1,
                   mirror=True, fixedrange=True),
        showlegend=False
    )

# ------------- Shot chart (unchanged)
def create_shot_chart(shots):
    fig = go.Figure()
    for tr in court_lines_traces(): fig.add_trace(tr)
    makes = [(s["x"], s["y"]) for s in shots if s["result"] == "Make"]
    misses = [(s["x"], s["y"]) for s in shots if s["result"] == "Miss"]
    if makes:
        x, y = zip(*makes)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(symbol='circle', size=10, color='green',
                                             line=dict(width=1, color='green')),
                                 name="Make"))
    if misses:
        x, y = zip(*misses)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(symbol='x', size=10, color='red'),
                                 name="Miss"))
    fig.update_layout(**base_layout())
    left_x  = RIM_X - FT_R
    right_x = RIM_X + FT_R
    for x in (left_x, right_x):
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=FT_CY, line=dict(color="black", width=2), layer="above")
    return fig

# --------- Zone-tiered color mapping by FG% (and white for 0 attempts)
def _zone_tier(zone_id: int) -> str:
    """Return 'close', 'mid', or 'three' based on zone id."""
    if zone_id in (1, 2, 3, 4):
        return "close"
    if zone_id in (5, 6, 7, 8, 9):
        return "mid"
    return "three"  # 10–14

def _rgba_for_zone(zone_id: int, attempts: int, pct: float) -> str:
    """
    Return fill color by zone tier and FG% thresholds.
      - Any zone with 0 attempts -> white.
      - CLOSE (1–4):   0–50 blue, 51–60 yellow, 61–100 red
      - MID (5–9):     0–35 blue, 36–45 yellow, 46–100 red
      - THREE (10–14): 0–25 blue, 26–35 yellow, 36–100 red

    Colors are bold/solid to mimic 2K-style:
      BLUE  = rgb(0, 102, 204)
      YELL  = rgb(255, 204, 0)
      RED   = rgb(204, 0, 0)
    """
    if attempts == 0:
        return "rgba(255,255,255,1.0)"  # stay white

    tier = _zone_tier(zone_id)
    # saturated fills (nearly opaque)
    BLUE   = "rgba(0,102,204,0.95)"
    YELLOW = "rgba(255,204,0,0.95)"
    RED    = "rgba(204,0,0,0.95)"

    if tier == "close":
        if 0.0 <= pct <= 50.0:   return BLUE
        if 51.0 <= pct <= 60.0:  return YELLOW
        return RED  # 61–100
    elif tier == "mid":
        if 0.0 <= pct <= 35.0:   return BLUE
        if 36.0 <= pct <= 45.0:  return YELLOW
        return RED  # 46–100
    else:  # three
        if 0.0 <= pct <= 25.0:   return BLUE
        if 26.0 <= pct <= 35.0:  return YELLOW
        return RED  # 36–100

def create_zone_chart():
    shots = safe_load_data()
    fig = go.Figure()

    # Baseline lines & guides (unchanged)
    for tr in court_lines_traces(): fig.add_trace(tr)
    for tr in first_zone_line_traces(): fig.add_trace(tr)
    for tr in mini_three_point_line(): fig.add_trace(tr)
    for tr in elbow_lines(): fig.add_trace(tr)
    for tr in diagonal_zone_lines(): fig.add_trace(tr)

    # Compute stats per zone
    zone_stats = calculate_zone_stats(shots)

    # Shade every zone based on its FG% and place 2-line text
    zone_centers = {
        1:(24.5,5.5),2:(16.5,5.5),3:(24.5,12),4:(33,6.5),
        5:(6,3.5),6:(10,14.5),7:(24.5,22),8:(39,14.5),9:(42,3),
        10:(1,3.5),11:(5.5,20),12:(24.5,29),13:(42.5,21),14:(48.5,2.5)
    }

    # Fills first (under text) — tiered by zone and white for 0 attempts
    for zone_id in range(1, 15):
        s = zone_stats.get(zone_id, {"makes":0,"attempts":0,"percentage":0.0})
        rgba = _rgba_for_zone(zone_id, s["attempts"], s["percentage"])
        add_zone_fill(fig, zone_id=zone_id, rgba=rgba, step=0.25)

    # 2-line labels (no boxes): "FGM/FGA" on top, "FG%" below
    for zone_id, center in zone_centers.items():
        s = zone_stats.get(zone_id, {"makes":0,"attempts":0,"percentage":0.0})
        makes, atts, pct = s["makes"], s["attempts"], s["percentage"]
        txt = f"{makes}/{atts}<br>{pct:.1f}%"
        fig.add_annotation(
            x=center[0], y=center[1], text=txt,
            showarrow=False,
            font=dict(size=12, color="black", family="Arial Black"),
        )

    fig.update_layout(**base_layout())
    return fig

# =========================
# Dash App
# =========================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"maxWidth":"1200px","margin":"0 auto","padding":"10px"},
    children=[
        html.Div([
            html.Div([
                html.Div("Shot Chart", style={"textAlign":"center","fontSize":"24px","fontWeight":700,"margin":"0 0 4px 0"}),
                dcc.Graph(id="shot_chart", config={"displayModeBar": False}),
                html.Div([html.Span("● Make", style={"color":"green","marginRight":"20px","fontWeight":600}),
                          html.Span("✖ Miss", style={"color":"red","fontWeight":600})],
                         style={"textAlign":"center","margin":"-4px 0 0 0"})
            ], style={"flex":"1 1 0","minWidth":"520px"}),

            html.Div([
                html.Div("Hot/Cold Map", style={"textAlign":"center","fontSize":"24px","fontWeight":700,"margin":"0 0 4px 0"}),
                dcc.Graph(id="zone_chart", config={"displayModeBar": False}),
            ], style={"flex":"1 1 0","minWidth":"520px"})
        ], style={"display":"flex","gap":"16px","alignItems":"flex-start","justifyContent":"center","flexWrap":"wrap"}),

        dcc.Interval(id="refresh", interval=20000, n_intervals=0),
        html.Div([
            html.Div(f"Data source: {DATA_PATH}", style={"color":"#666","fontSize":"12px","marginBottom":"4px"}),
            html.Div("Update method: Conservative polling (20s)", style={"color":"#888","fontSize":"10px"}),
            html.Div(id="status", style={"color":"#888","fontSize":"10px"}),
        ], style={"textAlign":"center","marginTop":"6px"}),
    ]
)

@app.callback(
    [Output("shot_chart","figure"),
     Output("zone_chart","figure"),
     Output("status","children")],
    [Input("refresh","n_intervals")]
)
def update_charts(n):
    try:
        shots = safe_load_data()
        return (
            create_shot_chart(shots),
            create_zone_chart(),
            f"Loaded {len(shots)} shots (Update #{n})"
        )
    except Exception as e:
        return create_shot_chart([]), create_zone_chart(), f"Error: {e}"

if __name__ == "__main__":
    print("Starting visualization server on http://localhost:8051")
    try:
        app.run(debug=False, port=8051, host="127.0.0.1")
    except Exception as e:
        print(f"Failed to start server on port 8051: {e}")
        print("Trying port 8052...")
        try:
            app.run(debug=False, port=8052, host="127.0.0.1")
        except Exception as e2:
            print(f"Also failed on port 8052: {e2}")
            print("Try running: python vz_mk1.py")
