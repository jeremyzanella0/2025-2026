# sc_mark_5.py — data entry app
# Mark 5 - runs with vz mk1 to display base shot charts and hot/cold map

import math
import json
import csv
import os
import re
import sys
import time
import tempfile
import subprocess
from datetime import datetime
import random
import string

import numpy as np
import dash
from dash import html, dcc, Output, Input, State, no_update, ctx, dash_table
import plotly.graph_objects as go

# =====================================================
# Configuration
# =====================================================
DATA_PATH = os.environ.get("BBALL_DATA", "data/possessions.json")
APP_TITLE = "Shot Chart — click anywhere on the court to enter a possession"

# Ensure data directory exists
os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)

# =====================================================
# Parser (pscoding18 must be next to this file)
# =====================================================
try:
    import pscoding18
except ImportError:
    pscoding18 = None
    print("Warning: pscoding18 not found - using dummy parser")

# =====================================================
# Helpers (Make/Miss from shorthand)
# =====================================================
_LAST_SYMBOL_RE = re.compile(r'(?:\+\+|\+|-)(?!.*(?:\+\+|\+|-))')

def result_from_shorthand(s: str):
    """
    Returns 'Make', 'Miss', or None based on the LAST scoring symbol in the shorthand.
    + or ++ = Make, - = Miss
    """
    if not s:
        return None
    m = _LAST_SYMBOL_RE.search(s)
    if not m:
        return None
    sym = m.group(0)
    if sym in ('+', '++'):
        return 'Make'
    if sym == '-':
        return 'Miss'
    return None

def _rand_suffix(n=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

# =====================================================
# Persistence (robust, non-blocking JSON file I/O with retry)
# =====================================================
_PENDING_DISK_ROWS = None  # holds latest rows to write if disk was busy
_LAST_WRITE_ATTEMPT = 0    # track when we last tried to write

def _coerce_float(v):
    try:
        f = float(v)
        if not np.isfinite(f):
            return None
        return float(f)
    except Exception:
        return None

def _normalize_row(r):
    """Return a normalized dict row."""
    x = _coerce_float(r.get("x", r.get("X (ft)")))
    y = _coerce_float(r.get("y", r.get("Y (ft)")))
    possession = (r.get("possession") or r.get("Shorthand") or "").strip()
    result = r.get("result") or r.get("Result") or result_from_shorthand(possession)
    ts = (r.get("timestamp") or r.get("Time") or "").strip()
    dist = _coerce_float(r.get("distance_ft", r.get("Shot Distance (ft)")))
    play = r.get("play_by_play") or r.get("Play-by-Play")
    rid = (r.get("id") or "").strip()

    if not ts:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    if not rid:
        rid = f"{ts}-{_rand_suffix()}"

    return {
        "id": rid,
        "timestamp": ts,
        "possession": possession or "",
        "x": x,
        "y": y,
        "distance_ft": dist,
        "result": result,
        "play_by_play": (play or "").strip(),
    }

def _atomic_write_json(obj, path, max_tries=5, wait=0.02):
    """
    Try atomic file replacement with more retries and shorter waits.
    Returns True on success, False on failure.
    """
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_, encoding='utf-8') as tf:
        json.dump(obj, tf, ensure_ascii=False, indent=2)
        tmp_name = tf.name

    last_err = None
    for i in range(max_tries):
        try:
            os.replace(tmp_name, path)  # atomic on Windows/POSIX
            return True
        except (PermissionError, OSError) as e:
            last_err = e
            # Use shorter waits with exponential backoff
            time.sleep(wait * (1.5 ** i))
    
    print(f"[atomic_write] busy after {max_tries} attempts; deferring persist. last_err={last_err}")
    # Clean up temp file if we failed
    try:
        os.unlink(tmp_name)
    except:
        pass
    return False

def _robust_load_json(path, retries=15, wait=0.03):
    """Load JSON with retries to avoid mid-write races."""
    last_err = None
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            last_err = e
            time.sleep(wait)
    print(f"[load_log_from_disk] JSON load failed after retries: {last_err}")
    return None

def load_log_from_disk():
    """
    Returns list of dict rows. Supports JSON list or CSV with headers.
    Retries reads to tolerate concurrent writes.
    Normalizes and deduplicates by 'id' (fallback to timestamp).
    """
    if not os.path.exists(DATA_PATH):
        return []

    rows = []
    try:
        if DATA_PATH.lower().endswith(".json"):
            data = _robust_load_json(DATA_PATH)
            if data is None:
                return []
            if isinstance(data, dict) and "rows" in data:
                data = data["rows"]
            for r in data or []:
                nr = _normalize_row(r)
                if nr:
                    rows.append(nr)
        elif DATA_PATH.lower().endswith(".csv"):
            with open(DATA_PATH, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    nr = _normalize_row(r)
                    if nr:
                        rows.append(nr)
        else:
            return []
    except Exception as e:
        print(f"Failed to load data: {e}")

    seen = set()
    deduped = []
    for r in rows:
        key = r.get("id") or r.get("timestamp")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped

def _json_safe_row(r: dict):
    """Ensure fields are JSON-serializable and within bounds."""
    out = dict(r)
    out["x"] = _coerce_float(out.get("x"))
    out["y"] = _coerce_float(out.get("y"))
    if out["x"] is not None:
        out["x"] = min(max(0.0, out["x"]), COURT_W)
    if out["y"] is not None:
        out["y"] = min(max(0.0, out["y"]), HALF_H)
    out["distance_ft"] = _coerce_float(out.get("distance_ft"))
    return out

def save_log_to_disk(rows):
    """
    Non-blocking persist with better retry strategy:
    - Try a quick atomic replace with more attempts.
    - If file is locked, stash rows in _PENDING_DISK_ROWS for automatic retry.
    Returns True if written now; False if deferred.
    """
    global _PENDING_DISK_ROWS, _LAST_WRITE_ATTEMPT
    try:
        payload = [_json_safe_row(r) for r in (rows or [])]
        _LAST_WRITE_ATTEMPT = time.time()
        
        if _atomic_write_json(payload, DATA_PATH):
            _PENDING_DISK_ROWS = None
            return True
        else:
            _PENDING_DISK_ROWS = payload
            return False
    except Exception as e:
        print(f"[save_log_to_disk] unexpected error: {e}")
        _PENDING_DISK_ROWS = payload if 'payload' in locals() else None
        _LAST_WRITE_ATTEMPT = time.time()
        return False

def try_flush_pending():
    """If a prior write was deferred, try once to flush it (non-blocking)."""
    global _PENDING_DISK_ROWS, _LAST_WRITE_ATTEMPT
    if _PENDING_DISK_ROWS is None:
        return True
        
    # Don't retry too aggressively
    if time.time() - _LAST_WRITE_ATTEMPT < 1.0:
        return False
        
    _LAST_WRITE_ATTEMPT = time.time()
    if _atomic_write_json(_PENDING_DISK_ROWS, DATA_PATH):
        _PENDING_DISK_ROWS = None
        return True
    return False

# =====================================================
# Court geometry (feet)
# =====================================================
COURT_W = 50.0
HALF_H  = 47.0

RIM_X   = 25.0
RIM_Y   = 4.25
RIM_R   = 0.75
BACKBOARD_Y = 3.0
RESTRICTED_R = 4.0

LANE_W  = 16.0
LANE_X0 = RIM_X - LANE_W/2.0
LANE_X1 = RIM_X + LANE_W/2.0

FT_CY   = 19.0
FT_R    = 6.0

THREE_R        = 22.15
SIDELINE_INSET = 3.0
LEFT_POST_X    = SIDELINE_INSET
RIGHT_POST_X   = COURT_W - SIDELINE_INSET

def court_lines():
    """Generate court lines as trace data"""
    traces = []

    # Court boundary
    boundary_x = [0, COURT_W, COURT_W, 0, 0]
    boundary_y = [0, 0, HALF_H, HALF_H, 0]
    traces.append(go.Scatter(
        x=boundary_x, y=boundary_y, mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='boundary'
    ))

    # Lane rectangle
    lane_x = [LANE_X0, LANE_X1, LANE_X1, LANE_X0, LANE_X0]
    lane_y = [0, 0, FT_CY, FT_CY, 0]
    traces.append(go.Scatter(
        x=lane_x, y=lane_y, mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='lane'
    ))

    # Free throw circle
    theta = np.linspace(0, 2*np.pi, 100)
    ft_circle_x = RIM_X + FT_R * np.cos(theta)
    ft_circle_y = FT_CY + FT_R * np.sin(theta)
    traces.append(go.Scatter(
        x=ft_circle_x, y=ft_circle_y, mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='ft_circle'
    ))

    # 3-point arc + corners
    def t_for_x(x_target):
        val = (x_target - RIM_X) / THREE_R
        val = max(-1.0, min(1.0, val))
        return math.asin(val)

    tL = t_for_x(LEFT_POST_X)
    tR = t_for_x(RIGHT_POST_X)
    yL = RIM_Y + THREE_R * math.cos(tL)
    yR = RIM_Y + THREE_R * math.cos(tR)

    traces.append(go.Scatter(
        x=[LEFT_POST_X, LEFT_POST_X], y=[0, yL], mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='left_corner'
    ))
    traces.append(go.Scatter(
        x=[RIGHT_POST_X, RIGHT_POST_X], y=[0, yR], mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='right_corner'
    ))

    ts = np.linspace(tL, tR, 100)
    arc_x = RIM_X + THREE_R * np.sin(ts)
    arc_y = RIM_Y + THREE_R * np.cos(ts)
    traces.append(go.Scatter(
        x=arc_x, y=arc_y, mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='three_arc'
    ))

    # Rim
    rim_theta = np.linspace(0, 2*np.pi, 50)
    rim_x = RIM_X + RIM_R * np.cos(rim_theta)
    rim_y = RIM_Y + RIM_R * np.sin(rim_theta)
    traces.append(go.Scatter(
        x=rim_x, y=rim_y, mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='rim'
    ))

    # Backboard
    traces.append(go.Scatter(
        x=[RIM_X - 3.0, RIM_X + 3.0], y=[BACKBOARD_Y, BACKBOARD_Y],
        mode='lines', line=dict(width=3, color='black'),
        showlegend=False, hoverinfo='skip', name='backboard'
    ))

    # Restricted area (semicircle)
    ra_theta = np.linspace(0, np.pi, 50)
    ra_x = RIM_X + RESTRICTED_R * np.cos(ra_theta)
    ra_y = RIM_Y + RESTRICTED_R * np.sin(ra_theta)
    traces.append(go.Scatter(
        x=ra_x, y=ra_y, mode='lines',
        line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='restricted'
    ))

    # Half court line
    traces.append(go.Scatter(
        x=[0, COURT_W], y=[HALF_H, HALF_H],
        mode='lines', line=dict(width=2, color='black'),
        showlegend=False, hoverinfo='skip', name='halfcourt'
    ))
    return traces

def make_click_dots():
    """Create dots covering the entire court area up to the boundaries"""
    xs, ys = [], []
    x = 0.0
    while x <= COURT_W:
        y = 0.0
        while y <= HALF_H:
            xs.append(x); ys.append(y)
            y += 0.5
        x += 0.5

    return go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(size=5, color='red', opacity=0.6),
        showlegend=False,
        hovertemplate='Click here to add shot<extra></extra>',
        name='click_zones'
    )

def base_fig():
    fig = go.Figure()
    for trace in court_lines():
        fig.add_trace(trace)
    fig.add_trace(make_click_dots())
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=720, height=1000,
        xaxis=dict(range=[0, COURT_W], showgrid=False, zeroline=False, ticks="", showticklabels=False, mirror=True, fixedrange=True),
        yaxis=dict(range=[0, HALF_H], showgrid=False, zeroline=False, ticks="", showticklabels=False, scaleanchor="x", scaleratio=1, mirror=True, fixedrange=True),
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",
        dragmode=False
    )
    return fig

# =====================================================
# Dash app
# =====================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE

# Pre-load any stored rows for continuity across sessions
_initial_log = load_log_from_disk()
try_flush_pending()  # opportunistic first flush (no-op if nothing pending)

def modal_style(show: bool):
    return {
        "display": "flex" if show else "none",
        "position": "fixed",
        "top": 0, "left": 0, "right": 0, "bottom": 0,
        "backgroundColor": "rgba(0,0,0,0.35)",
        "alignItems": "center",
        "justifyContent": "center",
        "zIndex": 1000,
    }

# ====== LAYOUT (Tabs: Shot Entry + Data Log) ======
app.layout = html.Div([
    html.H2(APP_TITLE, style={"textAlign": "center", "marginBottom": "16px"}),

    dcc.Tabs(id="tabs", value="chart", children=[
        dcc.Tab(label="Shot Entry", value="chart", children=[
            html.Div([
                html.Div(
                    dcc.Graph(
                        id="court",
                        figure=base_fig(),
                        config={"displayModeBar": False},
                        style={"width": "740px", "height": "1020px"}
                    ),
                    style={"flex": "0 0 760px", "display": "flex", "justifyContent": "center"}
                ),
                html.Div([
                    html.Div(id="debug_click", style={"fontSize": "14px", "color": "#666", "margin": "0 0 8px 0"}),
                    html.Div(id="output_block", style={"width": "100%", "minWidth": "520px"}),
                ], style={"flex": "1 1 0", "padding": "0 8px", "position": "relative"}),
            ], style={
                "display": "flex", "justifyContent": "flex-start", "gap": "28px",
                "alignItems": "flex-start", "maxWidth": "1600px", "margin": "0 auto",
                "paddingLeft": "24px", "paddingRight": "12px"
            }),

            # Modal
            html.Div(
                id="input_modal",
                style=modal_style(False),
                children=html.Div(
                    style={
                        "width": "560px", "background": "white", "borderRadius": "12px",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.2)", "padding": "18px"
                    },
                    children=[
                        html.H3("Enter possession shorthand"),
                        html.Div(id="click_xy_label", style={"color": "#666", "marginBottom": "6px"}),
                        dcc.Input(
                            id="possession_input", type="text",
                            placeholder="e.g. 1/2pnr3/45chd+",
                            style={"width": "100%", "padding": "12px", "fontSize": "16px"},
                            debounce=False
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "marginTop": "12px", "justifyContent": "flex-end"},
                            children=[
                                html.Button("Cancel", id="btn_cancel", n_clicks=0,
                                            style={"padding": "10px 16px", "borderRadius": "8px", "border": "1px solid #ccc", "background": "white"}),
                                html.Button("Submit", id="btn_submit", n_clicks=0,
                                            style={"padding": "10px 16px", "borderRadius": "8px", "border": "none", "background": "#2563eb", "color": "white"})
                            ]
                        )
                    ]
                )
            ),
        ]),

        dcc.Tab(label="Data Log", value="log", children=[
            html.Div(id="log_container", style={"maxWidth": "1200px", "margin": "18px auto", "padding": "0 12px"})
        ]),
    ]),

    # Stores
    dcc.Store(id="store_modal_open", data=False),
    dcc.Store(id="store_last_click_xy", data=None),
    dcc.Store(id="store_preview", data=None),
    dcc.Store(id="store_log", data=_initial_log),
    
    # Automatic retry interval for pending writes
    dcc.Interval(id="retry_interval", interval=2000, n_intervals=0),
])

# ---------- (kept, unused) CLI wrapper ----------
def get_ps_commentary_cli(possession_text: str) -> str:
    """Run pscoding18.py CLI and capture printed lines (left for future use)."""
    script_path = None
    if pscoding18 and getattr(pscoding18, "__file__", None):
        script_path = pscoding18.__file__
    else:
        candidate = os.path.join(os.path.dirname(__file__), "pscoding18.py")
        if os.path.exists(candidate):
            script_path = candidate

    if not script_path or not os.path.exists(script_path):
        return "Commentary: pscoding18.py not found next to this app."

    cmd = [sys.executable, script_path]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate(possession_text.strip() + "\nq\n", timeout=10)

        lines = []
        for line in (out or "").splitlines():
            s = line.strip()
            if not s or s.lower().startswith("enter possession"):
                continue
            lines.append(s)

        if not lines:
            if err:
                return f"Commentary error: {err.strip()}"
            return "Commentary: (no output captured from pscoding18)"

        return "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "Commentary error: pscoding18 timed out."
    except Exception as e:
        return f"Commentary error: {e!r}"
# ----------------------------------------------------

# ========= Callbacks =========

# Automatic retry callback for pending writes
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Input("retry_interval", "n_intervals"),
    State("store_log", "data"),
    prevent_initial_call=True
)
def auto_retry_pending_writes(_n, current_log):
    """Automatically retry pending disk writes."""
    if _PENDING_DISK_ROWS is not None and try_flush_pending():
        # Successfully flushed pending data, trigger a re-read to stay in sync
        refreshed_log = load_log_from_disk()
        return refreshed_log
    return no_update

# Debug callback
@app.callback(
    Output("debug_click", "children"),
    Input("court", "clickData"),
    prevent_initial_call=True
)
def debug_click_data(clickData):
    if clickData is None:
        return "No clicks detected"
    return f"Last click: x={clickData['points'][0]['x']:.1f}, y={clickData['points'][0]['y']:.1f}"

# Main controller (open/close modal)
@app.callback(
    Output("store_modal_open", "data"),
    Output("store_last_click_xy", "data"),
    Output("click_xy_label", "children"),
    Input("court", "clickData"),
    Input("btn_cancel", "n_clicks"),
    Input("btn_submit", "n_clicks"),
    Input("possession_input", "n_submit"),
    prevent_initial_call=True
)
def control_modal(clickData, cancel_clicks, submit_clicks, n_submit):
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Open on court click
    if trigger_id == "court" and clickData and clickData.get("points"):
        x = float(clickData["points"][0]["x"])
        y = float(clickData["points"][0]["y"])
        if 0.0 <= x <= COURT_W and 0.0 <= y <= HALF_H:
            return True, {"x": x, "y": y}, f"Selected: x={x:.2f}, y={y:.2f} ft"
        return no_update, no_update, "Please click inside the court boundaries."

    # Close on cancel OR submit
    if trigger_id in ("btn_cancel", "btn_submit", "possession_input"):
        return False, no_update, no_update

    return no_update, no_update, no_update

# Modal visibility
@app.callback(
    Output("input_modal", "style"),
    Input("store_modal_open", "data")
)
def toggle_modal(open_flag):
    return modal_style(bool(open_flag))

# Parse + Preview (also provide preview data for Save)
@app.callback(
    Output("output_block", "children"),
    Output("store_preview", "data"),
    Input("btn_submit", "n_clicks"),
    Input("possession_input", "n_submit"),
    State("possession_input", "value"),
    State("store_last_click_xy", "data"),
    prevent_initial_call=True
)
def parse_and_render(btn_clicks, n_submit, possession_text, xy):
    triggered = (btn_clicks or 0) > 0 or (n_submit or 0) > 0
    if not triggered:
        return no_update, no_update

    if not possession_text or not possession_text.strip():
        return html.Div("Please enter a possession string.", style={"color": "crimson"}), no_update

    try:
        if pscoding18:
            parsed = pscoding18.parse_possession_string(possession_text.strip())
            lines = [parsed] if isinstance(parsed, str) else list(parsed)
        else:
            lines = [f"Dummy parse: {possession_text.strip()}"]
    except Exception as e:
        return html.Div(f"Error while parsing: {e}", style={"color": "crimson"}), no_update

    # Location & Shot Distance
    x = y = None
    trailing = []
    if isinstance(xy, dict) and ("x" in xy and "y" in xy):
        x, y = float(xy["x"]), float(xy["y"])
        trailing.append(f"Location: x={x:.2f}, y={y:.2f} ft")
        dx, dy = (x - RIM_X), (y - RIM_Y)
        distance_ft = (dx**2 + dy**2) ** 0.5
        trailing.append(f"Shot distance: {distance_ft:.1f} ft from basket")
    else:
        distance_ft = None

    content_lines = lines + trailing

    # Build preview payload
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    preview = {
        "id": f"{now}-{_rand_suffix()}",
        "timestamp": now,
        "possession": possession_text.strip(),
        "x": x,
        "y": y,
        "distance_ft": round(distance_ft, 1) if distance_ft is not None else None,
        "play_by_play": "\n".join(lines),
        "result": result_from_shorthand(possession_text.strip())
    }

    preview_ui = html.Div([
        html.H4("Play-by-Play", style={"marginTop": 0}),
        html.Pre("\n".join(content_lines), style={
            "background": "#0b1021", "color": "#e6edf3",
            "padding": "16px", "borderRadius": "10px",
            "whiteSpace": "pre-wrap", "marginBottom": "10px", "fontSize": "16px",
            "width": "100%"
        }),
        html.Div([
            html.Button("Discard / Edit", id="btn_discard", n_clicks=0,
                        style={"padding": "10px 14px", "borderRadius": "8px",
                               "border": "1px solid #aaa", "background": "white", "marginRight": "8px"}),
            html.Button("Save possession", id="btn_confirm", n_clicks=0,
                        style={"padding": "10px 14px", "borderRadius": "8px",
                               "border": "none", "background": "#16a34a", "color": "white"})
        ], style={"display": "flex", "justifyContent": "flex-end", "gap": "8px"})
    ])

    return preview_ui, preview

# Save confirmed possession to log, reset input, switch to Data Log tab
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Output("possession_input", "value", allow_duplicate=True),
    Output("store_last_click_xy", "data", allow_duplicate=True),
    Output("output_block", "children", allow_duplicate=True),
    Output("tabs", "value", allow_duplicate=True),
    Input("btn_confirm", "n_clicks"),
    State("store_log", "data"),
    State("store_preview", "data"),
    prevent_initial_call=True
)
def save_possession(n_confirm, log_data, preview):
    if not n_confirm or not preview:
        return no_update, no_update, no_update, no_update, no_update

    log = list(log_data or [])
    log.append(_json_safe_row(preview))

    ok = save_log_to_disk(log)

    notice = ("Saved to Data Log (write pending)" if not ok else "Saved to Data Log")
    cleared_output = html.Div([
        html.Div(notice, style={"color": "#16a34a" if ok else "#b45309",
                                "marginBottom": "8px"})
    ])

    return log, "", None, cleared_output, "log"

# Discard: clear the preview UI
@app.callback(
    Output("output_block", "children", allow_duplicate=True),
    Input("btn_discard", "n_clicks"),
    prevent_initial_call=True
)
def discard_preview(n):
    if not n:
        return no_update
    return html.Div([
        html.Div("Discarded. Re-enter the possession when ready.", style={"color": "#b45309"})
    ])

# Render Data Log table when log changes
@app.callback(
    Output("log_container", "children"),
    Input("store_log", "data")
)
def render_log(log_rows):
    rows = log_rows or []
    if not rows:
        return html.Div("No possessions saved yet.", style={"color": "#666", "fontStyle": "italic", "marginTop": "12px"})

    columns = [
        {"name": "Time", "id": "timestamp"},
        {"name": "Shorthand", "id": "possession", "editable": True},
        {"name": "X (ft)", "id": "x", "type": "numeric"},
        {"name": "Y (ft)", "id": "y", "type": "numeric"},
        {"name": "Shot Distance (ft)", "id": "distance_ft", "type": "numeric"},
        {"name": "Result", "id": "result"},
        {"name": "Play-by-Play", "id": "play_by_play"},
        {"name": "Row ID", "id": "id"},  # internal; hidden via hidden_columns
    ]
    table = dash_table.DataTable(
        id="tbl_log",
        data=rows,
        columns=columns,
        hidden_columns=["id"],
        page_size=15,
        style_cell={"whiteSpace": "pre-line", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial", "fontSize": "15px"},
        style_header={"fontWeight": "600"},
        style_table={"overflowX": "auto"},
        sort_action="native",
        filter_action="native",
        editable=True,
        row_selectable="multi",
        selected_rows=[]
    )
    return html.Div([
        html.H4("Saved Possessions", style={"marginTop": "12px"}),
        table,
        html.Div(
            [html.Button("Delete selected row(s)",
                         id="btn_delete_rows",
                         n_clicks=0,
                         style={"marginTop": "10px", "padding": "8px 12px", "borderRadius": "8px"})],
            style={"display": "flex", "justifyContent": "flex-end"}
        )
    ])

# When a shorthand/x/y cell is edited, re-parse/update row, then persist
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Input("tbl_log", "data_timestamp"),
    State("tbl_log", "data"),
    State("store_log", "data"),
    prevent_initial_call=True
)
def update_row_from_edit(_ts, edited_rows, original_rows):
    if edited_rows is None or original_rows is None:
        return no_update

    edited_rows = list(edited_rows)
    updated = list(original_rows)

    # index by id (fallback to timestamp)
    index_by_key = {}
    for i, row in enumerate(updated):
        k = row.get("id") or row.get("timestamp")
        if k:
            index_by_key[k] = i

    changed = False
    for row in edited_rows:
        k = row.get("id") or row.get("timestamp")
        if k not in index_by_key:
            continue
        i = index_by_key[k]
        old = dict(updated[i])

        # Shorthand edit
        new_short = (row.get("possession") or "").strip()
        old_short = (old.get("possession") or "").strip()

        if new_short != old_short:
            try:
                if pscoding18:
                    parsed = pscoding18.parse_possession_string(new_short)
                    lines = [parsed] if isinstance(parsed, str) else list(parsed)
                else:
                    lines = [f"Dummy parse: {new_short}"]
            except Exception as e:
                print(f"[edit] parse failed; keeping old row. err={e!r}")
                continue

            x = _coerce_float(old.get("x"))
            y = _coerce_float(old.get("y"))
            if x is not None and y is not None:
                dx, dy = (x - RIM_X), (y - RIM_Y)
                distance_ft = round((dx * dx + dy * dy) ** 0.5, 1)
            else:
                distance_ft = old.get("distance_ft")

            new_row = dict(old)
            new_row["possession"]   = new_short
            new_row["play_by_play"] = "\n".join(lines)
            new_row["distance_ft"]  = distance_ft
            new_row["result"]       = result_from_shorthand(new_short)
            updated[i] = _json_safe_row(new_row)
            changed = True

        # x/y edited
        else:
            x = _coerce_float(row.get("x"))
            y = _coerce_float(row.get("y"))
            if x is not None and y is not None:
                x = min(max(0.0, x), COURT_W)
                y = min(max(0.0, y), HALF_H)
                dx, dy = (x - RIM_X), (y - RIM_Y)
                distance_ft = round((dx*dx + dy*dy) ** 0.5, 1)
                new_row = dict(old)
                new_row["x"] = x
                new_row["y"] = y
                new_row["distance_ft"] = distance_ft
                updated[i] = _json_safe_row(new_row)
                changed = True

    if changed:
        save_log_to_disk(updated)
        return updated
    return no_update

# Delete selected rows (and persist)
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Input("btn_delete_rows", "n_clicks"),
    State("tbl_log", "selected_rows"),
    State("tbl_log", "data"),
    prevent_initial_call=True
)
def delete_selected_rows(n_clicks, selected_rows, current_table_rows):
    if not n_clicks or not current_table_rows:
        return no_update
    sel = set(selected_rows or [])
    if not sel:
        return no_update
    new_rows = [row for idx, row in enumerate(current_table_rows) if idx not in sel]
    save_log_to_disk(new_rows)
    return new_rows

if __name__ == "__main__":
    # Keep the port free for vz mk1 (usually 8051). This app typically runs on 8050.
    app.run(debug=True, use_reloader=False, port=8050)
