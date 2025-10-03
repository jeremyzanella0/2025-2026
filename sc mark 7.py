# - sc_mark_7_fixed.py – Shot entry with Practice→Drill context and enhanced Data Log deletion features
# (First section — everything up to callbacks)

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
from dash.dependencies import ALL
import plotly.graph_objects as go

# =========================
# Config
# =========================
DATA_PATH  = os.environ.get("BBALL_DATA", "data/possessions.json")
APP_TITLE  = "CWB Practice Data Entry"
os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)

BASE_DIR    = os.path.dirname(DATA_PATH) or "."
ROSTER_PATH = os.path.join(BASE_DIR, "roster.json")
# --- NEW: practices metadata (absences) persisted next to possessions/roster
PRACTICES_PATH = os.path.join(BASE_DIR, "practices.json")

# --- NEW: Append-only JSONL path (for scalable, O(1) writes)
DATA_PATH_JSONL = os.environ.get("BBALL_DATA_JSONL", os.path.join(BASE_DIR, "possessions.jsonl"))

# --- NEW: Limit how many rows we keep in the client store (browser memory)
MAX_CLIENT_ROWS = int(os.environ.get("MAX_CLIENT_ROWS", "2000"))

# =========================
# Optional parser module
# =========================
try:
    import pscoding18
except ImportError:
    pscoding18 = None
    print("Warning: pscoding18 not found - using dummy parser")

# =========================
# Helpers
# =========================
# FIX: detect the last +/++/- from shorthand accurately
_LAST_SYMBOL_RE = re.compile(r'(?:\+\+|\+|-)(?!.*(?:\+\+|\+|-))')

def result_from_shorthand(s: str):
    if not s:
        return None
    m = _LAST_SYMBOL_RE.search(s)
    if not m:
        return None
    sym = m.group(0)
    if sym in ("+", "++"):
        return "Make"
    if sym == "-":
        return "Miss"
    return None

def _results_from_shorthand_tokens(short: str, shots_needed: int | None = None) -> list[str | None]:
    """
    Extract a sequence of per-shot results from the possession shorthand by
    scanning each token for its last +/-/++ sign. Example:
      '1/3 4/5- 10/11or+' -> ['Miss', 'Make']
    If shots_needed is provided, the list is trimmed or padded (with None) to that length.
    """
    seq: list[str | None] = []
    if not short:
        return [None] * (shots_needed or 0)

    tokens = re.split(r'\s+', short.strip())
    for tok in tokens:
        m = _LAST_SYMBOL_RE.search(tok)
        if not m:
            continue
        sym = m.group(0)
        if sym in ("+", "++"):
            seq.append("Make")
        elif sym == "-":
            seq.append("Miss")

    if shots_needed is not None:
        if len(seq) < shots_needed:
            seq = seq + [None] * (shots_needed - len(seq))
        else:
            seq = seq[:shots_needed]
    return seq

def _rand_suffix(n=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _new_group_id():
    return f"group-{datetime.now().strftime('%Y%m%d%H%M%S%f')}-{_rand_suffix(3)}"

def _slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', (s or '').strip().lower())

def _normalize_date_input(date_str: str) -> str:
    """
    Accepts 'MM/DD/YYYY', 'YYYY-MM-DD', or 'M/D/YYYY', returns 'YYYY-MM-DD'.
    Falls back to today if parsing fails.
    """
    s = (date_str or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        dt = datetime.strptime(s.replace('-', '/'), "%m/%d/%Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _new_practice_id(date_str: str) -> str:
    return f"practice-{_normalize_date_input(date_str)}"

def _new_drill_id(practice_id: str, drill_name: str) -> str:
    return f"{practice_id}__drill-{_slug(drill_name)}--{_rand_suffix(3)}"

# =========================
# Shooter/Result extraction from free text (play-by-play)
# =========================
# Permissive "guarded by" detector (handles commas/periods and variable spacing)
_SHOOT_GUARD_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9.\-'\s,]+?)\s*guarded\s*by\s*([A-Za-z][A-Za-z0-9.\-'\s,]+)",
    flags=re.IGNORECASE
)
# More tolerant result detectors
_MAKES_RE  = re.compile(r"\bmake(?:s|d)?\b", flags=re.IGNORECASE)   # make/makes/made
_MISSES_RE = re.compile(r"\bmiss(?:es)?\b",  flags=re.IGNORECASE)   # miss/misses

def _clean_name_fragment(s: str) -> str:
    # keep word chars, dot, dash, apostrophe and spaces; normalize spaces
    s = re.sub(r"[^\w.\-'\s]", " ", s or "")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _clean_shooter_line(s: str) -> str:
    """
    From text like 'Fontana, guarded by Lusk misses the shot' -> 'Fontana guarded by Lusk'
    Falls back to stripping trailing words after make/miss if regex doesn't hit.
    (Note: We do NOT store shooter anymore; we keep this helper solely to produce
    a clean, readable 'play_by_play_names' preview.)
    """
    if not s:
        return ""
    m = _SHOOT_GUARD_RE.search(s)
    if m:
        left  = _clean_name_fragment(m.group(1))
        right = _clean_name_fragment(m.group(2))
        return f"{left} guarded by {right}"
    # fallback: take text up to make/miss and normalize
    s2 = _MISSES_RE.split(s)[0]
    s2 = _MAKES_RE.split(s2)[0]
    s2 = re.sub(r'\bguarded\s*by\b', 'guarded by', s2, flags=re.IGNORECASE)
    return _clean_name_fragment(s2)

def _derive_result_from_text(s: str):
    if not s:
        return None
    if _MAKES_RE.search(s):
        return "Make"
    if _MISSES_RE.search(s):
        return "Miss"
    return None

def _first_nonempty(*vals):
    for v in vals:
        if isinstance(v, str):
            if v.strip():
                return v.strip()
        elif v is not None:
            return v
    return None

# =========================
# Persistence (log + roster)
# =========================
_PENDING_DISK_ROWS = None
_LAST_WRITE_ATTEMPT = 0

def _coerce_float(v):
    try:
        f = float(v)
        if not np.isfinite(f):
            return None
        return float(f)
    except Exception:
        return None

def _normalize_row(r):
    # Read common fields
    x = _coerce_float(r.get("x", r.get("X (ft)")))
    y = _coerce_float(r.get("y", r.get("Y (ft)")))
    possession = (r.get("possession") or r.get("Shorthand") or "").strip()
    result = r.get("result") or r.get("Result") or result_from_shorthand(possession)
    ts = (r.get("timestamp") or r.get("Time") or "").strip()
    dist = _coerce_float(r.get("distance_ft", r.get("Shot Distance (ft)")))
    play = r.get("play_by_play") or r.get("Play-by-Play")
    play_names = r.get("play_by_play_names") or r.get("Play-by-Play (Names)") or ""
    rid = (r.get("id") or "").strip()

    group_id = (r.get("group_id") or "").strip()
    group_size = r.get("group_size")
    shot_index = r.get("shot_index")
    possession_type = (r.get("possession_type") or "").strip()

    # Context fields (may be None for legacy rows)
    practice_id   = (r.get("practice_id") or "").strip() or None
    practice_date = (r.get("practice_date") or "").strip() or None
    drill_id      = (r.get("drill_id") or "").strip() or None
    drill_name    = (r.get("drill_name") or "").strip() or None

    if not ts:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    if not rid:
        rid = f"{ts}-{_rand_suffix()}"

    # Prefer the names-annotated text if present; otherwise fall back to raw play-by-play
    pbp_text = (play_names or play or "").strip()

    # Result: if missing/None, derive from text (make/miss tokens)
    if not result:
        result = _derive_result_from_text(pbp_text)

    # Coerce numeric-friendly fields
    gs = None
    try:
        if group_size is not None and str(group_size) != "":
            gs = int(group_size)
    except:
        pass

    si = None
    try:
        if shot_index is not None and str(shot_index) != "":
            si = int(shot_index)
    except:
        pass

    out = {
        "id": rid,
        "timestamp": ts,
        "possession": possession or "",
        "x": x, "y": y,
        "distance_ft": dist,
        "result": result,
        "play_by_play": (play or "").strip(),
        "play_by_play_names": (play_names or "").strip(),
        "group_id": group_id or None,
        "group_size": gs,
        "shot_index": si,
        "possession_type": possession_type or (("shots" if possession else None)),
        # context
        "practice_id": practice_id,
        "practice_date": practice_date,
        "drill_id": drill_id,
        "drill_name": drill_name,
    }
    return out

def _atomic_write_json(obj, path, max_tries=5, wait=0.02):
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_, encoding='utf-8') as tf:
        json.dump(obj, tf, ensure_ascii=False, indent=2)
        tmp_name = tf.name
    last_err = None
    for i in range(max_tries):
        try:
            os.replace(tmp_name, path)
            return True
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(wait * (1.5 ** i))
    print(f"[atomic_write] busy; deferring persist. err={last_err}")
    try:
        os.unlink(tmp_name)
    except:
        pass
    return False

def _robust_load_json(path, retries=15, wait=0.03):
    last_err = None
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            last_err = e
            time.sleep(wait)
    print(f"[load_log_from_disk] failed after retries: {last_err}")
    return None

# --- NEW: JSONL helpers (append, tail-read, and streaming iterate)
def _jsonl_path() -> str:
    return DATA_PATH_JSONL

def append_log_row(row: dict) -> bool:
    """
    Append a single possession row as JSONL (O(1) write).
    Keeps writes fast and avoids rewriting the entire dataset.
    """
    try:
        os.makedirs(os.path.dirname(_jsonl_path()) or ".", exist_ok=True)
        with open(_jsonl_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe_row(row), ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[append_log_row] {e}")
        return False

def _jsonl_tail(n: int) -> list[dict]:
    """
    Efficiently read the last n JSONL rows without loading entire file into memory.
    """
    path = _jsonl_path()
    if not os.path.exists(path) or n <= 0:
        return []
    # Simple block tail reader
    rows = []
    chunk_size = 8192
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        buffer = b""
        pos = file_size
        while pos > 0 and len(rows) < n:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            buffer = f.read(read_size) + buffer
            # split lines
            lines = buffer.split(b"\n")
            buffer = lines[0]  # incomplete head for next round
            for line in reversed(lines[1:]):  # process from end
                if not line.strip():
                    continue
                try:
                    r = json.loads(line.decode("utf-8"))
                    rows.append(_normalize_row(r))
                    if len(rows) >= n:
                        break
                except Exception:
                    continue
    rows.reverse()
    return rows

def iter_rows(date: str = None, drill: str = None):
    """
    Memory-light iterator over all possessions, optionally filtered.
    Streams from JSONL if available; otherwise falls back to JSON/CSV loader.
    """
    path_jsonl = _jsonl_path()
    if os.path.exists(path_jsonl):
        with open(path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = _normalize_row(json.loads(line))
                except Exception:
                    continue
                if date and r.get("practice_date") != date:
                    continue
                if drill and r.get("drill_name") != drill:
                    continue
                yield r
        return

    # Fallback: legacy JSON/CSV (may materialize into memory)
    for r in load_log_from_disk():
        if date and r.get("practice_date") != date:
            continue
        if drill and r.get("drill_name") != drill:
            continue
        yield r

def load_log_from_disk():
    """
    Backwards-compatible loader for existing JSON/CSV datasets.
    If a JSONL file exists, we *also* read it (only used in some legacy paths);
    the preferred way for large sets is to use iter_rows() or _jsonl_tail().
    """
    rows = []
    try:
        # Prefer JSON/CSV legacy path if present
        if os.path.exists(DATA_PATH):
            if DATA_PATH.lower().endswith(".json"):
                data = _robust_load_json(DATA_PATH) or []
                if isinstance(data, dict) and "rows" in data:
                    data = data["rows"]
                for r in data:
                    rows.append(_normalize_row(r))
            elif DATA_PATH.lower().endswith(".csv"):
                with open(DATA_PATH, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        rows.append(_normalize_row(r))
        # If JSONL exists (new path), extend with its contents
        if os.path.exists(_jsonl_path()):
            with open(_jsonl_path(), "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rows.append(_normalize_row(json.loads(line)))
                    except Exception:
                        continue
    except Exception as e:
        print(f"load error: {e}")

    # dedupe by id/timestamp
    seen, out = set(), []
    for r in rows:
        k = r.get("id") or r.get("timestamp")
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def _json_safe_row(r: dict):
    out = dict(r)
    out["x"] = _coerce_float(out.get("x"))
    out["y"] = _coerce_float(out.get("y"))
    if out["x"] is not None:
        out["x"] = min(max(0.0, out["x"]), 50.0)
    if out["y"] is not None:
        out["y"] = min(max(0.0, out["y"]), 47.0)
    out["distance_ft"] = _coerce_float(out.get("distance_ft"))
    for key in ("group_size", "shot_index"):
        if out.get(key) is not None:
            try:
                out[key] = int(out[key])
            except:
                out[key] = None
    for k in ("group_id", "possession_type", "practice_id", "drill_id", "practice_date", "drill_name"):
        if not out.get(k):
            out[k] = None
    if "play_by_play_names" not in out:
        out["play_by_play_names"] = ""
    # No 'shooter' field anymore — ensure it's dropped if present in legacy data
    out.pop("shooter", None)
    return out

def save_log_to_disk(rows):
    """
    Legacy full-file save (JSON). Kept for compatibility with existing flows.
    For scalability, callbacks should prefer append_log_row(new_row) to avoid
    rewriting large files. This function remains for explicit exports/migrations.
    """
    global _PENDING_DISK_ROWS, _LAST_WRITE_ATTEMPT
    try:
        payload = [_json_safe_row(r) for r in (rows or [])]
        _LAST_WRITE_ATTEMPT = time.time()
        if _atomic_write_json(payload, DATA_PATH):
            _PENDING_DISK_ROWS = None
            return True
        _PENDING_DISK_ROWS = payload
        return False
    except Exception as e:
        print(f"[save_log_to_disk] {e}")
        _PENDING_DISK_ROWS = payload if 'payload' in locals() else None
        _LAST_WRITE_ATTEMPT = time.time()
        return False

def try_flush_pending():
    global _PENDING_DISK_ROWS, _LAST_WRITE_ATTEMPT
    if _PENDING_DISK_ROWS is None:
        return True
    if time.time() - _LAST_WRITE_ATTEMPT < 1.0:
        return False
    _LAST_WRITE_ATTEMPT = time.time()
    if _atomic_write_json(_PENDING_DISK_ROWS, DATA_PATH):
        _PENDING_DISK_ROWS = None
        return True
    return False

# ---- Roster persistence
def load_roster_from_disk():
    try:
        if os.path.exists(ROSTER_PATH):
            with open(ROSTER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            out = {}
            for k, v in data.items():
                if v is None:
                    continue
                try:
                    kk = str(int(k))
                    name = str(v).strip()
                    if name:
                        out[kk] = name
                except:
                    continue
            return out
    except Exception as e:
        print(f"[roster load] {e}")
    return {}

def save_roster_to_disk(roster: dict):
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        clean = {str(int(k)): str(v).strip() for k, v in (roster or {}).items()
                 if str(v or "").strip() and str(k).strip().isdigit()}
        return _atomic_write_json(clean, ROSTER_PATH)
    except Exception as e:
        print(f"[roster save] {e}")
        return False

# --- NEW: Practices metadata (absences) persistence
def _load_practices_meta():
    if not os.path.exists(PRACTICES_PATH):
        return {}
    try:
        with open(PRACTICES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[practices load] {e}")
        return {}

def _save_practices_meta(meta: dict):
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        return _atomic_write_json(meta or {}, PRACTICES_PATH)
    except Exception as e:
        print(f"[practices save] {e}")
        return False

# =========================
# Name annotation (ensures “guarded by” is present/standard)
# =========================
# Matches jersey numbers in free text
_NUM_RE = re.compile(r'(?<!\w)(\d{1,2})(?!\w)')

# Standardize any capitalization/spacing for "guarded by"
_GUARDED_BY_STD_RE = re.compile(r'\bguarded\s*by\b', flags=re.IGNORECASE)

def _last_name(full: str) -> str:
    if not full:
        return ""
    parts = [p for p in re.split(r'\s+', full.strip()) if p]
    return parts[-1] if parts else full

def _build_display_name_map(roster: dict) -> dict:
    last_counts = {}
    for _, full in (roster or {}).items():
        ln = _last_name(full)
        last_counts[ln] = last_counts.get(ln, 0) + 1

    name_map = {}
    for num, full in (roster or {}).items():
        ln = _last_name(full)
        if last_counts.get(ln, 0) > 1:
            name_map[num] = full.strip()   # ambiguous -> full name
        else:
            name_map[num] = ln             # unique -> last name
    return name_map

def annotate_with_roster_lastnames(text: str, roster: dict) -> str:
    """
    1) Replace jersey numbers with display names (last name if unique, else full).
    2) Normalize the token to the exact phrase 'guarded by' (case/spacing).
       This guarantees downstream extraction works regardless of parser quirks.
    """
    if not text:
        return text
    s = str(text)

    # Normalize any capitalization/extra spaces: 'Guarded   By' -> 'guarded by'
    s = _GUARDED_BY_STD_RE.sub('guarded by', s)

    if not roster:
        return s

    name_map = _build_display_name_map(roster)
    s = re.sub(r'\bPlayer\s+', '', s)  # drop "Player " prefixes that sometimes appear

    def repl(m):
        num = m.group(1)
        return name_map.get(num, num)

    s = _NUM_RE.sub(repl, s)
    # collapse multiple spaces introduced by substitutions
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

# =========================
# Court geometry
# =========================
COURT_W = 50.0
HALF_H  = 47.0
RIM_X, RIM_Y, RIM_R = 25.0, 4.25, 0.75
BACKBOARD_Y = 3.0
RESTRICTED_R = 4.0
LANE_W = 16.0
LANE_X0, LANE_X1 = RIM_X - LANE_W/2.0, RIM_X + LANE_W/2.0
FT_CY, FT_R = 19.0, 6.0
THREE_R = 22.15
SIDELINE_INSET = 3.0
LEFT_POST_X, RIGHT_POST_X = SIDELINE_INSET, COURT_W - SIDELINE_INSET

def court_lines():
    traces = []
    boundary_x = [0, COURT_W, COURT_W, 0, 0]
    boundary_y = [0, 0, HALF_H, HALF_H, 0]
    traces.append(go.Scatter(x=boundary_x, y=boundary_y, mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    lane_x = [LANE_X0, LANE_X1, LANE_X1, LANE_X0, LANE_X0]
    lane_y = [0, 0, FT_CY, FT_CY, 0]
    traces.append(go.Scatter(x=lane_x, y=lane_y, mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))

    theta = np.linspace(0, 2*np.pi, 100)
    ft_x = RIM_X + FT_R*np.cos(theta)
    ft_y = FT_CY + FT_R*np.sin(theta)
    traces.append(go.Scatter(x=ft_x, y=ft_y, mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))

    def t_for_x(x_target):
        val = (x_target - RIM_X) / THREE_R
        val = max(-1.0, min(1.0, val))
        return math.asin(val)
    tL, tR = t_for_x(LEFT_POST_X), t_for_x(RIGHT_POST_X)
    yL = RIM_Y + THREE_R*math.cos(tL)
    yR = RIM_Y + THREE_R*math.cos(tR)

    traces.append(go.Scatter(x=[LEFT_POST_X, LEFT_POST_X], y=[0, yL], mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    traces.append(go.Scatter(x=[RIGHT_POST_X, RIGHT_POST_X], y=[0, yR], mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))

    ts = np.linspace(tL, tR, 100)
    arc_x = RIM_X + THREE_R*np.sin(ts)
    arc_y = RIM_Y + THREE_R*np.cos(ts)
    traces.append(go.Scatter(x=arc_x, y=arc_y, mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))

    rim_t = np.linspace(0, 2*np.pi, 50)
    rim_x = RIM_X + RIM_R*np.cos(rim_t)
    rim_y = RIM_Y + RIM_R*np.sin(rim_t)
    traces.append(go.Scatter(x=rim_x, y=rim_y, mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))

    traces.append(go.Scatter(x=[RIM_X-3.0, RIM_X+3.0], y=[BACKBOARD_Y, BACKBOARD_Y],
                             mode='lines', line=dict(width=3, color='black'),
                             showlegend=False, hoverinfo='skip'))

    ra_t = np.linspace(0, np.pi, 50)
    ra_x = RIM_X + RESTRICTED_R*np.cos(ra_t)
    ra_y = RIM_Y + RESTRICTED_R*np.sin(ra_t)
    traces.append(go.Scatter(x=ra_x, y=ra_y, mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))

    traces.append(go.Scatter(x=[0, COURT_W], y=[HALF_H, HALF_H], mode='lines',
                             line=dict(width=2, color='black'),
                             showlegend=False, hoverinfo='skip'))
    return traces

def make_click_dots():
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
        hovertemplate='Click here to add shot<extra></extra>'
    )

def base_fig():
    fig = go.Figure()
    for tr in court_lines():
        fig.add_trace(tr)
    fig.add_trace(make_click_dots())
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        width=720, height=1000,
        xaxis=dict(range=[0, COURT_W], showgrid=False, zeroline=False, ticks="", showticklabels=False, mirror=True, fixedrange=True),
        yaxis=dict(range=[0, HALF_H], showgrid=False, zeroline=False, ticks="", showticklabels=False, scaleanchor="x", scaleratio=1, mirror=True, fixedrange=True),
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select", dragmode=False
    )
    return fig

# =========================
# App
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE

_initial_log    = load_log_from_disk()
try_flush_pending()
_initial_roster = load_roster_from_disk()

# --- NEW: cap initial client log window to protect browser when datasets are large
if isinstance(_initial_log, list) and len(_initial_log) > MAX_CLIENT_ROWS:
    _initial_log = _initial_log[-MAX_CLIENT_ROWS:]

def modal_style(show: bool):
    return {
        "display": "flex" if show else "none",
        "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
        "backgroundColor": "rgba(0,0,0,0.35)",
        "alignItems": "center", "justifyContent": "center",
        "zIndex": 1000,
    }

# ------------- LAYOUT -------------
app.layout = html.Div([
    html.H2(APP_TITLE, style={"textAlign": "center", "marginBottom": "16px"}),

    dcc.Tabs(id="tabs", value="chart", children=[
        dcc.Tab(label="Shot Entry", value="chart", children=[

            # ---- Practice & Drill controls
            html.Div([
                html.Div([
                    html.Span("Practice date:", style={"fontWeight": 600, "marginRight": "6px"}),
                    dcc.Input(id="inp_practice_date", type="text", placeholder="MM/DD/YYYY",
                              style={"width": "150px", "padding": "6px"}),
                    html.Button("Start Practice", id="btn_start_practice", n_clicks=0,
                                style={"marginLeft": "8px", "padding": "6px 10px", "borderRadius": "8px",
                                       "background": "#2563eb", "color": "white", "border": "none"}),
                    html.Button("End Practice", id="btn_end_practice", n_clicks=0,
                                style={"marginLeft": "6px", "padding": "6px 10px", "borderRadius": "8px"})
                ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "6px"}),

                html.Div([
                    html.Span("Drill name:", style={"fontWeight": 600, "marginRight": "6px"}),
                    dcc.Input(id="inp_drill_name", type="text", placeholder="e.g., 3v3 pin down",
                              style={"width": "260px", "padding": "6px"}),
                    html.Button("Start Drill", id="btn_start_drill", n_clicks=0,
                                style={"marginLeft": "8px", "padding": "6px 10px", "borderRadius": "8px",
                                       "background": "#f59e0b", "color": "white", "border": "none"}),
                    html.Button("End Drill", id="btn_end_drill", n_clicks=0,
                                style={"marginLeft": "6px", "padding": "6px 10px", "borderRadius": "8px"})
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),

                html.Div(id="practice_status", style={"color": "#444", "marginTop": "4px"})
            ], style={"maxWidth": "1200px", "margin": "0 auto 8px", "padding": "0 12px"}),

            # Top row: court (left) + controls & play-by-play (right)
            html.Div([
                html.Div(
                    dcc.Graph(id="court", figure=base_fig(), config={"displayModeBar": False},
                              style={"width": "740px", "height": "1020px"}),
                    style={"flex": "0 0 760px", "display": "flex", "justifyContent": "center"}
                ),
                html.Div([
                    html.Div([
                        html.Div("Possession Type", style={"fontWeight": 600, "marginBottom": "6px"}),

                        html.Div([
                            html.Button("No-shot possession (TO)", id="btn_no_shot", n_clicks=0,
                                        style={"padding": "8px 10px", "borderRadius": "8px",
                                               "border": "1px solid #aaa", "background": "white"}),

                            dcc.Input(id="shots_count", type="number", min=1, max=12, placeholder="# shots",
                                      style={"width": "90px", "marginLeft": "10px", "padding": "6px"}),
                            html.Button("Start shots", id="btn_start_shots", n_clicks=0,
                                        style={"padding": "8px 10px", "borderRadius": "8px",
                                               "border": "none", "background": "#f59e0b",
                                               "color": "white", "marginLeft": "6px"}),
                        ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "8px"}),

                        html.Div(id="mode_hint", style={"fontSize": "13px", "color": "#444"})
                    ], style={"marginBottom": "12px"}),

                    html.Div(id="debug_click", style={"fontSize": "14px", "color": "#666", "margin": "0 0 8px 0"}),

                    # Preview / status messages (parser UI)
                    html.Div(id="output_block", style={"width": "100%", "minWidth": "520px", "marginBottom": "10px"}),

                    # Last saved possession card (filled by callbacks after save)
                    html.Div(id="last_saved", style={"width": "100%", "minWidth": "520px"}),

                ], style={"flex": "1 1 0", "padding": "0 8px", "position": "relative"}),
            ], style={"display": "flex", "gap": "28px", "alignItems": "flex-start",
                      "maxWidth": "1600px", "margin": "0 auto",
                      "paddingLeft": "24px", "paddingRight": "12px"}),

            # --- ROSTER BLOCK BELOW (full width) ---
            html.Div([
                html.Hr(),
                html.Div("Team Roster", style={"fontWeight": 700, "marginBottom": "6px"}),
                dash_table.DataTable(
                    id="tbl_roster",
                    columns=[
                        {"name": "Jersey", "id": "jersey", "type": "numeric"},
                        {"name": "Player Name", "id": "name"}
                    ],
                    data=[{"jersey": int(k), "name": v} for k, v in sorted(_initial_roster.items(), key=lambda kv: int(kv[0]))] or
                         [{"jersey": None, "name": ""}],
                    editable=True,
                    row_deletable=True,
                    page_size=12,
                    style_cell={"fontSize": "14px", "fontFamily": "system-ui"},
                    style_table={"maxWidth": "1000px", "margin": "0 auto", "overflowX": "auto"},
                ),
                html.Div(style={"display": "flex", "gap": "8px", "marginTop": "8px", "justifyContent": "center"}, children=[
                    html.Button("Add row", id="btn_roster_add", n_clicks=0,
                                style={"padding": "6px 10px", "borderRadius": "8px"}),
                    html.Button("Save roster", id="btn_roster_save", n_clicks=0,
                                style={"padding": "6px 10px", "borderRadius": "8px", "background": "#2563eb", "color": "white", "border": "none"}),
                    html.Div(id="roster_status", style={"marginLeft": "8px", "color": "666", "alignSelf": "center"})
                ])
            ], style={"maxWidth": "1200px", "margin": "16px auto 8px"}),

            # Modal for shorthand entry
            html.Div(id="input_modal", style=modal_style(False), children=html.Div(
                style={"width": "560px", "background": "white", "borderRadius": "12px",
                       "boxShadow": "0 10px 30px rgba(0,0,0,0.2)", "padding": "18px"},
                children=[
                    html.H3("Enter possession shorthand"),
                    html.Div(id="click_xy_label", style={"color": "#666", "marginBottom": "6px"}),
                    dcc.Input(id="possession_input", type="text",
                              placeholder="e.g., 1/3 4/5++",
                              style={"width": "100%", "padding": "12px", "fontSize": "16px"},
                              debounce=False),
                    html.Div(style={"display": "flex", "gap": "8px", "marginTop": "12px", "justifyContent": "flex-end"}, children=[
                        html.Button("Cancel", id="btn_cancel", n_clicks=0,
                                    style={"padding": "10px 16px", "borderRadius": "8px",
                                           "border": "1px solid #ccc", "background": "white"}),
                        html.Button("Submit", id="btn_submit", n_clicks=0,
                                    style={"padding": "10px 16px", "borderRadius": "8px",
                                           "border": "none", "background": "#2563eb", "color": "white"}),
                    ])
                ]
            )),

            # --- NEW: Missed Practice modal (count + jersey numbers)
            html.Div(id="missed_modal", style=modal_style(False), children=html.Div(
                style={"width": "520px", "background": "white", "borderRadius": "12px",
                       "boxShadow": "0 10px 30px rgba(0,0,0,0.2)", "padding": "18px"},
                children=[
                    html.H3("Players Missing Practice"),
                    html.Div("How many players missed this practice?", style={"marginBottom": "8px"}),
                    dcc.Input(id="inp_missed_count", type="number", min=0, step=1, placeholder="0",
                              style={"width":"120px","padding":"6px"}),
                    html.Div(id="missed_numbers_block", style={"marginTop":"12px"}),
                    html.Div(style={"display":"flex","gap":"8px","justifyContent":"flex-end","marginTop":"14px"}, children=[
                        html.Button("Cancel", id="btn_missed_cancel", n_clicks=0,
                                    style={"padding":"10px 16px","borderRadius":"8px","border":"1px solid #ccc","background":"white"}),
                        html.Button("Save", id="btn_missed_save", n_clicks=0,
                                    style={"padding":"10px 16px","borderRadius":"8px","border":"none","background":"#2563eb","color":"white"})
                    ])
                ]
            )),
        ]),

        dcc.Tab(label="Data Log", value="log", children=[
            # Filled by callbacks (dates → drills → table)
            html.Div(id="log_container",
                     style={"maxWidth": "1200px", "margin": "18px auto", "padding": "0 12px"}),
        ]),
    ]),

    # Stores
    dcc.Store(id="store_modal_open", data=False),
    dcc.Store(id="store_last_click_xy", data=None),
    dcc.Store(id="store_preview", data=None),
    dcc.Store(id="store_log", data=_initial_log),

    # Modes
    dcc.Store(id="store_mode", data=None),
    dcc.Store(id="store_shots_needed", data=0),
    dcc.Store(id="store_clicks", data=[]),

    # Roster store
    dcc.Store(id="store_roster", data=_initial_roster),

    # Practice/Drill session context
    dcc.Store(id="store_practice", data={"active": False, "date": None, "practice_id": None}),
    dcc.Store(id="store_drill",    data={"active": False, "name": None, "drill_id": None}),

    # --- NEW: controls visibility of missed practice modal
    dcc.Store(id="store_show_missed_modal", data=False),

    # Data Log navigation state
    dcc.Store(id="log_level", data="dates"),
    dcc.Store(id="sel_date", data=None),
    dcc.Store(id="sel_drill", data=None),

    # Confirmation modal stores
    dcc.Store(id="confirm_delete_type", data=None),
    dcc.Store(id="confirm_delete_target", data=None),
    dcc.Store(id="show_confirm_modal", data=False),

    # Confirmation modal
    html.Div(id="confirm_modal", style=modal_style(False), children=html.Div(
        style={"width": "450px", "background": "white", "borderRadius": "12px",
               "boxShadow": "0 10px 30px rgba(0,0,0,0.2)", "padding": "24px"},
        children=[
            html.H3("Confirm Deletion", style={"color": "#dc2626", "marginTop": 0}),
            html.Div(id="confirm_message", style={"marginBottom": "20px", "fontSize": "16px"}),
            html.Div(style={"display": "flex", "gap": "12px", "justifyContent": "flex-end"}, children=[
                html.Button("Cancel", id="btn_confirm_cancel", n_clicks=0,
                            style={"padding": "10px 20px", "borderRadius": "8px",
                                   "border": "1px solid #ccc", "background": "white"}),
                html.Button("Yes, Delete", id="btn_confirm_delete", n_clicks=0,
                            style={"padding": "10px 20px", "borderRadius": "8px",
                                   "border": "none", "background": "#dc2626", "color": "white"}),
            ])
        ]
    )),

    dcc.Interval(id="retry_interval", interval=2000, n_intervals=0),

    # ---------------------------------------------------------------------
    # Hidden placeholders so plain-ID & pattern-ID Inputs exist at all times.
    # This prevents "nonexistent object used in an Input/State" errors in Dash 3.x.
    # ---------------------------------------------------------------------
    html.Div(style={"display": "none"}, children=[
        # Table-level delete button and table (scope "main")
        html.Button(id={"role": "btn_delete_rows", "scope": "main"}, n_clicks=0),
        dash_table.DataTable(
            id={"role": "tbl_log", "scope": "main"},
            columns=[{"name": "dummy", "id": "dummy"}],
            data=[]
        ),
        # Representative practice/drill delete buttons used in dates/drills views
        html.Button(id={"type": "delete_practice_btn", "date": "__placeholder__"}, n_clicks=0),
        html.Button(id={"type": "delete_drill_btn", "drill": "__placeholder__", "date": "__placeholder__"}, n_clicks=0),

        # --- IMPORTANT: keep a *live* hidden copy of the legacy ID expected by some callbacks.
        # If callbacks still reference State('inp_missed_numbers', 'value'), this ensures it exists.
        dcc.Input(id="inp_missed_numbers", type="number", value=0),
    ]),
])

# ---- Validation layout: superset of all possible components
# NOTE: Pattern-ID placeholders MUST match the exact IDs used in callbacks.
app.validation_layout = html.Div([
    html.H2(APP_TITLE),
    dcc.Tabs(id="tabs", value="chart", children=[
        dcc.Tab(label="Shot Entry", value="chart"),
        dcc.Tab(label="Data Log", value="log"),
    ]),

    # Pattern ID placeholders used by table callbacks
    dash_table.DataTable(
        id={"role": "tbl_log", "scope": "main"},
        columns=[{"name": "dummy", "id": "dummy"}],
        data=[]
    ),
    html.Button(id={"role": "btn_delete_rows", "scope": "main"}, n_clicks=0),

    # Plain-ID placeholders used by nav callbacks
    html.Button(id="btn_back_drills", n_clicks=0),
    html.Button(id="btn_back_dates", n_clicks=0),

    # Containers
    html.Div(id="log_container"),
    html.Div(id="confirm_modal"),
    html.Div(id="last_saved"),

    # Stores
    dcc.Store(id="store_modal_open"),
    dcc.Store(id="store_last_click_xy"),
    dcc.Store(id="store_preview"),
    dcc.Store(id="store_log"),
    dcc.Store(id="store_mode"),
    dcc.Store(id="store_shots_needed"),
    dcc.Store(id="store_clicks"),
    dcc.Store(id="store_roster"),
    dcc.Store(id="store_practice"),
    dcc.Store(id="store_drill"),
    dcc.Store(id="store_show_missed_modal"),  # NEW
    dcc.Store(id="log_level"),
    dcc.Store(id="sel_date"),
    dcc.Store(id="sel_drill"),
    dcc.Store(id="confirm_delete_type"),
    dcc.Store(id="confirm_delete_target"),
    dcc.Store(id="show_confirm_modal"),

    dcc.Interval(id="retry_interval"),

    # --- NEW: validation placeholders for missed practice modal & inputs
    html.Div(id="missed_modal"),
    dcc.Input(id="inp_missed_count"),
    html.Div(id="missed_numbers_block"),
    dcc.Input(id="inp_missed_numbers"),  # legacy placeholder
    html.Button(id="btn_missed_cancel"),
    html.Button(id="btn_missed_save"),
])

# -------- Optional commentary wrapper (unchanged)
def get_ps_commentary_cli(possession_text: str) -> str:
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
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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




# =========================
# Callbacks
# =========================

# ---- helpers (local to callbacks) -------------------------------------------
_MAKES_RE  = re.compile(r"\bmake(?:s|d)?\b",  flags=re.IGNORECASE)
_MISSES_RE = re.compile(r"\bmiss(?:es)?\b",   flags=re.IGNORECASE)

def _extract_results(lines: list[str]) -> list[str | None]:
    """
    Return a list of 'Make'/'Miss' in the order they appear.
    Count a line as a shot ONLY if it includes a make/miss token.
    """
    out: list[str | None] = []
    for raw in (lines or []):
        s = (raw or "").strip()
        if not s:
            continue
        is_make = bool(_MAKES_RE.search(s))
        is_miss = bool(_MISSES_RE.search(s))
        if is_make:
            out.append("Make")
        elif is_miss:
            out.append("Miss")
    return out

def _per_shot_nones(n: int) -> list[None]:
    return [None] * max(0, int(n or 0))

# --- NEW (local): compact/overwrite storage after deletes, streaming-friendly ---
def _rewrite_storage_filtered(keep_predicate):
    """
    Rewrites on-disk storage to keep only rows for which keep_predicate(row) is True.
    Prefers JSONL compaction when available; falls back to legacy JSON if JSONL doesn't exist.
    """
    # Prefer JSONL path if it exists
    path_jsonl = _jsonl_path()
    if os.path.exists(path_jsonl):
        dir_ = os.path.dirname(path_jsonl) or "."
        tmp_path = os.path.join(dir_, f".possessions.tmp.{_rand_suffix(6)}.jsonl")
        try:
            with open(path_jsonl, "r", encoding="utf-8") as fin, \
                 open(tmp_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        r = _normalize_row(json.loads(line))
                    except Exception:
                        continue
                    if keep_predicate(r):
                        fout.write(json.dumps(_json_safe_row(r), ensure_ascii=False) + "\n")
            os.replace(tmp_path, path_jsonl)
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
            print(f"[compact jsonl] {e}")

    # Also handle legacy DATA_PATH if present (JSON/CSV); we keep JSON only here.
    if os.path.exists(DATA_PATH) and DATA_PATH.lower().endswith(".json"):
        try:
            legacy = _robust_load_json(DATA_PATH) or []
            if isinstance(legacy, dict) and "rows" in legacy:
                legacy = legacy["rows"]
            keep = []
            for r in legacy:
                try:
                    nr = _normalize_row(r)
                    if keep_predicate(nr):
                        keep.append(_json_safe_row(nr))
                except Exception:
                    continue
            _atomic_write_json(keep, DATA_PATH)
        except Exception as e:
            print(f"[compact json legacy] {e}")

def _refresh_client_window():
    """
    Refresh recent window for client from JSONL tail if available, else from legacy loader.
    """
    if os.path.exists(_jsonl_path()):
        try:
            return _jsonl_tail(MAX_CLIENT_ROWS)
        except Exception as e:
            print(f"[refresh_client_window.tail] {e}")
    # Fallback to legacy loader (will be sliced at init, but keep it small here too)
    rows = load_log_from_disk()
    return rows[-MAX_CLIENT_ROWS:] if isinstance(rows, list) and len(rows) > MAX_CLIENT_ROWS else rows

# -----------------------------------------------------------------------------


# Auto-retry pending writes (kept for legacy JSON full-file path)
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Input("retry_interval", "n_intervals"),
    State("store_log", "data"),
    prevent_initial_call=True
)
def auto_retry(_n, log):
    if _PENDING_DISK_ROWS is not None and try_flush_pending():
        # On success, refresh the client window from disk (tail when possible)
        return _refresh_client_window()
    return no_update


# -----------------------------
# Missed Practice Modal wiring
# -----------------------------

# Show/hide the missed-practice modal
@app.callback(
    Output("missed_modal", "style"),
    Input("store_show_missed_modal", "data")
)
def toggle_missed_modal(show):
    return modal_style(bool(show))

# Dynamically show the jersey-number field when count > 0
@app.callback(
    Output("missed_numbers_block", "children"),
    Input("inp_missed_count", "value")
)
def render_missed_numbers_block(count_val):
    try:
        n = int(count_val or 0)
    except Exception:
        n = 0
    if n <= 0:
        return html.Div("No one missed practice. Click Save to continue.",
                        style={"color":"#444","fontSize":"14px"})
    return html.Div([
        html.Div(f"Enter {n} jersey number(s), comma or space separated:", style={"marginBottom":"6px"}),
        # IMPORTANT: same id used everywhere; exists even when hidden thanks to the
        # hidden placeholder instance in Section 1.
        dcc.Input(id="inp_missed_numbers", type="text", placeholder="e.g., 12, 23 5",
                  style={"width":"100%","padding":"8px"})
    ])


# Start / End Practice
@app.callback(
    Output("store_practice","data", allow_duplicate=True),
    Output("practice_status","children", allow_duplicate=True),
    Output("store_show_missed_modal","data", allow_duplicate=True),
    Input("btn_start_practice","n_clicks"),
    State("inp_practice_date","value"),
    prevent_initial_call=True
)
def start_practice(n_start, date_str):
    if not n_start:
        return no_update, no_update, no_update
    if not (date_str and str(date_str).strip()):
        return no_update, "Enter a practice date to start.", False
    norm = _normalize_date_input(str(date_str))
    pid = _new_practice_id(norm)
    # Start as active, but prompt for absences
    practice = {"active": True, "date": norm, "practice_id": pid, "absent_numbers": []}
    status = f"Practice active: {norm} (id={pid}). Please confirm absences…"
    return practice, status, True

# Save/Cancel absences from the modal
@app.callback(
    Output("store_practice","data", allow_duplicate=True),
    Output("practice_status","children", allow_duplicate=True),
    Output("store_show_missed_modal","data", allow_duplicate=True),
    Input("btn_missed_save","n_clicks"),
    Input("btn_missed_cancel","n_clicks"),
    State("inp_missed_count","value"),
    State("inp_missed_numbers","value"),  # exists even when not visible
    State("store_practice","data"),
    prevent_initial_call=True
)
def finalize_practice_absences(n_save, n_cancel, count_val, numbers_raw, practice):
    trig = ctx.triggered_id if ctx.triggered_id else None
    if not practice or not practice.get("active"):
        return no_update, no_update, False

    if trig == "btn_missed_cancel":
        # Persist a record with zero absences for this practice
        meta = _load_practices_meta()
        meta[practice["practice_id"]] = {
            "practice_date": practice.get("date"),
            "absent_numbers": []
        }
        _save_practices_meta(meta)
        practice["absent_numbers"] = []
        status = f"Practice active: {practice['date']} (id={practice['practice_id']}). Absences: 0."
        return practice, status, False

    if trig != "btn_missed_save":
        return no_update, no_update, no_update

    # Parse count and jersey number list
    try:
        cnt = int(count_val or 0)
    except Exception:
        cnt = 0

    absent_nums = []
    if cnt > 0:
        raw = (numbers_raw or "").strip()
        if not raw:
            return no_update, "Enter jersey numbers or set count to 0.", True
        parts = re.split(r"[,\s]+", raw)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            try:
                absent_nums.append(str(int(p)))
            except Exception:
                # ignore non-numeric tokens
                pass
        # proceed even if lengths differ

    # Persist to practices.json
    meta = _load_practices_meta()
    meta[practice["practice_id"]] = {
        "practice_date": practice.get("date"),
        "absent_numbers": absent_nums
    }
    _save_practices_meta(meta)

    practice["absent_numbers"] = absent_nums
    status = (f"Practice active: {practice['date']} (id={practice['practice_id']}). "
              f"Absences: {len(absent_nums)}" + (f" ({', '.join(absent_nums)})" if absent_nums else "."))
    return practice, status, False


@app.callback(
    Output("store_practice","data", allow_duplicate=True),
    Output("store_drill","data", allow_duplicate=True),
    Output("practice_status","children", allow_duplicate=True),
    Input("btn_end_practice","n_clicks"),
    State("store_practice","data"),
    prevent_initial_call=True
)
def end_practice(n_end, practice):
    if not n_end:
        return no_update, no_update, no_update
    if not (practice and practice.get("active")):
        return no_update, no_update, "No active practice to end."
    return {"active": False, "date": practice.get("date"), "practice_id": practice.get("practice_id"), "absent_numbers": practice.get("absent_numbers")}, \
           {"active": False, "name": None, "drill_id": None}, \
           f"Practice ended: {practice.get('date')}."


# Start / End Drill
@app.callback(
    Output("store_drill","data", allow_duplicate=True),
    Output("practice_status","children", allow_duplicate=True),
    Input("btn_start_drill","n_clicks"),
    State("inp_drill_name","value"),
    State("store_practice","data"),
    prevent_initial_call=True
)
def start_drill(n_start, drill_name, practice):
    if not n_start:
        return no_update, no_update
    if not (practice and practice.get("active")):
        return no_update, "Start a practice first."
    dn = (drill_name or "").strip()
    if not dn:
        return no_update, "Enter a drill name to start."
    did = _new_drill_id(practice["practice_id"], dn)
    return {"active": True, "name": dn, "drill_id": did}, f"Drill active: {dn} (id={did})"

@app.callback(
    Output("store_drill","data", allow_duplicate=True),
    Output("practice_status","children", allow_duplicate=True),
    Input("btn_end_drill","n_clicks"),
    State("store_drill","data"),
    prevent_initial_call=True
)
def end_drill(n_end, drill):
    if not n_end:
        return no_update, no_update
    if not (drill and drill.get("active")):
        return no_update, "No active drill to end."
    return {"active": False, "name": drill.get("name"), "drill_id": drill.get("drill_id")}, \
           f"Drill ended: {drill.get('name')}."


# Click debug
@app.callback(
    Output("debug_click", "children"),
    Input("court", "clickData"),
    prevent_initial_call=True
)
def debug_click(clickData):
    if not clickData:
        return "No clicks detected"
    p = clickData['points'][0]
    return f"Last click: x={p['x']:.1f}, y={p['y']:.1f}"


# Set modes: guard that practice & drill are active
@app.callback(
    Output("store_mode", "data"),
    Output("store_shots_needed", "data"),
    Output("store_clicks", "data"),
    Output("mode_hint", "children"),
    Input("btn_no_shot", "n_clicks"),
    Input("btn_start_shots", "n_clicks"),
    State("shots_count", "value"),
    State("store_practice","data"),
    State("store_drill","data"),
    prevent_initial_call=True
)
def set_mode(n_to, n_start, shots_value, practice, drill):
    if not (practice and practice.get("active")):
        return no_update, no_update, no_update, "Start a practice first."
    if not (drill and drill.get("active")):
        return no_update, no_update, no_update, "Start a drill first."

    trig = ctx.triggered_id
    if trig == "btn_no_shot":
        return "no_shot", 0, [], "No-shot possession: a text box will open. No clicks needed."
    if trig == "btn_start_shots":
        n = int(shots_value or 0)
        if n >= 1:
            msg = f"Shots possession: need {n} shot location(s). Click the court in order."
            if n == 1: msg = "Shots possession: need 1 shot location. Click the court."
            return "shots", n, [], msg
        return no_update, no_update, no_update, "Enter a valid number of shots (>= 1) and press Start."
    return no_update, no_update, no_update, no_update


# Controller: open/close modal; collect clicks for shots; open for no-shot immediately
@app.callback(
    [Output("store_modal_open", "data"),
     Output("store_last_click_xy", "data"),
     Output("store_preview", "data", allow_duplicate=True),
     Output("click_xy_label", "children"),
     Output("store_clicks", "data", allow_duplicate=True)],
    [Input("court", "clickData"),
     Input("btn_no_shot", "n_clicks"),
     Input("btn_cancel", "n_clicks"),
     Input("btn_submit", "n_clicks"),
     Input("possession_input", "n_submit")],
    [State("store_mode", "data"),
     State("store_shots_needed", "data"),
     State("store_clicks", "data")],
    prevent_initial_call=True
)
def control(clickData, n_no_shot, n_cancel, n_submit_btn, n_submit_enter, mode, shots_needed, clicks):
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    trig = ctx.triggered_id

    if trig == "btn_no_shot":
        return True, None, None, "No-shot possession: enter the possession shorthand below.", no_update

    if trig in ("btn_cancel", "btn_submit", "possession_input"):
        return False, no_update, no_update, no_update, no_update

    if trig == "court" and clickData and clickData.get("points"):
        x = float(clickData["points"][0]["x"]); y = float(clickData["points"][0]["y"])
        if not (0.0 <= x <= 50.0 and 0.0 <= y <= 47.0):
            return no_update, no_update, no_update, "Please click inside the court boundaries.", no_update

        if mode != "shots":
            return no_update, no_update, no_update, "Press Start shots and set #shots first.", no_update

        clicks = list(clicks or []) + [{"x": x, "y": y}]
        if len(clicks) < shots_needed:
            return no_update, no_update, no_update, f"Collected {len(clicks)}/{shots_needed} shots. Continue clicking…", clicks

        # All clicks collected -> open modal; stash xy_list into preview
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        preview = {
            "id": f"{now}-{_rand_suffix()}",
            "timestamp": now,
            "possession": None,
            "xy_list": clicks,
            "distance_list": None,
            "result": None
        }
        label = f"Collected {shots_needed}/{shots_needed} shots. Enter possession shorthand for this possession."
        return True, None, preview, label, clicks

    return no_update, no_update, no_update, no_update, no_update


# Toggle modal visibility
@app.callback(
    [Output("input_modal", "style")],
    [Input("store_modal_open", "data")]
)
def toggle_modal(open_flag):
    return [modal_style(bool(open_flag))]


# Toggle confirmation modal visibility
@app.callback(
    [Output("confirm_modal", "style")],
    [Input("show_confirm_modal", "data")]
)
def toggle_confirm_modal(show):
    return [modal_style(bool(show))]


# Parse + Preview (shots/no-shot)
@app.callback(
    [Output("output_block", "children"),
     Output("store_preview", "data")],
    [Input("btn_submit", "n_clicks"),
     Input("possession_input", "n_submit")],
    [State("possession_input", "value"),
     State("store_last_click_xy", "data"),
     State("store_preview", "data"),
     State("store_mode", "data"),
     State("store_shots_needed", "data"),
     State("store_roster", "data")],
    prevent_initial_call=True
)
def parse_and_preview(n_btn, n_enter, possession_text, single_xy_unused, preview_state, mode, shots_needed, roster):
    triggered = (n_btn or 0) > 0 or (n_enter or 0) > 0
    if not triggered:
        return no_update, no_update
    if not possession_text or not possession_text.strip():
        return html.Div("Please enter a possession string.", style={"color": "crimson"}), no_update

    # parse
    try:
        if pscoding18:
            parsed = pscoding18.parse_possession_string(possession_text.strip())
            lines = [parsed] if isinstance(parsed, str) else list(parsed)
        else:
            lines = [f"Dummy parse: {possession_text.strip()}"]
    except Exception as e:
        return html.Div(f"Error while parsing: {e}", style={"color": "crimson"}), no_update

    roster = roster or {}
    short_line = f"Shorthand: {possession_text.strip()}"
    annotated_lines = [annotate_with_roster_lastnames(s, roster) for s in (lines or [])]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    group_id = _new_group_id()

    if mode == "shots" and isinstance(preview_state, dict) and preview_state.get("xy_list"):
        xy_list = preview_state["xy_list"]
        dist_list, trail = [], []
        for i, pt in enumerate(xy_list, start=1):
            x, y = float(pt["x"]), float(pt["y"])
            dx, dy = (x - RIM_X), (y - RIM_Y)
            dft = round((dx*dx + dy*dy) ** 0.5, 1)
            dist_list.append(dft)
            trail.append(f"Shot {i}: x={x:.2f}, y={y:.2f} ft (dist {dft:.1f} ft)")

        pre_text = "\n".join([
            f"{short_line}  (Shots: {len(xy_list)})",
            *annotated_lines,
            *([f"(Shots: {len(xy_list)})"] if len(xy_list) > 1 else []),
            *trail
        ])

        preview = {
            "id": f"{now}-{_rand_suffix()}",
            "timestamp": now,
            "possession": possession_text.strip(),
            "xy_list": xy_list,
            "distance_list": dist_list,
            "group_id": group_id,
            "group_size": len(xy_list),
            "possession_type": "shots",
            "play_by_play": "\n".join(lines),
            "play_by_play_names": "\n".join(annotated_lines),
            # keep shorthand-derived single result only as a fallback (we compute per-shot later)
            "result": result_from_shorthand(possession_text.strip())
        }

        ui = html.Div([
            html.H4("Play-by-Play", style={"marginTop": 0}),
            html.Pre(pre_text, style={
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
        return ui, preview

    # no-shot
    if mode == "no_shot":
        pre_text = "\n".join([
            f"{short_line}  (No-shot possession)",
            *[annotate_with_roster_lastnames(s, roster) for s in (lines or [])]
        ])
        preview = {
            "id": f"{now}-{_rand_suffix()}",
            "timestamp": now,
            "possession": possession_text.strip(),
            "x": None, "y": None, "distance_ft": None,
            "group_id": group_id,
            "group_size": 0,
            "possession_type": "no_shot",
            "play_by_play": "\n".join(lines),
            "play_by_play_names": "\n".join(annotated_lines),
            "result": None
        }
        ui = html.Div([
            html.H4("Play-by-Play (No-shot possession)", style={"marginTop": 0}),
            html.Pre(pre_text, style={
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
        return ui, preview

    return html.Div("Press Start shots (set #shots) or No-shot first."), no_update


# Save confirmed possession -> Data Log, reset controls (with Practice/Drill context)
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Output("possession_input", "value"),
    Output("store_last_click_xy", "data", allow_duplicate=True),
    Output("output_block", "children", allow_duplicate=True),
    Output("last_saved", "children"),
    Output("tabs", "value"),
    Output("store_mode", "data", allow_duplicate=True),
    Output("store_shots_needed", "data", allow_duplicate=True),
    Output("store_clicks", "data", allow_duplicate=True),
    Input("btn_confirm", "n_clicks"),
    State("store_log", "data"),
    State("store_preview", "data"),
    State("store_practice","data"),
    State("store_drill","data"),
    prevent_initial_call=True
)
def save_possession(n_confirm, log_data, preview, practice, drill):
    if not n_confirm or not preview:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if not (practice and practice.get("active") and drill and drill.get("active")):
        return no_update, no_update, no_update, html.Div("Start practice and drill first.", style={"color":"crimson"}), no_update, no_update, no_update, no_update, no_update

    log = list(log_data or [])
    pid, pdate = practice.get("practice_id"), practice.get("date")
    did, dname = drill.get("drill_id"), drill.get("name")

    def attach_context(row: dict) -> dict:
        row = dict(row)
        row["practice_id"]   = pid
        row["practice_date"] = pdate
        row["drill_id"]      = did
        row["drill_name"]    = dname
        return _json_safe_row(row)

    last_saved_card = None

    # --- Append rows to disk (JSONL) and trim client window ---
    def _append_and_window(row_dicts: list[dict]):
        nonlocal log
        for r in row_dicts:
            # Append to disk O(1)
            append_log_row(attach_context(r))
            # Update client window
            log.append(attach_context(r))
        if len(log) > MAX_CLIENT_ROWS:
            log = log[-MAX_CLIENT_ROWS:]

    if preview.get("possession_type") == "shots" and preview.get("xy_list"):
        gid = preview.get("group_id")
        raw_lines = (preview.get("play_by_play") or "").splitlines()
        raw_lines_names = (preview.get("play_by_play_names") or "").splitlines()

        # Text-derived results
        text_results = _extract_results(raw_lines_names or raw_lines)

        xy_list   = preview["xy_list"]
        dist_list = preview.get("distance_list") or [None] * len(xy_list)
        need_n = len(xy_list)

        # Shorthand-derived results (authoritative)
        short_results = _results_from_shorthand_tokens(preview.get("possession", ""), shots_needed=need_n)

        # Prefer shorthand token for each shot, then fall back to text
        merged_results = []
        for i in range(need_n):
            sv = short_results[i] if i < len(short_results) else None
            tv = text_results[i] if i < len(text_results) else None
            merged_results.append(sv or tv)

        ts = preview["timestamp"]
        new_rows = []
        for idx, (pt, dft) in enumerate(zip(xy_list, dist_list), start=1):
            new_rows.append({
                "id": f"{ts}-{_rand_suffix()}",
                "timestamp": ts,
                "possession": preview["possession"],
                "x": float(pt["x"]), "y": float(pt["y"]),
                "distance_ft": dft,
                "play_by_play": "\n".join(raw_lines) if raw_lines else preview.get("possession", ""),
                "play_by_play_names": "\n".join(raw_lines_names) if raw_lines_names else "",
                "result": merged_results[idx-1],     # 'Make'/'Miss' or None
                "group_id": gid,
                "group_size": len(xy_list),
                "shot_index": idx,
                "possession_type": "shots",
            })
        _append_and_window(new_rows)

        # Summary card
        xs = ", ".join(f"{pt['x']:.1f}" for pt in xy_list)
        ys = ", ".join(f"{pt['y']:.1f}" for pt in xy_list)
        dists = ", ".join(f"{d:.1f}" if d is not None else "" for d in dist_list)
        res_str = ", ".join(r for r in merged_results if r)
        last_saved_card = html.Div([
            html.Div("Last saved possession", style={"fontWeight":700, "marginBottom":"6px"}),
            html.Pre(
                f"Shorthand: {preview['possession']}\n"
                f"X (ft): {xs}\n"
                f"Y (ft): {ys}\n"
                f"Shot Distance (ft): {dists}\n"
                f"Result: {res_str}\n"
                f"Group Size: {len(xy_list)}\n"
                f"Drill: {dname}   Date: {pdate}",
                style={"background":"#0b1021","color":"#e6edf3","padding":"12px","borderRadius":"8px","whiteSpace":"pre-wrap"}
            )
        ])

    elif preview.get("possession_type") == "no_shot":
        row = {
            "id": preview["id"],
            "timestamp": preview["timestamp"],
            "possession": preview["possession"],
            "x": None, "y": None, "distance_ft": None,
            "play_by_play": preview.get("play_by_play", ""),
            "play_by_play_names": preview.get("play_by_play_names", ""),
            "result": None,
            "group_id": preview.get("group_id"),
            "group_size": 0,
            "shot_index": None,
            "possession_type": "no_shot",
        }
        _append_and_window([row])

        last_saved_card = html.Div([
            html.Div("Last saved possession", style={"fontWeight":700, "marginBottom":"6px"}),
            html.Pre(
                f"Shorthand: {preview['possession']}  (No-shot)\n"
                f"{preview.get('play_by_play_names','')}\n"
                f"Drill: {dname}   Date: {pdate}",
                style={"background":"#0b1021","color":"#e6edf3","padding":"12px","borderRadius":"8px","whiteSpace":"pre-wrap"}
            )
        ])

    else:
        _append_and_window([preview])

    notice = "Saved to Data Log"
    cleared_output = html.Div([html.Div(notice, style={"color": "#16a34a", "marginBottom": "8px"})])

    # Stay on "chart"
    return (log, "", None, cleared_output, last_saved_card, "chart",
            None, 0, [])


# Discard preview
@app.callback(
    [Output("output_block", "children", allow_duplicate=True)],
    [Input("btn_discard", "n_clicks")],
    prevent_initial_call=True
)
def discard_preview(n):
    if not n:
        return [no_update]
    return [html.Div([html.Div("Discarded. Re-enter the possession when ready.", style={"color": "#b45309"})])]


# =========================
# Data Log: Dates → Drills → Table with Delete Only (no edit)
# =========================

def _unique(items):
    seen, out = set(), []
    for v in items:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def _group_possessions(rows):
    """
    Collapse shot-rows belonging to the same possession (by group_id if present
    else by timestamp). Result column shows the per-shot results in order.
    """
    groups = {}
    for r in rows or []:
        key = r.get("group_id") or r.get("timestamp") or r.get("id")
        g = groups.setdefault(key, {
            "key": key,
            "timestamp": r.get("timestamp"),
            "possession": r.get("possession"),
            "xs": [], "ys": [], "dists": [],
            "res_seq": [],  # (shot_index, result)
            "group_size": 0,
            "practice_date": r.get("practice_date"),
            "practice_id": r.get("practice_id"),
            "drill_name": r.get("drill_name"),
            "drill_id": r.get("drill_id"),
        })
        x, y, d = r.get("x"), r.get("y"), r.get("distance_ft")
        if x is not None: g["xs"].append(x)
        if y is not None: g["ys"].append(y)
        if d is not None: g["dists"].append(d)

        res = r.get("result")
        g["res_seq"].append((r.get("shot_index") or 0, res if res else None))

        g["group_size"] = max(g["group_size"] or 0, r.get("group_size") or 0)

    consolidated = []
    for _, g in groups.items():
        results_ordered = [rv for _, rv in sorted(g["res_seq"], key=lambda t: (t[0] or 0)) if rv]
        consolidated.append({
            "id": g["key"],
            "timestamp": g["timestamp"],
            "possession": g["possession"] or "",
            "x": ", ".join(f"{v:.1f}" for v in g["xs"]) if g["xs"] else "",
            "y": ", ".join(f"{v:.1f}" for v in g["ys"]) if g["ys"] else "",
            "distance_ft": ", ".join(f"{v:.1f}" for v in g["dists"]) if g["dists"] else "",
            "result": ", ".join(results_ordered) if results_ordered else "",
            "group_size": g["group_size"] or (len(g["xs"]) if g["xs"] else 0),
            "practice_date": g["practice_date"],
            "practice_id": g["practice_id"],
            "drill_name": g["drill_name"] or "(unnamed)",
            "drill_id": g["drill_id"],
        })
    consolidated.sort(key=lambda r: r.get("timestamp",""))
    return consolidated

@app.callback(
    Output("log_container","children"),
    Input("store_log","data"),
    Input("log_level","data"),
    Input("sel_date","data"),
    Input("sel_drill","data"),
)
def render_log_view(_rows_ignored, level, sel_date, sel_drill):
    """
    Render from DISK via streaming (iter_rows), ignoring the full dataset in the client.
    This keeps UI snappy regardless of total rows.
    """
    level = level or "dates"
    wrap = lambda *kids: html.Div(list(kids), style={"maxWidth":"1200px","margin":"18px auto","padding":"0 12px"})

    if level == "dates":
        by_date = {}
        for r in iter_rows():
            d = r.get("practice_date")
            if not d:
                continue
            key = r.get("group_id") or r.get("timestamp") or r.get("id")
            by_date.setdefault(d, set()).add(key)
        dates = sorted(by_date.keys(), reverse=True)

        if not dates:
            return wrap(
                html.H4("Saved Possessions"),
                html.Div("No practice data yet. Start a practice and drill to begin logging possessions.", style={"color":"#666"})
            )

        items = []
        for d in dates:
            count = len(by_date[d])
            items.append(
                html.Div([
                    html.Button(
                        f"Practice: {d} ({count} possessions)",
                        id={"type":"date_btn","date":d}, n_clicks=0,
                        style={"display":"block","width":"calc(100% - 80px)","textAlign":"left","padding":"12px 16px",
                               "border":"1px solid #ddd","borderRadius":"8px",
                               "background":"white","cursor":"pointer","fontSize":"16px"}
                    ),
                    html.Button(
                        "Delete",
                        id={"type":"delete_practice_btn","date":d}, n_clicks=0,
                        style={"padding":"8px 12px","borderRadius":"6px","marginLeft":"8px",
                               "border":"1px solid #dc2626","color":"#dc2626","background":"white",
                               "cursor":"pointer","fontSize":"14px"}
                    )
                ], style={"display":"flex","alignItems":"center","margin":"6px 0"})
            )

        return wrap(
            html.H4("Practice Dates"),
            html.Div(items, role="list", style={"maxWidth":"800px"})
        )

    if level == "drills" and sel_date:
        by_drill = {}
        for r in iter_rows(date=sel_date):
            drill_name = r.get("drill_name") or "(unnamed)"
            key = r.get("group_id") or r.get("timestamp") or r.get("id")
            by_drill.setdefault(drill_name, set()).add(key)

        if not by_drill:
            return wrap(
                html.Div([
                    html.Button("← Back to dates", id="btn_back_dates", n_clicks=0,
                              style={"cursor":"pointer","color":"#2563eb","padding":"6px 12px",
                                     "border":"1px solid #2563eb","borderRadius":"6px","background":"white"}),
                    html.Span(f"Practice: {sel_date}", style={"fontWeight":600,"marginLeft":"15px"}),
                ], style={"marginBottom":"15px","display":"flex","alignItems":"center"}),
                html.Div("No drills found for this practice.", style={"color":"#666"})
            )

        items = []
        for drill_name in sorted(by_drill.keys()):
            count = len(by_drill[drill_name])
            items.append(
                html.Div([
                    html.Button(
                        f"{drill_name} ({count} possessions)",
                        id={"type":"drill_btn","drill":drill_name}, n_clicks=0,
                        style={"display":"block","width":"calc(100% - 80px)","textAlign":"left","padding":"12px 16px",
                               "border":"1px solid #ddd","borderRadius":"8px",
                               "background":"white","cursor":"pointer","fontSize":"16px"}
                    ),
                    html.Button(
                        "Delete",
                        id={"type":"delete_drill_btn","drill":drill_name,"date":sel_date}, n_clicks=0,
                        style={"padding":"8px 12px","borderRadius":"6px","marginLeft":"8px",
                               "border":"1px solid #dc2626","color":"#dc2626","background":"white",
                               "cursor":"pointer","fontSize":"14px"})
                ], style={"display":"flex","alignItems":"center","margin":"6px 0"})
            )

        return wrap(
            html.Div([
                html.Button("← Back to dates", id="btn_back_dates", n_clicks=0,
                          style={"cursor":"pointer","color":"#2563eb","padding":"6px 12px",
                                 "border":"1px solid #2563eb","borderRadius":"6px","background":"white"}),
                html.Span(f"Practice: {sel_date}", style={"fontWeight":600,"marginLeft":"15px"}),
            ], style={"marginBottom":"15px","display":"flex","alignItems":"center"}),
            html.H4("Drills"),
            html.Div(items, role="list", style={"maxWidth":"800px"})
        )

    if level == "table" and sel_date and sel_drill:
        raw = list(iter_rows(date=sel_date, drill=sel_drill))
        flt = _group_possessions(raw)
        if not flt:
            return wrap(
                html.Div([
                    html.Button("← Back to drills", id="btn_back_drills", n_clicks=0,
                              style={"cursor":"pointer","color":"#2563eb","padding":"6px 12px",
                                     "border":"1px solid #2563eb","borderRadius":"6px","background":"white"}),
                    html.Span(f"{sel_date} · {sel_drill}", style={"fontWeight":600,"marginLeft":"15px"}),
                ], style={"marginBottom":"15px","display":"flex","alignItems":"center"}),
                html.Div("No possessions for this drill.", style={"color":"#666"})
            )

        columns = [
            {"name": "#", "id": "__rownum__"},
            {"name": "Shorthand", "id": "possession"},
            {"name": "X (ft)", "id": "x"},
            {"name": "Y (ft)", "id": "y"},
            {"name": "Shot Distance (ft)", "id": "distance_ft"},
            {"name": "Result", "id": "result"},
            {"name": "Number of shots", "id": "group_size", "type": "numeric"},
            {"name": "Row ID", "id": "id"},
        ]

        for i, r in enumerate(flt, start=1):
            r["__rownum__"] = i

        tbl = dash_table.DataTable(
            id={"role":"tbl_log","scope":"main"},
            data=flt, columns=columns, hidden_columns=["id"],
            page_size=15,
            style_cell={"whiteSpace":"pre-line","fontFamily":"system-ui","fontSize":"15px"},
            style_header={"fontWeight":"600"}, style_table={"overflowX":"auto"},
            sort_action="native", filter_action="native",
            editable=False,  # <- no inline edit
            row_selectable="multi", selected_rows=[],
            # --- NEW: virtualize rendering so only visible rows are painted
            virtualization=True,
            fixed_rows={"headers": True}
        )
        return wrap(
            html.Div([
                html.Button("← Back to drills", id="btn_back_drills", n_clicks=0,
                          style={"cursor":"pointer","color":"#2563eb","padding":"6px 12px",
                                 "border":"1px solid #2563eb","borderRadius":"6px","background":"white"}),
                html.Span(f"{sel_date} · {sel_drill}", style={"fontWeight":600,"marginLeft":"15px"}),
            ], style={"marginBottom":"15px","display":"flex","alignItems":"center"}),
            tbl,
            html.Div([html.Button("Delete selected row(s)", id={"role":"btn_delete_rows","scope":"main"}, n_clicks=0,
                                  style={"marginTop":"10px","padding":"8px 12px","borderRadius":"8px",
                                         "border":"1px solid #dc2626","color":"#dc2626"})],
                     style={"display":"flex","justifyContent":"flex-end"})
        )

    return wrap(html.H4("Saved Possessions"), html.Div("Select a practice date."))


# Click a date -> go to drills
@app.callback(
    Output("log_level","data"),
    Output("sel_date","data"),
    Input({"type":"date_btn","date":ALL}, "n_clicks"),
    prevent_initial_call=True
)
def pick_date(n_clicks):
    if not n_clicks or not any(n_clicks):
        return no_update, no_update
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and "date" in triggered:
        return "drills", triggered["date"]
    return no_update, no_update

# Click a drill -> go to table
@app.callback(
    Output("log_level","data", allow_duplicate=True),
    Output("sel_drill","data"),
    Input({"type":"drill_btn","drill":ALL}, "n_clicks"),
    prevent_initial_call=True
)
def pick_drill(n_clicks):
    if not n_clicks or not any(n_clicks):
        return no_update, no_update
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and "drill" in triggered:
        return "table", triggered["drill"]
    return no_update, no_update

# Back links (simple navigation)
@app.callback(
    Output("log_level","data", allow_duplicate=True),
    Output("sel_drill","data", allow_duplicate=True),
    Input("btn_back_drills","n_clicks"),
    prevent_initial_call=True
)
def back_to_drills(n):
    if not n:
        return no_update, no_update
    return "drills", None

@app.callback(
    Output("log_level","data", allow_duplicate=True),
    Output("sel_date","data", allow_duplicate=True),
    Output("sel_drill","data", allow_duplicate=True),
    Input("btn_back_dates","n_clicks"),
    prevent_initial_call=True
)
def back_to_dates(n):
    if not n:
        return no_update, no_update, no_update
    return "dates", None, None


# Confirmation modal callbacks (PICKS LIVE TABLE INSTANCE)
@app.callback(
    Output("show_confirm_modal", "data"),
    Output("confirm_delete_type", "data"),
    Output("confirm_delete_target", "data"),
    Output("confirm_message", "children"),
    Input({"type":"delete_practice_btn","date":ALL}, "n_clicks"),
    Input({"type":"delete_drill_btn","drill":ALL,"date":ALL}, "n_clicks"),
    Input({"role":"btn_delete_rows","scope":ALL}, "n_clicks"),
    State({"role":"tbl_log","scope":ALL}, "selected_rows"),
    State({"role":"tbl_log","scope":ALL}, "data"),
    State("sel_date", "data"),
    State("sel_drill", "data"),
    prevent_initial_call=True
)
def show_confirmation_dialog(delete_practice_clicks, delete_drill_clicks, delete_rows_clicks,
                             selected_rows_all, table_data_all, current_date, current_drill):
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    trig = ctx.triggered_id
    trig_val = ctx.triggered[0].get("value", None)

    # Pick the LIVE table (scan from end; placeholders are earlier)
    selected_rows = []
    table_data = []
    for sr, td in zip((selected_rows_all or [])[::-1], (table_data_all or [])[::-1]):
        if isinstance(td, list) and len(td) > 0:
            selected_rows = sr or []
            table_data = td
            break

    if not trig_val:
        return no_update, no_update, no_update, no_update

    if isinstance(trig, dict) and trig.get("type") == "delete_practice_btn":
        date = trig.get("date")
        if not date:
            return no_update, no_update, no_update, no_update
        return True, "practice", date, (
            f"Are you sure you want to delete the entire practice from {date}? "
            "This will permanently remove all drills and possessions from this practice."
        )

    if isinstance(trig, dict) and trig.get("type") == "delete_drill_btn":
        drill = trig.get("drill"); date = trig.get("date")
        if not drill or not date:
            return no_update, no_update, no_update, no_update
        target = {"drill": drill, "date": date}
        return True, "drill", target, (
            f"Are you sure you want to delete the drill '{drill}' from {date}? "
            "This will permanently remove all possessions from this drill."
        )

    if isinstance(trig, dict) and trig.get("role") == "btn_delete_rows":
        if selected_rows and table_data:
            cnt = len(selected_rows)
            return True, "possession", selected_rows, (
                f"Are you sure you want to delete {cnt} selected "
                f"{'possession' if cnt == 1 else 'possessions'}? This action cannot be undone."
            )

    return no_update, no_update, no_update, no_update

# Cancel confirmation
@app.callback(
    Output("show_confirm_modal", "data", allow_duplicate=True),
    Input("btn_confirm_cancel", "n_clicks"),
    prevent_initial_call=True
)
def cancel_confirmation(n):
    if not n:
        return no_update
    return False

# Execute deletion after confirmation (stream-based compaction)
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Output("show_confirm_modal", "data", allow_duplicate=True),
    Output("log_level", "data", allow_duplicate=True),
    Output("sel_date", "data", allow_duplicate=True),
    Output("sel_drill", "data", allow_duplicate=True),
    Input("btn_confirm_delete", "n_clicks"),
    State("confirm_delete_type", "data"),
    State("confirm_delete_target", "data"),
    State({"role":"tbl_log","scope":ALL}, "data"),
    prevent_initial_call=True
)
def execute_deletion(n, delete_type, target, table_data_all):
    if not n or not delete_type or not target:
        return no_update, no_update, no_update, no_update, no_update

    # Pick the LIVE table's data
    table_data = []
    for td in (table_data_all or [])[::-1]:
        if isinstance(td, list) and len(td) > 0:
            table_data = td
            break

    if delete_type == "practice":
        practice_date = target
        def keep_pred(r):  # keep everything NOT in that practice
            return r.get("practice_date") != practice_date
        _rewrite_storage_filtered(keep_pred)
        # Refresh client window and nav back to dates
        return _refresh_client_window(), False, "dates", None, None

    elif delete_type == "drill":
        drill_name = target["drill"]
        practice_date = target["date"]
        def keep_pred(r):
            return not (r.get("practice_date") == practice_date and
                        (r.get("drill_name") or "(unnamed)") == drill_name)
        _rewrite_storage_filtered(keep_pred)
        return _refresh_client_window(), False, "drills", practice_date, None

    elif delete_type == "possession":
        if not table_data or not isinstance(target, list) or len(target) == 0:
            return no_update, False, no_update, no_update, no_update

        selected_keys = {table_data[i].get("id") for i in target if 0 <= i < len(table_data)}
        if not selected_keys:
            return no_update, False, no_update, no_update, no_update

        def keep_pred(r):
            key = r.get("group_id") or r.get("timestamp") or r.get("id")
            return key not in selected_keys

        _rewrite_storage_filtered(keep_pred)
        return _refresh_client_window(), False, no_update, no_update, no_update

    return no_update, no_update, no_update, no_update, no_update


# Roster editor callbacks
@app.callback(
    Output("tbl_roster", "data", allow_duplicate=True),
    Input("btn_roster_add", "n_clicks"),
    State("tbl_roster", "data"),
    prevent_initial_call=True
)
def add_roster_row(n, rows):
    if not n:
        return no_update
    rows = list(rows or [])
    rows.append({"jersey": None, "name": ""})
    return rows

@app.callback(
    Output("store_roster", "data", allow_duplicate=True),
    Output("roster_status", "children"),
    Input("btn_roster_save", "n_clicks"),
    State("tbl_roster", "data"),
    prevent_initial_call=True
)
def save_roster(n, rows):
    if not n:
        return no_update, no_update
    roster = {}
    for r in rows or []:
        try:
            j = r.get("jersey")
            nm = (r.get("name") or "").strip()
            if j is None or nm == "": continue
            j = int(j)
            if 0 < j < 100:
                roster[str(j)] = nm
        except Exception:
            continue
    ok = save_roster_to_disk(roster)
    msg = "Roster saved." if ok else "Failed to save roster."
    return roster, msg

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)
