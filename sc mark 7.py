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
# Detect the final +/++/- within a string
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
# Boundaries for where a name should STOP (before verbs/punctuation/end)
_NAME_BOUNDARY = r"(?=\s*(?:makes?|made|miss(?:es)?|scores?|shoots?|attempts?|[,.!]|$))"

# Permissive "guarded by" detector (handles commas/periods and variable spacing)
# — non-greedy name groups + boundary look-ahead so defender doesn't eat "makes the shot"
_SHOOT_GUARD_RE = re.compile(
    rf"([A-Za-z][\w.\-'\s,]+?)\s*guarded\s*by\s*([A-Za-z][\w.\-'\s,]+?){_NAME_BOUNDARY}",
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

def _strip_after_verbs(s: str) -> str:
    """Remove any trailing 'makes/made/misses ...' clause if present."""
    s = re.split(_MAKES_RE, s)[0]
    s = re.split(_MISSES_RE, s)[0]
    return s.strip()

def _clean_shooter_line(s: str) -> str:
    """
    From text like 'Fontana, guarded by Lusk misses the shot' -> 'Fontana guarded by Lusk'
    Uses the strict 'guarded by' extractor; falls back to trimming after verbs.
    """
    if not s:
        return ""
    m = _SHOOT_GUARD_RE.search(s)
    if m:
        left  = _clean_name_fragment(m.group(1))
        right = _clean_name_fragment(m.group(2))
        return f"{left} guarded by {right}"
    # fallback: take text up to make/miss and normalize
    s2 = _strip_after_verbs(s)
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

    # New structured fields (nullable; legacy rows may not have them)
    schema_version = r.get("schema_version")
    shooter_id = r.get("shooter_id"); shooter_name = r.get("shooter_name")
    primary_defender_id = r.get("primary_defender_id"); primary_defender_name = r.get("primary_defender_name")
    assist_id = r.get("assist_id"); assist_name = r.get("assist_name")
    on_ball_actions  = r.get("on_ball_actions", [])
    off_ball_actions = r.get("off_ball_actions", [])
    coverages_on     = r.get("coverages_on", [])
    coverages_off    = r.get("coverages_off", [])
    defense_context  = r.get("defense_context", [])
    shot_type        = r.get("shot_type")
    zone_id          = r.get("zone_id")

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

        # structured (nullable for legacy)
        "schema_version": int(schema_version) if str(schema_version or "").isdigit() else 1,
        "shooter_id": shooter_id or None,
        "shooter_name": (shooter_name or None),
        "primary_defender_id": primary_defender_id or None,
        "primary_defender_name": (primary_defender_name or None),
        "assist_id": assist_id or None,
        "assist_name": (assist_name or None),
        "on_ball_actions": list(on_ball_actions) if isinstance(on_ball_actions, list) else [],
        "off_ball_actions": list(off_ball_actions) if isinstance(off_ball_actions, list) else [],
        "coverages_on": list(coverages_on) if isinstance(coverages_on, list) else [],
        "coverages_off": list(coverages_off) if isinstance(coverages_off, list) else [],
        "defense_context": list(defense_context) if isinstance(defense_context, list) else [],
        "shot_type": shot_type or None,
        "zone_id": int(zone_id) if str(zone_id or "").isdigit() else None,
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

def load_log_from_disk():
    if not os.path.exists(DATA_PATH):
        return []
    rows = []
    try:
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
    """
    Ensure consistent JSON types and include new schema v2 fields as nullable lists/strings.
    """
    out = dict(r)

    # Numeric bounds
    out["x"] = _coerce_float(out.get("x"))
    out["y"] = _coerce_float(out.get("y"))
    if out["x"] is not None:
        out["x"] = min(max(0.0, out["x"]), 50.0)
    if out["y"] is not None:
        out["y"] = min(max(0.0, out["y"]), 47.0)
    out["distance_ft"] = _coerce_float(out.get("distance_ft"))

    for key in ("group_size", "shot_index", "zone_id"):
        if out.get(key) is not None and str(out.get(key)) != "":
            try:
                out[key] = int(out[key])
            except:
                out[key] = None

    # Nullable strings
    for k in ("group_id", "possession_type", "practice_id", "drill_id", "practice_date", "drill_name",
              "result", "shooter_id", "primary_defender_id", "assist_id",
              "shooter_name","primary_defender_name","assist_name","shot_type",
              "event_type","turnover_type","foul_type"):
        if not out.get(k):
            out[k] = None

    # Lists
    def _as_str_list(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x or "").strip()]
        return []
    for k in ("on_ball_actions","off_ball_actions","coverages_on","coverages_off","defense_context"):
        out[k] = _as_str_list(out.get(k))

    # Text presence (raw & normalized)
    if "play_by_play_names" not in out:
        out["play_by_play_names"] = ""
    if "play_by_play" not in out:
        out["play_by_play"] = out.get("possession","") or ""

    # Schema version default
    try:
        out["schema_version"] = int(out.get("schema_version") or 1)
    except:
        out["schema_version"] = 1

    # No 'shooter' legacy field
    out.pop("shooter", None)

    return out

def save_log_to_disk(rows):
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
# Lite structured extraction (IDs + actions/coverages)
# =========================
# To avoid forward references, use a local number regex and a safe last-name util.
_NUM_RE_LITE = re.compile(r'(?<!\w)(\d{1,2})(?!\w)')
def _last_name_lite(full: str) -> str:
    if not full:
        return ""
    parts = [p for p in re.split(r'\s+', str(full).strip()) if p]
    return parts[-1] if parts else str(full or "")

# Assist: allow a closing ) ] } before punctuation/end so "(assisted by X)!" matches
_ASSIST_RE       = re.compile(
    r"\bassis(?:t|ted)\s*by\s+([A-Za-z0-9.\-'\s,]+?)(?=\s*[\)\]\}]*[,.!]|$)",
    flags=re.IGNORECASE
)
_ASSIST_COLON_RE = re.compile(
    r"\bassist\s*:\s*([A-Za-z0-9.\-'\s,]+?)(?=\s*[\)\]\}]*[,.!]|$)",
    flags=re.IGNORECASE
)

_ON_BALL_TOKENS = ["h","d","p","pnr","pnp","slp","gst","rj","dho","ho"]
_OFF_BALL_TOKENS= ["bd","pn","fl","bk","awy","crs","wdg","rip","ucla","stg","ivs","elv"]
_COVERAGE_TOKENS= ["ch","ct","sw","bz","cs","ice","tl"]
_DEF_CTX_TOKENS = ["hp","sw","rot"]  # help / switch / rotation

_NUM_ONLY_RE = re.compile(r'^\d{1,2}$')

def _player_id_from_num(num: str) -> str | None:
    try:
        return f"player_{int(str(num).strip())}"
    except Exception:
        return None

def _build_player_indexes(roster: dict):
    """
    Input roster like {"12":"John Smith","23":"Mike Jones"}.
    Returns tuples of lookup dicts.
    """
    num_to_id, id_to_name, full_to_id, last_to_ids = {}, {}, {}, {}
    for k, full in (roster or {}).items():
        if full is None:
            continue
        name = str(full).strip()
        if not name:
            continue
        num = str(k).strip()
        if not num.isdigit():
            continue
        pid = _player_id_from_num(num)
        num_to_id[num] = pid
        id_to_name[pid] = name
        full_to_id[name.casefold()] = pid
        ln = _last_name_lite(name).casefold()
        last_to_ids.setdefault(ln, set()).add(pid)
    return num_to_id, id_to_name, full_to_id, last_to_ids

def _clean_token_for_match(token: str) -> str:
    return _clean_name_fragment(token)

def _resolve_player_token(token: str, roster: dict):
    """
    Resolve "12" or "Smith" or "John Smith" -> (player_id, canonical_name).
    Ambiguity/unknown => (None, cleaned_token or None).
    """
    if not token:
        return (None, None)
    s = _clean_token_for_match(token)

    num_to_id, id_to_name, full_to_id, last_to_ids = _build_player_indexes(roster)

    # jersey number
    if _NUM_ONLY_RE.fullmatch(s):
        pid = num_to_id.get(str(int(s)))
        return (pid, id_to_name.get(pid)) if pid else (None, s)

    # exact full-name
    pid = full_to_id.get(s.casefold())
    if pid:
        return (pid, id_to_name.get(pid))

    # unique last-name
    ln = _last_name_lite(s).casefold()
    ids = list(last_to_ids.get(ln, []))
    if len(ids) == 1:
        pid = ids[0]
        return (pid, id_to_name.get(pid))

    return (None, s)

def _scan_tokens(line: str, tokens: list[str]) -> list[str]:
    s = (line or "").lower()
    found = []
    for t in tokens:
        if re.search(rf'(?<!\w){re.escape(t)}(?!\w)', s):
            found.append(t)
    # stable unique order
    out, seen = [], set()
    for t in found:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def parse_shot_line_lite(line_text: str, roster: dict) -> dict:
    """
    Extract shooter/defender/assist + on/off-ball actions + coverages + defense context.
    NOTE: Defender is taken ONLY from 'guarded by <Name>' to avoid false positives.
    Missing pieces -> None / [] (never blocks a save).
    """
    out = {
        "shooter_id": None, "shooter_name": None,
        "primary_defender_id": None, "primary_defender_name": None,
        "assist_id": None, "assist_name": None,
        "on_ball_actions": [], "off_ball_actions": [],
        "coverages_on": [], "coverages_off": [], "defense_context": []
    }
    s = (line_text or "").strip()
    if not s:
        return out

    # Shooter & primary defender: "<A> guarded by <B>"
    m = _SHOOT_GUARD_RE.search(s)
    if m:
        sh_raw = _clean_name_fragment(m.group(1))
        df_raw = _clean_name_fragment(m.group(2))
        pid, pname = _resolve_player_token(sh_raw, roster)
        did, dname = _resolve_player_token(df_raw, roster)
        out["shooter_id"], out["shooter_name"] = pid, pname
        # Defender ONLY from 'guarded by'
        out["primary_defender_id"], out["primary_defender_name"] = did, dname
    else:
        # Fallback shooter guess: first token before 'make/miss' or 'guarded by'
        head = re.split(r'\bguarded\s*by\b', s, flags=re.IGNORECASE)[0]
        head = _MISSES_RE.split(head)[0]
        head = _MAKES_RE.split(head)[0]
        toks = re.findall(r"[A-Za-z0-9.\-']+", head)
        if toks:
            pid, pname = _resolve_player_token(toks[0], roster)
            out["shooter_id"], out["shooter_name"] = pid, pname
        # IMPORTANT: do NOT guess a defender if we don't find "guarded by"

    # Assist (now robust to closing parentheses before punctuation)
    m2 = _ASSIST_RE.search(s) or _ASSIST_COLON_RE.search(s)
    if m2:
        a_raw = _clean_name_fragment(m2.group(1))
        aid, aname = _resolve_player_token(a_raw, roster)
        out["assist_id"], out["assist_name"] = aid, aname

    # Action / coverage tags (from shorthand tokens if present in text)
    on_tokens  = _scan_tokens(s, _ON_BALL_TOKENS)
    off_tokens = _scan_tokens(s, _OFF_BALL_TOKENS)
    cov_tokens = _scan_tokens(s, _COVERAGE_TOKENS)
    def_tokens = _scan_tokens(s, _DEF_CTX_TOKENS)

    out["on_ball_actions"]  = on_tokens
    out["off_ball_actions"] = off_tokens
    if cov_tokens:
        if on_tokens:
            out["coverages_on"] = cov_tokens
        else:
            out["coverages_off"] = cov_tokens
    out["defense_context"] = def_tokens
    return out

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

# Optional derived at save-time (kept here for callbacks to call)
def shot_type_from_xy(x, y) -> str | None:
    """Return '2pt'/'3pt' based on 3PT geometry (handles corner 3s)."""
    try:
        x = float(x); y = float(y)
    except Exception:
        return None

    def t_for_x(x_target):
        val = (x_target - RIM_X) / THREE_R
        val = max(-1.0, min(1.0, val))
        return math.asin(val)

    tL, tR = t_for_x(LEFT_POST_X), t_for_x(RIGHT_POST_X)
    yL = RIM_Y + THREE_R*math.cos(tL)
    yR = RIM_Y + THREE_R*math.cos(tR)

    # Corner exception: outside posts & below intersection → 3pt
    if (x <= LEFT_POST_X and y <= yL) or (x >= RIGHT_POST_X and y <= yR):
        return "3pt"

    dist = math.hypot(x - RIM_X, y - RIM_Y)
    return "3pt" if dist >= THREE_R else "2pt"

# =========================
# App
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE

_initial_log    = load_log_from_disk()
try_flush_pending()
_initial_roster = load_roster_from_disk()

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

                        # ---- Debug toggle + JSON Inspector
                        html.Div([
                            dcc.Checklist(
                                id="debug_toggle",
                                options=[{"label": " Debug mode", "value": "on"}],
                                value=[],
                                style={"userSelect": "none"}
                            ),
                            html.Button("JSON Inspector", id="btn_open_json", n_clicks=0,
                                        style={"marginLeft": "10px","padding":"6px 10px","borderRadius":"8px",
                                               "border":"1px solid #aaa","background":"white"})
                        ], style={"display":"flex","alignItems":"center","gap":"6px","margin":"4px 0 8px"}),

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
                        html.Button("Test Parse", id="btn_test_parse", n_clicks=0,
                                    style={"padding":"10px 16px","borderRadius":"8px",
                                           "border":"1px solid #ccc","background":"white","marginRight":"auto"}),
                        html.Button("Cancel", id="btn_cancel", n_clicks=0,
                                    style={"padding": "10px 16px", "borderRadius": "8px",
                                           "border": "1px solid #ccc", "background": "white"}),
                        html.Button("Submit", id="btn_submit", n_clicks=0,
                                    style={"padding": "10px 16px", "borderRadius": "8px",
                                           "border": "none", "background": "#2563eb", "color": "white"}),
                    ])
                ]
            )),

            # JSON Inspector modal (read-only)
            html.Div(id="json_modal", style=modal_style(False), children=html.Div(
                style={"width":"740px","background":"white","borderRadius":"12px",
                       "boxShadow":"0 10px 30px rgba(0,0,0,0.2)","padding":"18px"},
                children=[
                    html.H3("Last 20 saved rows (read-only)"),
                    html.Div(id="json_modal_body", style={"maxHeight":"60vh","overflow":"auto",
                                                          "background":"#0b1021","color":"#e6edf3",
                                                          "padding":"12px","borderRadius":"8px",
                                                          "fontFamily":"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
                                                          "fontSize":"13px"}),
                    html.Div(style={"textAlign":"right","marginTop":"10px"}, children=[
                        html.Button("Close", id="btn_close_json", n_clicks=0,
                                    style={"padding":"8px 12px","borderRadius":"8px",
                                           "border":"1px solid #aaa","background":"white"})
                    ])
                ]
            )),
        ]),

        dcc.Tab(label="Data Log", value="log", children=[
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

    # Data Log navigation state
    dcc.Store(id="log_level", data="dates"),
    dcc.Store(id="sel_date", data=None),
    dcc.Store(id="sel_drill", data=None),

    # Confirmation modal stores
    dcc.Store(id="confirm_delete_type", data=None),
    dcc.Store(id="confirm_delete_target", data=None),
    dcc.Store(id="show_confirm_modal", data=False),

    # Debug/Inspector stores
    dcc.Store(id="store_debug", data=False),
    dcc.Store(id="show_json_modal", data=False),

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
    # Hidden placeholders so pattern-matching Inputs exist at initial load.
    # ---------------------------------------------------------------------
    html.Div(style={"display": "none"}, children=[
        html.Button(id={"role": "btn_delete_rows", "scope": "main"}, n_clicks=0),
        dash_table.DataTable(
            id={"role": "tbl_log", "scope": "main"},
            columns=[{"name": "dummy", "id": "dummy"}],
            data=[]
        ),
        html.Button(id={"type": "delete_practice_btn", "date": "__placeholder__"}, n_clicks=0),
        html.Button(id={"type": "delete_drill_btn", "drill": "__placeholder__", "date": "__placeholder__"}, n_clicks=0),
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

    dash_table.DataTable(
        id={"role": "tbl_log", "scope": "main"},
        columns=[{"name": "dummy", "id": "dummy"}],
        data=[]
    ),
    html.Button(id={"role": "btn_delete_rows", "scope": "main"}, n_clicks=0),

    html.Button(id="btn_back_drills", n_clicks=0),
    html.Button(id="btn_back_dates", n_clicks=0),

    html.Div(id="log_container"),
    html.Div(id="confirm_modal"),
    html.Div(id="last_saved"),
    html.Div(id="json_modal"),
    html.Div(id="json_modal_body"),

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
    dcc.Store(id="log_level"),
    dcc.Store(id="sel_date"),
    dcc.Store(id="sel_drill"),
    dcc.Store(id="confirm_delete_type"),
    dcc.Store(id="confirm_delete_target"),
    dcc.Store(id="show_confirm_modal"),
    dcc.Store(id="store_debug"),
    dcc.Store(id="show_json_modal"),

    dcc.Interval(id="retry_interval"),
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

# general name token
_NAME_CHARS = r"[A-Za-z0-9.\-'\s,]+?"

# screen/screener extraction
_SCREEN_FROM_RE = re.compile(
    rf"(?:pick(?:\s*-\s*|(?:\s*and\s*))?roll|pick\s*and\s*pop|screen)\s*(?:from|by)\s+({_NAME_CHARS})(?=\s*(?:guarded\s*by|[,.!]|$))",
    re.IGNORECASE
)
# finds "<someone> guarded by <defender>"
_GUARDED_BY_PAIR_RE = re.compile(
    rf"({_NAME_CHARS})\s*guarded\s*by\s+({_NAME_CHARS})(?=[,.!]|$)",
    re.IGNORECASE
)
# screen assist
_SCREEN_ASSIST_RE = re.compile(
    rf"\bscreen\s+assist\s+by\s+({_NAME_CHARS})(?=[,.!]|$)",
    re.IGNORECASE
)
# handoff
_HANDOFF_RE = re.compile(
    rf"(?:dribble\s*hand\s*off|handoff|hand\s*off|dho|ho)\s*(?:from\s+({_NAME_CHARS}))?(?:\s*(?:to|with)\s+({_NAME_CHARS}))?",
    re.IGNORECASE
)

# coverage phrases -> friendly text + compact tag
_COVERAGE_PHRASES = [
    (re.compile(r"\bchases?\s+over\b", re.IGNORECASE), ("Chases Over", "ch")),
    (re.compile(r"\bgo(?:es|ing)?\s+over\b", re.IGNORECASE), ("Chases Over", "ch")),
    (re.compile(r"\bgo(?:es|ing)?\s+under\b", re.IGNORECASE), ("Under", "un")),
    (re.compile(r"\bswitch(?:ed|es|ing)?\b", re.IGNORECASE), ("Switch", "sw")),
    (re.compile(r"\bblitz(?:ed|es|ing)?\b", re.IGNORECASE), ("Blitz", "bz")),
    (re.compile(r"\bcontain(?:ed|s|ing)?\b", re.IGNORECASE), ("Contain", "ct")),
    (re.compile(r"\bice\b", re.IGNORECASE), ("Ice", "ice")),
    (re.compile(r"\bdrop(?:ped)?\b", re.IGNORECASE), ("Drop", "dp")),
    (re.compile(r"\btrap(?:ped)?\b", re.IGNORECASE), ("Trap", "tr")),
]

def _extract_results(lines: list[str]) -> list[str | None]:
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

def _align_lines_for_shots(all_lines: list[str], need_n: int) -> list[str]:
    """
    Choose the most relevant line for each shot:
    1) If there are lines with make/miss tokens, align shots to those (in order).
    2) If fewer such lines than shots, use the last make/miss line for the remaining shots.
    3) If none contain make/miss, use the last line; if not present, use empty strings.
    """
    need_n = int(need_n or 0)
    if need_n <= 0:
        return []
    all_lines = list(all_lines or [])
    mm_lines = [ln for ln in all_lines if (_MAKES_RE.search(ln) or _MISSES_RE.search(ln))]

    aligned: list[str] = []
    if mm_lines:
        for i in range(need_n):
            aligned.append(mm_lines[i] if i < len(mm_lines) else mm_lines[-1])
        return aligned

    fallback = all_lines[-1] if all_lines else ""
    return [fallback] * need_n

def _fmt_idname(pid, pname):
    return (pname or pid or "") or ""

def _resolve_name_pair(name_raw: str, roster: dict):
    pid, pname = _resolve_player_token(_clean_name_fragment(name_raw), roster)
    return {"id": pid, "name": pname}

def _friendly_coverage_from_text(full_text: str) -> tuple[str | None, str | None]:
    """
    Returns (friendly_text, short_tag) if a coverage phrase is found anywhere in the possession text.
    e.g., "Lusk chases over the screen" -> ("Chases Over", "ch")
    """
    s = full_text or ""
    for rx, (label, tag) in _COVERAGE_PHRASES:
        if rx.search(s):
            return label, tag
    return None, None

def _augment_on_ball_detail_from_context(all_lines: list[str], chosen_line: str, base: dict, roster: dict) -> dict | None:
    """
    Build a rich on-ball detail using BOTH:
      - the specific line aligned to the shot (chosen_line),
      - the entire possession text (all_lines) to pick up screener/coverage that may be on other lines.
    """
    actions = base.get("on_ball_actions") or []
    if not actions:
        return None

    # pick main on-ball type
    priority = ["pnr", "pnp", "dho", "ho", "slp", "gst", "rj", "h", "d", "p"]
    on_type = next((a for a in priority if a in actions), actions[0])

    # merge text for context
    full_text = " ".join([ln.strip() for ln in (all_lines or []) if ln and ln.strip()])
    s_line = (chosen_line or "").strip()

    detail = {"type": on_type}

    # ball handler & defender (default to shooter/primary defender from base)
    bh = {"id": base.get("shooter_id"), "name": base.get("shooter_name")}
    bhd= {"id": base.get("primary_defender_id"), "name": base.get("primary_defender_name")}
    if bh.get("id") or bh.get("name"):
        detail["ball_handler"] = bh
    if bhd.get("id") or bhd.get("name"):
        detail["ball_handler_defender"] = bhd

    # coverage from full context (friendly label + tag)
    friendly_cov, cov_tag = _friendly_coverage_from_text(full_text)
    if friendly_cov:
        detail["coverage_friendly"] = friendly_cov
        detail["coverage"] = cov_tag

    # ---- SCREEN / PNR / PNP / SLP / GST / RJ ----
    if on_type in ("pnr", "pnp", "slp", "gst", "rj"):
        # screener from full context
        m = _SCREEN_FROM_RE.search(full_text)
        if m:
            scr = _resolve_name_pair(m.group(1), roster)
            detail["screener"] = scr

            # screener defender: find "<screener> guarded by X" anywhere
            if scr.get("name"):
                mdef = re.search(
                    rf"{re.escape(_clean_name_fragment(scr['name']))}\s*guarded\s*by\s+({_NAME_CHARS})(?=[,.!]|$)",
                    full_text,
                    re.IGNORECASE,
                )
                if mdef:
                    detail["screener_defender"] = _resolve_name_pair(mdef.group(1), roster)

        # screen assist (from the result line is fine, but scan whole text)
        msa = _SCREEN_ASSIST_RE.search(full_text)
        if msa:
            detail["screen_assist"] = _resolve_name_pair(msa.group(1), roster)

    # ---- HANDOFF ----
    if on_type in ("dho", "ho"):
        m = _HANDOFF_RE.search(full_text)
        if m:
            frm_raw, to_raw = m.group(1), m.group(2)
            if frm_raw:
                detail["handoff_from"] = _resolve_name_pair(frm_raw, roster)
            if to_raw:
                detail["handoff_to"] = _resolve_name_pair(to_raw, roster)
        # defenders for those players, if the text has "<name> guarded by <defender>"
        for key in ("handoff_from", "handoff_to"):
            if key in detail and (detail[key].get("name") or detail[key].get("id")):
                nm = detail[key].get("name") or ""
                mdef = re.search(
                    rf"{re.escape(_clean_name_fragment(nm))}\s*guarded\s*by\s+({_NAME_CHARS})(?=[,.!]|$)",
                    full_text,
                    re.IGNORECASE,
                )
                if mdef:
                    detail[f"{key}_defender"] = _resolve_name_pair(mdef.group(1), roster)

    # ---- Simple on-ball: h / d / p ----
    if on_type in ("h", "d", "p"):
        if bhd.get("id") or bhd.get("name"):
            detail["on_ball_defender"] = bhd

    return detail

# -----------------------------------------------------------------------------


# Auto-retry pending writes
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Input("retry_interval", "n_intervals"),
    State("store_log", "data"),
    prevent_initial_call=True
)
def auto_retry(_n, log):
    if _PENDING_DISK_ROWS is not None and try_flush_pending():
        return load_log_from_disk()
    return no_update


# Start / End Practice
@app.callback(
    Output("store_practice","data", allow_duplicate=True),
    Output("practice_status","children", allow_duplicate=True),
    Input("btn_start_practice","n_clicks"),
    State("inp_practice_date","value"),
    prevent_initial_call=True
)
def start_practice(n_start, date_str):
    if not n_start: return no_update, no_update
    if not (date_str and str(date_str).strip()):
        return no_update, "Enter a practice date to start."
    norm = _normalize_date_input(str(date_str))
    pid = _new_practice_id(norm)
    return {"active": True, "date": norm, "practice_id": pid}, f"Practice active: {norm} (id={pid})"

@app.callback(
    Output("store_practice","data", allow_duplicate=True),
    Output("store_drill","data", allow_duplicate=True),
    Output("practice_status","children", allow_duplicate=True),
    Input("btn_end_practice","n_clicks"),
    State("store_practice","data"),
    prevent_initial_call=True
)
def end_practice(n_end, practice):
    if not n_end: return no_update, no_update, no_update
    if not (practice and practice.get("active")):
        return no_update, no_update, "No active practice to end."
    return {"active": False, "date": practice.get("date"), "practice_id": practice.get("practice_id")}, \
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
    if not n_start: return no_update, no_update
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
    if not n_end: return no_update, no_update
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
    if not clickData: return "No clicks detected"
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
    if not ctx.triggered: return no_update, no_update, no_update, no_update, no_update
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


# ---------------------------
# Debug toggle + JSON Inspector
# ---------------------------
@app.callback(
    Output("store_debug", "data"),
    Input("debug_toggle", "value"),
    prevent_initial_call=False
)
def set_debug_mode(vals):
    return bool(vals and "on" in vals)

@app.callback(
    Output("show_json_modal", "data"),
    Input("btn_open_json", "n_clicks"),
    Input("btn_close_json", "n_clicks"),
    State("show_json_modal", "data"),
    prevent_initial_call=True
)
def toggle_json_modal(n_open, n_close, cur):
    trig = ctx.triggered_id
    if trig == "btn_open_json":
        return True
    if trig == "btn_close_json":
        return False
    return cur

@app.callback(
    Output("json_modal", "style"),
    Output("json_modal_body", "children"),
    Input("show_json_modal", "data"),
    State("store_log", "data")
)
def render_json_modal(show, rows):
    style = modal_style(bool(show))
    if not show:
        return style, no_update
    rows = rows or []
    tail = rows[-20:]
    try:
        txt = json.dumps(tail, ensure_ascii=False, indent=2)
    except Exception:
        txt = "(error rendering json)"
    return style, html.Pre(txt)

# ---------------------------
# Parse + Preview (shots/no-shot) + Test Parse debug surface
# ---------------------------
@app.callback(
    [Output("output_block", "children"),
     Output("store_preview", "data")],
    [Input("btn_submit", "n_clicks"),
     Input("possession_input", "n_submit"),
     Input("btn_test_parse", "n_clicks")],
    [State("possession_input", "value"),
     State("store_last_click_xy", "data"),
     State("store_preview", "data"),
     State("store_mode", "data"),
     State("store_shots_needed", "data"),
     State("store_roster", "data"),
     State("store_debug", "data")],
    prevent_initial_call=True
)
def parse_and_preview(n_btn, n_enter, n_test, possession_text, single_xy_unused, preview_state,
                      mode, shots_needed, roster, debug_mode):
    if not ((n_btn or 0) > 0 or (n_enter or 0) > 0 or (n_test or 0) > 0):
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
            "result": result_from_shorthand(possession_text.strip())
        }

        debug_panel = None
        try_test = (ctx.triggered_id == "btn_test_parse")
        if debug_mode or try_test:
            per = []
            shot_lines = _align_lines_for_shots(annotated_lines or lines or [], len(xy_list))

            def _flat(p):
                if not isinstance(p, dict): return ""
                return p.get("name") or p.get("id") or ""

            for i, pt in enumerate(xy_list, start=1):
                line_txt = shot_lines[i-1] if i-1 < len(shot_lines) else (shot_lines[-1] if shot_lines else "")
                base = parse_shot_line_lite(line_txt, roster)
                # enrich using the full possession text to pick up screener/coverage/etc.
                detail = _augment_on_ball_detail_from_context(annotated_lines, line_txt, base, roster) or {}

                row = {
                    "#": i,
                    "shooter": base.get("shooter_name") or base.get("shooter_id"),
                    "defender": base.get("primary_defender_name") or base.get("primary_defender_id"),
                    "assist": base.get("assist_name") or base.get("assist_id"),
                    "on": ",".join(base.get("on_ball_actions") or []),
                    "off": ",".join(base.get("off_ball_actions") or []),
                    "cov_on": ",".join(base.get("coverages_on") or []),
                    "cov_off": ",".join(base.get("coverages_off") or []),
                    "ctx": ",".join(base.get("defense_context") or []),
                    "on_type": detail.get("type") or "",
                    "ball_handler": _flat(detail.get("ball_handler")),
                    "bh_defender": _flat(detail.get("ball_handler_defender")),
                }
                if detail.get("type") in ("pnr","pnp","slp","gst","rj"):
                    row.update({
                        "screener": _flat(detail.get("screener")),
                        "scr_defender": _flat(detail.get("screener_defender")),
                        "coverage": detail.get("coverage_friendly") or (detail.get("coverage") or ""),
                        "screen_assist": _flat(detail.get("screen_assist")),
                    })
                if detail.get("type") in ("dho","ho"):
                    row.update({
                        "handoff_from": _flat(detail.get("handoff_from")),
                        "handoff_from_def": _flat(detail.get("handoff_from_defender")),
                        "handoff_to": _flat(detail.get("handoff_to")),
                        "handoff_to_def": _flat(detail.get("handoff_to_defender")),
                        "coverage": detail.get("coverage_friendly") or (detail.get("coverage") or ""),
                    })
                if detail.get("type") in ("h","d","p"):
                    row.update({
                        "on_ball_defender": _flat(detail.get("on_ball_defender")),
                    })
                per.append(row)

            # dynamic columns so all detail fields show
            col_ids = []
            for r in per:
                for k in r.keys():
                    if k not in col_ids:
                        col_ids.append(k)

            debug_panel = html.Div([
                html.Div("Structured preview (per shot)", style={"fontWeight":600, "margin":"8px 0 4px"}),
                dash_table.DataTable(
                    columns=[{"name":k,"id":k} for k in col_ids],
                    data=per or [],
                    page_size=10,
                    style_cell={"fontFamily":"system-ui","fontSize":"13px","whiteSpace":"pre","padding":"6px"},
                    style_header={"fontWeight":"600"}
                )
            ])

        ui_children = [
            html.H4("Play-by-Play", style={"marginTop": 0}),
            html.Pre(pre_text, style={
                "background": "#0b1021", "color": "#e6edf3",
                "padding": "16px", "borderRadius": "10px",
                "whiteSpace": "pre-wrap", "marginBottom": "10px", "fontSize": "16px",
                "width": "100%"
            })
        ]
        if debug_panel is not None:
            ui_children.append(debug_panel)
        ui_children.append(
            html.Div([
                html.Button("Discard / Edit", id="btn_discard", n_clicks=0,
                            style={"padding": "10px 14px", "borderRadius": "8px",
                                   "border": "1px solid #aaa", "background": "white", "marginRight": "8px"}),
                html.Button("Save possession", id="btn_confirm", n_clicks=0,
                            style={"padding": "10px 14px", "borderRadius": "8px",
                                   "border": "none", "background": "#16a34a", "color": "white"})
            ], style={"display": "flex", "justifyContent": "flex-end", "gap": "8px"})
        )

        return html.Div(ui_children), preview

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
    State("store_roster","data"),
    prevent_initial_call=True
)
def save_possession(n_confirm, log_data, preview, practice, drill, roster):
    if not n_confirm or not preview:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if not (practice and practice.get("active") and drill and drill.get("active")):
        return no_update, no_update, no_update, html.Div("Start practice and drill first.", style={"color":"crimson"}), no_update, no_update, no_update, no_update, no_update

    log = list(log_data or [])
    pid, pdate = practice.get("practice_id"), practice.get("date")
    did, dname = drill.get("drill_id"), drill.get("name")
    roster = roster or {}

    def attach_context(row: dict) -> dict:
        row = dict(row)
        row["practice_id"]   = pid
        row["practice_date"] = pdate
        row["drill_id"]      = did
        row["drill_name"]    = dname
        return _json_safe_row(row)

    last_saved_card = None

    if preview.get("possession_type") == "shots" and preview.get("xy_list"):
        gid = preview.get("group_id")
        raw_lines = (preview.get("play_by_play") or "").splitlines()
        raw_lines_names = (preview.get("play_by_play_names") or "").splitlines()

        text_results = _extract_results(raw_lines_names or raw_lines)

        xy_list   = preview["xy_list"]
        dist_list = preview.get("distance_list") or [None] * len(xy_list)
        need_n = len(xy_list)

        short_results = _results_from_shorthand_tokens(preview.get("possession", ""), shots_needed=need_n)

        merged_results = []
        for i in range(need_n):
            sv = short_results[i] if i < len(short_results) else None
            tv = text_results[i] if i < len(text_results) else None
            merged_results.append(sv or tv)

        lines_src = raw_lines_names if raw_lines_names else raw_lines
        shot_lines = _align_lines_for_shots(lines_src, need_n)

        ts = preview["timestamp"]
        for idx, (pt, dft) in enumerate(zip(xy_list, dist_list), start=1):
            line_txt = shot_lines[idx-1] if idx-1 < len(shot_lines) else (shot_lines[-1] if shot_lines else "")
            base = parse_shot_line_lite(line_txt, roster)
            detail = _augment_on_ball_detail_from_context(lines_src, line_txt, base, roster) or {}

            row = {
                "id": f"{ts}-{_rand_suffix()}",
                "timestamp": ts,
                "schema_version": 2,

                "possession": preview["possession"],
                "x": float(pt["x"]), "y": float(pt["y"]),
                "distance_ft": dft,

                "play_by_play": "\n".join(raw_lines) if raw_lines else preview.get("possession", ""),
                "play_by_play_names": "\n".join(raw_lines_names) if raw_lines_names else "",

                "result": merged_results[idx-1],
                "group_id": gid,
                "group_size": len(xy_list),
                "shot_index": idx,
                "possession_type": "shots",

                "shooter_id": base.get("shooter_id"),
                "shooter_name": base.get("shooter_name"),
                "primary_defender_id": base.get("primary_defender_id"),
                "primary_defender_name": base.get("primary_defender_name"),
                "assist_id": base.get("assist_id") if merged_results[idx-1] == "Make" else None,
                "assist_name": base.get("assist_name") if merged_results[idx-1] == "Make" else None,
                "on_ball_actions": base.get("on_ball_actions") or [],
                "off_ball_actions": base.get("off_ball_actions") or [],
                "coverages_on": base.get("coverages_on") or [],
                "coverages_off": base.get("coverages_off") or [],
                "defense_context": base.get("defense_context") or [],

                # Persist richer on-ball info + screen_assist separate from assist
                "on_ball_detail": detail or None,
                "screen_assist_id": (detail.get("screen_assist") or {}).get("id"),
                "screen_assist_name": (detail.get("screen_assist") or {}).get("name"),

                "shot_type": shot_type_from_xy(pt["x"], pt["y"]),
            }
            log.append(attach_context(row))

        xs = ", ".join(f"{pt['x']:.1f}" for pt in xy_list)
        ys = ", ".join(f"{pt['y']:.1f}" for pt in xy_list)
        dists = ", ".join(f"{d:.1f}" if d is not None else "" for d in dist_list)

        first_line = shot_lines[0] if shot_lines else (raw_lines_names[0] if raw_lines_names else (raw_lines[0] if raw_lines else ""))
        p0 = parse_shot_line_lite(first_line, roster)

        res_str = ", ".join(r for r in merged_results if r)
        banner = f"Saved possession · results=[{res_str}] · shooter={_fmt_idname(p0.get('shooter_id'), p0.get('shooter_name'))} · " \
                 f"defender={_fmt_idname(p0.get('primary_defender_id'), p0.get('primary_defender_name'))} · " \
                 f"assist={_fmt_idname(p0.get('assist_id'), p0.get('assist_name'))}"

        last_saved_card = html.Div([
            html.Div("Last saved possession", style={"fontWeight":700, "marginBottom":"6px"}),
            html.Pre(
                f"{banner}\n"
                f"Shorthand: {preview['possession']}\n"
                f"X (ft): {xs}\n"
                f"Y (ft): {ys}\n"
                f"Shot Distance (ft): {dists}\n"
                f"Group Size: {len(xy_list)}\n"
                f"Drill: {dname}   Date: {pdate}",
                style={"background":"#0b1021","color":"#e6edf3","padding":"12px","borderRadius":"8px","whiteSpace":"pre-wrap"}
            )
        ])

    elif preview.get("possession_type") == "no_shot":
        row = {
            "id": preview["id"],
            "timestamp": preview["timestamp"],
            "schema_version": 2,
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
        log.append(attach_context(row))

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
        preview = dict(preview)
        preview["schema_version"] = 2
        log.append(attach_context(preview))

    ok = save_log_to_disk(log)
    notice = ("Saved to Data Log (write pending)" if not ok else "Saved to Data Log")
    cleared_output = html.Div([html.Div(notice, style={"color": "#16a34a" if ok else "#b45309", "marginBottom": "8px"})])

    return (log, "", None, cleared_output, last_saved_card, "chart",
            None, 0, [])


# Discard preview
@app.callback(
    [Output("output_block", "children", allow_duplicate=True)],
    [Input("btn_discard", "n_clicks")],
    prevent_initial_call=True
)
def discard_preview(n):
    if not n: return [no_update]
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
    groups = {}
    for r in rows or []:
        key = r.get("group_id") or r.get("timestamp") or r.get("id")
        g = groups.setdefault(key, {
            "key": key,
            "timestamp": r.get("timestamp"),
            "possession": r.get("possession"),
            "xs": [], "ys": [], "dists": [],
            "res_seq": [],
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
        g["res_seq"].append((r.get("shot_index") or 0, r.get("result") if r.get("result") else None))
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
def render_log_view(rows, level, sel_date, sel_drill):
    rows = rows or []
    level = level or "dates"

    wrap = lambda *kids: html.Div(list(kids), style={"maxWidth":"1200px","margin":"18px auto","padding":"0 12px"})

    if level == "dates":
        by_date = {}
        for r in rows:
            d = r.get("practice_date")
            if not d: continue
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
        for r in rows:
            if r.get("practice_date") != sel_date: continue
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
        raw = [r for r in rows if r.get("practice_date")==sel_date and (r.get("drill_name") or "(unnamed)") == sel_drill]
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
            editable=False,
            row_selectable="multi", selected_rows=[]
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
    if not n: return no_update, no_update
    return "drills", None

@app.callback(
    Output("log_level","data", allow_duplicate=True),
    Output("sel_date","data", allow_duplicate=True),
    Output("sel_drill","data", allow_duplicate=True),
    Input("btn_back_dates","n_clicks"),
    prevent_initial_call=True
)
def back_to_dates(n):
    if not n: return no_update, no_update, no_update
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
    if not n: return no_update
    return False

# Execute deletion after confirmation (USES LIVE TABLE DATA)
@app.callback(
    Output("store_log", "data", allow_duplicate=True),
    Output("show_confirm_modal", "data", allow_duplicate=True),
    Output("log_level", "data", allow_duplicate=True),
    Output("sel_date", "data", allow_duplicate=True),
    Output("sel_drill", "data", allow_duplicate=True),
    Input("btn_confirm_delete", "n_clicks"),
    State("confirm_delete_type", "data"),
    State("confirm_delete_target", "data"),
    State("store_log", "data"),
    State({"role":"tbl_log","scope":ALL}, "data"),
    prevent_initial_call=True
)
def execute_deletion(n, delete_type, target, all_rows, table_data_all):
    if not n or not delete_type or not target:
        return no_update, no_update, no_update, no_update, no_update

    all_rows = list(all_rows or [])

    table_data = []
    for td in (table_data_all or [])[::-1]:
        if isinstance(td, list) and len(td) > 0:
            table_data = td
            break

    if delete_type == "practice":
        practice_date = target
        new_rows = [r for r in all_rows if r.get("practice_date") != practice_date]
        save_log_to_disk(new_rows)
        return new_rows, False, "dates", None, None

    elif delete_type == "drill":
        drill_name = target["drill"]
        practice_date = target["date"]
        new_rows = [r for r in all_rows if not (r.get("practice_date") == practice_date and
                                               (r.get("drill_name") or "(unnamed)") == drill_name)]
        save_log_to_disk(new_rows)
        return new_rows, False, "drills", practice_date, None

    elif delete_type == "possession":
        if not table_data or not isinstance(target, list) or len(target) == 0:
            return no_update, False, no_update, no_update, no_update

        selected_keys = {table_data[i].get("id") for i in target if 0 <= i < len(table_data)}
        if not selected_keys:
            return no_update, False, no_update, no_update, no_update

        new_rows = [r for r in all_rows
                    if (r.get("group_id") or r.get("timestamp") or r.get("id")) not in selected_keys]
        save_log_to_disk(new_rows)
        return new_rows, False, no_update, no_update, no_update

    return no_update, no_update, no_update, no_update, no_update


# Roster editor callbacks
@app.callback(
    Output("tbl_roster", "data", allow_duplicate=True),
    Input("btn_roster_add", "n_clicks"),
    State("tbl_roster", "data"),
    prevent_initial_call=True
)
def add_roster_row(n, rows):
    if not n: return no_update
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
    if not n: return no_update, no_update
    roster = {}
    for r in rows or []:
        try:
            j = r.get("jersey")
            nm = (r.get("name") or "").strip()
            if j is None or nm == "": continue
            j = int(j)
            if 0 < j < 100:
                roster[str(j)] = nm
        except:
            continue
    ok = save_roster_to_disk(roster)
    msg = "Roster saved." if ok else "Failed to save roster."
    return roster, msg

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)
