# ------------------SECTION 1--------------------------------------------------

# vz_mk5 — adds shooting stats and stats table, adds filters

import os
import json
import math
import re
import numpy as np
import dash
from dash import html, dcc, Output, Input, State, no_update
from dash import callback_context
from dash import ALL
import plotly.graph_objects as go
import time  # NEW: for startup warm tick
from copy import deepcopy  # FIX: ensure callers never mutate cached rows

# =========================
# Config
# =========================
DATA_PATH = os.environ.get("BBALL_DATA", "data/possessions.json")

# NEW: match the entry app’s roster location scheme
BASE_DIR    = os.path.dirname(DATA_PATH) or "."
ROSTER_PATH = os.path.join(BASE_DIR, "roster.json")

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

# NEW (mk5 perf): also cache by file modified time to avoid needless re-parses
_DATA_LAST_MTIME = 0.0
_CACHED_DATA_BY_MTIME = []  # same content as CACHED_DATA but keyed by mtime

# NEW: startup warm state
_STARTUP_WARMED = False
_STARTUP_TICK = None  # monotonic tick captured at warmup; harmless global for consumers that may cache


def result_from_shorthand(s: str):
    if not s:
        return None
    # find the last +/++ or - in the string
    m = re.search(r'(?:\+\+|\+|-)(?!.*(?:\+\+|\+|-))', s)
    if not m:
        return None
    return "Make" if m.group(0) in ("+","++") else ("Miss" if m.group(0)=="-" else None)

# NEW: helper to convert full rows into plottable shot points when needed.
def rows_to_shots(rows):
    shots = []
    for row in (rows or []):
        try:
            x = float(row.get("x", 0))
            y = float(row.get("y", 0))
            res = row.get("result") or result_from_shorthand(row.get("possession", ""))
            if 0 <= x <= COURT_W and 0 <= y <= HALF_H and res in ("Make", "Miss"):
                shots.append({"x": x, "y": y, "result": res, "row_ref": row})
        except Exception:
            continue
    return shots

def _deepcopy_rows(rows):
    """Return a deep copy so downstream code cannot mutate our cache."""
    try:
        # deepcopy is fine here; rows are small dicts
        return deepcopy(rows or [])
    except Exception:
        # ultra-safe fallback
        return [dict(r) for r in (rows or [])]

def safe_load_data(force: bool = False):
    """Load and cache the *raw possession rows* with caching on failure.

    mk5 improvement:
      - Only re-parse when the DATA_PATH's mtime changes, unless force=True
        or caches are empty (first call on app start).
      - If file is missing or mid-write/corrupt, return last good cache.
      - IMPORTANT: returns the original possession ROWS (dicts), not {x,y,result} shots.
      - FIX: always return a **deep** copy so Dash callbacks see a new object and
        downstream code cannot mutate the cached snapshot.
      - NEW: if no cached data exists on first invocation, attempt an eager read
        so that initial page render has data without requiring any user interaction.
    """
    global CACHED_DATA, _DATA_LAST_MTIME, _CACHED_DATA_BY_MTIME

    # If the file doesn't exist, keep serving whatever we had last.
    if not os.path.exists(DATA_PATH):
        return _deepcopy_rows(_CACHED_DATA_BY_MTIME or CACHED_DATA)

    try:
        mtime = os.path.getmtime(DATA_PATH)
    except Exception:
        # If we can't stat it, keep the last known good.
        return _deepcopy_rows(_CACHED_DATA_BY_MTIME or CACHED_DATA)

    # Re-parse if forced, or if we have no cached data yet (fresh startup),
    # or if the file changed.
    need_parse = (
        force
        or not _CACHED_DATA_BY_MTIME  # <-- critical for first paint
        or mtime != _DATA_LAST_MTIME
    )
    if not need_parse:
        # return a deep copy to trigger downstream recompute and prevent mutation
        return _deepcopy_rows(_CACHED_DATA_BY_MTIME)

    # Try to (re)load and parse
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both {"rows":[...]} and [...] shapes
        rows = data.get("rows", data) if isinstance(data, dict) else (data or [])
        # Ensure it's a list[dict]
        rows = [r for r in (rows or []) if isinstance(r, dict)]

        # update both caches on success (store an immutable snapshot)
        CACHED_DATA = _deepcopy_rows(rows)
        _CACHED_DATA_BY_MTIME = _deepcopy_rows(rows)
        _DATA_LAST_MTIME = mtime

        # return a fresh deep copy so each caller gets a unique object graph
        return _deepcopy_rows(rows)

    except Exception:
        # If load/parse fails (e.g., mid-write), return last good cache
        return _deepcopy_rows(_CACHED_DATA_BY_MTIME or CACHED_DATA)

# ------------------------- Roster loading (same as entry app) -------------------------

def _load_roster_from_disk():
    """
    Read roster.json produced by the data-entry app.
    Returns dict like {"12": "Joleen Lusk", "21": "Adrianna Fontana", ...}
    """
    try:
        if os.path.exists(ROSTER_PATH):
            with open(ROSTER_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            out = {}
            for k, v in raw.items():
                try:
                    kk = str(int(k))  # jersey numbers as strings
                    name = (v or "").strip()
                    if name:
                        out[kk] = name
                except:
                    continue
            return out
    except Exception as e:
        print(f"[roster load] {e}")
    return {}

def _last_name(full: str) -> str:
    if not full: return ""
    parts = [p for p in re.split(r"\s+", full.strip()) if p]
    return parts[-1] if parts else full

def _first_name(full: str) -> str:
    if not full: return ""
    parts = [p for p in re.split(r"\s+", full.strip()) if p]
    return parts[0] if parts else full

# Build helper maps once per possession
def _build_name_maps(roster_dict: dict):
    """
    Returns:
      last_map:  last -> [full, full, ...]
      first_map: first -> [full, ...]
      full_set:  set of normalized full names (lower)
    """
    last_map = {}
    first_map = {}
    full_set = set()
    for full in roster_dict.values():
        ln = _last_name(full)
        fn = _first_name(full)
        last_map.setdefault(ln.lower(), []).append(full)
        first_map.setdefault(fn.lower(), []).append(full)
        full_set.add(full.strip().lower())
    return last_map, first_map, full_set

# === NEW: lightweight, global roster cache + normalization helpers ===
_ROSTER_CACHE = None
_ROSTER_LAST_MTIME = 0.0  # FIX: reload roster when roster.json changes

def _get_roster_cache():
    global _ROSTER_CACHE, _ROSTER_LAST_MTIME
    try:
        mtime = os.path.getmtime(ROSTER_PATH) if os.path.exists(ROSTER_PATH) else 0.0
    except Exception:
        mtime = _ROSTER_LAST_MTIME

    # Load if missing or file changed
    if (_ROSTER_CACHE is None) or (mtime != _ROSTER_LAST_MTIME):
        _ROSTER_CACHE = _load_roster_from_disk() or {}
        _ROSTER_LAST_MTIME = mtime
    return _ROSTER_CACHE

def _normalize_to_roster(nm: str, roster_dict: dict | None = None) -> str:
    """Map 'Fontana' or 'Lusk' or partials to a full 'First Last' when unambiguous."""
    nm = (nm or "").strip()
    if not nm:
        return nm
    roster_dict = roster_dict or _get_roster_cache()
    last_map, first_map, full_set = _build_name_maps(roster_dict)

    low = nm.lower()
    # If already a full exact match, return the canonical full
    if " " in nm and low in full_set:
        for full in roster_dict.values():
            if full.strip().lower() == low:
                return full

    # Single-token: resolve uniquely by last or first
    if " " not in nm:
        if low in last_map and len(last_map[low]) == 1:
            return last_map[low][0]
        if low in first_map and len(first_map[low]) == 1:
            return first_map[low][0]
        # substring unique match
        subs = [full for full in roster_dict.values() if low in full.lower()]
        if len(subs) == 1:
            return subs[0]
        return nm

    # Two tokens: try "First ... Last" by exact last and prefix first
    parts = nm.split()
    first_tok, last_tok = parts[0].lower(), parts[-1].lower()
    cands = []
    for full in roster_dict.values():
        fparts = full.lower().split()
        if fparts and fparts[0].startswith(first_tok) and fparts[-1] == last_tok:
            cands.append(full)
    if len(cands) == 1:
        return cands[0]
    return nm

def _normalize_list(names):
    roster = _get_roster_cache()
    out, seen = [], set()
    for n in (names or []):
        nn = _normalize_to_roster(n, roster)
        key = (nn or "").strip().lower()
        if nn and key not in seen:
            out.append(nn)
            seen.add(key)
    return out

# ------------------------- Shooter / Defender / Assister parsing -------------------------
_CAP = r"[A-Z][a-zA-Z0-9.\-']+"
_FULLNAME = rf"{_CAP}(?:\s+{_CAP})?"

_SHOOT_GUARD_RE = re.compile(rf"({_FULLNAME})\s*guarded\s*by\s*({_FULLNAME})", re.IGNORECASE)
_MAKE_RE  = re.compile(r"\bmake(?:s|d)?\b", re.IGNORECASE)
_MISS_RE  = re.compile(r"\bmiss(?:es|ed)?\b", re.IGNORECASE)

# True assist only (ignore "screen assist by ...")
_ASSIST_BY_RE   = re.compile(rf"(?<!screen )\bassist(?:ed)?\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
_ASSIST_WITH_RE = re.compile(rf"({_FULLNAME})\s+(?:with|get[s]?s?\s+the)\s+assist\b", re.IGNORECASE)

# UPDATED: support multiple names after "screen assist by" (commas / 'and') and allow closing punctuation
_SCREEN_ASSIST_LIST_RE = re.compile(
    rf"\bscreen\s+assist\s+by\s+((?:{_FULLNAME}(?:\s*(?:,|and)\s*)?)+)[\s\).,!;:]*",
    re.IGNORECASE
)

_FROM_GUARDED_RE = re.compile(rf"\bfrom\s+({_FULLNAME})\s+guarded\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
_TO_GUARDED_RE   = re.compile(rf"\bto\s+({_FULLNAME})\s+guarded\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
_FOR_GUARDED_RE  = re.compile(rf"\bfor\s+({_FULLNAME})\s+guarded\s+by\s+({_FULLNAME})\b", re.IGNORECASE)

# ---- scrub trailing verbs accidentally glued to a name
_VERB_TAILS = {
    "makes","made","misses","missed","brings","bring","passes","pass","drives","drive",
    "posts","posting","screens","screen","cuts","cut","hands","handoff","handoffs","with","gets","get","keeps","keep",
    # extras seen in off-ball text
    "comes","rolls","pops","slips","ghosts","rejects",
    # NEW: rescreen variants so "Lusk rescreens" -> "Lusk"
    "rescreen","rescreens","rescreened","rescreening","re-screen","re-screens","re-screened","re-screening"
}
def _trim_trailing_verb(name: str) -> str:
    if not name: return ""
    parts = name.split()
    while len(parts) >= 2 and parts[-1].lower() in _VERB_TAILS:
        parts = parts[:-1]
    return " ".join(parts)

# NEW: strip leading prepositions/glue before names ("and Boyd" -> "Boyd")
def _strip_leading_preps(s: str) -> str:
    return re.sub(r"^(?:from|by|to|off|of|the|and)\s+", "", (s or "").strip(), flags=re.IGNORECASE)

def _clean_frag(s: str) -> str:
    s = re.sub(r"[^\w.\-'\s,()#]", " ", s or "")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _is_shot_line(line: str) -> bool:
    return bool(_MAKE_RE.search(line) or _MISS_RE.search(line))

def _nth_shot_line(pbp_text: str, n_index: int) -> str:
    lines = [ln.strip() for ln in (pbp_text or "").splitlines() if ln.strip()]
    shot_lines = [ln for ln in lines if _is_shot_line(ln)]
    if 1 <= int(n_index or 0) <= len(shot_lines):
        return shot_lines[int(n_index)-1]
    if len(shot_lines) == 1:
        return shot_lines[0]
    return pbp_text or ""

# === NEW: normalize any name outputs against roster ===
def _norm(nm: str) -> str:
    return _normalize_to_roster(_strip_leading_preps(_trim_trailing_verb((nm or "").strip())))

def _first_guard_pair(line: str):
    m = _SHOOT_GUARD_RE.search(_clean_frag(line))
    if not m:
        return ("","")
    a = _norm(m.group(1))
    b = _norm(m.group(2))
    return a, b

def _from_pair(line: str):
    m = _FROM_GUARDED_RE.search(_clean_frag(line))
    if not m: return ("","")
    return (_norm(m.group(1)), _norm(m.group(2)))

def _to_pair(line: str):
    m = _TO_GUARDED_RE.search(_clean_frag(line))
    if not m: return ("","")
    return (_norm(m.group(1)), _norm(m.group(2)))

def _for_pair(line: str):
    m = _FOR_GUARDED_RE.search(_clean_frag(line))
    if not m: return ("","")
    return (_norm(m.group(1)), _norm(m.group(2)))

def _parse_assister(text: str, prefer_line: str = "") -> str:
    for src in (prefer_line, text):
        t = _clean_frag(src)
        if not t:
            continue
        m = _ASSIST_BY_RE.search(t)
        if m: return _norm(m.group(1))
        m = _ASSIST_WITH_RE.search(t)
        if m: return _norm(m.group(1))
    return ""

# ===================== NEW: Blocker extraction from PBP text =====================
# These helpers DO NOT change existing parsing behavior; they enable downstream
# code to render "Block: <Name>" when shorthand lacks a jersey (e.g., "blks").
_BLOCKS_SUBJ_RE = re.compile(rf"({_FULLNAME})\s+block(?:s|ed|ing)?\s+(?:the\s+)?shot\b", re.IGNORECASE)
_BLOCKS_BY_RE   = re.compile(rf"\b(?:shot\s+)?block(?:s|ed|ing)?\s+by\s+({_FULLNAME})\b", re.IGNORECASE)

def parse_blockers_from_pbp(pbp_text: str, prefer_line: str = "") -> list[str]:
    """
    Find shot blocker name(s) from play-by-play natural language, e.g.:
      - 'Lusk blocks the shot'
      - 'Shot blocked by Lusk'
    Returns a de-duped, roster-normalized list of full names.
    """
    names = []
    seen = set()
    for src in (prefer_line, pbp_text or ""):
        t = _clean_frag(src)
        if not t:
            continue
        for m in re.finditer(_BLOCKS_SUBJ_RE, t):
            nm = _norm(m.group(1))
            k = nm.lower()
            if nm and k not in seen:
                names.append(nm); seen.add(k)
        for m in re.finditer(_BLOCKS_BY_RE, t):
            nm = _norm(m.group(1))
            k = nm.lower()
            if nm and k not in seen:
                names.append(nm); seen.add(k)
    return names
# =================== END: Blocker extraction from PBP text =======================

# ---- helpers for multi-name parsing ----

def _split_fullname_list(blob: str) -> list[str]:
    """Split a blob like 'Fontana and Tillotson' or 'Adrianna Fontana, Brooke Tillotson'."""
    names = []
    # allow trailing punctuation/paren
    blob = re.sub(r"[)\].,!;:\s]+$", "", blob or "")
    for nm in re.findall(_FULLNAME, blob or "", flags=re.IGNORECASE):
        nm = _strip_leading_preps(_trim_trailing_verb(nm.strip()))
        if nm:
            names.append(nm)
    # NEW: normalize and de-dupe against roster
    return _normalize_list(names)

def _parse_screen_assister(text: str, prefer_line: str = "") -> list[str]:
    """Collect screen assists from the shot line or possession text, de-duped."""
    names = []
    seen = set()
    for src in (prefer_line, text):
        t = _clean_frag(src)
        if not t:
            continue
        for mm in re.finditer(_SCREEN_ASSIST_LIST_RE, t):
            blob = mm.group(1)
            for nm in _split_fullname_list(blob):
                low = nm.lower()
                if low and low not in seen:
                    names.append(nm)
                    seen.add(low)
    return names

# Fallback shooter: last full name before make/miss
def _guess_shooter_from_make_miss(line: str) -> str:
    if not line: return ""
    m = _MAKE_RE.search(line) or _MISS_RE.search(line)
    if not m: return ""
    prefix = line[:m.start()]
    names = re.findall(_FULLNAME, prefix)
    return _norm(names[-1]) if names else ""

# ================== NEW: multi-defender & "rotating over" parsing ==================
# Detect a guarded-by list like: "guarded by Campbell and McAnally"
# and optional per-name tag "rotating over" e.g. "Lusk rotating over"
_GUARDED_BY_BLOCK_RE = re.compile(r"\bguarded\s+by\s+(.+?)\s*(?:$|[.!)]|\bmake|\bmiss|\bpasses|\bdriv|\bpost|\bwith|\bassisted)", re.IGNORECASE)
_GUARDED_DEF_ITEM_RE = re.compile(rf"({_FULLNAME})(?:\s+rotating\s+over)?", re.IGNORECASE)
_HELP_LINE_RE = re.compile(r"\bhelp(?:s|ed|ing)?\b|\bsteps\s+in\s+to\s+help\b", re.IGNORECASE)

def _parse_defenders_with_tags_from_line(line: str):
    """
    Return list of dicts: [{"name": Full Name, "rotating_over": bool}, ...]
    from a shot line like "Fontana guarded by Lusk rotating over" or
    "Lusk guarded by Campbell and McAnally misses the shot".
    """
    t = _clean_frag(line or "")
    out = []
    seen = set()
    m = _GUARDED_BY_BLOCK_RE.search(t)
    if not m:
        # fallback to single pair, for back-compat
        a, b = _first_guard_pair(line)
        if b:
            low = b.lower()
            if low not in seen:
                out.append({"name": b, "rotating_over": False})
                seen.add(low)
        return out

    blob = m.group(1) or ""
    # Split on commas/ands but rely on name matcher to extract cleanly
    for mm in re.finditer(_GUARDED_DEF_ITEM_RE, blob):
        nm = _norm(mm.group(1))
        rot = bool(re.search(r"\s+rotating\s+over\s*$", mm.group(0), re.IGNORECASE))
        if nm:
            low = nm.lower()
            if low not in seen:
                out.append({"name": nm, "rotating_over": rot})
                seen.add(low)
    return out

def _format_defenders_for_display(def_list):
    """
    Join defenders into a single display string, appending ' rotating over'
    to any name tagged as rotating_over=True.
    """
    if not def_list:
        return ""
    parts = []
    for d in def_list:
        nm = d.get("name","")
        if not nm: 
            continue
        if d.get("rotating_over"):
            parts.append(f"{nm} rotating over")
        else:
            parts.append(nm)
    return ", ".join(parts)

def _possession_has_help_text(pbp_text: str) -> bool:
    """Lightweight flag so downstream UI can highlight 'shot out of help' if needed."""
    t = _clean_frag(pbp_text or "")
    return bool(_HELP_LINE_RE.search(t))

def extract_roles_for_shot(pbp_text: str, shot_index: int):
    """Return (shooter, onball_def, assister, screen_assists[list], candidate_action_lines[])"""
    line = _nth_shot_line(pbp_text or "", shot_index)
    shooter, onball_def = _first_guard_pair(line)
    if not shooter:
        shooter = _guess_shooter_from_make_miss(line)

    # --- NEW: upgrade on-ball defender to support multi-defenders and 'rotating over'
    defenders = _parse_defenders_with_tags_from_line(line)
    if defenders:
        onball_def = _format_defenders_for_display(defenders)

    assister = _parse_assister(pbp_text or "", prefer_line=line)
    screen_ast_list = _parse_screen_assister(pbp_text or "", prefer_line=line)

    # include lines that plausibly contain actions OR coverage cues
    candidates = []
    for ln in (pbp_text or "").splitlines():
        lcl = ln.lower()
        if any(k in lcl for k in [
            # on-ball phrases
            "pick and roll","pick and pop","dribble hand","hands off","hand off","handoff",
            "slip","ghost","reject","bring","drive",
            "post up","posting up","posts up",
            "keep the handoff","handoff keep",
            "rescreen","re-screen",
            # off-ball phrases:
            "backdoor","backdoor cut","pin down","pindown","flare screen","back screen","away screen","ucla screen",
            "cross screen","wedge screen","rip screen","stagger screen","stagger screens","iverson screen",
            "elevator screen","elevator screens",
            # coverage-only cues (include common inflections!)
            "switch", "switches", "switched", "switching",
            "chase over", "chases over",
            "cut under", "cuts under",
            "caught on screen", "top lock", "ice", "blitz",
            # help-defense cues
            "help", "steps in to help"
        ]):
            candidates.append(ln.strip())

    return shooter, onball_def, assister, screen_ast_list, candidates

# ------------------------- On-ball actions parsing + coverage additions -------------------------
_GHOST_RE  = re.compile(r"\bghost(?:s|ed|ing)?(?:\s+(?:the\s+)?)?screen\b", re.IGNORECASE)
_SLIP_RE   = re.compile(r"\bslip(?:s|ped|ping)?\b", re.IGNORECASE)
_REJECT_RE = re.compile(r"\breject(?:s|ed|ing)?\s+(?:the\s+)?(?:ball\s+)?screen\b", re.IGNORECASE)

_DHO_RE = re.compile(
    r"\bdribbl(?:e|es|ed|ing)\b.*\bhand(?:s)?[\s-]?off\s+to\b|\bdribble\s+hand[\s-]?off\b",
    re.IGNORECASE
)
_HO_RE  = re.compile(r"\bhand(?:s)?\s*off\s+to\b|\bhand-?off\s+to\b|\bhandoff\s+to\b", re.IGNORECASE)
_KP_RE  = re.compile(r"\bkeep[s]?\s+the\s+hand[\s-]?off\b|\bhandoff\s+keep\b", re.IGNORECASE)

# NEW: rescreen detector
_RESCR_RE = re.compile(r"\brescreen(?:s|ed|ing)?\b|\bre\-screen(?:s|ed|ing)?\b", re.IGNORECASE)

_COV_CH = re.compile(r"\bchas(?:e|es|ed|ing)\s+over\b", re.IGNORECASE)        # ch
_COV_CT = re.compile(r"\bcut(?:s|ting)?\s+(?:under|below|around)\b", re.IGNORECASE)  # ct
_COV_SW = re.compile(r"\bswitch(?:es|ed|ing)?\b(?:\s+(?:onto|on)\s+({_FULLNAME}))?", re.IGNORECASE)  # sw
_COV_BZ = re.compile(r"\bblitz(?:es|ed|ing)?\b", re.IGNORECASE)               # bz
_COV_CS = re.compile(r"\bcaught\s+on\s+screen\b", re.IGNORECASE)              # cs
_COV_TL = re.compile(r"\btop\s+lock(?:s|ed|ing)?\b", re.IGNORECASE)           # tl
_COV_ICE = re.compile(r"\bice(?:s|d|ing)?\b", re.IGNORECASE)

def _parse_coverages(line: str):
    t = _clean_frag(line)
    out = []
    if _COV_CH.search(t): out.append({"cov":"ch", "label":"Chase over"})
    if _COV_CT.search(t): out.append({"cov":"ct", "label":"Cut under"})
    if _COV_BZ.search(t): out.append({"cov":"bz", "label":"Blitz"})
    if _COV_CS.search(t): out.append({"cov":"cs", "label":"Caught on screen"})
    if _COV_TL.search(t): out.append({"cov":"tl", "label":"Top lock"})
    if _COV_ICE.search(t): out.append({"cov":"ice", "label":"Ice"})
    m = _COV_SW.search(t)
    if m:
        who = _trim_trailing_verb(m.group(1).strip()) if m and m.group(1) else ""
        who = _strip_leading_preps(who)
        out.append({"cov":"sw", "label":"Switch", "onto": _normalize_to_roster(who)})
    return out

# ---- helper: all guarded-by pairs in a line, in order (BH/def first, then any screeners/defs)
def _all_guard_pairs_in_line(line: str):
    out = []
    for m in re.finditer(_SHOOT_GUARD_RE, _clean_frag(line)):
        a = _norm(m.group(1))
        b = _norm(m.group(2))
        out.append((a, b))
    return out

def parse_onball_actions_from_pbp(lines, screen_ast_in_possession):
    actions = []
    last_bh = ""
    last_bh_def = ""

    def _remember(d):
        nonlocal last_bh, last_bh_def
        if d.get("bh"):
            last_bh = d["bh"]; last_bh_def = d.get("bh_def","")
        elif d.get("giver"):
            last_bh = d["giver"]; last_bh_def = d.get("giver_def","")
        elif d.get("keeper"):
            last_bh = d["keeper"]; last_bh_def = d.get("keeper_def","")

    for ln in lines:
        lc = ln.lower()
        added_action = False
        covs_line = _parse_coverages(ln)

        first_a, first_b = _first_guard_pair(ln)
        from_a, from_b = _from_pair(ln)
        to_a,   to_b   = _to_pair(ln)
        for_a,  for_b  = _for_pair(ln)

        # gather ALL guarded-by pairs on the line (BH first, then screeners)
        guard_pairs = _all_guard_pairs_in_line(ln)

        if _KP_RE.search(lc):
            d = {"type":"kp","label":"Keep",
                 "keeper": first_a, "keeper_def": first_b,
                 "intended": (for_a or to_a), "intended_def": (for_b or to_b),
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        elif _DHO_RE.search(lc):
            d = {"type":"dho","label":"Dribble handoff",
                 "giver": first_a, "giver_def": first_b,
                 "receiver": to_a, "receiver_def": to_b,
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        elif _HO_RE.search(lc):
            d = {"type":"ho","label":"Hand off",
                 "giver": first_a, "giver_def": first_b,
                 "receiver": to_a, "receiver_def": to_b,
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        elif "pick and roll" in lc:
            d = {
                "type":"pnr", "label":"Pick and roll",
                "bh": first_a, "bh_def": first_b,
                "screener": from_a, "screener_def": from_b,
                "screen_assist": (screen_ast_in_possession or ""),
                "coverages": covs_line
            }
            # NEW: support multiple screeners by using all guarded pairs after BH/def
            if guard_pairs and len(guard_pairs) > 1:
                scr_pairs = guard_pairs[1:]
                d["screeners"] = [{"name": a, "def": b} for (a,b) in scr_pairs]
                # keep single fields for back-compat (first screener)
                d["screener"], d["screener_def"] = scr_pairs[0]
            actions.append(d); _remember(d); added_action = True

        elif "pick and pop" in lc:
            d = {
                "type":"pnp", "label":"Pick and pop",
                "bh": first_a, "bh_def": first_b,
                "screener": from_a, "screener_def": from_b,
                "screen_assist": (screen_ast_in_possession or ""),
                "coverages": covs_line
            }
            if guard_pairs and len(guard_pairs) > 1:
                scr_pairs = guard_pairs[1:]
                d["screeners"] = [{"name": a, "def": b} for (a,b) in scr_pairs]
                d["screener"], d["screener_def"] = scr_pairs[0]
            actions.append(d); _remember(d); added_action = True

        elif _RESCR_RE.search(lc):
            bh, bh_def = (for_a or last_bh), (for_b or last_bh_def)
            scr, scr_def = first_a, first_b
            d = {"type":"rs","label":"Rescreen",
                 "bh": bh, "bh_def": bh_def,
                 "screener": scr, "screener_def": scr_def,
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        elif _SLIP_RE.search(lc) or _GHOST_RE.search(lc) or _REJECT_RE.search(lc):
            label = "Slip" if _SLIP_RE.search(lc) else ("Ghost" if _GHOST_RE.search(lc) else "Reject")
            bh, bh_def = "", ""
            scr, scr_def = "", ""

            if for_a:
                bh, bh_def = for_a, for_b
                if from_a:
                    scr, scr_def = from_a, from_b
                elif first_a:
                    scr, scr_def = first_a, first_b
            else:
                if from_a:
                    bh, bh_def = (first_a, first_b)
                    scr, scr_def = (from_a, from_b)
                else:
                    scr, scr_def = (first_a, first_b)
                    bh, bh_def = (last_bh, last_bh_def)

            d = {"type":("slp" if label=="Slip" else "gst" if label=="Ghost" else "rj"),
                 "label":label, "bh":bh, "bh_def":bh_def,
                 "screener":scr, "screener_def":scr_def,
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        elif re.search(r"bring[s]?\s+.*over\s+half\s*court", lc):
            d = {"type":"h","label":"Bring over halfcourt","bh":first_a,"bh_def":first_b,
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        elif re.search(r"\bdriv(?:e|es|ing)\b", lc):
            d = {"type":"d","label":"Drive","bh":first_a,"bh_def":first_b,
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        elif re.search(r"\bpost(?:s)?\s+up\b|\bposting\s+up\b", lc):
            d = {"type":"p","label":"Post up","bh":first_a,"bh_def":first_b,
                 "coverages": covs_line}
            actions.append(d); _remember(d); added_action = True

        # if this line had only coverage, attach to the most recent action
        if (not added_action) and covs_line and actions:
            existed = actions[-1].get("coverages", [])
            def _k(c): return (c.get("cov"), c.get("label"), (c.get("onto") or "").lower())
            seen = {_k(c) for c in existed}
            for c in covs_line:
                if _k(c) not in seen:
                    existed.append(c)
            actions[-1]["coverages"] = existed

    return actions


# =========================
# NEW: eager warm-up on import
# =========================
def _warm_on_import():
    """Ensure roster and possessions are ready for first render.

    This is intentionally side-effectful at import time so that components
    that compute stats on page load have data immediately (no user action needed).
    """
    global _STARTUP_WARMED, _STARTUP_TICK
    if _STARTUP_WARMED:
        return
    try:
        # Load roster once so name normalization is ready
        _ = _get_roster_cache()
        # Force-load possessions the first time regardless of mtime
        _ = safe_load_data(force=True)
        # capture a monotonic-ish tick (not used elsewhere; kept for completeness)
        _STARTUP_TICK = time.time()
        _STARTUP_WARMED = True
    except Exception as e:
        # Do not crash on warmup; downstream calls will still attempt to load.
        print(f"[startup warm] non-fatal: {e}")

# Call warmup immediately so first page paint has data
_warm_on_import()



#-------------------------------Section 2-------------------------------------------------

# ------------------------- OFF-BALL actions + coverages -------------------------
_OFFBALL_TYPES = {
    "bd":  {"label": "Backdoor cut", "keys": [r"\bbackdoor\b", r"backdoor\s+cut"]},
    "pn":  {"label": "Pin down", "keys": [r"\bpin\s*down\b", r"\bpindown\b"]},
    "fl":  {"label": "Flare screen", "keys": [r"\bflare\s+screen\b"]},
    "bk":  {"label": "Back screen", "keys": [r"\bback\s+screen\b"]},
    "awy": {"label": "Away screen", "keys": [r"\baway\s+screen\b"]},
    "ucla":{"label": "UCLA screen", "keys": [r"\bucla\s+screen\b"]},
    "crs": {"label": "Cross screen", "keys": [r"\bcross\s+screen\b"]},
    "wdg": {"label": "Wedge screen", "keys": [r"\bwedge\s+screen\b"]},
    "rip": {"label": "Rip screen", "keys": [r"\brip\s+screen\b"]},
    "stg": {"label": "Stagger screen", "keys": [r"\bstagger\s+screen[s]?\b"]},
    "ivs": {"label": "Iverson screen", "keys": [r"\biverson\s+screen[s]?\b"]},
    "elv": {"label": "Elevator screen", "keys": [r"\belevator\s+screen[s]?\b"]},
}

def _matches_any(text_lc: str, patt_list):
    for p in patt_list:
        if re.search(p, text_lc, re.IGNORECASE):
            return True
    return False

def _parse_guarded_pairs_in_order(text: str):
    out = []
    for m in re.finditer(_SHOOT_GUARD_RE, _clean_frag(text)):
        a = _norm(m.group(1))
        b = _norm(m.group(2))
        out.append((a, b))
    return out

def parse_offball_actions_from_pbp(lines):
    actions = []
    for ln in (lines or []):
        lc = ln.lower()
        matched_key = None
        for k, meta in _OFFBALL_TYPES.items():
            if _matches_any(lc, meta["keys"]):
                matched_key = k
                break
        if not matched_key:
            continue

        pairs = _parse_guarded_pairs_in_order(ln)
        coming_off, coming_off_def = ("","")
        screeners = []

        if pairs:
            coming_off, coming_off_def = pairs[0]
            for a,b in pairs[1:]:
                screeners.append({"name": a, "def": b})

        d = {
            "type": matched_key,
            "label": _OFFBALL_TYPES[matched_key]["label"],
            "coming_off": coming_off,
            "coming_off_def": coming_off_def,
            "screeners": screeners,
            "coverages": _parse_coverages(ln),
        }
        actions.append(d)

    return actions


# ========================== NEW: FILTER INFRASTRUCTURE ==========================
# These helpers fix the "conditional filters wipe out stats" bug by:
#  - NEVER mutating the base dataframe
#  - Applying subfilters ONLY when they are selected (None/''/[]/'All' => no-op)
#  - Being resilient to missing columns

def _is_falsy_filter(val):
    """Standardize placeholders (None, '', 'All', 'Man / Zone', etc.) as 'no filter'."""
    PLACEHOLDER_STRS = {
        "", "all", "any", "none",
        "man / zone", "man/zone",
        "select", "select…", "select...",
        "select on-ball actions", "select off-ball actions",
    }
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in PLACEHOLDER_STRS
    # Treat single-item lists like [''] or ['All'] as no-ops
    try:
        vals = list(val)
    except Exception:
        return False
    if len(vals) == 0:
        return True
    if len(vals) == 1:
        v = vals[0]
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in PLACEHOLDER_STRS:
            return True
    return False

def _as_list(val):
    """Normalize scalar or iterable filter to a clean list, dropping placeholders."""
    if _is_falsy_filter(val):
        return []
    if isinstance(val, str):
        return [val]
    try:
        out = list(val)
    except Exception:
        return [val]
    clean = []
    for v in out:
        if _is_falsy_filter([v]):
            continue
        clean.append(v)
    return clean

def _norm_code_set(val):
    """Normalize to a lowercased set of codes."""
    return {str(x).strip().lower() for x in _as_list(val)}

# Action → subfilters mapping (for UI to decide what to show later sections)
ONBALL_ACTION_SUBFILTERS = {
    "d":   ["ball_handler", "ball_handler_defender"],  # Drive
    "pnr": ["ball_handler", "ball_handler_defender", "screener", "screener_defender", "pnr_coverage"],
    "pnp": ["ball_handler", "ball_handler_defender", "screener", "screener_defender", "pnr_coverage"],
    "dho": ["ball_handler", "ball_handler_defender", "handoff_giver", "handoff_giver_defender", "pnr_coverage"],
    "ho":  ["ball_handler", "ball_handler_defender", "handoff_giver", "handoff_giver_defender", "pnr_coverage"],
    "kp":  ["ball_handler", "ball_handler_defender"],
    "rs":  ["ball_handler", "ball_handler_defender", "screener", "screener_defender", "pnr_coverage"],
    "rj":  ["ball_handler", "ball_handler_defender", "screener", "screener_defender", "pnr_coverage"],
    "slp": ["ball_handler", "ball_handler_defender", "screener", "screener_defender", "pnr_coverage"],
    "gst": ["ball_handler", "ball_handler_defender", "screener", "screener_defender", "pnr_coverage"],
    "p":   ["ball_handler", "ball_handler_defender"],
    "h":   ["ball_handler", "ball_handler_defender"],
}

OFFBALL_ACTION_SUBFILTERS = {
    "bd":  ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "pn":  ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "fl":  ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "bk":  ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "awy": ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "ucla":["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "crs": ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "wdg": ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "rip": ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "stg": ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "ivs": ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
    "elv": ["off_mover", "off_mover_defender", "off_screener", "off_screener_defender"],
}

def _col_ok(df, name):
    try:
        return (name in df.columns)
    except Exception:
        return False

def _isin_or_all(series, values):
    """
    If 'values' is falsy/placeholder, return all True; else series.isin(values).
    """
    import pandas as _pd
    if _is_falsy_filter(values):
        return _pd.Series(True, index=series.index)
    vals = _as_list(values)
    if len(vals) == 0:
        return _pd.Series(True, index=series.index)
    return series.isin(vals)

def apply_action_filters(df,
                         on_action=None,
                         off_action=None,
                         # on-ball subfilters
                         ball_handler=None, ball_handler_defender=None,
                         screener=None, screener_defender=None,
                         pnr_coverage=None,
                         handoff_giver=None, handoff_giver_defender=None,
                         # off-ball subfilters
                         off_mover=None, off_mover_defender=None,
                         off_screener=None, off_screener_defender=None):
    """
    Non-mutating, column-safe filtering that only applies subfilters when present.
    Returns a *view* of df (never overwrites df).
    """
    import pandas as _pd
    if df is None or getattr(df, "empty", False):
        return df

    mask = _pd.Series(True, index=df.index)

    # ---------------- On-ball ----------------
    if not _is_falsy_filter(on_action):
        if _col_ok(df, "on_ball_action"):
            mask &= (df["on_ball_action"].str.lower() == str(on_action).strip().lower())
        else:
            return df.iloc[0:0]

        if _col_ok(df, "ball_handler"):
            mask &= _isin_or_all(df["ball_handler"], ball_handler)
        if _col_ok(df, "ball_handler_defender"):
            mask &= _isin_or_all(df["ball_handler_defender"], ball_handler_defender)

        if str(on_action).strip().lower() in ("pnr", "pnp", "rs", "rj", "slp", "gst"):
            if _col_ok(df, "screener"):
                mask &= _isin_or_all(df["screener"], screener)
            if _col_ok(df, "screener_defender"):
                mask &= _isin_or_all(df["screener_defender"], screener_defender)
            if _col_ok(df, "pnr_coverage"):
                mask &= _isin_or_all(df["pnr_coverage"], pnr_coverage)

        if str(on_action).strip().lower() in ("dho", "ho", "kp"):
            if _col_ok(df, "handoff_giver"):
                mask &= _isin_or_all(df["handoff_giver"], handoff_giver)
            if _col_ok(df, "handoff_giver_defender"):
                mask &= _isin_or_all(df["handoff_giver_defender"], handoff_giver_defender)
            if _col_ok(df, "pnr_coverage"):
                mask &= _isin_or_all(df["pnr_coverage"], pnr_coverage)

    # ---------------- Off-ball ----------------
    if not _is_falsy_filter(off_action):
        if _col_ok(df, "off_ball_action"):
            mask &= (df["off_ball_action"].str.lower() == str(off_action).strip().lower())
        else:
            return df.iloc[0:0]

        if _col_ok(df, "off_mover"):
            mask &= _isin_or_all(df["off_mover"], off_mover)
        if _col_ok(df, "off_mover_defender"):
            mask &= _isin_or_all(df["off_mover_defender"], off_mover_defender)
        if _col_ok(df, "off_screener"):
            mask &= _isin_or_all(df["off_screener"], off_screener)
        if _col_ok(df, "off_screener_defender"):
            mask &= _isin_or_all(df["off_screener_defender"], off_screener_defender)

    return df.loc[mask]

def safe_rate(numer, denom):
    try:
        d = float(denom)
        if d == 0.0:
            return 0.0
        return 100.0 * float(numer) / d
    except Exception:
        return 0.0

def df_from_store(store_data, df_base):
    """
    If no filtered data has been computed yet (None/[]/''), return the base df.
    Downstream use keeps stats/legends present at app start.
    """
    import pandas as _pd
    if store_data is None:
        return df_base
    if isinstance(store_data, (list, tuple)) and len(store_data) == 0:
        return df_base
    if store_data == "":
        return df_base
    return _pd.DataFrame(store_data)
# ======================== END FILTER INFRASTRUCTURE ============================


# =======================  mk5 ADDITIONS START  =========================

# ---- Filter option sets (used by UI + predicates)
ONBALL_ACTION_CODES = {"h","d","p","pnr","pnp","slp","rj","gst","dho","ho","kp","rs"}
OFFBALL_ACTION_CODES = {"bd","pn","fl","bk","awy","crs","wdg","rip","ucla","stg","ivs","elv"}
DEFENSE_FILTERS = {"Man", "Zone"}  # 'Man' vs 'Zone'

# ---- Robust, flexible getters from possession rows ---------------------------------

_DATE_TOKEN_RE = re.compile(r"(?P<m>\d{1,2})[/-](?P<d>\d{1,2})(?:[/-](?P<y>\d{2,4}))?", re.IGNORECASE)

def _coerce_int(x, default=1):
    try:
        return int(x)
    except:
        return default

def _normalize_date_str(s: str) -> str:
    """
    Normalize few common shapes to 'YYYY-MM-DD' if possible; else return original-ish.
    """
    if not s:
        return ""
    s = str(s)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = _DATE_TOKEN_RE.search(s)
    if not m:
        return s.strip()
    mm = int(m.group("m") or 0)
    dd = int(m.group("d") or 0)
    yy = m.group("y")
    if yy and len(yy) == 2:
        yy = ("20" + yy)
    if not yy:
        return f"{mm:02d}-{dd:02d}"
    return f"{yy}-{mm:02d}-{dd:02d}"

def _row_practice_date_key(row: dict) -> str:
    for k in ("practice_date","date","group_label","group_id","session","label","title"):
        v = row.get(k)
        if v:
            return _normalize_date_str(v)
    return ""

def _row_drill_label(row: dict) -> str:
    for k in ("drill","drill_label","group_label","label","title"):
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""

# IMPORTANT: renamed to avoid shadowing by later sections
_DRILL_SIZE_PAIR_RE = re.compile(r"\b(\d+)\s*[vV]\s*(\d+)\b")

def _row_drill_size(row: dict) -> str:
    """
    Parse and normalize 'NvM' from the drill label.
    Returns '5v5', '3v3', or '' if not found. Defensive against malformed tokens.
    """
    lbl = _row_drill_label(row) or ""
    m = _DRILL_SIZE_PAIR_RE.search(lbl)
    if not m:
        return ""
    # m.groups() are digits by regex; keep as strings to avoid int() crashes
    return f"{m.group(1)}v{m.group(2)}"

def _row_possession_text(row: dict) -> str:
    return (row.get("possession") or row.get("pbp") or row.get("text") or "").strip()

def _row_shot_index(row: dict) -> int:
    """Provide the nth shot in the possession; defaults to 1."""
    return _coerce_int(row.get("shot_index"), 1)

def _row_xy(row: dict):
    """Return (x,y) if present (floatable), else (None,None)."""
    try:
        x = float(row.get("x", None)); y = float(row.get("y", None))
        if 0 <= x <= COURT_W and 0 <= y <= HALF_H:
            return x, y
    except:
        pass
    return None, None

def _row_result(row: dict) -> str:
    res = row.get("result")
    if res in ("Make","Miss"):
        return res
    rs = result_from_shorthand(_row_possession_text(row))
    return rs or ""

def _row_shot_value(x: float, y: float) -> int:
    """2 or 3 based on distance/line."""
    if x is None or y is None:
        return 0
    dist = math.hypot(x - RIM_X, y - RIM_Y)
    if dist > THREE_R:
        left_t = math.asin(max(-1,min(1,(LEFT_POST_X - RIM_X)/THREE_R)))
        right_t = math.asin(max(-1,min(1,(RIGHT_POST_X - RIM_X)/THREE_R)))
        y_left  = RIM_Y + THREE_R*math.cos(left_t)
        y_right = RIM_Y + THREE_R*math.cos(right_t)
        if (x <= LEFT_POST_X and y <= y_left) or (x >= RIGHT_POST_X and y <= y_right):
            return 2
        return 3
    return 2

# ---- Role & action extraction wrappers --------------------------------------------

def _row_roles_and_lines(row: dict):
    """
    Uses helpers to derive roles for the nth shot:
      shooter, defenders_display, assister, screen_assists[list], candidate_lines[list]
    """
    text = _row_possession_text(row)
    idx  = _row_shot_index(row)
    return extract_roles_for_shot(text, idx)

def _row_onball_offball_actions(row: dict):
    shooter, ondef, ast, screen_list, cand_lines = _row_roles_and_lines(row)
    onball = parse_onball_actions_from_pbp(cand_lines, (screen_list[0] if screen_list else ""))
    offball = parse_offball_actions_from_pbp(cand_lines)
    return onball, offball

def _row_defense_label(row: dict) -> str:
    """
    Return 'Man' or 'Zone' depending on zone_for_shot() classifier.
    """
    try:
        zlbl = zone_for_shot(_row_possession_text(row), _row_shot_index(row))
    except Exception:
        zlbl = "Man to Man"
    return "Man" if (str(zlbl).strip().lower().startswith("man")) else "Zone"

# ---- Filter predicate --------------------------------------------------------------

def _str_in_ci(needle: str, hay: str) -> bool:
    return (needle or "").strip().lower() in (hay or "").strip().lower()

def _any_match_ci(options: set[str] | list[str], value: str) -> bool:
    v = (value or "").lower()
    for opt in (options or []):
        if (opt or "").lower() == v:
            return True
    return False

def _collect_action_codes(actions: list[dict]) -> set[str]:
    return { (a.get("type") or "").lower() for a in (actions or []) if a.get("type") }

def _passes_date_range(row: dict, start_key: str, end_key: str) -> bool:
    """
    start_key/end_key are normalized strings produced externally (e.g., '2025-09-12').
    We compare on best-effort basis: if row key lacks year ('MM-DD'), we match month/day.
    """
    k = _row_practice_date_key(row)
    if _is_falsy_filter(start_key) and _is_falsy_filter(end_key):
        return True
    if not k:
        return False

    def _split_isoish(s):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            y,m,d = s.split("-"); return (y,int(m),int(d))
        if re.match(r"^\d{2}-\d{2}$", s):
            m,d = s.split("-"); return ("",int(m),int(d))
        s2 = _normalize_date_str(s)
        return _split_isoish(s2) if s2 != s else ("",0,0)

    ry, rm, rd = _split_isoish(k)

    def _cmp(a, b):
        ay,am,ad = a; by,bm,bd = b
        if not ay or not by:
            return (am,ad) < (bm,bd)
        return (ay,am,ad) < (by, bm, bd)

    rk = (ry, rm, rd)

    if not _is_falsy_filter(start_key):
        sk = _split_isoish(start_key)
        if _cmp(rk, sk):
            return False
    if not _is_falsy_filter(end_key):
        ek = _split_isoish(end_key)
        if _cmp(ek, rk):
            return False
    return True

# ===== NEW: helpers for subfilter matching on parsed actions =====

def _names_match_any(filter_set, candidates):
    """Case-insensitive substring match: True if any filter value appears in any candidate."""
    if _is_falsy_filter(filter_set):
        return True
    cand_list = [c for c in (candidates or []) if c]
    if not cand_list:
        return False
    for f in _as_list(filter_set):
        if _is_falsy_filter(f):
            continue
        for c in cand_list:
            if _str_in_ci(str(f), c):
                return True
    return False

def _coverage_match_any(filter_set, coverage_items):
    """
    Match on coverage code or label. Accepts things like 'sw' or 'switch' or 'Ice'.
    """
    if _is_falsy_filter(filter_set):
        return True
    cov_labels = []
    for c in (coverage_items or []):
        lbl = (c.get("label") or "").strip()
        code = (c.get("cov") or "").strip()
        if lbl: cov_labels.append(lbl)
        if code: cov_labels.append(code)
    if not cov_labels:
        return False
    for f in _as_list(filter_set):
        if _is_falsy_filter(f):
            continue
        for lbl in cov_labels:
            if _str_in_ci(str(f), lbl):
                return True
    return False

def _onball_action_satisfies_subfilters(action: dict, subf: dict) -> bool:
    """
    Given a single on-ball action dict and a subfilter dict with keys
      on_bh, on_bh_def, on_screener, on_screener_def, on_cov
    return True iff the action matches all provided subfilters.
    """
    if not action:
        return False

    bh_names = [action.get("bh") or action.get("giver") or action.get("keeper")]
    bh_def_names = [action.get("bh_def") or action.get("giver_def") or action.get("keeper_def")]

    if not _names_match_any(subf.get("on_bh"), bh_names):
        return False
    if not _names_match_any(subf.get("on_bh_def"), bh_def_names):
        return False

    scr_names = []
    scr_def_names = []
    if action.get("screener"):
        scr_names.append(action.get("screener"))
    if action.get("screener_def"):
        scr_def_names.append(action.get("screener_def"))
    for pair in (action.get("screeners") or []):
        nm = pair.get("name"); df = pair.get("def")
        if nm: scr_names.append(nm)
        if df: scr_def_names.append(df)

    if not _names_match_any(subf.get("on_screener"), scr_names):
        return False
    if not _names_match_any(subf.get("on_screener_def"), scr_def_names):
        return False

    if not _coverage_match_any(subf.get("on_cov"), action.get("coverages")):
        return False

    return True

def _offball_action_satisfies_subfilters(action: dict, subf: dict) -> bool:
    """
    For off-ball actions with keys coming_off, coming_off_def, screeners[{name,def}]
      subfilters: off_mover, off_mover_def, off_screener, off_screener_def
    """
    if not action:
        return False
    mover_names = [action.get("coming_off")]
    mover_def_names = [action.get("coming_off_def")]
    scr_names = [p.get("name") for p in (action.get("screeners") or []) if p.get("name")]
    scr_def_names = [p.get("def") for p in (action.get("screeners") or []) if p.get("def")]

    if not _names_match_any(subf.get("off_mover"), mover_names):
        return False
    if not _names_match_any(subf.get("off_mover_def"), mover_def_names):
        return False
    if not _names_match_any(subf.get("off_screener"), scr_names):
        return False
    if not _names_match_any(subf.get("off_screener_def"), scr_def_names):
        return False

    return True

def row_passes_shooting_filters(row: dict, filters: dict) -> bool:
    """
    Shooting-tab filters (dates, drill size/label, roles, on/off-ball actions+subfilters, defense).
    See docstring in original for full schema of 'filters'.
    """
    filters = filters or {}

    if not _passes_date_range(row, filters.get("date_start",""), filters.get("date_end","")):
        return False

    ds = _row_drill_size(row)
    ds_allowed = set(_as_list(filters.get("drill_size")))
    if len(ds_allowed) and ds not in ds_allowed:
        return False

    full_allowed = set(_as_list(filters.get("drill_full")))
    if len(full_allowed) and _row_drill_label(row) not in full_allowed:
        return False

    shooter, defenders_disp, assister, screen_list, cand_lines = _row_roles_and_lines(row)

    if len(_as_list(filters.get("shooter"))):
        if not any(_str_in_ci(s, shooter) for s in _as_list(filters.get("shooter"))):
            return False

    if len(_as_list(filters.get("defenders"))):
        if not any(_str_in_ci(d, defenders_disp) for d in _as_list(filters.get("defenders"))):
            return False

    if len(_as_list(filters.get("assister"))):
        if not any(_str_in_ci(a, assister) for a in _as_list(filters.get("assister"))):
            return False

    if len(_as_list(filters.get("screen_ast"))):
        joined = ", ".join(screen_list or [])
        if not any(_str_in_ci(s, joined) for s in _as_list(filters.get("screen_ast"))):
            return False

    onball, offball = _row_onball_offball_actions(row)

    on_sel = _norm_code_set(filters.get("onball"))
    if len(on_sel):
        candidates = [a for a in (onball or []) if (a.get("type","").lower() in on_sel)]
        if not candidates:
            return False
        subf = {
            "on_bh": filters.get("on_bh") or [],
            "on_bh_def": filters.get("on_bh_def") or [],
            "on_screener": filters.get("on_screener") or [],
            "on_screener_def": filters.get("on_screener_def") or [],
            "on_cov": filters.get("on_cov") or [],
        }
        if any(len(_as_list(v)) for v in subf.values()):
            ok = any(_onball_action_satisfies_subfilters(a, subf) for a in candidates)
            if not ok:
                return False

    off_sel = _norm_code_set(filters.get("offball"))
    if len(off_sel):
        candidates = [a for a in (offball or []) if (a.get("type","").lower() in off_sel)]
        if not candidates:
            return False
        subf = {
            "off_mover": filters.get("off_mover") or [],
            "off_mover_def": filters.get("off_mover_def") or [],
            "off_screener": filters.get("off_screener") or [],
            "off_screener_def": filters.get("off_screener_def") or [],
        }
        if any(len(_as_list(v)) for v in subf.values()):
            ok = any(_offball_action_satisfies_subfilters(a, subf) for a in candidates)
            if not ok:
                return False

    def_sel = {str(x).strip().title() for x in _as_list(filters.get("defense"))}
    if len(def_sel):
        if _row_defense_label(row) not in def_sel:
            return False

    return True





#------------------------Section 3------------------------------------------------


# ---- Shot collection & Shooting Stats ---------------------------------------------

def collect_shots_for_filters(possession_rows: list[dict], filters: dict):
    """
    Returns list of dicts for plotting + statting:
      { x, y, result('Make'/'Miss'), value(2/3), shooter, defenders, assister, screen_ast[list], row_ref }
    Only includes rows that pass row_passes_shooting_filters.
    """
    # NEW: normalize filters so initial render (None / blanks / placeholders) behaves as "no filters"
    f_in = filters or {}
    try:
        # If every provided value is effectively "no filter", use the canonical empty set
        all_blank = True
        for v in f_in.values():
            # _is_falsy_filter is defined in Section 2; tolerate absence gracefully
            if "_is_falsy_filter" in globals():
                if not _is_falsy_filter(v):
                    all_blank = False
                    break
            else:
                # fallback: consider None/''/[] as blank
                if v not in (None, "", [], ()):
                    all_blank = False
                    break
        filters = empty_shooting_filters() if all_blank else f_in
    except Exception:
        filters = f_in or empty_shooting_filters()

    out = []
    for row in (possession_rows or []):
        if not row_passes_shooting_filters(row, filters or {}):
            continue
        x, y = _row_xy(row)
        res = _row_result(row)
        if x is None or y is None or res not in ("Make","Miss"):
            # This shot cannot contribute to chart/percentages
            continue
        shooter, defenders_disp, assister, screen_list, _cand = _row_roles_and_lines(row)
        val = _row_shot_value(x,y)
        out.append({
            "x": x, "y": y, "result": res, "value": val,
            "shooter": shooter, "defenders": defenders_disp,
            "assister": assister, "screen_assists": (screen_list or []),
            "row_ref": row
        })
    return out

def compute_shooting_totals(filtered_shots: list[dict]) -> dict:
    """
    Compute the left-panel shooting metrics:
      FGM, FGA, FG%, 2PM, 2PA, 2P%, 3PM, 3PA, 3P%
    """
    shots = filtered_shots or []
    FGA = len(shots)
    FGM = sum(1 for s in shots if s.get("result") == "Make")
    twos = [s for s in shots if s.get("value") == 2]
    thrs = [s for s in shots if s.get("value") == 3]
    twoA = len(twos); twoM = sum(1 for s in twos if s.get("result") == "Make")
    thrA = len(thrs); thrM = sum(1 for s in thrs if s.get("result") == "Make")

    def pct(m,a):
        return round((100.0*m/a), 1) if a else 0.0

    return {
        "FGM": FGM,
        "FGA": FGA,
        "FGP": pct(FGM, FGA),
        "2PM": twoM,
        "2PA": twoA,
        "2PP": pct(twoM, twoA),
        "3PM": thrM,
        "3PA": thrA,
        "3PP": pct(thrM, thrA),
    }

# ---- Convenience builder for default (no filters) ---------------------------------

def empty_shooting_filters() -> dict:
    """Blank filter set for the Shooting tab. (Lists, not sets—JSON serializable for dcc.Store.)"""
    return {
        "date_start": "", "date_end": "",
        "drill_size": [], "drill_full": [],
        "shooter": [], "defenders": [], "assister": [], "screen_ast": [],
        "onball": [], "offball": [],
        # NEW optional subfilters (leave empty by default)
        "on_bh": [], "on_bh_def": [], "on_screener": [], "on_screener_def": [], "on_cov": [],
        "off_mover": [], "off_mover_def": [], "off_screener": [], "off_screener_def": [],
        "defense": [],  # subset of {"Man","Zone"}
    }

# ========================= NEW: Defensive / Rebound / Special tokens (from shorthand) =========================
# Supports tokens:
#   r (defensive rebound), or (offensive rebound),
#   f (defensive foul), of (offensive foul),
#   lbto (live-ball TO), dbto (dead-ball TO),
#   stlN (steal by N), blkN (block by N), defN (deflection by N)
# Numbers may be attached before the code (e.g., "10r", "10/11or", "5f"), and
# for TO/foul codes the number may be omitted; we then use current possession context.

_NUM_ONLY_RE   = re.compile(r"^\d+$")
# minimal on-ball shorthand recognizer to advance "current ballhandler/defender"
_ONBALL_SH_RE  = re.compile(r"(?<!\S)(?P<bh>\d+)(?:/(?P<def>\d+))?(?P<act>h|d|p|pnr|pnp|dho|ho|kp|rj|slp|gst)\b", re.IGNORECASE)
# rebound / specials
_DEF_REB_RE    = re.compile(r"(?<!\S)(?P<p>\d+)(?:/\d+)?r(?!\w)", re.IGNORECASE)
_OFF_REB_RE    = re.compile(r"(?<!\S)(?P<p>\d+)(?:/\d+)?or(?!\w)", re.IGNORECASE)
_STEAL_RE      = re.compile(r"(?<!\S)(?P<p>\d+)stl(?!\w)", re.IGNORECASE)
_BLOCK_RE      = re.compile(r"(?<!\S)(?P<p>\d+)blk(?!\w)", re.IGNORECASE)
_DEFLECT_RE    = re.compile(r"(?<!\S)(?P<p>\d+)def(?!\w)", re.IGNORECASE)
_LBTO_RE       = re.compile(r"(?<!\S)(?:(?P<p>\d+)?)lbto(?!\w)", re.IGNORECASE)
_DBTO_RE       = re.compile(r"(?<!\S)(?:(?P<p>\d+)?)dbto(?!\w)", re.IGNORECASE)
_DFOUL_RE      = re.compile(r"(?<!\S)(?:(?P<p>\d+)?)f(?!\w)", re.IGNORECASE)
_OFOUL_RE      = re.compile(r"(?<!\S)(?:(?P<p>\d+)?)of(?!\w)", re.IGNORECASE)

def _name_from_num(num: str) -> str:
    """Return roster display name for a jersey number string, else the number."""
    roster = _get_roster_cache()
    s = (num or "").strip()
    if not s:
        return ""
    try:
        key = str(int(s))
    except:
        key = s
    return roster.get(key, key)

def parse_special_stats_from_shorthand(short_text: str):
    """
    Parse shorthand string (e.g., '10r 1/3h 4/5d- 10/11or def5 lbto stl11') and return a dict:
      {
        "def_rebounds": [names...],
        "off_rebounds": [names...],
        "deflections":  [names...],
        "steals":       [names...],
        "blocks":       [names...],
        "live_ball_to": [names...],  # turnover by ...
        "dead_ball_to": [names...],
        "def_fouls":    [names...],
        "off_fouls":    [names...],
        "events":       [ {"code": "or", "player": "Name"}, ... ]  # in-order, if useful
      }

    Resolution rules for unnumbered tokens:
      - f (defensive foul): use current on-ball defender if available; else None.
      - of (offensive foul): use current ballhandler/offensive possessor if available.
      - lbto/dbto: use current offensive possessor (last on-ball BH or last OR rebounder).
    """
    s = (short_text or "").strip()
    if not s:
        return {
            "def_rebounds": [], "off_rebounds": [], "deflections": [],
            "steals": [], "blocks": [], "live_ball_to": [], "dead_ball_to": [],
            "def_fouls": [], "off_fouls": [], "events": []
        }

    # tracking context as we scan left->right
    cur_bh_num = None           # last explicit ballhandler number (from on-ball action)
    cur_bh_def_num = None       # last explicit on-ball defender
    cur_off_possessor = None    # best guess at current offensive possessor (BH or OR rebounder)

    out = {
        "def_rebounds": [], "off_rebounds": [], "deflections": [],
        "steals": [], "blocks": [], "live_ball_to": [], "dead_ball_to": [],
        "def_fouls": [], "off_fouls": [], "events": []
    }

    tokens = re.split(r"\s+", s)
    for tok in tokens:
        if not tok:
            continue

        # 1) update BH/DEF from on-ball micro-actions
        m_bh = _ONBALL_SH_RE.search(tok)
        if m_bh:
            cur_bh_num = m_bh.group("bh")
            cur_bh_def_num = m_bh.group("def") or cur_bh_def_num
            cur_off_possessor = cur_bh_num or cur_off_possessor

        # 2) rebounds
        m = _DEF_REB_RE.match(tok)
        if m:
            p = m.group("p")
            name = _name_from_num(p)
            out["def_rebounds"].append(name)
            out["events"].append({"code":"r", "player": name})
            # possession flips; offensive possessor becomes None
            cur_off_possessor = None
            continue

        m = _OFF_REB_RE.match(tok)
        if m:
            p = m.group("p")
            name = _name_from_num(p)
            out["off_rebounds"].append(name)
            out["events"].append({"code":"or", "player": name})
            # offense keeps; new possessor is rebounder
            cur_off_possessor = p
            continue

        # 3) steals, blocks, deflections
        m = _STEAL_RE.match(tok)
        if m:
            p = m.group("p"); name = _name_from_num(p)
            out["steals"].append(name)
            out["events"].append({"code":"stl", "player": name})
            continue

        m = _BLOCK_RE.match(tok)
        if m:
            p = m.group("p"); name = _name_from_num(p)
            out["blocks"].append(name)
            out["events"].append({"code":"blk", "player": name})
            continue

        m = _DEFLECT_RE.match(tok)
        if m:
            p = m.group("p"); name = _name_from_num(p)
            out["deflections"].append(name)
            out["events"].append({"code":"def", "player": name})
            continue

        # 4) turnovers (may be unnumbered)
        m = _LBTO_RE.match(tok)
        if m:
            p = m.group("p") or cur_off_possessor or cur_bh_num
            name = _name_from_num(p or "")
            if name:
                out["live_ball_to"].append(name)
                out["events"].append({"code":"lbto", "player": name})
            continue

        m = _DBTO_RE.match(tok)
        if m:
            p = m.group("p") or cur_off_possessor or cur_bh_num
            name = _name_from_num(p or "")
            if name:
                out["dead_ball_to"].append(name)
                out["events"].append({"code":"dbto", "player": name})
            continue

        # 5) fouls (may be unnumbered)
        m = _DFOUL_RE.match(tok)
        if m:
            p = m.group("p") or cur_bh_def_num
            name = _name_from_num(p or "")
            if name:
                out["def_fouls"].append(name)
                out["events"].append({"code":"f", "player": name})
            continue

        m = _OFOUL_RE.match(tok)
        if m:
            p = m.group("p") or cur_off_possessor or cur_bh_num
            name = _name_from_num(p or "")
            if name:
                out["off_fouls"].append(name)
                out["events"].append({"code":"of", "player": name})
            continue

        # (ignore other tokens; this function is focused on specials)

    # de-dup lists while preserving order (optional but tidy)
    def _dedupe(seq):
        seen = set(); out_list = []
        for x in seq:
            k = (x or "").lower()
            if k and k not in seen:
                seen.add(k); out_list.append(x)
        return out_list

    for k in ("def_rebounds","off_rebounds","deflections","steals","blocks",
              "live_ball_to","dead_ball_to","def_fouls","off_fouls"):
        out[k] = _dedupe(out[k])

    return out

# ===================== NEW: Zone-defense & participant harvesting =====================

# Recognize zone tags like "2-3[...]", "3-2[...]", "1-3-1[...]", or "matchup[...]"
_ZONE_TAG_RE = re.compile(r"(?i)\b(?P<label>(?:\d(?:-\d){1,3}|matchup))\s*\[(?P<body>.*?)\]", re.DOTALL)

def _normspace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _normalize_zone_label(lbl: str) -> str:
    if not lbl:
        return ""
    lbl = lbl.strip().lower()
    if lbl == "matchup":
        return "Matchup Zone"
    # keep numeric shapes as-is (e.g., 2-3, 1-3-1)
    return f"{lbl.upper()} Zone" if re.match(r"^\d(?:-\d){1,3}$", lbl) else f"{lbl.title()} Zone"

def zone_for_shot(pbp_text: str, shot_index: int) -> str:
    """
    Determine defense for the nth shot in a possession:
      - If the shot line is contained within any <ZONE>[ ... ] chunk, return "<ZONE> Zone"
      - Otherwise default to "Man to Man"
    """
    shot_line = _nth_shot_line(pbp_text or "", shot_index)
    if not shot_line:
        return "Man to Man"
    shot_norm = _normspace(shot_line)

    txt = pbp_text or ""
    for m in re.finditer(_ZONE_TAG_RE, txt):
        lbl = _normalize_zone_label(m.group("label") or "")
        body = _normspace(m.group("body") or "")
        # direct containment check on normalized whitespace
        if shot_norm and shot_norm in body:
            return lbl or "Man to Man"
    return "Man to Man"

def collect_players_from_possession(pbp_text: str, shorthand_text: str):
    """
    Harvest lists of offensive and defensive players the parser can see
    from a possession via:
      - Guarded-by pairs across all lines (on-ball + off-ball)
      - Assister + screen-assist names (offense)
      - Shot-blockers in PBP text (defense)
      - Shorthand pairs like '10/11h' (left=offense, right=defense)
    Returns (offense_list, defense_list) de-duped, order preserved.
    """
    off_list, def_list = [], []
    seen_off, seen_def = set(), set()

    def _add_off(nm: str):
        nm = _normalize_to_roster(nm)
        k = (nm or "").lower()
        if nm and k not in seen_off:
            off_list.append(nm); seen_off.add(k)

    def _add_def(nm: str):
        nm = _normalize_to_roster(nm)
        k = (nm or "").lower()
        if nm and k not in seen_def:
            def_list.append(nm); seen_def.add(k)

    # 1) From all 'guarded by' pairs in the full text (covers on-ball & off-ball lines)
    for ln in (pbp_text or "").splitlines():
        for a, b in _all_guard_pairs_in_line(ln):
            if a: _add_off(a)
            if b: _add_def(b)

    # 2) From assist + screen assist cues (offense)
    #    We scan entire possession text to avoid missing pre/post shot mentions
    assister = _parse_assister(pbp_text or "", prefer_line="")
    if assister: _add_off(assister)
    for nm in _parse_screen_assister(pbp_text or "", prefer_line=""):
        _add_off(nm)

    # 3) From natural-language blockers (defense)
    for nm in parse_blockers_from_pbp(pbp_text or "", prefer_line=""):
        _add_def(nm)

    # 4) From shorthand micro-actions like '10/11h' (left=offense, right=defense)
    for tok in re.split(r"\s+", shorthand_text or ""):
        m = _ONBALL_SH_RE.search(tok or "")
        if not m:
            continue
        bh = m.group("bh")
        df = m.group("def")
        if bh:
            _add_off(_name_from_num(bh))
        if df:
            _add_def(_name_from_num(df))

    return off_list, def_list

# ------------------------- Geometry precompute -------------------------
mini_radius = FT_CY - RIM_Y
max_angle = math.pi / 2.2
LEFT_DIAG_START = (RIM_X + mini_radius * math.sin(-max_angle), RIM_Y + mini_radius * math.cos(-max_angle))
RIGHT_DIAG_START = (RIM_X + mini_radius * math.sin(+max_angle), RIM_Y + mini_radius * math.cos(+max_angle))
LEFT_DIAG_SLOPE = -0.8
RIGHT_DIAG_SLOPE = +0.8

def _three_point_intersection(start_x, start_y, slope, go_right: bool):
    step = 0.25 if go_right else -0.25
    x, y = start_x, start_y
    for _ in range(600):
        dist = math.hypot(x - RIM_X, y - RIM_Y)
        if abs(dist - THREE_R) < 0.05:
            return x, y
        x += step
        y = start_y + slope * (x - start_x)
    x = start_x + (8 if go_right else -8)
    y = start_y + slope * (x - start_x)
    return x, y

LEFT_DIAG_IX, LEFT_DIAG_IY = _three_point_intersection(*LEFT_DIAG_START, LEFT_DIAG_SLOPE, go_right=False)
RIGHT_DIAG_IX, RIGHT_DIAG_IY = _three_point_intersection(*RIGHT_DIAG_START, RIGHT_DIAG_SLOPE, go_right=True)


# ------------------------- Zone classifier -------------------------
def point_in_zone(x, y, zone_id):
    dist_from_rim = math.hypot(x - RIM_X, y - RIM_Y)

    mini_R = FT_CY - RIM_Y

    x0L, y0L = (RIM_X - RESTRICTED_R), RIM_Y
    x1L, y1L = LANE_X0, FT_CY
    slope_L = (y1L - y0L) / (x1L - x0L)
    def y_elbow_L(xq): return y0L + slope_L * (xq - x0L)

    x0R, y0R = (RIM_X + RESTRICTED_R), RIM_Y
    x1R, y1R = LANE_X1, FT_CY
    slope_R = (y1R - y0R) / (x1R - x0R)
    def y_elbow_R(xq): return y0R + slope_R * (xq - x0R)

    if zone_id == 1:
        in_circle = dist_from_rim <= RESTRICTED_R
        in_posts  = (RIM_X - RESTRICTED_R <= x <= RIM_X + RESTRICTED_R) and (0.0 <= y <= RIM_Y)
        return in_circle or in_posts
    elif zone_id == 2:
        mini_left_x = RIM_X + mini_R * math.sin(-math.pi/2.2)
        if not ((x < RIM_X) and (x >= mini_left_x) and (x <= RIM_X - RESTRICTED_R) and (y >= 0.0) and (y <= y_elbow_L(x))):
            return False
        if dist_from_rim <= RESTRICTED_R: return False
        if y < 2.0: return True
        return dist_from_rim <= mini_R
    elif zone_id == 3:
        in_circle = dist_from_rim <= RESTRICTED_R
        in_posts = (RIM_X - RESTRICTED_R <= x <= RIM_X + RESTRICTED_R) and (0.0 <= y <= RIM_Y)
        if in_circle or in_posts: return False
        return (
            (dist_from_rim > RESTRICTED_R) and
            (dist_from_rim <= mini_R) and
            (y >= y_elbow_L(x)) and
            (y >= y_elbow_R(x))
        )
    elif zone_id == 4:
        mini_right_x = RIM_X + mini_R * math.sin(math.pi/2.2)
        if not ((x > RIM_X) and (x <= mini_right_x) and (x >= RIM_X + RESTRICTED_R) and (y >= 0.0) and (y <= y_elbow_R(x))):
            return False
        if dist_from_rim <= RESTRICTED_R: return False
        if y < 2.0: return True
        return dist_from_rim <= mini_R
    elif zone_id == 5:
        left_start_x, left_start_y = LEFT_DIAG_START
        def y_diagonal_line(xq):
            return left_start_y + LEFT_DIAG_SLOPE * (xq - left_start_x)
        return (
            (y >= 0.0) and
            (x >= LEFT_POST_X) and
            (x <= left_start_x) and
            (y <= y_diagonal_line(x)) and
            (dist_from_rim < THREE_R)
        )
    elif zone_id == 6:
        left_start_x, left_start_y = LEFT_DIAG_START
        def y_diagonal_line(xq):
            return left_start_y + LEFT_DIAG_SLOPE * (xq - left_start_x)
        return (
            (dist_from_rim >= mini_R) and
            (dist_from_rim < THREE_R) and
            (y < y_elbow_L(x)) and
            (y >= y_diagonal_line(x)) and
            (x < RIM_X)
        )
    elif zone_id == 7:
        return (
            (dist_from_rim >= mini_R) and
            (dist_from_rim < THREE_R) and
            (y >= y_elbow_L(x)) and
            (y >= y_elbow_R(x))
        )
    elif zone_id == 8:
        right_start_x, right_start_y = RIGHT_DIAG_START
        def y_diagonal_line_right(xq):
            return right_start_y + RIGHT_DIAG_SLOPE * (xq - right_start_x)
        return (
            (dist_from_rim >= mini_R) and
            (dist_from_rim < THREE_R) and
            (y < y_elbow_R(x)) and
            (y >= y_diagonal_line_right(x)) and
            (x > RIM_X)
        )
    elif zone_id == 9:
        right_start_x, right_start_y = RIGHT_DIAG_START
        def y_diagonal_line_right(xq):
            return right_start_y + RIGHT_DIAG_SLOPE * (xq - right_start_x)
        return (
            (y >= 0.0) and
            (x <= RIGHT_POST_X) and
            (x >= right_start_x) and
            (y <= y_diagonal_line_right(x)) and
            (dist_from_rim < THREE_R)
        )
    elif zone_id == 10:
        if x <= 0 or y <= 0:
            return False
        left_start_x, left_start_y = LEFT_DIAG_START
        def y_diagonal_top(xq):
            if xq <= LEFT_DIAG_IX: return LEFT_DIAG_IY
            else: return left_start_y + LEFT_DIAG_SLOPE * (xq - left_start_x)
        eps = 1e-6
        if y >= y_diagonal_top(x) - eps: return False
        if (x <= LEFT_POST_X - eps) or (math.hypot(x - RIM_X, y - RIM_Y) >= THREE_R + eps): return True
        return False
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
        left_start_x_diag, left_start_y_diag = LEFT_DIAG_START
        def y_diagonal_line(xq):
            if xq <= LEFT_DIAG_IX: return LEFT_DIAG_IY
            else: return left_start_y_diag + LEFT_DIAG_SLOPE * (xq - left_start_x_diag)
        if y <= y_diagonal_line(x): return False
        return True
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
        right_start_x_diag, right_start_y_diag = RIGHT_DIAG_START
        def y_diagonal_line(xq):
            if xq >= RIGHT_DIAG_IX: return RIGHT_DIAG_IY
            else: return right_start_y_diag + RIGHT_DIAG_SLOPE * (xq - right_start_x_diag)
        if y <= y_diagonal_line(x): return False
        return True
    elif zone_id == 14:
        if x >= COURT_W or y <= 0:
            return False
        right_start_x, right_start_y = RIGHT_DIAG_START
        def y_diagonal_top(xq):
            if xq >= RIGHT_DIAG_IX: return RIGHT_DIAG_IY
            else: return right_start_y + RIGHT_DIAG_SLOPE * (xq - right_start_x)
        eps = 1e-6
        if y >= y_diagonal_top(x) - eps: return False
        if (x >= RIGHT_POST_X + eps) or (math.hypot(x - RIM_X, y - RIM_Y) >= THREE_R + eps): return True
        return False

    return False



#----------------------Section 4----------------------------------------------------------

# ---- stats
def calculate_zone_stats(shots):
    zone_stats = {}
    for zone_id in range(1, 15):
        makes = attempts = 0
        for shot in (shots or []):
            # Count only valid shots (Make/Miss with legal x/y)
            try:
                x = float(shot.get("x"))
                y = float(shot.get("y"))
                res = (shot.get("result") or "").strip()
            except Exception:
                continue
            if x is None or y is None or res not in ("Make", "Miss"):
                continue
            if 0 <= x <= COURT_W and 0 <= y <= HALF_H and point_in_zone(x, y, zone_id):
                attempts += 1
                if res == "Make":
                    makes += 1
        pct = (makes / attempts) * 100 if attempts else 0.0
        zone_stats[zone_id] = {"makes": makes, "attempts": attempts, "percentage": round(pct, 1)}
    return zone_stats


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


# ======================================================================
# =======================  mk5 ADDITIONS (Section 2)  ===================
# Per-player stat aggregation for the Stats tab + filters
# ======================================================================

# ---- Stats-tab filters (subset: dates, drill size/label, defense) ----
def empty_stats_filters() -> dict:
    # IMPORTANT: use lists (not sets) so nothing non-JSON-serializable lands in Stores/layout.
    return {
        "date_start": "",
        "date_end": "",
        "drill_size": [],   # e.g., ["3v3", "5v5"]
        "drill_full": [],   # e.g., ["5v5 Stags"]
        "defense": [],      # values like "Man" or "Zone"
    }

def row_passes_stats_filters(row: dict, filters: dict) -> bool:
    filters = filters or {}
    ds_list    = list(filters.get("drill_size") or [])
    dfull_list = list(filters.get("drill_full") or [])
    def_list   = list(filters.get("defense") or [])

    if not _passes_date_range(row, filters.get("date_start",""), filters.get("date_end","")):
        return False
    if ds_list:
        if _row_drill_size(row) not in ds_list:
            return False
    if dfull_list:
        if _row_drill_label(row) not in dfull_list:
            return False
    if def_list:
        if _row_defense_label(row) not in def_list:
            return False
    return True


# ---- Per-player stat accumulator ----------------------------------------------------
def _ensure_player(bucket: dict, player: str) -> dict:
    """
    Ensure a dict for player exists with all counters initialized.
    Returns the player's dict for in-place updates.
    """
    name = (player or "").strip() or "Unknown"
    if name not in bucket:
        bucket[name] = {
            "Player": name,
            # shooting
            "FGM":0, "FGA":0, "2PM":0, "2PA":0, "3PM":0, "3PA":0,
            # passing
            "AST":0,
            # rebounding
            "DRB":0, "ORB":0,
            # turnovers & fouls
            "LBTO":0, "DBTO":0, "TO":0, "PF":0, "OF":0, "F":0,
            # defense/specials
            "STL":0, "DEF":0, "BLK":0,
            # possession counts
            "OP":0, "DP":0,
        }
    return bucket[name]

def _pct(m, a):
    try:
        return round((100.0 * m / a), 1) if a else 0.0
    except Exception:
        return 0.0


# ---- Main aggregator ----------------------------------------------------------------
def compute_player_stats_table(possession_rows: list[dict], filters: dict) -> list[dict]:
    """
    Build a table of per-player totals for the Stats tab.
    Columns:
      Player, FGM,FGA,FG%, 2PM,2PA,2P%, 3PM,3PA,3P%, AST,
      DRB, ORB, TRB, LBTO, DBTO, TO, PF, OF, F, STL, DEF, BLK, OP, DP
    """
    f_in = filters or {}
    try:
        # if all filter values are "empty", use fully empty defaults (lists)
        all_blank = True
        for v in f_in.values():
            if v not in (None, "", [], ()):
                all_blank = False
                break
        filters = empty_stats_filters() if all_blank else f_in
    except Exception:
        filters = f_in or empty_stats_filters()

    players = {}

    for row in (possession_rows or []):
        if not row_passes_stats_filters(row, filters):
            continue

        pbp = _row_possession_text(row)
        shorthand = pbp
        shot_x, shot_y = _row_xy(row)
        shot_res = _row_result(row)
        shooter, _ondef_disp, assister, _screen_list, _cand = _row_roles_and_lines(row)

        # Offensive / Defensive possessions from text + shorthand
        off_list, def_list = collect_players_from_possession(pbp, shorthand)
        for nm in off_list:
            _ensure_player(players, nm)["OP"] += 1
        for nm in def_list:
            _ensure_player(players, nm)["DP"] += 1

        # Shooting events credited to shooter
        if shooter and shot_res in ("Make", "Miss") and shot_x is not None and shot_y is not None:
            val = _row_shot_value(shot_x, shot_y)
            P = _ensure_player(players, shooter)
            P["FGA"] += 1
            if shot_res == "Make":
                P["FGM"] += 1
                if val == 2:
                    P["2PM"] += 1
                elif val == 3:
                    P["3PM"] += 1
            if val == 2:
                P["2PA"] += 1
            elif val == 3:
                P["3PA"] += 1

        # Assists
        if assister:
            _ensure_player(players, assister)["AST"] += 1

        # Specials (rebounds, TOs, fouls, steals/blocks/deflections)
        sp = parse_special_stats_from_shorthand(shorthand) or {}

        for nm in sp.get("def_rebounds", []):
            _ensure_player(players, nm)["DRB"] += 1
        for nm in sp.get("off_rebounds", []):
            _ensure_player(players, nm)["ORB"] += 1

        for nm in sp.get("live_ball_to", []):
            _ensure_player(players, nm)["LBTO"] += 1
        for nm in sp.get("dead_ball_to", []):
            _ensure_player(players, nm)["DBTO"] += 1

        for nm in sp.get("def_fouls", []):
            _ensure_player(players, nm)["PF"] += 1
        for nm in sp.get("off_fouls", []):
            _ensure_player(players, nm)["OF"] += 1

        for nm in sp.get("steals", []):
            _ensure_player(players, nm)["STL"] += 1
        for nm in sp.get("blocks", []):
            _ensure_player(players, nm)["BLK"] += 1
        for nm in sp.get("deflections", []):
            _ensure_player(players, nm)["DEF"] += 1

    # Post-processing: compute derived fields (FG%, 2P%, 3P%, TRB, TO, F)
    table = []
    for _nm, rec in players.items():
        rec = dict(rec)  # shallow copy
        rec["FG%"] = _pct(rec["FGM"], rec["FGA"])
        rec["2P%"] = _pct(rec["2PM"], rec["2PA"])
        rec["3P%"] = _pct(rec["3PM"], rec["3PA"])
        rec["TRB"] = rec["DRB"] + rec["ORB"]
        rec["TO"]  = rec["LBTO"] + rec["DBTO"]
        rec["F"]   = rec["PF"] + rec["OF"]
        table.append(rec)

    return table


# ------------ drawing helpers (unchanged visuals)
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
    tL = -math.pi / 2.2
    tR =  math.pi / 2.2
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
    left_start_x, left_start_y = LEFT_DIAG_START
    right_start_x, right_start_y = RIGHT_DIAG_START
    style = dict(width=2, color="black")
    return [
        go.Scatter(x=[left_start_x, LEFT_DIAG_IX],   y=[left_start_y, LEFT_DIAG_IY],   mode="lines", line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[LEFT_DIAG_IX, 0],              y=[LEFT_DIAG_IY, LEFT_DIAG_IY],   mode="lines", line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[right_start_x, RIGHT_DIAG_IX], y=[right_start_y, RIGHT_DIAG_IY], mode="lines", line=style, hoverinfo="skip", showlegend=False),
        go.Scatter(x=[RIGHT_DIAG_IX, COURT_W],       y=[RIGHT_DIAG_IY, RIGHT_DIAG_IY], mode="lines", line=style, hoverinfo="skip", showlegend=False),
    ]


def base_layout(fig_w=600, fig_h=720):
    return dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=fig_w,
        height=fig_h,
        margin=dict(l=10, r=10, t=0, b=0),   # no top margin
        xaxis=dict(
            range=[0, COURT_W],
            showgrid=False,
            zeroline=False,
            ticks="",
            showticklabels=False,
            mirror=True,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, HALF_H],
            showgrid=False,
            zeroline=False,
            ticks="",
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            mirror=True,
            fixedrange=True
        ),
        showlegend=False,
        title=dict(text=None, pad=dict(t=0, b=0, l=0, r=0))
    )


# ----- helpers to ensure clean inputs -----
def _coerce_to_shots(maybe_shots):
    """Return a safe list of plottable shots [{x,y,result}], filtering invalid rows."""
    if not maybe_shots:
        return []
    # Convert sets/tuples -> list
    if isinstance(maybe_shots, (set, tuple)):
        maybe_shots = list(maybe_shots)
    out = []
    for s in maybe_shots:
        if not isinstance(s, dict):
            continue
        try:
            x = float(s.get("x"))
            y = float(s.get("y"))
            res = (s.get("result") or "").strip()
        except Exception:
            continue
        if res not in ("Make", "Miss"):
            continue
        if not (0 <= x <= COURT_W and 0 <= y <= HALF_H):
            continue
        out.append({"x": x, "y": y, "result": res})
    return out


# ------------- Shot chart
def create_shot_chart(shots=None, highlight_coords=None):
    fig = go.Figure()
    for tr in court_lines_traces():
        fig.add_trace(tr)

    shots_list = _coerce_to_shots(shots)

    if shots_list:
        makes = [(s["x"], s["y"]) for s in shots_list if s["result"] == "Make"]
        misses = [(s["x"], s["y"]) for s in shots_list if s["result"] == "Miss"]
        if makes:
            x, y = zip(*makes)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(symbol='circle', size=10, color='green', line=dict(width=1, color='green')),
                name="Make"
            ))
        if misses:
            x, y = zip(*misses)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(symbol='x', size=10, color='red'),
                name="Miss"
            ))

    # highlight boxes (if any)
    if highlight_coords:
        try:
            coords = list(highlight_coords)  # tolerate tuples/sets
        except Exception:
            coords = []
        L = 1.2
        for (hx, hy) in coords:
            try:
                fig.add_shape(
                    type="rect",
                    x0=float(hx) - L/2, y0=float(hy) - L/2,
                    x1=float(hx) + L/2, y1=float(hy) + L/2,
                    line=dict(color="#888", width=1),
                    fillcolor="#e6e6e6",
                    layer="below"
                )
            except Exception:
                continue

    fig.update_layout(**base_layout())
    fig.update_layout(clickmode="event+select")

    # mini lane guides
    left_x  = RIM_X - FT_R
    right_x = RIM_X + FT_R
    for x in (left_x, right_x):
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=FT_CY, line=dict(color="black", width=2), layer="above")
    return fig


# --------- Zone-tiered color mapping by FG%
def _zone_tier(zone_id: int) -> str:
    if zone_id in (1, 2, 3, 4):
        return "close"
    if zone_id in (5, 6, 7, 8, 9):
        return "mid"
    return "three"

def _rgba_for_zone(zone_id: int, attempts: int, pct: float) -> str:
    if attempts == 0:
        return "rgba(255,255,255,1.0)"
    tier = _zone_tier(zone_id)
    BLUE   = "rgba(0,102,204,0.95)"
    YELLOW = "rgba(255,204,0,0.95)"
    RED    = "rgba(204,0,0,0.95)"
    if tier == "close":
        if 0.0 <= pct <= 50.0:   return BLUE
        if 51.0 <= pct <= 60.0:  return YELLOW
        return RED
    elif tier == "mid":
        if 0.0 <= pct <= 35.0:   return BLUE
        if 36.0 <= pct <= 45.0:  return YELLOW
        return RED
    else:
        if 0.0 <= pct <= 25.0:   return BLUE
        if 26.0 <= pct <= 35.0:  return YELLOW
        return RED


def create_zone_chart(shots=None):
    """
    Build the hot/cold zone chart. If 'shots' is None, render a neutral court (no data).
    """
    fig = go.Figure()

    for tr in court_lines_traces(): fig.add_trace(tr)
    for tr in first_zone_line_traces(): fig.add_trace(tr)
    for tr in mini_three_point_line(): fig.add_trace(tr)
    for tr in elbow_lines(): fig.add_trace(tr)
    for tr in diagonal_zone_lines(): fig.add_trace(tr)

    shots_list = _coerce_to_shots(shots)
    zone_stats = calculate_zone_stats(shots_list)

    zone_centers = {
        1:(24.5,5.5),2:(16.5,5.5),3:(24.5,12),4:(33,6.5),
        5:(6,3.5),6:(10,14.5),7:(24.5,22),8:(39,14.5),9:(42,3),
        10:(1,3.5),11:(5.5,20),12:(24.5,29),13:(42.5,21),14:(48.5,2.5)
    }

    for zone_id in range(1, 15):
        s = zone_stats.get(zone_id, {"makes":0,"attempts":0,"percentage":0.0})
        rgba = _rgba_for_zone(zone_id, s["attempts"], s["percentage"])
        add_zone_fill(fig, zone_id=zone_id, step=0.25, rgba=rgba)

    for zone_id, center in zone_centers.items():
        s = zone_stats.get(zone_id, {"makes":0,"attempts":0,"percentage":0.0})
        txt = f"{s['makes']}/{s['attempts']}<br>{s['percentage']:.1f}%"
        fig.add_annotation(x=center[0], y=center[1], text=txt,
                           showarrow=False,
                           font=dict(size=12, color="black", family="Arial Black"))

    fig.update_layout(**base_layout())
    return fig


# =========================
# Dash App
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # expose Flask server for deployment environments

# IMPORTANT: Do NOT load data or create figures from data at import time.
# Build neutral, static figures to avoid any non-serializable objects in the initial layout.
_initial_shot_fig = create_shot_chart(None)
_initial_zone_fig = create_zone_chart(None)

# ---- mk5 ADD: bring in DataTable for sortable Stats tab
from dash import dash_table  # (Dash 2+) correct import

# ====================== mk5: helpers used by callbacks (moved here so Section 4 sees them) ======================




# Ensure initial figures preserve UI state (no flicker) without using Graph.uirevision
try:
    _initial_shot_fig.update_layout(uirevision="keep")
    _initial_zone_fig.update_layout(uirevision="keep")
except Exception:
    pass


# ---------- Legend component (colors only; labels filled later in Section 5) ----------
def zone_legend_component():
    """
    Visual key placed under the Hot/Cold Zones chart.
    Shows % ranges for each color; the text is filled by callbacks in Section 5.
    """
    BLUE   = "rgba(0,102,204,0.95)"
    YELLOW = "rgba(255,204,0,0.95)"
    RED    = "rgba(204,0,0,0.95)"
    WHITE  = "rgba(255,255,255,1.0)"

    BOX = {
        "display": "inline-block",
        "width": "14px",
        "height": "14px",
        "border": "1px solid #333",
        "marginRight": "6px",
        "verticalAlign": "-2px",
    }

    def tier_row(span_id_blue, span_id_yellow, span_id_red):
        # three colored bins + labels (labels are injected via callbacks)
        return html.Div(
            [
                html.Span(style={**BOX, "background": BLUE}),
                html.Span(id=span_id_blue,   children="—", style={"marginRight": "14px"}),

                html.Span(style={**BOX, "background": YELLOW}),
                html.Span(id=span_id_yellow, children="—", style={"marginRight": "14px"}),

                html.Span(style={**BOX, "background": RED}),
                html.Span(id=span_id_red,    children="—", style={"marginRight": "14px"}),

                html.Span(style={**BOX, "background": WHITE}),
                html.Span("No attempts", style={"opacity": 0.8}),
            ],
            style={"marginTop": "2px"}
        )

    block_style = {"display": "inline-block", "margin": "0 18px", "textAlign": "left"}

    return html.Div(
        [
            html.Div(
                [html.Div("Close Shots", style={"fontWeight": 800, "marginBottom": "2px"}),
                 tier_row("legend_close_low", "legend_close_mid", "legend_close_high")],
                style=block_style
            ),
            html.Div(
                [html.Div("Mid Range Shots", style={"fontWeight": 800, "marginBottom": "2px"}),
                 tier_row("legend_mid_low", "legend_mid_mid", "legend_mid_high")],
                style=block_style
            ),
            html.Div(
                [html.Div("3 Point Shots", style={"fontWeight": 800, "marginBottom": "2px"}),
                 tier_row("legend_three_low", "legend_three_mid", "legend_three_high")],
                style=block_style
            ),
        ],
        id="zone_legend",
        style={"textAlign": "center", "marginTop": "8px", "fontSize": "14px"}
    )





#----------------------------Section 5---------------------------------------------------

# ---------------- roster harvesting & normalization ----------------
def _harvest_fullnames_from_any(value):
    out = []
    if isinstance(value, str):
        if " " in value and re.match(rf"^{_FULLNAME}$", value.strip()):
            out.append(value.strip())
    elif isinstance(value, list):
        for v in value:
            out += _harvest_fullnames_from_any(v)
    elif isinstance(value, dict):
        for v in value.values():
            out += _harvest_fullnames_from_any(v)
    return out


def _collect_roster_for_group(rows_for_plot, key):
    disk_roster = _load_roster_from_disk()
    names = list(disk_roster.values())
    if names:
        seen = set()
        ordered = []
        for n in names:
            if n.lower() not in seen:
                ordered.append(n)
                seen.add(n.lower())
        return ordered

    for rr in rows_for_plot:
        k = rr.get("group_id") or rr.get("timestamp") or rr.get("id")
        if k != key:
            continue
        for field in ("roster", "roster_names", "names", "home_names", "away_names",
                      "home_roster", "away_roster", "roster_map"):
            if field in rr:
                vals = rr[field]
                names = _harvest_fullnames_from_any(vals)
                if names:
                    seen = set()
                    ordered = []
                    for n in names:
                        if n.lower() not in seen:
                            ordered.append(n)
                            seen.add(n.lower())
                    return ordered

    pool = []
    for rr in rows_for_plot:
        k = rr.get("group_id") or rr.get("timestamp") or rr.get("id")
        if k != key:
            continue
        for v in rr.values():
            pool += _harvest_fullnames_from_any(v)

    seen = set()
    ordered = []
    for n in pool:
        if n.lower() not in seen:
            ordered.append(n)
            seen.add(n.lower())
    return ordered


# --- RENAMED to avoid clobbering Section-1's _normalize_to_roster ---
def _normalize_to_roster_for_list(name, roster_full_list):
    nm = _trim_trailing_verb(name or "").strip()
    nm = _strip_leading_preps(nm)
    nm = re.sub(r"(?:\s*(?:,|and|&)\s*)+$", "", nm, flags=re.IGNORECASE)
    if not nm:
        return nm

    roster_dict = {str(i): full for i, full in enumerate(roster_full_list or [], 1)}
    last_map, first_map, full_set = _build_name_maps(roster_dict)

    low = nm.lower()
    if " " not in nm:
        if low in last_map and len(last_map[low]) == 1:
            return last_map[low][0]
        if low in first_map and len(first_map[low]) == 1:
            return first_map[low][0]
        subs = [full for full in roster_dict.values() if low in full.lower()]
        if len(subs) == 1:
            return subs[0]
        return nm
    else:
        if low in full_set:
            for full in roster_dict.values():
                if full.strip().lower() == low:
                    return full
        parts = nm.split()
        first_tok, last_tok = parts[0].lower(), parts[-1].lower()
        cands = []
        for full in roster_dict.values():
            fparts = full.lower().split()
            if fparts and fparts[0].startswith(first_tok) and fparts[-1] == last_tok:
                cands.append(full)
        if len(cands) == 1:
            return cands[0]
        return nm


def _norm_block(value, roster_full_list):
    return _normalize_to_roster_for_list(value, roster_full_list)


# ---------- helper: patch "Bring over halfcourt" using 'X picks up Y' if needed
_BRING_SUBJ_RE = re.compile(rf"({_FULLNAME})\s+bring[s]?\s+(?:the\s+ball\s+)?over\s+half\s*court", re.IGNORECASE)
_PICKS_UP_RE   = re.compile(rf"({_FULLNAME})\s+picks\s+up\s+({_FULLNAME})", re.IGNORECASE)


def _patch_bring_over_halfcourt(actions, pbp_text):
    """Fill bh/bh_def for 'Bring over halfcourt' when PBP uses 'X picks up Y'."""
    if not actions:
        return actions
    try:
        txt = _clean_frag(pbp_text or "")
        bh_from_brings = ""
        m_bring = _BRING_SUBJ_RE.search(txt)
        if m_bring:
            bh_from_brings = _trim_trailing_verb(m_bring.group(1).strip())

        bh_def_from_pick = ""
        bh_from_pick = ""
        for m in re.finditer(_PICKS_UP_RE, txt):
            d_name = _trim_trailing_verb(m.group(1).strip())
            o_name = _trim_trailing_verb(m.group(2).strip())
            if bh_from_brings and o_name and o_name.lower() == bh_from_brings.lower():
                bh_def_from_pick = d_name
                bh_from_pick = o_name
                break
            bh_def_from_pick, bh_from_pick = d_name, o_name

        for a in actions:
            if a.get("type") == "h":
                if not a.get("bh"):
                    a["bh"] = bh_from_brings or bh_from_pick or a.get("bh", "")
                if not a.get("bh_def"):
                    if a.get("bh") and bh_from_pick and a["bh"].lower() == bh_from_pick.lower():
                        a["bh_def"] = bh_def_from_pick
                    else:
                        a["bh_def"] = bh_def_from_pick or a.get("bh_def", "")
        return actions
    except Exception:
        return actions


# ================== NEW HELPERS (used by callbacks) ==================
_ROTATING_OVER_RE = re.compile(r"\brotating\s+over\b", re.IGNORECASE)
_HELP_VERBS_RE    = re.compile(r"\bhelp(?:s|ed|ing)?\b|\bsteps\s+in\s+to\s+help\b", re.IGNORECASE)


def shot_defender_display(pbp_text: str, shot_index: int) -> str:
    """Format defenders for the Nth shot line, preserving 'rotating over' tags."""
    line = _nth_shot_line(pbp_text or "", shot_index)
    return _format_defenders_for_display(_parse_defenders_with_tags_from_line(line))


def shot_has_help(pbp_text: str, shot_index: int) -> bool:
    """True if possession text/shot line contains help cues."""
    txt = _clean_frag(pbp_text or "")
    if _HELP_VERBS_RE.search(txt):
        return True
    line = _nth_shot_line(txt, shot_index)
    return bool(_HELP_VERBS_RE.search(line) or _ROTATING_OVER_RE.search(line))


# ========================= Special-stats display helpers =========================
_SPECIAL_LABELS = {
    "def_rebounds": "Defensive Rebound",
    "off_rebounds": "Offensive Rebound",
    "deflections":  "Deflection",
    "steals":       "Steal",
    "blocks":       "Block",
    "live_ball_to": "Live Ball Turnover",
    "dead_ball_to": "Dead Ball Turnover",
    "def_fouls":    "Defensive Foul",
    "off_fouls":    "Offensive Foul",
}


def special_stats_for_display(short_text: str):
    """Return label+players rows ready for rendering from shorthand."""
    try:
        parsed = parse_special_stats_from_shorthand(short_text or "")
    except Exception:
        parsed = None
    rows = []
    if not parsed:
        return rows
    for key, label in _SPECIAL_LABELS.items():
        players = parsed.get(key, []) or []
        if players:
            rows.append({"type": key, "label": label, "players": players})
    return rows


def blockers_from_pbp_for_display(pbp_text: str) -> list[str]:
    """Wrapper for parse_blockers_from_pbp(...)."""
    try:
        return parse_blockers_from_pbp(pbp_text or "")
    except Exception:
        return []


def special_stats_with_pbp_blocks(short_text: str, pbp_text: str):
    """Merge shorthand Block with PBP-detected blockers."""
    rows = special_stats_for_display(short_text or "")
    by_label = {r["label"]: list(r.get("players", []) or []) for r in rows}
    pbp_blocks = blockers_from_pbp_for_display(pbp_text or "")
    if pbp_blocks:
        base = by_label.get("Block", [])
        base_l = {b.lower() for b in base}
        merged = base + [n for n in pbp_blocks if n and n.lower() not in base_l]
        by_label["Block"] = merged

    order = ["Defensive Rebound", "Offensive Rebound", "Deflection", "Steal",
             "Live Ball Turnover", "Dead Ball Turnover", "Defensive Foul",
             "Offensive Foul", "Block"]
    out = []
    for lbl in order:
        players = by_label.get(lbl, [])
        if players:
            key = None
            for k, v in _SPECIAL_LABELS.items():
                if v == lbl:
                    key = k
                    break
            out.append({"type": key or "", "label": lbl, "players": players})
    return out


# ========================= Zone defense + participants wrappers =========================
def defense_label_for_shot(pbp_text: str, shot_index: int) -> str:
    """Wrapper for zone_for_shot(...)."""
    try:
        return zone_for_shot(pbp_text or "", shot_index)
    except Exception:
        return "Man to Man"


def participants_for_possession(pbp_text: str, shorthand_text: str):
    """Wrapper for collect_players_from_possession(...)."""
    try:
        return collect_players_from_possession(pbp_text or "", shorthand_text or "")
    except Exception:
        return ([], [])


# ------------------------- OFF-BALL actions + coverages (mk5 version with stagger merge) -------------------------
_OFFBALL_TYPES = {
    "bd":  {"label": "Backdoor cut", "keys": [r"\bbackdoor\b", r"backdoor\s+cut"]},
    "pn":  {"label": "Pin down", "keys": [r"\bpin\s*down\b", r"\bpindown\b"]},
    "fl":  {"label": "Flare screen", "keys": [r"\bflare\s+screen\b"]},
    "bk":  {"label": "Back screen", "keys": [r"\bback\s+screen\b"]},
    "awy": {"label": "Away screen", "keys": [r"\baway\s+screen\b"]},
    "ucla":{"label": "UCLA screen", "keys": [r"\bucla\s+screen\b"]},
    "crs": {"label": "Cross screen", "keys": [r"\bcross\s+screen\b"]},
    "wdg": {"label": "Wedge screen", "keys": [r"\bwedge\s+screen\b"]},
    "rip": {"label": "Rip screen", "keys": [r"\brip\s+screen\b"]},
    "stg": {"label": "Stagger screen", "keys": [r"\bstagger\s+screen[s]?\b"]},
    "ivs": {"label": "Iverson screen", "keys": [r"\biverson\s+screen[s]?\b"]},
    "elv": {"label": "Elevator screen", "keys": [r"\belevator\s+screen[s]?\b"]},
}


def _matches_any(text_lc: str, patt_list):
    for p in patt_list:
        if re.search(p, text_lc, re.IGNORECASE):
            return True
    return False


def _parse_guarded_pairs_in_order(text: str):
    out = []
    for m in re.finditer(_SHOOT_GUARD_RE, _clean_frag(text)):
        a = _trim_trailing_verb(m.group(1).strip())
        b = _trim_trailing_verb(m.group(2).strip())
        a = _strip_leading_preps(a)
        b = _strip_leading_preps(b)
        out.append((a, b))
    return out


def parse_offball_actions_from_pbp(lines):
    """Merge multiple stagger lines into one; other types remain one-per-line."""
    stg_agg = None
    stg_seen_scr = set()
    stg_seen_cov = set()
    actions = []

    for ln in (lines or []):
        lc = ln.lower()
        matched_key = None
        for k, meta in _OFFBALL_TYPES.items():
            if _matches_any(lc, meta["keys"]):
                matched_key = k
                break
        if not matched_key:
            continue

        pairs = _parse_guarded_pairs_in_order(ln)
        coming_off, coming_off_def = ("", "")
        screeners = []
        if pairs:
            coming_off, coming_off_def = pairs[0]
            for a, b in pairs[1:]:
                screeners.append({"name": a, "def": b})

        covs = _parse_coverages(ln)

        if matched_key == "stg":
            if stg_agg is None:
                stg_agg = {
                    "type": "stg",
                    "label": _OFFBALL_TYPES["stg"]["label"],
                    "coming_off": coming_off,
                    "coming_off_def": coming_off_def,
                    "screeners": [],
                    "coverages": [],
                }
            if not stg_agg.get("coming_off") and coming_off:
                stg_agg["coming_off"] = coming_off
            if not stg_agg.get("coming_off_def") and coming_off_def:
                stg_agg["coming_off_def"] = coming_off_def

            for s in screeners:
                key = (s.get("name", "").lower(), s.get("def", "").lower())
                if key not in stg_seen_scr and (s.get("name") or s.get("def")):
                    stg_agg["screeners"].append({"name": s.get("name", ""), "def": s.get("def", "")})
                    stg_seen_scr.add(key)

            for c in (covs or []):
                key = (c.get("label", "").lower(), (c.get("onto", "") or "").lower())
                if key not in stg_seen_cov:
                    stg_agg["coverages"].append(c)
                    stg_seen_cov.add(key)
            continue

        actions.append({
            "type": matched_key,
            "label": _OFFBALL_TYPES[matched_key]["label"],
            "coming_off": coming_off,
            "coming_off_def": coming_off_def,
            "screeners": screeners,
            "coverages": covs,
        })

    if stg_agg:
        actions.append(stg_agg)
    return actions


# ================== defender-rotation decoration across ANY possession line ==================
def _last_token(n: str) -> str:
    n = (n or "").strip()
    if not n:
        return ""
    parts = re.split(r"\s+", n)
    return parts[-1] if parts else n


def _word_or_last_regex(name_full: str) -> str:
    full = re.escape((name_full or "").strip())
    last = re.escape(_last_token(name_full))
    if not full and not last:
        return r""
    if full and last and full.lower() != last.lower():
        return rf"(?:{full}|{last})"
    return full or last


def _decorate_defenders_with_rotation(def_str: str, shooter_name: str, pbp_text: str) -> str:
    if not def_str:
        return def_str
    txt = _clean_frag(pbp_text or "")
    names = _split_fullname_list(def_str) or [def_str.strip()]
    decorated = []
    shooter_pat = _word_or_last_regex(shooter_name)
    for nm in names:
        def_pat = _word_or_last_regex(nm)
        has_rot = False
        if shooter_pat and def_pat:
            patt1 = re.compile(rf"\b{shooter_pat}\s+guarded\s+by\s+{def_pat}\s+rotating\s+over\b", re.IGNORECASE)
            if patt1.search(txt):
                has_rot = True
        if not has_rot and def_pat:
            patt2 = re.compile(rf"\bguarded\s+by\s+{def_pat}\s+rotating\s+over\b", re.IGNORECASE)
            if patt2.search(txt):
                has_rot = True
        decorated.append(f"{nm} rotating over" if has_rot else nm)
    return ", ".join(decorated)


# ================== sanitize trailing modifiers for participant lists ==================
def _strip_trailing_modifiers(nm: str) -> str:
    s = (nm or "").strip()
    s = re.sub(r"\s+(rotating\s+over|rotating|grabs?|tips?|for|to|helps?|picks\s+up)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[)\].,!;:]+$", "", s)
    return s


# ====================== mk5 UI (tabs, filters, tables) ======================
# ---- mk5 ADD: option helpers (populated in callbacks)
def _mk_multi(label, cid, placeholder="", options=None):
    return dcc.Dropdown(
        id=cid, options=(options or []), multi=True, placeholder=placeholder,
        style={"minWidth": "160px"}
    )


def _pill(title, child):
    return html.Div(
        [html.Div(title, style={"fontSize": "12px", "fontWeight": 700, "marginBottom": "4px"}), child],
        style={"display": "flex", "flexDirection": "column", "gap": "4px"}
    )


# ---- mk5 ADD: constants for action filters
ONBALL_OPTIONS = [
    {"label": "H (bring over)", "value": "h"},
    {"label": "D (drive)", "value": "d"},
    {"label": "P (post up)", "value": "p"},
    {"label": "PNR", "value": "pnr"},
    {"label": "PNP", "value": "pnp"},
    {"label": "DHO", "value": "dho"},
    {"label": "HO", "value": "ho"},
    {"label": "KP (keep)", "value": "kp"},
    {"label": "RJ (reject)", "value": "rj"},
    {"label": "SLP (slip)", "value": "slp"},
    {"label": "GST (ghost)", "value": "gst"},
]
OFFBALL_OPTIONS = [
    {"label": "Backdoor (bd)", "value": "bd"},
    {"label": "Pin down (pn)", "value": "pn"},
    {"label": "Flare (fl)", "value": "fl"},
    {"label": "Back screen (bk)", "value": "bk"},
    {"label": "Away (awy)", "value": "awy"},
    {"label": "UCLA (ucla)", "value": "ucla"},
    {"label": "Cross (crs)", "value": "crs"},
    {"label": "Wedge (wdg)", "value": "wdg"},
    {"label": "Rip (rip)", "value": "rip"},
    {"label": "Stagger (stg)", "value": "stg"},
    {"label": "Iverson (ivs)", "value": "ivs"},
    {"label": "Elevator (elv)", "value": "elv"},
]
# values must be "Man"/"Zone" (strings) to match filter predicates
DEFENSE_OPTIONS = [
    {"label": "Man to Man", "value": "Man"},
    {"label": "Zone", "value": "Zone"},
]


# ---- JSON-safe empty filters for Shooting store (aligned with Section 2/3)
def _safe_empty_shooting_filters():
    # names here MUST match row_passes_shooting_filters / collect_shots_for_filters
    return {
        "date_start": "",
        "date_end": "",
        "drill_size": [],
        "drill_full": [],
        "shooter": [],          # was 'shooters'
        "defenders": [],        # same key as predicate
        "assister": [],         # was 'assisters'
        "screen_ast": [],       # was 'screen_assisters'
        "onball": [],           # action codes
        "offball": [],
        "on_bh": [], "on_bh_def": [], "on_screener": [], "on_screener_def": [], "on_cov": [],
        "off_mover": [], "off_mover_def": [], "off_screener": [], "off_screener_def": [],
        "defense": [],
    }


# ---- LAYOUT (no Loading spinners, no timer-based Interval)
app.layout = html.Div(
    style={"maxWidth": "1600px", "margin": "0 auto", "padding": "10px"},
    children=[
        html.Div(
            [
                html.Div(f"Data source: {DATA_PATH}", style={"color": "#666", "fontSize": "12px", "marginBottom": "2px"}),
                html.Div("Charts update when data or filters change", style={"color": "#888", "fontSize": "10px"}),
                html.Div(id="status", style={"color": "#888", "fontSize": "10px"}),
            ],
            style={"textAlign": "center", "marginBottom": "8px"},
        ),

        dcc.Tabs(
            id="tabs",
            value="tab_shooting",
            children=[
                # ===================== Shooting TAB =====================
                dcc.Tab(
                    label="Shooting",
                    value="tab_shooting",
                    children=[
                        # ---------- Row 1: Core filters (Defense moved here)
                        html.Div(
                            [
                                _pill("Practice Date(s)", dcc.DatePickerRange(
                                    id="flt_date_range_shoot",
                                    min_date_allowed=None,
                                    max_date_allowed=None,
                                    start_date=None,
                                    end_date=None,
                                    minimum_nights=0,
                                    display_format="YYYY-MM-DD",
                                    style={"background": "white"},
                                )),
                                _pill("Drill Size", _mk_multi("Drill Size", "flt_drill_size_shoot", "e.g. 3v3 / 5v5")),
                                _pill("Drill", _mk_multi("Drill", "flt_drill_full_shoot", "e.g. 5v5 Stags")),
                                _pill("Shooter", _mk_multi("Shooter", "flt_shooter", "Filter by shooter")),
                                _pill("Defender(s)", _mk_multi("Defender(s)", "flt_defenders", "Who contested the shot")),
                                _pill("Assister", _mk_multi("Assister", "flt_assister", "Passer on made FG")),
                                _pill("Screen Assister", _mk_multi("Screen Assister", "flt_screen_assister", "Who set the screen")),
                                _pill("Defense", dcc.Dropdown(
                                    id="flt_defense_shoot",
                                    options=DEFENSE_OPTIONS,
                                    multi=True,
                                    placeholder="Man / Zone",
                                    style={"minWidth": "140px"},
                                )),
                                html.Button(
                                    "Clear",
                                    id="btn_clear_shoot",
                                    n_clicks=0,
                                    style={"height": "34px", "alignSelf": "flex-end", "marginLeft": "8px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "12px",
                                "alignItems": "end",
                                "background": "#fafafa",
                                "border": "1px solid #eee",
                                "borderRadius": "8px",
                                "padding": "10px",
                                "marginBottom": "10px",
                            },
                        ),

                        # ---------- Row 2: On-Ball (subfilters appear after selection)
                        html.Div(
                            [
                                _pill("On-Ball Action", dcc.Dropdown(
                                    id="flt_onball",
                                    options=ONBALL_OPTIONS,
                                    multi=True,
                                    placeholder="Select on-ball actions",
                                    style={"minWidth": "240px"},
                                )),
                                html.Div(
                                    id="flt_onball_dynamic",
                                    style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "alignItems": "end"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "12px",
                                "alignItems": "end",
                                "background": "#fcfcfc",
                                "border": "1px dashed #ddd",
                                "borderRadius": "8px",
                                "padding": "10px",
                                "marginBottom": "10px",
                            },
                        ),

                        # ---------- Row 3: Off-Ball (subfilters appear after selection)
                        html.Div(
                            [
                                _pill("Off-Ball Action", dcc.Dropdown(
                                    id="flt_offball",
                                    options=OFFBALL_OPTIONS,
                                    multi=True,
                                    placeholder="Select off-ball actions",
                                    style={"minWidth": "240px"},
                                )),
                                html.Div(
                                    id="flt_offball_dynamic",
                                    style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "alignItems": "end"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "12px",
                                "alignItems": "end",
                                "background": "#fcfcfc",
                                "border": "1px dashed #ddd",
                                "borderRadius": "8px",
                                "padding": "10px",
                                "marginBottom": "12px",
                            },
                        ),

                        # ---------- Single row: Stats | Shot Chart | Hot/Cold
                        html.Div(
                            [
                                # --- 1) Shooting Stats (left) ---
                                html.Div(
                                    [
                                        html.Div(
                                            "Shooting Stats",
                                            style={"textAlign": "center", "fontSize": "20px", "fontWeight": 700, "marginBottom": "6px"},
                                        ),
                                        html.Div(
                                            id="shooting_stats_box",
                                            style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px"},
                                        ),
                                    ],
                                    style={
                                        "width": "300px",
                                        "padding": "8px",
                                        "border": "1px solid #eee",
                                        "borderRadius": "8px",
                                        "background": "white",
                                    },
                                ),

                                # --- 2) Shot Chart (middle) ---
                                html.Div(
                                    [
                                        html.Div(
                                            "Shot Chart",
                                            style={"textAlign": "center", "fontSize": "26px", "fontWeight": 800, "marginBottom": "4px"},
                                        ),
                                        dcc.Graph(
                                            id="shot_chart",
                                            config={"displayModeBar": False},
                                            figure=_initial_shot_fig,
                                            animate=False,
                                            clear_on_unhover=False,
                                        ),
                                        html.Div(
                                            [
                                                html.Span("● Make", style={"color": "green", "marginRight": "20px", "fontWeight": 600}),
                                                html.Span("✖ Miss", style={"color": "red", "fontWeight": 600}),
                                            ],
                                            style={"textAlign": "center", "marginTop": "-4px"},
                                        ),
                                    ],
                                    style={"width": "600px"},
                                ),

                                # --- 3) Hot/Cold Zones (right) ---
                                html.Div(
                                    [
                                        html.Div(
                                            "Hot/Cold Zones",
                                            style={"textAlign": "center", "fontSize": "26px", "fontWeight": 800, "marginBottom": "4px"},
                                        ),
                                        dcc.Graph(
                                            id="zone_chart",
                                            config={"displayModeBar": False},
                                            figure=_initial_zone_fig,
                                            animate=False,
                                            clear_on_unhover=False,
                                        ),
                                        zone_legend_component(),
                                    ],
                                    style={"width": "600px"},
                                ),
                            ],
                            id="shooting-row",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "300px 600px 600px",
                                "columnGap": "20px",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "margin": "0 auto",
                                "maxWidth": "1600px",
                                "overflowX": "visible",
                            },
                        ),

                        html.Div(id="shot_details", style={"maxWidth": "920px", "margin": "14px auto 0 auto"}),
                    ],
                ),

                # ===================== Stats TAB =====================
                dcc.Tab(
                    label="Stats",
                    value="tab_stats",
                    children=[
                        # ---------- Stats tab filters (subset)
                        html.Div(
                            [
                                _pill("Practice Date(s)", dcc.DatePickerRange(
                                    id="flt_date_range_stats",
                                    min_date_allowed=None,
                                    max_date_allowed=None,
                                    start_date=None,
                                    end_date=None,
                                    minimum_nights=0,
                                    display_format="YYYY-MM-DD",
                                    style={"background": "white"},
                                )),
                                _pill("Drill Size", _mk_multi("Drill Size", "flt_drill_size_stats", "e.g. 3v3 / 5v5")),
                                _pill("Drill", _mk_multi("Drill", "flt_drill_full_stats", "e.g. 5v5 Stags")),
                                _pill("Defense", dcc.Dropdown(
                                    id="flt_defense_stats",
                                    options=DEFENSE_OPTIONS,
                                    multi=True,
                                    placeholder="Man / Zone",
                                    style={"minWidth": "140px"},
                                )),
                                html.Button(
                                    "Clear",
                                    id="btn_clear_stats",
                                    n_clicks=0,
                                    style={"height": "34px", "alignSelf": "flex-end", "marginLeft": "8px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "12px",
                                "alignItems": "end",
                                "background": "#fafafa",
                                "border": "1px solid #eee",
                                "borderRadius": "8px",
                                "padding": "10px",
                                "marginBottom": "10px",
                            },
                        ),

                        # ---------- Stats table
                        html.Div(
                            [
                                html.Div(
                                    "Team Stats (click headers to sort)",
                                    style={"fontSize": "18px", "fontWeight": 700, "marginBottom": "6px"},
                                ),
                                dash_table.DataTable(
                                    id="stats_table",
                                    columns=[{"name": c, "id": c} for c in [
                                        "Player", "FGM", "FGA", "FG%", "2PM", "2PA", "2P%", "3PM", "3PA", "3P%",
                                        "AST", "DRB", "ORB", "TRB", "LBTO", "DBTO", "TO", "PF", "OF", "F",
                                        "STL", "DEF", "BLK", "OP", "DP"
                                    ]],
                                    data=[],
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontFamily": "Arial", "fontSize": "13px", "padding": "6px", "textAlign": "center"},
                                    style_header={"fontWeight": "700", "backgroundColor": "#f5f5f5"},
                                    fixed_rows={"headers": True},
                                    page_size=50,
                                ),
                            ],
                            style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "8px", "background": "white"},
                        ),
                    ],
                ),
            ],
        ),

        # ---- state stores (callbacks in Sections 6–7 will update on data/filters only)
        dcc.Store(id="sel_pos", data=[]),
        # Use JSON-safe defaults here:
        dcc.Store(id="filters_shoot_state", data=_safe_empty_shooting_filters()),
        dcc.Store(id="filters_stats_state", data=empty_stats_filters()),
        dcc.Store(id="options_cache", data={}),
        dcc.Store(id="action_subfilters_state", data={}),
    ],
)



#--------------------------------------Section 6-------------------------------------------

# ---------------- callbacks ----------------
from datetime import datetime, date
from dash import no_update, callback_context, callback
from dash.dependencies import ALL
from dash import Input, Output, State

# ---------- small utils ----------
def _parse_date_any(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(s)[:10], fmt).date()
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(s)[:10]).date()
    except Exception:
        return None

# IMPORTANT: do NOT name this _DRILL_SIZE_RE — Section 2 uses _DRILL_SIZE_PAIR_RE
_DRILL_SIZE_TOKEN_RE = re.compile(r"\b(\d+v\d+)\b", re.IGNORECASE)

def _get_drill_size(drill: str) -> str:
    if not drill: return ""
    m = _DRILL_SIZE_TOKEN_RE.search(drill)
    return (m.group(1) if m else "").lower()

def _distance_from_rim(x, y):
    try:
        return math.hypot(float(x) - RIM_X, float(y) - RIM_Y)
    except Exception:
        return 0.0

def _is_three(x, y):
    return _distance_from_rim(x, y) >= THREE_R - 1e-9

def _as_list(v):
    if v is None: return []
    if isinstance(v, (list, tuple, set)): return list(v)
    return [v]

def _safe_str(s):
    return (s or "").strip()

def _serialize_date(d):
    if not d: return ""
    try:
        return str(d)[:10]
    except Exception:
        return ""

# ---------- load rows + light parsing ----------
def _rows_full():
    """
    Load rows from DATA_PATH once per render. Includes first-run diagnostics so it’s
    obvious if the file isn't there yet. Also tolerant to common field variants from
    the entry app (different x/y keys, scaled coords, different result keys/labels).
    """
    # Print the data path once so we can see what the app is trying to read
    if not getattr(_rows_full, "_printed_path", False):
        try:
            print("[BBALL] DATA_PATH ->", DATA_PATH)
        except Exception:
            pass
        _rows_full._printed_path = True

    rows_for_plot = []

    if not os.path.exists(DATA_PATH):
        print("[BBALL] possessions file missing at:", DATA_PATH)
        return rows_for_plot

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("[BBALL] Failed to load JSON:", e)
        return rows_for_plot

    # Accept either {"rows":[...]} or a top-level list
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]

    def _coerce_xy(rr):
        """Handle x/y in several shapes: absolute, pixel, or normalized 0..1."""
        x = rr.get("x"); y = rr.get("y")
        # Scaled 0..1 -> court units
        if x in (None, "") and y in (None, ""):
            x = rr.get("sx"); y = rr.get("sy")
            if x not in (None, "") and y not in (None, ""):
                try:
                    x = float(x) * float(COURT_W)
                    y = float(y) * float(HALF_H)
                except Exception:
                    x = y = None
        # Pixel coords -> court units (if canvas sizes provided)
        if x in (None, "") and y in (None, ""):
            x = rr.get("px"); y = rr.get("py")
            cw = rr.get("canvas_w"); ch = rr.get("canvas_h")
            if x not in (None, "") and y not in (None, "") and cw and ch:
                try:
                    x = float(x) / float(cw) * float(COURT_W)
                    y = float(y) / float(ch) * float(HALF_H)
                except Exception:
                    x = y = None
        # Final numeric coercion
        try:
            x = float(x) if x not in (None, "", "None") else None
            y = float(y) if y not in (None, "", "None") else None
        except Exception:
            x = y = None
        return x, y

    def _coerce_result(rr):
        """Accept 'result' or common boolean/text variants from entry app."""
        res = rr.get("result")
        if isinstance(res, str) and res.strip():
            r = res.strip().lower()
            if r in ("make", "made", "m", "fgm", "good", "true", "1"): return "Make"
            if r in ("miss", "x", "0", "false", "no", "fga", "missed"): return "Miss"
        if isinstance(res, (bool, int)):
            return "Make" if bool(res) else "Miss"
        # Other common field names
        for k in ("shot_result", "fg_result", "fg_make", "made", "is_make"):
            v = rr.get(k)
            if isinstance(v, str) and v.strip():
                r = v.strip().lower()
                if r in ("make", "made", "m", "fgm", "good", "true", "1"): return "Make"
                if r in ("miss", "x", "0", "false", "no", "missed"): return "Miss"
            if isinstance(v, (bool, int)):
                return "Make" if bool(v) else "Miss"
        # Try to infer from shorthand if present
        return result_from_shorthand(rr.get("possession", ""))

    def _coerce_drill_size(rr):
        """
        Normalize drill size to 'NvM'. Accepts '5v5', '5 v 5', '5x5', '3X3', etc.
        Prefer an existing rr['drill_size'] if present.
        """
        raw = (rr.get("drill_size") or
               rr.get("drill") or rr.get("practice_drill") or rr.get("drill_name") or "")
        s = str(raw).strip().lower().replace(" ", "").replace("x", "v")
        m = re.search(r"(\d+)\s*v\s*(\d+)", s)
        if m:
            try:
                return f"{int(m.group(1))}v{int(m.group(2))}"
            except Exception:
                pass
        m2 = re.search(r"\b\d+v\d+\b", s)
        return m2.group(0) if m2 else ""

    # Diagnostics
    total = 0
    with_xy = 0
    with_shot = 0

    for rr in (data or []):
        total += 1
        x, y = _coerce_xy(rr)
        res = _coerce_result(rr)

        if x is not None and y is not None:
            with_xy += 1

        # Only keep rows that are real shots within court bounds
        if x is not None and y is not None and (0 <= x <= COURT_W) and (0 <= y <= HALF_H) and res in ("Make", "Miss"):
            with_shot += 1

            # PBP sources (optional)
            pbp_names_src = (rr.get("play_by_play_names") or "")
            pbp_raw_src   = (rr.get("play_by_play") or "")
            pbp_src_for_roles = pbp_names_src or pbp_raw_src
            idx = rr.get("shot_index") or 1

            # Role extraction (safe)
            try:
                shooter_p, onball_def_p, assister_p, screen_ast_list_p, action_lines = extract_roles_for_shot(pbp_src_for_roles, idx)
            except Exception:
                shooter_p = onball_def_p = assister_p = None
                screen_ast_list_p = []
                action_lines = []

            try:
                _def_disp_from_shot_line = shot_defender_display(pbp_src_for_roles, idx)
            except Exception:
                _def_disp_from_shot_line = None
            if _def_disp_from_shot_line:
                onball_def_p = _def_disp_from_shot_line

            shooter    = rr.get("shooter")    or rr.get("shooter_raw")    or shooter_p
            onball_def = rr.get("defenders")  or rr.get("defender")       or rr.get("defenders_raw") or onball_def_p
            assister   = rr.get("assister")   or rr.get("assister_raw")   or assister_p
            screen_ast_list = (rr.get("screen_assisters") or rr.get("screen_assister") or
                               rr.get("screen_assisters_raw") or screen_ast_list_p or [])

            # Parse actions for type sets
            try:
                on_actions  = parse_onball_actions_from_pbp(action_lines, (screen_ast_list[0] if screen_ast_list else ""))
            except Exception:
                on_actions = []
            try:
                off_actions = parse_offball_actions_from_pbp(action_lines)
            except Exception:
                off_actions = []

            on_types  = { (a.get("type") or "").lower() for a in on_actions }
            off_types = { (a.get("type") or "").lower() for a in off_actions }

            # Defense label (tolerant)
            try:
                def_label = defense_label_for_shot(pbp_src_for_roles, idx) or "Man to Man"
            except Exception:
                def_label = "Man to Man"
            short = (rr.get("possession") or "")
            if "[" in short and "]" in short:
                m_zone = re.search(r"\b(\d(?:-\d){1,3})\s*\[", short)
                if m_zone:
                    def_label = f"{m_zone.group(1)} Zone"

            practice = rr.get("practice_date") or rr.get("practice") or rr.get("date") or ""
            drill    = rr.get("drill") or rr.get("practice_drill") or rr.get("drill_name") or ""
            drill_sz = _coerce_drill_size(rr)

            # Ensure fields expected downstream are present
            row_out = {
                **rr,
                "x": x, "y": y, "result": res,
                "is_three": _is_three(x, y),
                "practice_date_str": str(practice),
                "practice_date_obj": _parse_date_any(practice),
                "drill": drill,
                "drill_size": drill_sz,
                "shooter_raw": shooter,
                "defenders_raw": onball_def,
                "assister_raw": assister,
                "screen_assisters_raw": screen_ast_list or [],
                "onball_set": sorted(on_types),
                "offball_set": sorted(off_types),
                # names some older code expects:
                "onball_types": sorted(on_types),
                "offball_types": sorted(off_types),
                "defense_label": def_label,
                "pbp_src_for_roles": pbp_src_for_roles,
                "action_lines": action_lines,
            }
            # Provide a safe default for logic that checks possession type
            row_out.setdefault("possession_type", "shots")
            rows_for_plot.append(row_out)

    # One-line diagnostics to terminal
    try:
        print(f"[BBALL] Loaded rows: total={total} with_xy={with_xy} plottable_shots={with_shot}")
    except Exception:
        pass

    return rows_for_plot



# ---------- options cache (shooters/defenders/assisters/drills/sizes) ----------
def _uniq_sorted(seq):
    seen = set(); out = []
    for s in (seq or []):
        if not s:
            continue
        key = str(s).strip()
        l = key.lower()
        if l and l not in seen:
            seen.add(l); out.append(key)
    return sorted(out, key=lambda z: z.lower())

def _split_names(s_or_list):
    if not s_or_list:
        return []
    if isinstance(s_or_list, list):
        raw = []
        for v in s_or_list:
            raw += _split_names(v)
        return _uniq_sorted(raw)
    return _uniq_sorted(_split_fullname_list(str(s_or_list)))

def _build_options_cache(rows):
    drills = _uniq_sorted([r.get("drill") for r in rows])
    sizes  = _uniq_sorted([r.get("drill_size") for r in rows if r.get("drill_size")])
    shooters = _uniq_sorted(sum([_split_names(r.get("shooter_raw")) for r in rows], []))
    defenders = _uniq_sorted(sum([_split_names(r.get("defenders_raw")) for r in rows], []))
    assisters = _uniq_sorted(sum([_split_names(r.get("assister_raw")) for r in rows], []))
    screeners = _uniq_sorted(sum([_split_names(r.get("screen_assisters_raw")) for r in rows], []))
    return {
        "drill_full": [{"label": d, "value": d} for d in drills],
        "drill_size": [{"label": s, "value": s} for s in sizes],
        "shooter":    [{"label": n, "value": n} for n in shooters],
        "defenders":  [{"label": n, "value": n} for n in defenders],
        "assister":   [{"label": n, "value": n} for n in assisters],
        "screen_ast": [{"label": n, "value": n} for n in screeners],
    }

@callback(Output("options_cache","data"), Input("tabs","value"))
def init_options_cache(_tab):
    rows = _rows_full()
    return _build_options_cache(rows)

# ---------- populate main filter dropdowns from cache ----------
@callback(
    Output("flt_drill_full_shoot","options"),
    Output("flt_drill_size_shoot","options"),
    Output("flt_shooter","options"),
    Output("flt_defenders","options"),
    Output("flt_assister","options"),
    Output("flt_screen_assister","options"),
    Input("options_cache","data"),
    prevent_initial_call=False
)
def populate_main_filters(cache):
    cache = cache or {}
    return (
        cache.get("drill_full", []),
        cache.get("drill_size", []),
        cache.get("shooter", []),
        cache.get("defenders", []),
        cache.get("assister", []),
        cache.get("screen_ast", []),
    )

# ---------- CLEAR buttons ----------
@callback(
    Output("flt_date_range_shoot","start_date"),
    Output("flt_date_range_shoot","end_date"),
    Output("flt_drill_size_shoot","value"),
    Output("flt_drill_full_shoot","value"),
    Output("flt_shooter","value"),
    Output("flt_defenders","value"),
    Output("flt_assister","value"),
    Output("flt_screen_assister","value"),
    Output("flt_onball","value"),
    Output("flt_offball","value"),
    Output("flt_defense_shoot","value"),
    Input("btn_clear_shoot","n_clicks"),
    prevent_initial_call=True
)
def clear_shooting_filters(n):
    return (None, None, [], [], [], [], [], [], [], [], [])

@callback(
    Output("flt_date_range_stats","start_date"),
    Output("flt_date_range_stats","end_date"),
    Output("flt_drill_size_stats","value"),
    Output("flt_drill_full_stats","value"),
    Output("flt_defense_stats","value"),
    Input("btn_clear_stats","n_clicks"),
    prevent_initial_call=True
)
def clear_stats_filters(n):
    return (None, None, [], [], [])

# ---------- Build Shooting filters -> Store (JSON-safe; keys match Section 3) ----------
@callback(
    Output("filters_shoot_state","data"),
    Input("flt_date_range_shoot","start_date"),
    Input("flt_date_range_shoot","end_date"),
    Input("flt_drill_size_shoot","value"),
    Input("flt_drill_full_shoot","value"),
    Input("flt_shooter","value"),
    Input("flt_defenders","value"),
    Input("flt_assister","value"),
    Input("flt_screen_assister","value"),
    Input("flt_onball","value"),
    Input("flt_offball","value"),
    # on-ball subfilters
    Input("flt_onball_bh","value"),
    Input("flt_onball_bh_def","value"),
    Input("flt_onball_scr","value"),
    Input("flt_onball_scr_def","value"),
    Input("flt_onball_cov","value"),
    # off-ball subfilters (if present)
    Input("flt_offball_mover","value"),
    Input("flt_offball_mover_def","value"),
    Input("flt_offball_scr","value"),
    Input("flt_offball_scr_def","value"),
    Input("flt_defense_shoot","value"),
)
def build_shoot_filters(sd, ed, ds, df, sh, dfnd, ast, scr_ast, onb, offb,
                        on_bh, on_bh_def, on_scr, on_scr_def, on_cov,
                        off_mover, off_mover_def, off_scr, off_scr_def,
                        defense_vals):
    return {
        "date_start": _serialize_date(sd),
        "date_end":   _serialize_date(ed),
        "drill_size": _as_list(ds),
        "drill_full": _as_list(df),
        "shooter":    _as_list(sh),
        "defenders":  _as_list(dfnd),
        "assister":   _as_list(ast),
        "screen_ast": _as_list(scr_ast),
        "onball":     _as_list(onb),
        "offball":    _as_list(offb),
        # subfilters
        "on_bh": _as_list(on_bh), "on_bh_def": _as_list(on_bh_def),
        "on_screener": _as_list(on_scr), "on_screener_def": _as_list(on_scr_def),
        "on_cov": _as_list(on_cov),
        "off_mover": _as_list(off_mover), "off_mover_def": _as_list(off_mover_def),
        "off_screener": _as_list(off_scr), "off_screener_def": _as_list(off_scr_def),
        "defense": _as_list(defense_vals),
    }

# ---------- Shooting: charts + tiles from Store ----------
def _rows_to_possession_rows(rows):
    # Section 3’s collectors work fine with the enriched raw rows we kept in _rows_full()
    return rows

@callback(
    Output("shot_chart","figure"),
    Output("zone_chart","figure"),
    Output("shooting_stats_box","children"),
    Input("filters_shoot_state","data"),
)
def update_shooting_outputs(filters):
    rows = _rows_full()
    poss_rows = _rows_to_possession_rows(rows)
    shots = collect_shots_for_filters(poss_rows, filters or {})
    # Shot chart
    shot_fig = create_shot_chart(shots)
    # Zone chart
    zone_fig = create_zone_chart(shots)
    # Tiles
    totals = compute_shooting_totals(shots)
    metrics = [
        ("FGM", totals["FGM"]), ("FGA", totals["FGA"]), ("FG%", f"{totals['FGP']:.1f}%"),
        ("2PM", totals["2PM"]), ("2PA", totals["2PA"]), ("2P%", f"{totals['2PP']:.1f}%"),
        ("3PM", totals["3PM"]), ("3PA", totals["3PA"]), ("3P%", f"{totals['3PP']:.1f}%"),
    ]
    boxes = [
        html.Div([html.Div(name, style={"fontSize": "12px", "color": "#555"}),
                  html.Div(str(val), style={"fontSize": "18px", "fontWeight": 800})],
                 style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "8px",
                        "textAlign": "center", "background": "white"})
        for name, val in metrics
    ]
    tiles = html.Div(boxes, style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(80px, 1fr))", "gap": "8px"})
    return shot_fig, zone_fig, tiles

# ---------- Stats tab: build Store and table ----------
@callback(
    Output("filters_stats_state","data"),
    Input("flt_date_range_stats","start_date"),
    Input("flt_date_range_stats","end_date"),
    Input("flt_drill_size_stats","value"),
    Input("flt_drill_full_stats","value"),
    Input("flt_defense_stats","value"),
)
def build_stats_filters(sd, ed, ds, df, defv):
    return {
        "date_start": _serialize_date(sd),
        "date_end":   _serialize_date(ed),
        "drill_size": _as_list(ds),
        "drill_full": _as_list(df),
        "defense":    _as_list(defv),
    }

@callback(
    Output("stats_table","data"),
    Input("filters_stats_state","data")
)
def update_stats_table(filters):
    rows = _rows_full()
    table = compute_player_stats_table(rows, filters or {})
    return table

# ===================== ON-BALL SUBFILTERS =====================

_ONBALL_FACETS = {
    "pnr": [
        ("Ball handler",           "flt_onball_bh",      "Select ball handler",         False),
        ("Ball handler defender",  "flt_onball_bh_def",  "Select BH defender",          False),
        ("Screener(s)",            "flt_onball_scr",     "Select screener(s)",          True),
        ("Screener defender(s)",   "flt_onball_scr_def", "Select screener defender(s)", True),
        ("Coverage",               "flt_onball_cov",     "Select coverage",             True),
    ],
    "d":   [("Ball handler","flt_onball_bh","Select ball handler",False),
            ("Ball handler defender","flt_onball_bh_def","Select BH defender",False)],
    "p":   [("Post player","flt_onball_bh","Select post player",False),
            ("Primary defender","flt_onball_bh_def","Select defender",False)],
    "h":   [("Ball handler","flt_onball_bh","Select ball handler",False),
            ("Ball handler defender","flt_onball_bh_def","Select BH defender",False)],
    "dho": [("Ball handler","flt_onball_bh","Select ball handler",False),
            ("Ball handler defender","flt_onball_bh_def","Select BH defender",False),
            ("Handoffer/Screen", "flt_onball_scr","Select handoffer",True),
            ("Handoffer defender", "flt_onball_scr_def","Select handoffer defender",True)],
    "ho":  [("Ball handler","flt_onball_bh","Select ball handler",False),
            ("Ball handler defender","flt_onball_bh_def","Select BH defender",False),
            ("Handoff partner", "flt_onball_scr","Select partner",True),
            ("Partner defender","flt_onball_scr_def","Select partner defender",True)],
}
_PNR_LIKE = {"pnp","gst","slp","ho","kp","rj","rs"}

def _mk_sub_dd(label, cid, placeholder, multi=True):
    return _pill(label, dcc.Dropdown(id=cid, options=[], value=None, multi=multi, placeholder=placeholder,
                                     style={"minWidth":"160px"}))

@callback(Output("flt_onball_dynamic","children"),
          Input("flt_onball","value"), prevent_initial_call=False)
def show_onball_subfilters(selected):
    if not selected: return []
    sel = set(selected if isinstance(selected, list) else [selected])
    need_keys = set()
    for s in sel:
        if s in _ONBALL_FACETS: need_keys.add(s)
        elif s in _PNR_LIKE:    need_keys.add("pnr")
    by_id = {}
    for k in need_keys:
        for label, cid, ph, multi in _ONBALL_FACETS.get(k, []):
            by_id.setdefault(cid, (label, ph, multi))
    order = ["flt_onball_bh","flt_onball_bh_def","flt_onball_scr","flt_onball_scr_def","flt_onball_cov"]
    out = []
    for cid in order:
        if cid in by_id:
            label, ph, multi = by_id[cid]
            out.append(_mk_sub_dd(label, cid, ph, multi))
    return out

# ---------- Build available ON-BALL facet options ----------
def _iter_onball_facets_for_row(row):
    """Yield tuples ('bh'|'bh_def'|'scr'|'scr_def'|'cov', value) for this row across all actions."""
    lines = row.get("action_lines") or []
    sa_list = row.get("screen_assisters_raw") or []
    actions = parse_onball_actions_from_pbp(lines, (sa_list[0] if sa_list else ""))
    gid_key = row.get("group_id") or row.get("timestamp") or row.get("id")
    roster_full_list = _collect_roster_for_group([row], gid_key)
    def _n(n): return _norm_block(n, roster_full_list)

    for a in actions:
        if a.get("bh"):      yield ("bh", _n(a.get("bh")))
        if a.get("bh_def"):  yield ("bh_def", _n(a.get("bh_def")))
        if a.get("keeper"):  yield ("bh", _n(a.get("keeper")))
        if a.get("keeper_def"): yield ("bh_def", _n(a.get("keeper_def")))
        if a.get("screener"):    yield ("scr", _n(a.get("screener")))
        if a.get("screener_def"):yield ("scr_def", _n(a.get("screener_def")))
        for s in (a.get("screeners") or []):
            if s.get("name"): yield ("scr", _n(s.get("name")))
            if s.get("def"):  yield ("scr_def", _n(s.get("def")))
        for c in (a.get("coverages") or []):
            lbl = c.get("label")
            onto = _n(c.get("onto",""))
            yield ("cov", f"{lbl}→{onto}" if onto else lbl)

def _collect_onball_facet_options(rows):
    opts = {"bh": set(), "bh_def": set(), "scr": set(), "scr_def": set(), "cov": set()}
    for r in rows:
        for key, val in _iter_onball_facets_for_row(r):
            if val: opts[key].add(val)
    return {k: [{"label": v, "value": v} for v in sorted(vals, key=lambda z: z.lower())] for k, vals in opts.items()}

@callback(
    Output("flt_onball_bh", "options", allow_duplicate=True),
    Output("flt_onball_bh_def", "options", allow_duplicate=True),
    Output("flt_onball_scr", "options", allow_duplicate=True),
    Output("flt_onball_scr_def", "options", allow_duplicate=True),
    Output("flt_onball_cov", "options", allow_duplicate=True),
    Input("flt_date_range_shoot","start_date"),
    Input("flt_date_range_shoot","end_date"),
    Input("flt_drill_size_shoot","value"),
    Input("flt_drill_full_shoot","value"),
    Input("flt_shooter","value"),
    Input("flt_defenders","value"),
    Input("flt_assister","value"),
    Input("flt_screen_assister","value"),
    Input("flt_onball","value"),
    Input("flt_offball","value"),
    Input("flt_defense_shoot","value"),
    prevent_initial_call="initial_duplicate"
)
def populate_onball_subfilter_options(sd, ed, ds, df, shtr, defs, asts, scrs, onball, offball, def_sh):
    # Pre-filter by the current *base* filters so facet options are relevant
    rows = _rows_full()
    base = {
        "date_start": _serialize_date(sd),
        "date_end":   _serialize_date(ed),
        "drill_size": _as_list(ds),
        "drill_full": _as_list(df),
        "shooter":    _as_list(shtr),
        "defenders":  _as_list(defs),
        "assister":   _as_list(asts),
        "screen_ast": _as_list(scrs),
        "onball":     _as_list(onball),
        "offball":    _as_list(offball),
        "defense":    _as_list(def_sh),
        # zero out the facet subfilters while building options
        "on_bh": [], "on_bh_def": [], "on_screener": [], "on_screener_def": [], "on_cov": [],
        "off_mover": [], "off_mover_def": [], "off_screener": [], "off_screener_def": [],
    }
    prelim_shots = collect_shots_for_filters(rows, base)  # enforces base filters
    kept_rows = [s.get("row_ref") for s in prelim_shots if isinstance(s.get("row_ref"), dict)]
    coll = _collect_onball_facet_options(kept_rows or rows)
    return (coll["bh"], coll["bh_def"], coll["scr"], coll["scr_def"], coll["cov"])

# ===================== OFF-BALL SUBFILTERS =====================
_OFFBALL_FACETS = [
    ("Coming off",             "flt_offball_mover",     "Select mover",            False),
    ("Mover defender",         "flt_offball_mover_def", "Select mover defender",   False),
    ("Screener(s)",            "flt_offball_scr",       "Select screener(s)",      True),
    ("Screener defender(s)",   "flt_offball_scr_def",   "Select screener defender(s)", True),
]

@callback(Output("flt_offball_dynamic","children"),
          Input("flt_offball","value"), prevent_initial_call=False)
def show_offball_subfilters(selected):
    if not selected: return []
    # Any off-ball selection reveals same 4 subfilters
    out = []
    for label, cid, ph, multi in _OFFBALL_FACETS:
        out.append(_mk_sub_dd(label, cid, ph, multi))
    return out

def _iter_offball_facets_for_row(row):
    """Yield tuples ('mover'|'mover_def'|'scr'|'scr_def'|'cov', value) for this row."""
    actions = parse_offball_actions_from_pbp(row.get("action_lines") or [])
    gid_key = row.get("group_id") or row.get("timestamp") or row.get("id")
    roster_full_list = _collect_roster_for_group([row], gid_key)
    def N(n): return _norm_block(n, roster_full_list)
    for a in (actions or []):
        if a.get("coming_off"):      yield ("mover", N(a.get("coming_off")))
        if a.get("coming_off_def"):  yield ("mover_def", N(a.get("coming_off_def")))
        for s in (a.get("screeners") or []):
            if s.get("name"): yield ("scr", N(s.get("name")))
            if s.get("def"):  yield ("scr_def", N(s.get("def")))
        for c in (a.get("coverages") or []):
            lbl = c.get("label")
            onto = N(c.get("onto",""))
            yield ("cov", f"{lbl}→{onto}" if onto else lbl)

def _collect_offball_facet_options(rows, selected_types):
    sel = {s.lower() for s in _as_list(selected_types)} if selected_types else set()
    opts = {"mover": set(), "mover_def": set(), "scr": set(), "scr_def": set(), "cov": set()}
    for r in rows:
        for key, val in _iter_offball_facets_for_row(r):
            if sel and key in {"mover","mover_def","scr","scr_def"}:
                # (optional) filter by selected off-ball types if you later thread action type into this stage
                pass
            if val: opts[key].add(val)
    to_opts = lambda s: [{"label": v, "value": v} for v in sorted(list(s), key=lambda z: z.lower())]
    return {k: to_opts(opts[k]) for k in opts}

@callback(
    Output("flt_offball_mover", "options", allow_duplicate=True),
    Output("flt_offball_mover_def", "options", allow_duplicate=True),
    Output("flt_offball_scr", "options", allow_duplicate=True),
    Output("flt_offball_scr_def", "options", allow_duplicate=True),
    Output("flt_offball_cov", "options", allow_duplicate=True),
    Input("flt_date_range_shoot","start_date"),
    Input("flt_date_range_shoot","end_date"),
    Input("flt_drill_size_shoot","value"),
    Input("flt_drill_full_shoot","value"),
    Input("flt_shooter","value"),
    Input("flt_defenders","value"),
    Input("flt_assister","value"),
    Input("flt_screen_assister","value"),
    Input("flt_onball","value"),
    Input("flt_offball","value"),
    Input("flt_defense_shoot","value"),
    prevent_initial_call="initial_duplicate",
)
def populate_offball_subfilter_options(sd, ed, sz, dr, shtr, defs, asts, scrs, onball, offball, def_sh):
    rows = _rows_full()
    base = {
        "date_start": _serialize_date(sd),
        "date_end":   _serialize_date(ed),
        "drill_size": _as_list(sz),
        "drill_full": _as_list(dr),
        "shooter":    _as_list(shtr),
        "defenders":  _as_list(defs),
        "assister":   _as_list(asts),
        "screen_ast": _as_list(scrs),
        "onball":     _as_list(onball),
        "offball":    _as_list(offball),
        "defense":    _as_list(def_sh),
        # zero out subfilters while building options
        "on_bh": [], "on_bh_def": [], "on_screener": [], "on_screener_def": [], "on_cov": [],
        "off_mover": [], "off_mover_def": [], "off_screener": [], "off_screener_def": [],
    }
    prelim_shots = collect_shots_for_filters(rows, base)
    kept_rows = [s.get("row_ref") for s in prelim_shots if isinstance(s.get("row_ref"), dict)]
    coll = _collect_offball_facet_options(kept_rows or rows, offball)
    return (coll["mover"], coll["mover_def"], coll["scr"], coll["scr_def"], coll["cov"])

# ---------- Zone legend labels ----------
@callback(
    Output("legend_close_low","children"),
    Output("legend_close_mid","children"),
    Output("legend_close_high","children"),
    Output("legend_mid_low","children"),
    Output("legend_mid_mid","children"),
    Output("legend_mid_high","children"),
    Output("legend_three_low","children"),
    Output("legend_three_mid","children"),
    Output("legend_three_high","children"),
    Input("zone_chart","figure"),
    prevent_initial_call=False
)
def fill_zone_legend_labels(_):
    # Must mirror _rgba_for_zone bins
    return (
        "≤ 50%", "51–60%", "≥ 61%",
        "≤ 35%", "36–45%", "≥ 46%",
        "≤ 25%", "26–35%", "≥ 36%",
    )





#--------------------------------------Section 7---------------------------------------------------

# ---------- helper: coerce any dropdown/date/state into a clean list/None ----------
def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [x for x in v if x not in (None, "")]
    return [v] if v not in ("", None) else []

# Safe default if earlier constants aren't loaded
if "_PNR_LIKE" not in globals():
    _PNR_LIKE = set()

# ---------- SHIMS / BACK-COMPAT (fix undefined-name warnings) ----------
def _rows_to_shots(rows):
    try:
        return rows_to_shots(rows or [])
    except Exception:
        return rows or []

def _aggregate_stats_table(rows):
    try:
        poss = _rows_to_possession_rows(rows or [])
    except Exception:
        poss = rows or []
    try:
        return compute_player_stats_table(poss, {}) or []
    except Exception:
        return []

def _shooting_tiles(rows):
    try:
        shots = _rows_to_shots(rows)
        totals = compute_shooting_totals(shots)
        metrics = [
            ("FGM", totals.get("FGM", 0)), ("FGA", totals.get("FGA", 0)), ("FG%", f"{totals.get('FGP', 0):.1f}%"),
            ("2PM", totals.get("2PM", 0)), ("2PA", totals.get("2PA", 0)), ("2P%", f"{totals.get('2PP', 0):.1f}%"),
            ("3PM", totals.get("3PM", 0)), ("3PA", totals.get("3PA", 0)), ("3P%", f"{totals.get('3PP', 0):.1f}%"),
        ]
        return [
            html.Div([
                html.Div(name,  style={"fontSize": "12px", "color": "#555"}),
                html.Div(str(val), style={"fontSize": "18px", "fontWeight": "600"})
            ], style={"display": "inline-block", "padding": "8px 12px", "margin": "4px",
                      "border": "1px solid #eee", "borderRadius": "8px"})
            for (name, val) in metrics
        ]
    except Exception:
        return [html.Div("No data", style={"padding":"8px 12px","border":"1px solid #eee","borderRadius":"8px"})]

def _filter_rows(rows, sd, ed, sz, dr, sh, df, ast, sa, onb, offb, deflbls, for_stats_tab=False):
    try:
        if 'filter_rows' in globals() and callable(globals()['filter_rows']):
            return globals()['filter_rows'](
                rows, sd, ed, sz, dr, sh, df, ast, sa, onb, offb, deflbls, for_stats_tab=for_stats_tab
            )
    except Exception:
        pass

    rows = rows or []

    def _date_ok(r):
        d = r.get("practice_date_obj")
        if sd and d and isinstance(d, date) and d < sd:  return False
        if ed and d and isinstance(d, date) and d > ed:  return False
        return True

    def _any_match_strlist(field_val, wanted):
        if not wanted:
            return True
        vals = []
        if isinstance(field_val, (list, tuple, set)):
            vals = [str(x) for x in field_val if x not in (None, "")]
        elif field_val not in (None, ""):
            vals = [str(field_val)]
        have = {v.strip().lower() for v in vals}
        need = {str(w).strip().lower() for w in wanted if w not in (None, "")}
        return bool(have & need)

    def _name_split(raw):
        try:
            return _split_names(raw)
        except Exception:
            if not raw:
                return []
            return [raw] if isinstance(raw, str) else list(raw)

    out = []
    for r in rows:
        if not _date_ok(r):
            continue
        if sz and str(r.get("drill_size","")) not in set(sz):
            continue
        if dr and str(r.get("drill","")) not in set(dr):
            continue
        if sh and not _any_match_strlist(_name_split(r.get("shooter_raw")), sh):
            continue
        if df and not _any_match_strlist(_name_split(r.get("defenders_raw")), df):
            continue
        if ast and not _any_match_strlist(_name_split(r.get("assister_raw")), ast):
            continue
        if sa and not _any_match_strlist(_name_split(r.get("screen_assisters_raw")), sa):
            continue
        if deflbls:
            def_val = (r.get("defense_label") or r.get("defense") or "").strip()
            if def_val and def_val not in deflbls:
                continue
        if onb:
            have_on = [str(x).lower() for x in (r.get("onball_types") or [])]
            if not (set(have_on) & set(onb)) and not ("pnr" in onb and (set(have_on) & set(_PNR_LIKE))):
                continue
        if offb:
            have_off = [str(x).lower() for x in (r.get("offball_types") or [])]
            if not set(have_off) & set(offb):
                continue
        out.append(r)
    return out


# ---------- apply selected on-ball subfilters ----------
def _row_matches_onball_facets(row, need_types, bh_v, bhd_v, scr_v, scrd_v, cov_v):
    """True if row contains a required on-ball action matching all selected facets."""
    if not need_types:
        return True

    lines = row.get("action_lines") or []
    sa_list = row.get("screen_assisters_raw") or []
    actions = parse_onball_actions_from_pbp(lines, (sa_list[0] if sa_list else ""))

    gid_key = row.get("group_id") or row.get("timestamp") or row.get("id")
    roster = _collect_roster_for_group([row], gid_key)
    def N(x): return _norm_block(x, roster)

    sel_bh   = {N(x) for x in _as_list(bh_v)}
    sel_bhd  = {N(x) for x in _as_list(bhd_v)}
    sel_scr  = {N(x) for x in _as_list(scr_v)}
    sel_scrd = {N(x) for x in _as_list(scrd_v)}
    sel_cov  = set(_as_list(cov_v))

    def match_cov(covs):
        if not sel_cov:
            return True
        have = set()
        for c in (covs or []):
            lbl = c.get("label")
            onto = N(c.get("onto", ""))
            have.add(f"{lbl}→{onto}" if onto else lbl)
        return bool(have & sel_cov)

    pnr_like = _PNR_LIKE if "_PNR_LIKE" in globals() else set()
    for a in actions:
        t = (a.get("type") or "").lower()
        if t in need_types or (t in pnr_like and "pnr" in need_types):
            bh   = N(a.get("bh") or a.get("keeper")  or a.get("giver")     or a.get("receiver"))
            bh_d = N(a.get("bh_def") or a.get("keeper_def") or a.get("giver_def") or a.get("receiver_def"))

            scrs = set(); scrd = set()
            if a.get("screener"):     scrs.add(N(a.get("screener")))
            if a.get("screener_def"): scrd.add(N(a.get("screener_def")))
            for s in (a.get("screeners") or []):
                if s.get("name"): scrs.add(N(s.get("name")))
                if s.get("def"):  scrd.add(N(s.get("def")))

            if sel_bh   and (bh   not in sel_bh):   continue
            if sel_bhd  and (bh_d not in sel_bhd):  continue
            if sel_scr  and not (scrs  & sel_scr):  continue
            if sel_scrd and not (scrd  & sel_scrd): continue
            if not match_cov(a.get("coverages")):   continue
            return True
    return False


# ====== (DISABLED) duplicate option population — handled in Section 6 ======
def _disabled_populate_filter_options(_tab_value):
    # Kept for reference only; Section 6 provides the live version.
    rows = _rows_full()
    drill_sizes = _uniq_sorted([r.get("drill_size") for r in rows])
    drills_full = _uniq_sorted([r.get("drill") for r in rows])

    shooters  = _uniq_sorted(sum([_split_names(r.get("shooter_raw"))          for r in rows], []))
    defenders = _uniq_sorted(sum([_split_names(r.get("defenders_raw"))        for r in rows], []))
    assisters = _uniq_sorted(sum([_split_names(r.get("assister_raw"))         for r in rows], []))
    screeners = _uniq_sorted(sum([_split_names(r.get("screen_assisters_raw")) for r in rows], []))

    dates = sorted([d for d in [r.get("practice_date_obj") for r in rows] if isinstance(d, date)])
    dmin = dates[0] if dates else None
    dmax = dates[-1] if dates else None
    def _opts(lst): return [{"label": x, "value": x} for x in lst]
    return (_opts(drill_sizes), _opts(drills_full),
            _opts(shooters), _opts(defenders), _opts(assisters), _opts(screeners),
            _opts(drill_sizes), _opts(drills_full),
            dmin, dmax, dmin, dmax)


# ====== (DISABLED) duplicate clear button — handled in Section 6 ======
def _disabled_clear_shoot_filters(_nc):
    return (None, None, [], [], [], [], [], [], [], [], [],
            None, None, None, None, None,
            None, None, None, None, None)


# ====== (DISABLED) duplicate “compute all” — charts & stats are handled in Section 6 ======
def _disabled_compute_all(*args, **kwargs):
    return no_update, no_update, no_update, no_update, no_update


# ---------------- Legend text population (DISABLED: handled in Section 6) ----------------
def _disabled_update_zone_legend(_zone_fig):
    close_bins = ("0–50%", "51–60%", "61%+")
    mid_bins   = ("0–35%", "36–45%", "46%+")
    three_bins = ("0–25%", "26–35%", "36%+")
    return (*close_bins, *mid_bins, *three_bins)


# ===== Off-Ball subfilter OPTIONS (DISABLED: handled in Section 6) =====
def _iter_offball_facets_for_row(row):
    lines = row.get("action_lines") or []
    actions = parse_offball_actions_from_pbp(lines)
    gid_key = row.get("group_id") or row.get("timestamp") or row.get("id")
    roster_full_list = _collect_roster_for_group([row], gid_key)
    def N(n): return _norm_block(n, roster_full_list)

    for a in actions:
        if a.get("coming_off"):      yield ("mover", N(a.get("coming_off")))
        if a.get("coming_off_def"):  yield ("mover_def", N(a.get("coming_off_def")))
        for s in (a.get("screeners") or []):
            if s.get("name"): yield ("scr", N(s.get("name")))
            if s.get("def"):  yield ("scr_def", N(s.get("def")))
        for c in (a.get("coverages") or []):
            lbl = c.get("label")
            onto = N(c.get("onto",""))
            yield ("cov", f"{lbl}→{onto}" if onto else lbl)

def _collect_offball_facet_options(rows):
    opts = {"mover": set(), "mover_def": set(), "scr": set(), "scr_def": set(), "cov": set()}
    for r in rows:
        for key, val in _iter_offball_facets_for_row(r):
            if val: opts[key].add(val)
    return {k: [{"label": v, "value": v} for v in sorted(vals, key=lambda z: z.lower())] for k, vals in opts.items()}

def _disabled_populate_offball_subfilter_options(sd, ed, sz, dr, shtr, defs, asts, scrs, onball, offball, def_sh):
    rows = _rows_full()
    prelim = _filter_rows(
        rows,
        _parse_date_any(sd), _parse_date_any(ed),
        (sz or []), (dr or []),
        (shtr or []), (defs or []), (asts or []), (scrs or []),
        (onball or []), (offball or []), (def_sh or []),
        for_stats_tab=False
    )
    coll = _collect_offball_facet_options(prelim)
    return (coll["mover"], coll["mover_def"], coll["scr"], coll["scr_def"], coll["cov"])


# ===== details panel (ACTIVE) =====
@callback(
    [Output("shot_details", "children"),
     Output("sel_pos", "data")],
    [Input("shot_chart", "clickData"),
     Input({"type":"close_details","idx":ALL}, "n_clicks")],
    prevent_initial_call=False
)
def show_shot_details(clickData, close_clicks):
    ctx = callback_context
    if ctx and ctx.triggered and "close_details" in ctx.triggered[0]["prop_id"]:
        return ("", [])

    try:
        if not clickData or "points" not in clickData or not clickData["points"]:
            return no_update, no_update

        p = clickData["points"][0]
        x_clicked, y_clicked = float(p["x"]), float(p["y"])

        rows_for_plot = _rows_full()
        if not rows_for_plot:
            return no_update, no_update

        def dist2(r):
            return (r["x"] - x_clicked)**2 + (r["y"] - y_clicked)**2
        r = sorted(rows_for_plot, key=dist2)[0]

        idx = r.get("shot_index")
        total = r.get("group_size")
        gid_key = r.get("group_id") or r.get("timestamp") or r.get("id")
        def same_possession(rr):
            key_rr = rr.get("group_id") or rr.get("timestamp") or rr.get("id")
            return key_rr == gid_key and (rr.get("possession_type") == "shots")
        if not total:
            total = sum(1 for rr in rows_for_plot if same_possession(rr))
        shot_num_display = f"{int(idx)}/{int(total)}" if (idx and total) else (str(idx) if idx else "")
        pos_coords = [(rr["x"], rr["y"]) for rr in rows_for_plot if same_possession(rr)]

        pbp_names_src = (r.get("play_by_play_names") or "")
        pbp_raw_src   = (r.get("play_by_play") or "")
        pbp_src_for_roles = pbp_names_src or pbp_raw_src

        shooter, onball_def, assister, screen_ast_list, action_lines = extract_roles_for_shot(pbp_src_for_roles, idx or 1)
        _def_disp_from_shot_line = shot_defender_display(pbp_src_for_roles, idx or 1)
        if _def_disp_from_shot_line:
            onball_def = _def_disp_from_shot_line
        _shot_out_of_help = shot_has_help(pbp_src_for_roles, idx or 1)

        onball_actions = parse_onball_actions_from_pbp(action_lines, (screen_ast_list[0] if screen_ast_list else ""))
        onball_actions = _patch_bring_over_halfcourt(onball_actions, pbp_src_for_roles)
        offball_actions = parse_offball_actions_from_pbp(action_lines)

        if not assister:
            for a in reversed(onball_actions):
                if a.get("type") in ("ho","dho") and a.get("receiver","").lower() == (shooter or "").lower():
                    assister = a.get("giver",""); break

        roster_full_list = _collect_roster_for_group(rows_for_plot, gid_key)

        shooter = _norm_block(shooter, roster_full_list)
        onball_def = _norm_block(onball_def, roster_full_list)
        assister = _norm_block(assister, roster_full_list)

        _base_defs = _split_fullname_list(onball_def) or ([onball_def] if onball_def else [])
        _base_defs = [_norm_block(nm, roster_full_list) for nm in _base_defs if nm]
        onball_def_display = _decorate_defenders_with_rotation(", ".join(_base_defs), shooter, pbp_src_for_roles)
        _def_list_for_count = _split_fullname_list(onball_def_display)
        _multi_defenders = len(_def_list_for_count) >= 2

        uniq_screens = []
        seen_sa = set()
        for nm in (screen_ast_list or []):
            for piece in _split_fullname_list(nm):
                nm2 = _norm_block(piece, roster_full_list)
                low = (nm2 or "").lower()
                if low and low not in seen_sa:
                    uniq_screens.append(nm2); seen_sa.add(low)
        screen_ast_list = uniq_screens

        for a in onball_actions:
            for k in ("bh","bh_def","screener","screener_def","giver","giver_def","receiver","receiver_def","keeper","keeper_def","intended","intended_def"):
                if k in a: a[k] = _norm_block(a[k], roster_full_list)
            if a.get("screeners"):
                for s in a["screeners"]:
                    s["name"] = _norm_block(s.get("name",""), roster_full_list)
                    s["def"]  = _norm_block(s.get("def",""), roster_full_list)
            if a.get("coverages"):
                for c in a["coverages"]:
                    if "onto" in c and c["onto"]:
                        c["onto"] = _norm_block(c["onto"], roster_full_list)

        for a in offball_actions:
            for k in ("coming_off","coming_off_def"):
                if k in a: a[k] = _norm_block(a[k], roster_full_list)
            for s in a.get("screeners", []):
                s["name"] = _norm_block(s.get("name",""), roster_full_list)
                s["def"]  = _norm_block(s.get("def",""), roster_full_list)
            if a.get("coverages"):
                for c in a["coverages"]:
                    if "onto" in c and c["onto"]:
                        c["onto"] = _norm_block(c["onto"], roster_full_list)

        for a in onball_actions:
            if a.get("type") == "h":
                a["coverages"] = []

        _screen_handoff_types = {"pnr","pnp","rs","slp","gst","rj","dho","ho","kp"}
        for a in onball_actions:
            t = (a.get("type") or "").lower()
            if t not in _screen_handoff_types and a.get("coverages"):
                a["coverages"] = [c for c in a["coverages"] if (c.get("cov") != "sw")]

        sa_phrase = "screen assist" in ((pbp_names_src or pbp_raw_src or "").lower())
        if sa_phrase:
            scr_from_actions = []
            for a in onball_actions:
                if a.get("type") in ("pnr", "pnp", "rs"):
                    if a.get("screeners"):
                        scr_from_actions.extend([s.get("name","") for s in a["screeners"] if s.get("name")])
                    elif a.get("screener"):
                        scr_from_actions.append(a.get("screener"))
            scr_from_actions = [n for n in scr_from_actions if n]
            scr_seen = set()
            scr_from_actions = [n for n in scr_from_actions if not (n.lower() in scr_seen or scr_seen.add(n.lower()))]
            if scr_from_actions and len(screen_ast_list) < len(scr_from_actions):
                screen_ast_list = scr_from_actions

        def line(label, val):
            return html.Div([html.Span(f"{label}:", style={"fontWeight":600, "marginRight":"6px"}),
                             html.Span((val or ""), style={"whiteSpace":"pre-wrap"})],
                            style={"marginBottom":"6px"})

        def mini_table(rows):
            return html.Table(
                [html.Tbody([html.Tr([html.Td(html.B(lbl + ":"), style={"paddingRight":"8px","verticalAlign":"baseline"}),
                                       html.Td(val, style={"verticalAlign":"baseline"})]) for (lbl, val) in rows])],
                style={"borderCollapse":"collapse","width":"auto","margin":"6px 0"}
            )

        def cov_text(cov):
            if not cov: return ""
            out = []
            for c in cov:
                onto = c.get("onto","")
                lbl = c.get("label","")
                out.append(f"{lbl} (onto {onto})" if onto else lbl)
            return "; ".join(out)

        def action_block(a):
            t = a.get("type","").lower()
            blocks = [line("Action", a.get("label",""))]
            if t in ("h","d","kp"):
                blocks += [line("Ball Handler", a.get("bh", a.get("keeper",""))),
                           line("Ball Handler Defender", a.get("bh_def", a.get("keeper_def","")))]
                if t == "kp":
                    if a.get("intended"): blocks.append(line("Intended receiver", a.get("intended","")))
                    if a.get("intended_def"): blocks.append(line("Intended defender", a.get("intended_def","")))
                    if a.get("coverages"): blocks.append(line("Coverage", cov_text(a.get("coverages"))))
                return html.Div(blocks, style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})
            if t == "p":
                rows = [("Action", a.get("label","")), ("Posting up", a.get("bh","")), ("Defending Post up", a.get("bh_def",""))]
                if a.get("coverages"): rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})
            if t in ("pnr","pnp","rs"):
                rows = [("Action", a.get("label","")), ("Ball Handler", a.get("bh","")), ("Ball Handler Defender", a.get("bh_def",""))]
                scr_list = a.get("screeners") or []
                if scr_list:
                    rows += [("Screener(s)", ", ".join(s.get("name","") for s in scr_list)),
                             ("Screener(s) Defender(s)", ", ".join(s.get("def","") for s in scr_list))]
                else:
                    rows += [("Screener", a.get("screener","")), ("Screener defender", a.get("screener_def",""))]
                if a.get("coverages"): rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})
            if t in ("slp","gst","rj"):
                rows = [("Action", a.get("label","")), ("Ball Handler", a.get("bh","")), ("Ball Handler Defender", a.get("bh_def","")),
                        ("Screener", a.get("screener","")), ("Screener defender", a.get("screener_def",""))]
                if a.get("coverages"): rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})
            if t in ("dho","ho"):
                rows = [("Action", a.get("label","")), ("Giver", a.get("giver","")), ("Giver defender", a.get("giver_def","")),
                        ("Receiver", a.get("receiver","")), ("Receiver defender", a.get("receiver_def",""))]
                if a.get("coverages"): rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})
            if t in ("bd","pn","fl","bk","awy","ucla","crs","wdg","rip","stg","ivs","elv"):
                rows = [("Action", (a.get("label","") or "").lower())]
                if t == "bd":
                    rows += [("Cutter", a.get("coming_off","")), ("Cutter defender", a.get("coming_off_def",""))]
                else:
                    rows += [("Coming off screen", a.get("coming_off","")), ("Defender on coming-off player", a.get("coming_off_def",""))]
                    scr_txt = ", ".join(s.get("name","") for s in (a.get("screeners") or []))
                    scr_def_txt = ", ".join(s.get("def","") for s in (a.get("screeners") or []))
                    if scr_txt: rows.append(("Screener(s)", scr_txt))
                    if scr_def_txt: rows.append(("Screener(s) Defender(s)", scr_def_txt))
                if a.get("coverages"): rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})
            return html.Div(blocks, style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

        practice = r.get("practice_date_str") or ""
        drill = r.get("drill") or ""

        try:
            def_label = defense_label_for_shot(pbp_src_for_roles, idx or 1) or "Man to Man"
        except Exception:
            def_label = "Man to Man"
        _short = (r.get("possession") or "")
        if "[" in _short and "]" in _short:
            m_zone = re.search(r"\b(\d(?:-\d){1,3})\s*\[", _short)
            def_label = f"{m_zone.group(1)} Zone" if m_zone else def_label

        try:
            _op, _dp = participants_for_possession(pbp_src_for_roles, r.get("possession",""))
        except Exception:
            _op, _dp = ([], [])

        roster_full_list = _collect_roster_for_group(rows_for_plot, gid_key)

        op_list = []
        seen_op = set()
        for nm in (_op or []):
            nm = _strip_trailing_modifiers(nm)
            nn = _norm_block(nm, roster_full_list)
            k = (nn or "").lower()
            if nn and k not in seen_op: seen_op.add(k); op_list.append(nn)

        dp_list = []
        seen_dp = set()
        for nm in (_dp or []):
            nm = _strip_trailing_modifiers(nm)
            nn = _norm_block(nm, roster_full_list)
            k = (nn or "").lower()
            if nn and k not in seen_dp: seen_dp.add(k); dp_list.append(nn)

        header_top = html.Div([
            html.Div([
                html.Span("Shot details", style={"fontWeight":700,"fontSize":"18px","marginRight":"8px"}),
                html.Span(f"({shot_num_display})" if shot_num_display else "", style={"color":"#666"}),
                html.Span(f" • Defense: {def_label}", style={"marginLeft":"6px","color":"#444"}),
                html.Span(f" • Result: {r.get('result','')}", style={"marginLeft":"6px","color":"#444"}),
            ]),
            html.Div([
                html.Span(f"Practice: {practice}" if practice else "", style={"marginRight":"16px","color":"#555"}),
                html.Span(f"Drill: {drill}" if drill else "", style={"color":"#555"}),
            ]),
            html.Div([
                html.Span("Offensive Players: ", style={"fontWeight":600,"marginRight":"6px","color":"#333"}),
                html.Span(", ".join(op_list) if op_list else "—", style={"color":"#333"})
            ], style={"marginTop":"2px"}),
            html.Div([
                html.Span("Defensive Players: ", style={"fontWeight":600,"marginRight":"6px","color":"#333"}),
                html.Span(", ".join(dp_list) if dp_list else "—", style={"color":"#333"})
            ], style={"marginTop":"2px"})
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"baseline","gap":"10px","flexWrap":"wrap"})

        top_close = html.Div(
            html.Button("Close", id={"type":"close_details","idx":0}, n_clicks=0,
                        style={"padding":"6px 10px","borderRadius":"8px","border":"1px solid #aaa","background":"white"}),
            style={"display":"flex","justifyContent":"flex-end","marginBottom":"6px"}
        )

        header = html.Div(header_top, style={"display":"flex","justifyContent":"space-between","alignItems":"center"})

        ident_rows = [("Shooter", shooter),
                      ("Defender(s)" if _multi_defenders else "Defender", onball_def_display)]
        if assister: ident_rows.append(("Assisted by", assister))
        if screen_ast_list: ident_rows.append(("Screen Assist", ", ".join(screen_ast_list)))
        if _shot_out_of_help and not _multi_defenders: ident_rows.append(("Out of help", "Yes"))

        specials_rows = special_stats_with_pbp_blocks(r.get("possession",""), pbp_src_for_roles) or []

        def _clean_display_name(tok: str) -> str:
            s = (tok or "").strip()
            s = re.sub(r"^(?:rebound|defensive\s+rebound|offensive\s+rebound|"
                       r"turnover|live\s+ball\s+turnover|dead\s+ball\s+turnover|"
                       r"steal|deflection|block|offensive\s+foul|defensive\s+foul|"
                       r"from|by|commits|grabs|the)\s+", "", s, flags=re.IGNORECASE)
            m = re.search(rf"({_CAP}(?:\s+{_CAP})?)\s*$", s)
            if m: s = m.group(1)
            return _norm_block(s, roster_full_list)

        _spec_by_label = {}
        for row in (specials_rows or []):
            lbl = row.get("label", "")
            raw_players = row.get("players", []) or []
            names = []
            seen = set()
            for p in raw_players:
                nn = _clean_display_name(p)
                k = (nn or "").lower()
                if nn and k not in seen:
                    names.append(nn); seen.add(k)
            if lbl and names:
                _spec_by_label[lbl] = names

        txt_for_infer = _clean_frag(pbp_names_src or pbp_raw_src or "")
        if txt_for_infer:
            _DEFLECT_SUBJ_RE = re.compile(rf"({_FULLNAME})\s+deflects\b", re.IGNORECASE)
            def_names = [ _clean_display_name(m.group(1)) for m in re.finditer(_DEFLECT_SUBJ_RE, txt_for_infer) ]
            if def_names:
                base = _spec_by_label.get("Deflection", []); base_l = {b.lower() for b in base}
                _spec_by_label["Deflection"] = base + [n for n in def_names if n and n.lower() not in base_l]

            _STEAL_SUBJ_RE    = re.compile(rf"({_FULLNAME})\s+(?:steal(?:s|ed)?|stole)\b", re.IGNORECASE)
            _STEAL_PASSIVE_RE = re.compile(rf"\bsteal(?:s|ed)?\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
            stl_names = [ _clean_display_name(m.group(1)) for m in re.finditer(_STEAL_SUBJ_RE, txt_for_infer) ]
            stl_names += [ _clean_display_name(m.group(1)) for m in re.finditer(_STEAL_PASSIVE_RE, txt_for_infer) ]
            if stl_names:
                base = _spec_by_label.get("Steal", []); base_l = {b.lower() for b in base}
                _spec_by_label["Steal"] = base + [n for n in stl_names if n and n.lower() not in base_l]

            _OFF_FOUL_SUBJ_RE = re.compile(rf"({_FULLNAME})\s+commits\s+an?\s+offensive\s+foul\b", re.IGNORECASE)
            _OFF_FOUL_BY_RE   = re.compile(rf"\boffensive\s+foul\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
            off_foul_names = [ _clean_display_name(m.group(1)) for m in re.finditer(_OFF_FOUL_SUBJ_RE, txt_for_infer) ]
            off_foul_names += [ _clean_display_name(m.group(1)) for m in re.finditer(_OFF_FOUL_BY_RE, txt_for_infer) ]
            if off_foul_names:
                base = _spec_by_label.get("Offensive Foul", []); base_l = {b.lower() for b in base}
                _spec_by_label["Offensive Foul"] = base + [n for n in off_foul_names if n and n.lower() not in base_l]

            _DEF_FOUL_SUBJ_RE = re.compile(rf"({_FULLNAME})\s+commits\s+an?\s+defensive\s+foul\b", re.IGNORECASE)
            _DEF_FOUL_BY_RE   = re.compile(rf"\bdefensive\s+foul\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
            def_foul_names = [ _clean_display_name(m.group(1)) for m in re.finditer(_DEF_FOUL_SUBJ_RE, txt_for_infer) ]
            def_foul_names += [ _clean_display_name(m.group(1)) for m in re.finditer(_DEF_FOUL_BY_RE, txt_for_infer) ]
            if def_foul_names:
                base = _spec_by_label.get("Defensive Foul", []); base_l = {b.lower() for b in base}
                _spec_by_label["Defensive Foul"] = base + [n for n in def_foul_names if n and n.lower() not in base_l]

        if _spec_by_label.get("Block"):
            ident_rows.append(("Block", ", ".join(_spec_by_label["Block"])))
        _special_order = ["Defensive Rebound","Offensive Rebound","Deflection","Steal",
                          "Live Ball Turnover","Dead Ball Turnover","Defensive Foul",
                          "Offensive Foul","Block"]
        for lbl in _special_order:
            if lbl == "Block": continue
            ppl = _spec_by_label.get(lbl, [])
            if ppl: ident_rows.append((lbl, ", ".join(ppl)))

        identity = html.Div([mini_table(ident_rows)],
                            style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px","background":"#fafafa"})

        oblocks = [action_block(a) for a in onball_actions]
        fblocks = [action_block(a) for a in offball_actions]

        pre_lines = [f"Shorthand:\n  {(r.get('possession') or '').strip()}"]
        if (pbp_names_src or "").strip():
            pre_lines.append(f"\nPlay-by-play:\n{(pbp_names_src or '').strip()}")

        pre = html.Div([
            html.Pre(
                "\n".join(pre_lines),
                style={
                    "background": "#eef2ff",
                    "color": "#111827",
                    "padding": "10px",
                    "borderRadius": "8px",
                    "whiteSpace": "pre-wrap",
                    "border": "1px solid #c7d2fe",
                    "marginBottom": "8px",
                }
            )
        ])

        return html.Div([top_close, pre, header, identity,
                         html.Div([html.Div("On-ball Actions", style={"fontWeight":700,"margin":"6px 0"}), *oblocks]) if oblocks else html.Div(),
                         html.Div([html.Div("Off-ball Actions", style={"fontWeight":700,"margin":"10px 0 6px"}), *fblocks]) if fblocks else html.Div()
                         ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"10px","background":"#fff"}), pos_coords

    except Exception as e:
        return html.Div(f"Error: {e}", style={"color":"crimson"}), []


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
