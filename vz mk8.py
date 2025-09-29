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

# =========================
# Config
# =========================
DATA_PATH = os.environ.get("BBALL_DATA", "data/possessions.json")

# NEW: match the entry app’s roster location scheme
BASE_DIR    = os.path.dirname(DATA_PATH) or "."
ROSTER_PATH = os.path.join(BASE_DIR, "roster.json")
# --- NEW: practices metadata (written by sc_mark_6 when starting a practice)
PRACTICES_PATH = os.path.join(BASE_DIR, "practices.json")

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

def safe_load_data():
    """Load and cache the *raw possession rows* with caching on failure.

    mk5 improvement:
      - Only re-parse when the DATA_PATH's mtime changes.
      - If file is missing or mid-write/corrupt, return last good cache.
      - IMPORTANT: returns the original possession ROWS (dicts), not {x,y,result} shots.
      - FIX: always return a shallow copy so Dash callbacks see a new object and recompute stats.
    """
    global CACHED_DATA, _DATA_LAST_MTIME, _CACHED_DATA_BY_MTIME

    # If the file doesn't exist, keep serving whatever we had last.
    if not os.path.exists(DATA_PATH):
        return list(_CACHED_DATA_BY_MTIME or CACHED_DATA)

    try:
        mtime = os.path.getmtime(DATA_PATH)
    except Exception:
        # If we can't stat it, keep the last known good.
        return list(_CACHED_DATA_BY_MTIME or CACHED_DATA)

    # If file hasn't changed, return the mtime-cached parse (super fast).
    if mtime == _DATA_LAST_MTIME and _CACHED_DATA_BY_MTIME:
        # return a shallow copy to trigger downstream recompute without reparsing
        return list(_CACHED_DATA_BY_MTIME)

    # Otherwise, try to (re)load and parse
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both {"rows":[...]} and [...] shapes
        rows = data.get("rows", data) if isinstance(data, dict) else (data or [])
        # Ensure it's a list[dict]
        rows = [r for r in (rows or []) if isinstance(r, dict)]

        # update both caches on success
        CACHED_DATA = rows
        _CACHED_DATA_BY_MTIME = rows
        _DATA_LAST_MTIME = mtime
        # return a shallow copy so callbacks see a new object
        return list(rows)

    except Exception:
        # If load/parse fails (e.g., mid-write), return last good cache
        return list(_CACHED_DATA_BY_MTIME or CACHED_DATA)

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
def _get_roster_cache():
    global _ROSTER_CACHE
    if _ROSTER_CACHE is None:
        _ROSTER_CACHE = _load_roster_from_disk() or {}
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

# --- NEW: practices meta loader + date helpers + PRAC computation -------------

def load_practices_meta(path=PRACTICES_PATH):
    """
    Load practices metadata written by the entry app when a practice starts:
      {
        "practice-YYYY-MM-DD": {
          "practice_date": "YYYY-MM-DD",
          "absent_numbers": ["12","23", ...]
        },
        ...
      }
    Returns {} if file missing or unreadable.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[practices load] {e}")
        return {}

def _norm_date_yyyy_mm_dd(s):
    """Accept 'YYYY-MM-DD' or 'MM/DD/YYYY' and return 'YYYY-MM-DD' or None."""
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        return datetime.strptime(s.replace("-", "/"), "%m/%d/%Y").strftime("%Y-%m-%d")
    except Exception:
        return None

def _date_in_range(d, start_d, end_d):
    """All args are 'YYYY-MM-DD' or None. Inclusive if bounds provided."""
    if not d:
        return False
    if start_d and d < start_d:
        return False
    if end_d and d > end_d:
        return False
    return True

def compute_practices_played_counts(roster_map, practices_meta, start_date=None, end_date=None):
    """
    Compute PRAC per player (by jersey):
      - For each practice in range, every jersey NOT listed in absent_numbers gets +1.
    Returns dict { "12": 5, "23": 4, ... }.
    """
    # local import to avoid changing your global imports footprint elsewhere
    from datetime import datetime  # safe: only used in helpers above
    start_d = _norm_date_yyyy_mm_dd(start_date) if start_date else None
    end_d   = _norm_date_yyyy_mm_dd(end_date) if end_date else None

    jerseys = [str(int(j)) for j in (roster_map or {}).keys() if str(j).strip().isdigit()]
    counts = {j: 0 for j in jerseys}

    for _pid, meta in (practices_meta or {}).items():
        d = _norm_date_yyyy_mm_dd((meta or {}).get("practice_date"))
        if not _date_in_range(d, start_d, end_d):
            continue
        absents = set()
        for x in (meta or {}).get("absent_numbers", []):
            try:
                absents.add(str(int(x)))
            except:
                pass
        for j in jerseys:
            if j not in absents:
                counts[j] += 1
    return counts

# ------------------------------------------------------------------------------

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
_FOR_GUARDED_RE  = re.compile(r"\bfor\s+({_FULLNAME})\s+guarded\s+by\s+({_FULLNAME})\b", re.IGNORECASE)

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
            "post up","posting up","posts up",   # <-- added posts up
            "keep the handoff","handoff keep",
            "rescreen","re-screen",
            # off-ball phrases:
            "backdoor","backdoor cut","pin down","pindown","flare screen","back screen","away screen", "hammer screen", "ucla screen",
            "cross screen","wedge screen","rip screen","stagger screen","stagger screens","iverson screen",
            "elevator screen","elevator screens",
            # coverage-only cues (include common inflections!)
            "switch", "switches", "switched", "switching",
            "chase over", "chases over",
            "cut under", "cuts under",
            "caught on screen", "top lock", "ice", "blitz",
            # help-defense cues (still candidates, but no separate “Out of help” block)
            "help", "steps in to help"
        ]):
            candidates.append(ln.strip())

    # (Optional downstream use) You can also check _possession_has_help_text(pbp_text)
    # to tag "shots out of help" in your UI if you’d like.

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

# ------------------------- OFF-BALL actions + coverages -------------------------
_OFFBALL_TYPES = {
    "bd":  {"label": "Backdoor cut", "keys": [r"\bbackdoor\b", r"backdoor\s+cut"]},
    "pn":  {"label": "Pin down", "keys": [r"\bpin\s*down\b", r"\bpindown\b"]},
    "fl":  {"label": "Flare screen", "keys": [r"\bflare\s+screen\b"]},
    "bk":  {"label": "Back screen", "keys": [r"\bback\s+screen\b"]},
    "awy": {"label": "Away screen", "keys": [r"\baway\s+screen\b"]},
    "hm":  {"label": "Hammer screen", "keys": [r"\bhammer\s+screen\b"]},
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



#--------------------------------------SECTION 2-----------------------------------

# =======================  mk5 ADDITIONS START  =========================
from datetime import datetime  # --- NEW: required by Section 1 date helpers (PRAC support)

# ---- Filter option sets (used by UI + predicates)
ONBALL_ACTION_CODES = {"h","d","p","pnr","pnp","slp","rj","gst","dho","ho","kp","rs"}
OFFBALL_ACTION_CODES = {"bd","pn","fl","bk","awy", "hm", "crs","wdg","rip","ucla","stg","ivs","elv"}
DEFENSE_FILTERS = {"Man", "Zone"}  # 'Man' vs 'Zone' (Zone includes 2-3, 3-2, 1-3-1, Matchup, etc.)

# ---- Robust, flexible getters from possession rows ---------------------------------

_DATE_TOKEN_RE = re.compile(r"(?P<m>\d{1,2})[/-](?P<d>\d{1,2})(?:[/-](?P<y>\d{2,4}))?", re.IGNORECASE)

def _coerce_int(x, default=1):
    try:
        return int(x)
    except:
        return default

def _normalize_date_str(s: str) -> str:
    """
    Normalize a few common date shapes to ISO-ish 'YYYY-MM-DD' strings if possible.
    Accepts things like '9/12 practice', '09-12-2025', '2025/09/12', '9/12'.
    Falls back to original string if we cannot normalize.
    """
    if not s:
        return ""
    s = str(s)
    # Already looks ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # Find first mm/dd[/yy]
    m = _DATE_TOKEN_RE.search(s)
    if not m:
        return s.strip()
    mm = int(m.group("m") or 0)
    dd = int(m.group("d") or 0)
    yy = m.group("y")
    if yy and len(yy) == 2:
        yy = ("20" + yy)  # naive 20xx assumption for practice logs
    if not yy:
        # No year provided; leave off (we'll compare with month/day only if needed)
        return f"{mm:02d}-{dd:02d}"
    return f"{yy}-{mm:02d}-{dd:02d}"

def _row_practice_date_key(row: dict) -> str:
    """
    Extract a comparable date key for filtering:
      - Prefer row['practice_date'] or row['date']
      - Else try row['group_label'] / row['group_id'] text like '9/12 practice'
      - Returns ISO-like 'YYYY-MM-DD' when possible; otherwise a best-effort token.
    """
    for k in ("practice_date","date","group_label","group_id","session","label","title"):
        v = row.get(k)
        if v:
            return _normalize_date_str(v)
    return ""

def _row_drill_label(row: dict) -> str:
    """
    Extract the full drill label (e.g., '5v5 Stags', '3v3 Pin Down').
    Tries fields: drill, drill_label, group_label, label.
    """
    for k in ("drill","drill_label","group_label","label","title"):
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""

_DRILL_SIZE_RE = re.compile(r"\b(\d+)\s*[vV]\s*(\d+)\b")

def _row_drill_size(row: dict) -> str:
    """
    Parse and normalize the 'NvM' size token from the drill label.
    Returns strings like '5v5', '3v3', or '' if not found.
    """
    lbl = _row_drill_label(row)
    m = _DRILL_SIZE_RE.search(lbl)
    if not m:
        return ""
    return f"{int(m.group(1))}v{int(m.group(2))}"

def _row_possession_text(row: dict) -> str:
    return (row.get("possession") or row.get("pbp") or row.get("text") or "").strip()

def _row_shot_index(row: dict) -> int:
    """
    Provide the nth shot in the possession for parsing.
    Defaults to 1 if unspecified. If the entry app logs a 'shot_index', we'll use it.
    """
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
    """Return 'Make'/'Miss' if present/derivable; else ''."""
    res = row.get("result")
    if res in ("Make","Miss"):
        return res
    rs = result_from_shorthand(_row_possession_text(row))
    return rs or ""

def _row_shot_value(x: float, y: float) -> int:
    """2 or 3 based on distance/line. Simple arc check with side lines."""
    if x is None or y is None:
        return 0
    # Three if beyond arc and beyond straight lines near corners
    dist = math.hypot(x - RIM_X, y - RIM_Y)
    if dist > THREE_R:
        # Corner lines from LEFT_POST_X and RIGHT_POST_X up to their y limits
        left_t = math.asin(max(-1,min(1,(LEFT_POST_X - RIM_X)/THREE_R)))
        right_t = math.asin(max(-1,min(1,(RIGHT_POST_X - RIM_X)/THREE_R)))
        y_left  = RIM_Y + THREE_R*math.cos(left_t)
        y_right = RIM_Y + THREE_R*math.cos(right_t)
        # If outside side line region or above the arc endpoints
        if (x <= LEFT_POST_X and y <= y_left) or (x >= RIGHT_POST_X and y <= y_right):
            return 2
        return 3
    return 2

# ---- Role & action extraction wrappers (using your existing parsers) ---------------

def _row_roles_and_lines(row: dict):
    """
    Uses your mk4 helpers to derive roles for the nth shot:
      shooter, defenders_display, assister, screen_assists[list], candidate_lines[list]
    """
    text = _row_possession_text(row)
    idx  = _row_shot_index(row)
    return extract_roles_for_shot(text, idx)

def _row_onball_offball_actions(row: dict):
    """
    Returns (onball_actions[list[dict]], offball_actions[list[dict]]) parsed from
    candidate lines for this possession/shot.
    """
    shooter, ondef, ast, screen_list, cand_lines = _row_roles_and_lines(row)
    onball = parse_onball_actions_from_pbp(cand_lines, (screen_list[0] if screen_list else ""))
    offball = parse_offball_actions_from_pbp(cand_lines)
    return onball, offball

def _row_defense_label(row: dict) -> str:
    """
    Return 'Man' or 'Zone' depending on your zone_for_shot() classifier
    (which is defined later in the file). If zone_for_shot returns 'Man to Man'
    we map to 'Man', else 'Zone'.
    """
    try:
        zlbl = zone_for_shot(_row_possession_text(row), _row_shot_index(row))  # defined later
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
    k = _row_practice_date_key(row)  # e.g., '2025-09-12' or '09-12'
    if not (start_key or end_key):
        return True
    if not k:
        return False

    def _split_isoish(s):
        # returns (Y?, M, D) where Y may be ''
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            y,m,d = s.split("-"); return (y,int(m),int(d))
        if re.match(r"^\d{2}-\d{2}$", s):
            m,d = s.split("-"); return ("",int(m),int(d))
        # last resort: try again
        s2 = _normalize_date_str(s)
        return _split_isoish(s2) if s2 != s else ("",0,0)

    ry, rm, rd = _split_isoish(k)

    def _cmp(a, b):
        # compare dates with optional year ('' treated as wildcard)
        ay,am,ad = a; by,bm,bd = b
        # If either has '', compare month/day only
        if not ay or not by:
            return (am,ad) < (bm,bd)
        return (ay,am,ad) < (by,bm,bd)

    rk = (ry, rm, rd)

    if start_key:
        sk = _split_isoish(start_key)
        if _cmp(rk, sk):  # row < start
            return False
    if end_key:
        ek = _split_isoish(end_key)
        if _cmp(ek, rk):  # end < row
            return False
    return True

def row_passes_shooting_filters(row: dict, filters: dict) -> bool:
    """
    Shooting-tab filters (practice dates, drill size & label, shooter, defenders,
    assister, screen assister, on-ball action, off-ball action, defense Man/Zone).
    The 'filters' dict can contain:
      {
        "date_start": "YYYY-MM-DD" or "MM-DD" or free-text,
        "date_end":   same as above,
        "drill_size": set[str] like {"5v5","3v3"},
        "drill_full": set[str] like {"5v5 Stags"},
        "shooter":    set[str] of full names or partials,
        "defenders":  set[str] of full names or partials (any match),
        "assister":   set[str],
        "screen_ast": set[str],
        "onball":     set[str] of codes (e.g., {"pnr","pnp"}),
        "offball":    set[str] of codes (e.g., {"pn","bd"}),
        "defense":    set[str] subset of {"Man","Zone"},
      }
    """
    # Dates
    if not _passes_date_range(row, filters.get("date_start",""), filters.get("date_end","")):
        return False

    # Drill size / full label
    ds = _row_drill_size(row)
    if filters.get("drill_size"):
        if ds not in filters["drill_size"]:
            return False
    if filters.get("drill_full"):
        if _row_drill_label(row) not in filters["drill_full"]:
            return False

    # Roles
    shooter, defenders_disp, assister, screen_list, cand_lines = _row_roles_and_lines(row)

    if filters.get("shooter"):
        # exact-or-partial match against shooter full name
        if not any(_str_in_ci(s, shooter) for s in filters["shooter"]):
            return False

    if filters.get("defenders"):
        # FIX: single any(...) (the previous any(any(...)) could break callbacks)
        if not any(_str_in_ci(d, defenders_disp) for d in filters["defenders"]):
            return False

    if filters.get("assister"):
        if not any(_str_in_ci(a, assister) for a in filters["assister"]):
            return False

    if filters.get("screen_ast"):
        joined = ", ".join(screen_list or [])
        if not any(_str_in_ci(s, joined) for s in filters["screen_ast"]):
            return False

    # Actions
    onball, offball = _row_onball_offball_actions(row)
    if filters.get("onball"):
        codes = _collect_action_codes(onball)
        if not (codes & set([c.lower() for c in (filters["onball"] or [])])):
            return False

    if filters.get("offball"):
        codes = _collect_action_codes(offball)
        if not (codes & set([c.lower() for c in (filters["offball"] or [])])):
            return False

    # Defense
    if filters.get("defense"):
        if _row_defense_label(row) not in filters["defense"]:
            return False

    return True

# ---- Shot collection & Shooting Stats ---------------------------------------------

def collect_shots_for_filters(possession_rows: list[dict], filters: dict):
    """
    Returns list of dicts for plotting + statting:
      { x, y, result('Make'/'Miss'), value(2/3), shooter, defenders, assister, screen_ast[list], row_ref }
    Only includes rows that pass row_passes_shooting_filters.
    """
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
            "assister": assister, "screen_assists": screen_list or [],
            "row_ref": row
        })
    return out

def compute_shooting_totals(filtered_shots: list[dict]) -> dict:
    """
    Compute the left-panel shooting metrics:
      FGM, FGA, FG%, 2PM, 2PA, 2P%, 3PM, 3PA, 3P%
    """
    FGA = len(filtered_shots or [])
    FGM = sum(1 for s in filtered_shots if s.get("result") == "Make")
    twos = [s for s in (filtered_shots or []) if s.get("value") == 2]
    thrs = [s for s in (filtered_shots or []) if s.get("value") == 3]
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
    """Blank filter set for the Shooting tab."""
    return {
        "date_start": "", "date_end": "",
        "drill_size": set(), "drill_full": set(),
        "shooter": set(), "defenders": set(), "assister": set(), "screen_ast": set(),
        "onball": set(), "offball": set(),
        "defense": set(),  # subset of {"Man","Zone"}
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


#-------------------------------------------SECTION 3---------------------------


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


# ---- stats
def calculate_zone_stats(shots):
    zone_stats = {}
    for zone_id in range(1, 15):
        makes = attempts = 0
        for shot in shots:
            # Harden to only count valid shots (Make/Miss with legal x/y)
            try:
                x = float(shot.get("x")); y = float(shot.get("y"))
                res = shot.get("result") or ""
            except Exception:
                continue
            if x is None or y is None or res not in ("Make","Miss"):
                continue
            if point_in_zone(x, y, zone_id):
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

# ---- Stats-tab filters (subset: dates, drill size/label, defense) -------------------
def empty_stats_filters() -> dict:
    return {
        "date_start": "", "date_end": "",
        "drill_size": set(), "drill_full": set(),
        "defense": set(),  # {"Man","Zone"}
    }

def row_passes_stats_filters(row: dict, filters: dict) -> bool:
    if not _passes_date_range(row, filters.get("date_start",""), filters.get("date_end","")):
        return False
    if filters.get("drill_size"):
        if _row_drill_size(row) not in filters["drill_size"]:
            return False
    if filters.get("drill_full"):
        if _row_drill_label(row) not in filters["drill_full"]:
            return False
    if filters.get("defense"):
        if _row_defense_label(row) not in filters["defense"]:
            return False
    return True

# ---- Per-player stat accumulator ----------------------------------------------------
def _ensure_player(bucket: dict, player: str) -> dict:
    """
    Ensure a dict for player exists with all counters initialized.
    Returns the player's dict for in-place updates.
    """
    name = (player or "").strip()
    if not name:
        name = "Unknown"
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
            "LBTO":0, "DBTO":0, "TO":0, 
            # defense/specials
            "STL":0, "DEF":0, "BLK":0,
            # possession counts
            "OP":0, "DP":0,
            # --- NEW: practices played
            "PRAC":0,
        }
    return bucket[name]

def _pct(m,a):
    try:
        return round((100.0*m/a), 1) if a else 0.0
    except:
        return 0.0

# -------------------- NEW: Practice/absence helpers (PRAC) --------------------
# We accept several possible keys the entry app may store absences under.
_ABSENCE_KEYS = ("absences","absence_numbers","absent","absent_numbers","absent_list","missed")

def _extract_absent_names_from_row(row: dict) -> set[str]:
    """
    Try to pull absences from a row via common keys. Handles:
      - list of jersey numbers or names
      - comma/space separated strings
    Returns a set of canonical roster names.
    """
    out = set()
    roster = _get_roster_cache()

    def _add_name_token(tok: str):
        tok = (tok or "").strip()
        if not tok:
            return
        # if it's a number, map by jersey; else normalize to roster
        if re.fullmatch(r"\d{1,2}", tok):
            out.add(_name_from_num(tok))
        else:
            out.add(_normalize_to_roster(tok, roster))

    for k in _ABSENCE_KEYS:
        if k in row and row.get(k) is not None:
            val = row.get(k)
            if isinstance(val, list):
                for v in val:
                    # list may contain numbers or names
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        _add_name_token(str(int(v)))
                    else:
                        _add_name_token(str(v))
            else:
                # split any free text by commas/space
                for tok in re.findall(r"[A-Za-z.'\-]+(?:\s+[A-Za-z.'\-]+)?|\d{1,2}", str(val)):
                    _add_name_token(tok)
    # clean out empties
    return {n for n in (out or set()) if n}

def _gather_practice_absences(possession_rows: list[dict], date_start: str, date_end: str):
    """
    Group by normalized practice-date token and collect absent-name sets.
    Only date range is applied here (drill/defense filters do NOT affect PRAC).
    Returns:
      practices: list of date keys counted
      abs_by_date: { date_key -> set(absent names) }
    """
    abs_by_date = {}
    dates = set()

    for r in (possession_rows or []):
        # get the normalized practice date key
        dk = _row_practice_date_key(r)
        if not dk:
            continue

        # apply ONLY date-range
        if not _passes_date_range({"practice_date": dk}, date_start, date_end):
            continue

        dates.add(dk)
        abset = abs_by_date.setdefault(dk, set())
        abset |= _extract_absent_names_from_row(r)

    practices = sorted(dates)
    return practices, abs_by_date

def _accumulate_practices(players: dict, possession_rows: list[dict], filters: dict):
    """
    Increment PRAC once per counted practice for every roster player
    who is NOT absent for that practice date.
    """
    roster = _get_roster_cache() or {}
    all_players = list(roster.values())

    practices, abs_by_date = _gather_practice_absences(
        possession_rows,
        filters.get("date_start",""),
        filters.get("date_end","")
    )

    for dk in practices:
        absent = abs_by_date.get(dk, set())
        for full_name in all_players:
            if not full_name:
                continue
            if full_name in absent:
                continue
            _ensure_player(players, full_name)["PRAC"] += 1
# ------------------ END: Practice/absence helpers (PRAC) ----------------------


# ---- Main aggregator ----------------------------------------------------------------
def compute_player_stats_table(possession_rows: list[dict], filters: dict) -> list[dict]:
    """
    Build a table of per-player totals for the Stats tab.
    Columns:
      Player, FGM,FGA,FG%, 2PM,2PA,2P%, 3PM,3PA,3P%, AST,
      DRB, ORB, TRB, LBTO, DBTO, TO, STL, DEF, BLK, OP, DP
      + PRAC (NEW)
    """
    players = {}

    for row in (possession_rows or []):
        if not row_passes_stats_filters(row, filters or {}):
            continue

        pbp = _row_possession_text(row)
        shorthand = pbp  # tokens can be mixed into the possession text; parser is token-based
        shot_x, shot_y = _row_xy(row)
        shot_res = _row_result(row)
        shooter, ondef_disp, assister, _screen_list, _cand = _row_roles_and_lines(row)

        # Offensive / Defensive possessions from text + shorthand
        off_list, def_list = collect_players_from_possession(pbp, shorthand)
        for nm in off_list:
            _ensure_player(players, nm)["OP"] += 1
        for nm in def_list:
            _ensure_player(players, nm)["DP"] += 1

        # Shooting events credited to shooter
        if shooter and shot_res in ("Make","Miss") and shot_x is not None and shot_y is not None:
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
        sp = parse_special_stats_from_shorthand(shorthand)

        for nm in sp.get("def_rebounds", []):
            _ensure_player(players, nm)["DRB"] += 1
        for nm in sp.get("off_rebounds", []):
            _ensure_player(players, nm)["ORB"] += 1

        for nm in sp.get("live_ball_to", []):
            _ensure_player(players, nm)["LBTO"] += 1
        for nm in sp.get("dead_ball_to", []):
            _ensure_player(players, nm)["DBTO"] += 1
        for nm in sp.get("steals", []):
            _ensure_player(players, nm)["STL"] += 1
        for nm in sp.get("blocks", []):
            _ensure_player(players, nm)["BLK"] += 1
        for nm in sp.get("deflections", []):
            _ensure_player(players, nm)["DEF"] += 1

    # --- NEW: accumulate practices played (PRAC) using date range only
    _accumulate_practices(players, possession_rows, filters)

    # Post-processing: compute derived fields (FG%, 2P%, 3P%, TRB, TO, F)
    table = []
    for nm, rec in players.items():
        rec = dict(rec)  # shallow copy
        rec["FG%"] = _pct(rec["FGM"], rec["FGA"])
        rec["2P%"] = _pct(rec["2PM"], rec["2PA"])
        rec["3P%"] = _pct(rec["3PM"], rec["3PA"])
        rec["TRB"] = rec["DRB"] + rec["ORB"]
        rec["TO"]  = rec["LBTO"] + rec["DBTO"]
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
        title=dict(text=None, pad=dict(t=0, b=0, l=0, r=0))  # 🔑 ensure no hidden title padding
    )


# ------------- Shot chart
def create_shot_chart(shots, highlight_coords=None):
    fig = go.Figure()
    for tr in court_lines_traces(): fig.add_trace(tr)

    # Ensure we are dealing with plottable shots, not raw rows
    shots_list = rows_to_shots(shots) if shots and isinstance(shots[0], dict) and "row_ref" not in shots[0] else (shots or [])

    makes = [(s["x"], s["y"]) for s in shots_list if s["result"] == "Make"]
    misses = [(s["x"], s["y"]) for s in shots_list if s["result"] == "Miss"]
    if makes:
        x, y = zip(*makes)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(symbol='circle', size=10, color='green',
                                             line=dict(width=1, color='green'))),
                                 )
    if misses:
        x, y = zip(*misses)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(symbol='x', size=10, color='red')),
                                 )

    if highlight_coords:
        L = 1.2
        for (hx, hy) in highlight_coords:
            fig.add_shape(
                type="rect",
                x0=hx - L/2, y0=hy - L/2, x1=hx + L/2, y1=hy + L/2,
                line=dict(color="#888", width=1),
                fillcolor="#e6e6e6",
                layer="below"
            )

    fig.update_layout(**base_layout())
    fig.update_layout(clickmode="event+select")

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


def create_zone_chart():
    # Load raw rows, convert to shots so result/x/y are guaranteed
    possession_rows = safe_load_data()
    shots = rows_to_shots(possession_rows)

    fig = go.Figure()

    for tr in court_lines_traces(): fig.add_trace(tr)
    for tr in first_zone_line_traces(): fig.add_trace(tr)
    for tr in mini_three_point_line(): fig.add_trace(tr)
    for tr in elbow_lines(): fig.add_trace(tr)
    for tr in diagonal_zone_lines(): fig.add_trace(tr)

    zone_stats = calculate_zone_stats(shots)

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
        makes, atts, pct = s["makes"], s["attempts"], s["percentage"]
        txt = f"{makes}/{atts}<br>{pct:.1f}%"
        fig.add_annotation(x=center[0], y=center[1], text=txt,
                           showarrow=False,
                           font=dict(size=12, color="black", family="Arial Black"))

    fig.update_layout(**base_layout())
    return fig


# =========================
# Dash App
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

_initial_shots = safe_load_data()
_initial_shot_fig = create_shot_chart(_initial_shots)
_initial_zone_fig = create_zone_chart()

# ---- mk5 ADD: bring in DataTable for sortable Stats tab
from dash import dash_table  # (Dash 2+) correct import

# ====================== mk5: helpers used by callbacks (moved here so Section 4 sees them) ======================





#--------------------------SECTION 4---------------------------------------------------

# Ensure initial figures preserve UI state (no flicker) without using Graph.uirevision
# (These figures are defined earlier; this only adjusts their layout.)
try:
    _initial_shot_fig.update_layout(uirevision="keep")
    _initial_zone_fig.update_layout(uirevision="keep")
except Exception:
    pass

# ---------- NEW: compact legend component for Hot/Cold Zones (colors only; labels filled later in Section 5) ----------
# ---------- Legend component (updated labels & headers) ----------
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

    block_style = {"display":"inline-block","margin":"0 18px","textAlign":"left"}

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
        style={"textAlign":"center","marginTop":"8px","fontSize":"14px"}
    )

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
        seen = set(); ordered = []
        for n in names:
            if n.lower() not in seen:
                ordered.append(n); seen.add(n.lower())
        return ordered
    for rr in rows_for_plot:
        k = rr.get("group_id") or rr.get("timestamp") or rr.get("id")
        if k != key:
            continue
        for field in ("roster","roster_names","names","home_names","away_names","home_roster","away_roster","roster_map"):
            if field in rr:
                vals = rr[field]
                names = _harvest_fullnames_from_any(vals)
                if names:
                    seen = set(); ordered = []
                    for n in names:
                        if n.lower() not in seen:
                            ordered.append(n); seen.add(n.lower())
                    return ordered
    pool = []
    for rr in rows_for_plot:
        k = rr.get("group_id") or rr.get("timestamp") or rr.get("id")
        if k != key:
            continue
        for v in rr.values():
            pool += _harvest_fullnames_from_any(v)
    seen = set(); ordered = []
    for n in pool:
        if n.lower() not in seen:
            ordered.append(n); seen.add(n.lower())
    return ordered

# --- RENAMED to avoid clobbering Section-1's _normalize_to_roster ---
def _normalize_to_roster_for_list(name, roster_full_list):
    nm = _trim_trailing_verb(name or "").strip()
    nm = _strip_leading_preps(nm)
    # Strip trailing connectors that break normalization (e.g., "Lusk and")
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
    if not actions: return actions
    try:
        txt = _clean_frag(pbp_text or "")
        bh_from_brings = ""
        m_bring = _BRING_SUBJ_RE.search(txt)
        if m_bring:
            bh_from_brings = _trim_trailing_verb(m_bring.group(1).strip())

        # find "defender picks up ballhandler"
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
            if a.get("type") == "h":  # bring over halfcourt
                if not a.get("bh"):
                    a["bh"] = bh_from_brings or bh_from_pick or a.get("bh","")
                if not a.get("bh_def"):
                    if a.get("bh") and bh_from_pick and a["bh"].lower() == bh_from_pick.lower():
                        a["bh_def"] = bh_def_from_pick
                    else:
                        a["bh_def"] = bh_def_from_pick or a.get("bh_def","")
        return actions
    except:
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
    if _HELP_VERBS_RE.search(txt): return True
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

    order = ["Defensive Rebound","Offensive Rebound","Deflection","Steal",
             "Live Ball Turnover","Dead Ball Turnover","Defensive Foul",
             "Offensive Foul","Block"]
    out = []
    for lbl in order:
        players = by_label.get(lbl, [])
        if players:
            key = None
            for k, v in _SPECIAL_LABELS.items():
                if v == lbl:
                    key = k; break
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
    "hm":  {"label": "Hammer screen", "keys": [r"\bhammer\s+screen\b"]},
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
        coming_off, coming_off_def = ("","")
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
                key = (s.get("name","").lower(), s.get("def","").lower())
                if key not in stg_seen_scr and (s.get("name") or s.get("def")):
                    stg_agg["screeners"].append({"name": s.get("name",""), "def": s.get("def","")})
                    stg_seen_scr.add(key)

            for c in (covs or []):
                key = (c.get("label","").lower(), (c.get("onto","") or "").lower())
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
    if not n: return ""
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
            if patt1.search(txt): has_rot = True
        if not has_rot and def_pat:
            patt2 = re.compile(rf"\bguarded\s+by\s+{def_pat}\s+rotating\s+over\b", re.IGNORECASE)
            if patt2.search(txt): has_rot = True
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
        style={"minWidth":"160px"}
    )

def _pill(title, child):
    return html.Div(
        [html.Div(title, style={"fontSize":"12px","fontWeight":700,"marginBottom":"4px"}), child],
        style={"display":"flex","flexDirection":"column","gap":"4px"}
    )

# ---- mk5 ADD: constants for action filters
ONBALL_OPTIONS = [
    {"label":"H (bring over)","value":"h"},
    {"label":"D (drive)","value":"d"},
    {"label":"P (post up)","value":"p"},
    {"label":"PNR","value":"pnr"},
    {"label":"PNP","value":"pnp"},
    {"label":"DHO","value":"dho"},
    {"label":"HO","value":"ho"},
    {"label":"KP (keep)","value":"kp"},
    {"label":"RJ (reject)","value":"rj"},
    {"label":"SLP (slip)","value":"slp"},
    {"label":"GST (ghost)","value":"gst"},
]
OFFBALL_OPTIONS = [
    {"label":"Backdoor (bd)","value":"bd"},
    {"label":"Pin down (pn)","value":"pn"},
    {"label":"Flare (fl)","value":"fl"},
    {"label":"Back screen (bk)","value":"bk"},
    {"label":"Away (awy)","value":"awy"},
    {"label":"Hammer (hm)","value":"hm"},
    {"label":"UCLA (ucla)","value":"ucla"},
    {"label":"Cross (crs)","value":"crs"},
    {"label":"Wedge (wdg)","value":"wdg"},
    {"label":"Rip (rip)","value":"rip"},
    {"label":"Stagger (stg)","value":"stg"},
    {"label":"Iverson (ivs)","value":"ivs"},
    {"label":"Elevator (elv)","value":"elv"},
]
# FIX: values must be {"Man","Zone"} to match filter predicates; keep label for UX
DEFENSE_OPTIONS = [
    {"label":"Man to Man","value":"Man"},
    {"label":"Zone","value":"Zone"},
]

# ---- mk5 ADD: layout (no Loading spinners, no timer-based Interval)
app.layout = html.Div(
    style={"maxWidth":"1600px","margin":"0 auto","padding":"10px"},
    children=[
        html.Div([
            html.Div(f"Data source: {DATA_PATH}", style={"color":"#666","fontSize":"12px","marginBottom":"2px"}),
            html.Div("Charts update when data or filters change", style={"color":"#888","fontSize":"10px"}),
            html.Div(id="status", style={"color":"#888","fontSize":"10px"}),
        ], style={"textAlign":"center","marginBottom":"8px"}),

        dcc.Tabs(id="tabs", value="tab_shooting", children=[
            dcc.Tab(label="Shooting", value="tab_shooting", children=[
                # ---------- Shooting filters
                html.Div([
                    _pill("Practice Date(s)", dcc.DatePickerRange(
                        id="flt_date_range_shoot",
                        min_date_allowed=None, max_date_allowed=None,
                        start_date=None, end_date=None,
                        minimum_nights=0,                 # <<< allow same-day range
                        display_format="YYYY-MM-DD",
                        style={"background":"white"}
                    )),
                    _pill("Drill Size", _mk_multi("Drill Size","flt_drill_size_shoot","e.g. 3v3 / 5v5")),
                    _pill("Drill", _mk_multi("Drill","flt_drill_full_shoot","e.g. 5v5 Stags")),
                    _pill("Shooter", _mk_multi("Shooter","flt_shooter","Filter by shooter")),
                    _pill("Defender(s)", _mk_multi("Defender(s)","flt_defenders","Who contested the shot")),
                    _pill("Assister", _mk_multi("Assister","flt_assister","Passer on made FG")),
                    _pill("Screen Assister", _mk_multi("Screen Assister","flt_screen_assister","Who set the screen")),
                    _pill("On-Ball Action", dcc.Dropdown(
                        id="flt_onball", options=ONBALL_OPTIONS, multi=True, placeholder="Select on-ball actions",
                        style={"minWidth":"200px"}
                    )),
                    _pill("Off-Ball Action", dcc.Dropdown(
                        id="flt_offball", options=OFFBALL_OPTIONS, multi=True, placeholder="Select off-ball actions",
                        style={"minWidth":"200px"}
                    )),
                    _pill("Defense", dcc.Dropdown(
                        id="flt_defense_shoot", options=DEFENSE_OPTIONS, multi=True, placeholder="Man / Zone",
                        style={"minWidth":"140px"}
                    )),
                    html.Button("Clear", id="btn_clear_shoot", n_clicks=0,
                                style={"height":"34px","alignSelf":"flex-end","marginLeft":"8px"})
                ], style={"display":"flex","flexWrap":"wrap","gap":"12px",
                          "alignItems":"end","background":"#fafafa","border":"1px solid #eee",
                          "borderRadius":"8px","padding":"10px","marginBottom":"10px"}),

                # Single row: Stats | Shot Chart | Hot/Cold
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
                            style={"width": "300px", "padding": "8px", "border": "1px solid #eee",
                                   "borderRadius": "8px", "background": "white"},
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
                                # NEW: legend key placed directly under the zones chart
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

                html.Div(id="shot_details", style={"maxWidth":"920px","margin":"14px auto 0 auto"}),
            ]),

            dcc.Tab(label="Stats", value="tab_stats", children=[
                # ---------- Stats tab filters (subset)
                html.Div([
                    _pill("Practice Date(s)", dcc.DatePickerRange(
                        id="flt_date_range_stats",
                        min_date_allowed=None, max_date_allowed=None,
                        start_date=None, end_date=None,
                        minimum_nights=0,                 # <<< allow same-day range
                        display_format="YYYY-MM-DD",
                        style={"background":"white"}
                    )),
                    _pill("Drill Size", _mk_multi("Drill Size","flt_drill_size_stats","e.g. 3v3 / 5v5")),
                    _pill("Drill", _mk_multi("Drill","flt_drill_full_stats","e.g. 5v5 Stags")),
                    _pill("Defense", dcc.Dropdown(
                        id="flt_defense_stats", options=DEFENSE_OPTIONS, multi=True, placeholder="Man / Zone",
                        style={"minWidth":"140px"}
                    )),
                    html.Button("Clear", id="btn_clear_stats", n_clicks=0,
                                style={"height":"34px","alignSelf":"flex-end","marginLeft":"8px"})
                ], style={"display":"flex","flexWrap":"wrap","gap":"12px",
                          "alignItems":"end","background":"#fafafa","border":"1px solid #eee",
                          "borderRadius":"8px","padding":"10px","marginBottom":"10px"}),

                # ---------- Stats table
                html.Div([
                    html.Div("Team Stats (click headers to sort)", style={"fontSize":"18px","fontWeight":700,"marginBottom":"6px"}),
                    dash_table.DataTable(
                        id="stats_table",
                        columns=[{"name":c,"id":c} for c in [
                            "Player","FGM","FGA","FG%","2PM","2PA","2P%","3PM","3PA","3P%",
                            "AST","DRB","ORB","TRB","LBTO","DBTO","TO","STL","DEF","BLK","OP","DP","PRAC"
                        ]],
                        data=[],
                        sort_action="native",
                        style_table={"overflowX":"auto"},
                        style_cell={"fontFamily":"Arial","fontSize":"13px","padding":"6px","textAlign":"center"},
                        style_header={"fontWeight":"700","backgroundColor":"#f5f5f5"},
                        fixed_rows={"headers": True},
                        page_size=50
                    )
                ], style={"border":"1px solid #eee","borderRadius":"8px","padding":"8px","background":"white"})
            ]),
        ]),

        # ---- state stores (no timer; callbacks in Section 5 will update on data/filters only)
        dcc.Store(id="sel_pos", data=[]),
        dcc.Store(id="filters_shoot_state"),
        dcc.Store(id="filters_stats_state"),
        dcc.Store(id="options_cache", data={}),
    ]
)





#----------------------SECTION 5-------------------------------

# ---------------- callbacks ----------------
from datetime import datetime, date
from collections import defaultdict


def _parse_date_any(s):
    if not s: return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try: return datetime.strptime(str(s)[:10], fmt).date()
        except: pass
    try:
        return datetime.fromisoformat(str(s)[:10]).date()
    except:
        return None

_DRILL_SIZE_RE = re.compile(r"\b(\d+v\d+)\b", re.IGNORECASE)

def _get_drill_size(drill: str) -> str:
    if not drill: return ""
    m = _DRILL_SIZE_RE.search(drill)
    return (m.group(1) if m else "").lower()

def _distance_from_rim(x,y):
    try:
        return math.hypot(float(x) - RIM_X, float(y) - RIM_Y)
    except:
        return 0.0

def _is_three(x,y):
    return _distance_from_rim(x,y) >= THREE_R - 1e-9

def _rows_full():
    rows_for_plot = []
    if not os.path.exists(DATA_PATH):
        return rows_for_plot

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]

    for rr in (data or []):
        try:
            x = float(rr.get("x")) if str(rr.get("x")).strip() not in ("", "None") else None
            y = float(rr.get("y")) if str(rr.get("y")).strip() not in ("", "None") else None
        except:
            x = y = None
        res = rr.get("result") or result_from_shorthand(rr.get("possession",""))
        if x is not None and y is not None and (0 <= x <= COURT_W) and (0 <= y <= HALF_H) and res in ("Make","Miss"):
            pbp_names_src = (rr.get("play_by_play_names") or "")
            pbp_raw_src   = (rr.get("play_by_play") or "")
            pbp_src_for_roles = pbp_names_src or pbp_raw_src
            idx = rr.get("shot_index") or 1

            # Parse roles from PBP…
            shooter_p, onball_def_p, assister_p, screen_ast_list_p, action_lines = extract_roles_for_shot(pbp_src_for_roles, idx)
            _def_disp_from_shot_line = shot_defender_display(pbp_src_for_roles, idx)
            if _def_disp_from_shot_line:
                onball_def_p = _def_disp_from_shot_line

            # …but FALL BACK to explicit row fields when present.
            shooter = rr.get("shooter") or shooter_p
            onball_def = rr.get("defenders") or rr.get("defender") or onball_def_p
            assister = rr.get("assister") or assister_p
            screen_ast_list = rr.get("screen_assisters") or rr.get("screen_assister") or screen_ast_list_p or []

            on_actions  = parse_onball_actions_from_pbp(action_lines, (screen_ast_list[0] if screen_ast_list else ""))
            off_actions = parse_offball_actions_from_pbp(action_lines)

            on_types  = { (a.get("type") or "").lower() for a in on_actions }
            off_types = { (a.get("type") or "").lower() for a in off_actions }

            try:
                def_label = defense_label_for_shot(pbp_src_for_roles, idx) or "Man to Man"
            except:
                def_label = "Man to Man"

            short = (rr.get("possession") or "")
            if "[" in short and "]" in short:
                m_zone = re.search(r"\b(\d(?:-\d){1,3})\s*\[", short)
                if m_zone:
                    def_label = f"{m_zone.group(1)} Zone"

            practice = rr.get("practice_date") or rr.get("practice") or rr.get("date") or ""
            drill    = rr.get("drill") or rr.get("practice_drill") or rr.get("drill_name") or ""
            drill_sz = _get_drill_size(drill)

            rows_for_plot.append({
                **rr,
                "x": x, "y": y, "result": res,
                "is_three": _is_three(x,y),
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
                "defense_label": def_label,
                "pbp_src_for_roles": pbp_src_for_roles,
            })
    return rows_for_plot

def _uniq_sorted(seq):
    seen = set(); out = []
    for s in (seq or []):
        if not s: continue
        key = str(s).strip()
        l = key.lower()
        if l and l not in seen:
            seen.add(l); out.append(key)
    return sorted(out, key=lambda z: z.lower())

def _split_names(s_or_list):
    if not s_or_list: return []
    if isinstance(s_or_list, list):
        raw = []
        for v in s_or_list:
            raw += _split_names(v)
        return _uniq_sorted(raw)
    return _split_fullname_list(str(s_or_list))

def _filter_rows(rows,
                 d_start, d_end,
                 drill_sizes, drills_full,
                 shooters, defenders, assisters, screen_assisters,
                 onball, offball, defense_opts,
                 for_stats_tab=False):
    out = []
    # normalize defense opts: accept "Man", "Man to Man", or any specific zone label, plus "Zone"
    def_norm = set((defense_opts or []))
    def_norm = {str(x).strip().lower() for x in def_norm}
    if "man" in def_norm or "man to man" in def_norm:
        def_norm.add("man to man")
    if "zone" in def_norm:
        def_norm.add("zone")

    for r in rows:
        dt = r.get("practice_date_obj")
        if d_start and dt and dt < d_start: continue
        if d_end and dt and dt > d_end: continue

        if drill_sizes:
            if (r.get("drill_size") or "").lower() not in {s.lower() for s in (drill_sizes or [])}:
                continue
        if drills_full:
            if (r.get("drill") or "").lower() not in {s.lower() for s in (drills_full or [])}:
                continue

        if def_norm:
            lab  = (r.get("defense_label") or "").lower()  # e.g., "man to man" or "2-3 zone"
            is_zone = (lab != "man to man")
            good = False
            if "man to man" in def_norm and lab == "man to man":
                good = True
            if "zone" in def_norm and is_zone:
                good = True
            # exact (e.g., "2-3 zone") still supported if user ever supplies a specific value
            if any(lab == x for x in def_norm):
                good = True
            if not good:
                continue

        if not for_stats_tab:
            if shooters:
                shots = set([n.lower() for n in _split_names(r.get("shooter_raw"))])
                if not shots.intersection({n.lower() for n in shooters}):
                    continue
            if defenders:
                defs = set([n.lower() for n in _split_names(r.get("defenders_raw"))])
                if not defs.intersection({n.lower() for n in defenders}):
                    continue
            if assisters:
                asts = set([n.lower() for n in _split_names(r.get("assister_raw"))])
                if not asts.intersection({n.lower() for n in assisters}):
                    continue
            if screen_assisters:
                sas = set([n.lower() for n in _split_names(r.get("screen_assisters_raw"))])
                if not sas.intersection({n.lower() for n in screen_assisters}):
                    continue
            if onball:
                if not set(r.get("onball_set") or []).intersection({a.lower() for a in onball}):
                    continue
            if offball:
                if not set(r.get("offball_set") or []).intersection({a.lower() for a in offball}):
                    continue

        out.append(r)
    return out

def _rows_to_shots(rows):
    return [{"x": r["x"], "y": r["y"], "result": r["result"]} for r in rows]

def _shooting_tiles(rows):
    fga = len(rows)
    fgm = sum(1 for r in rows if r["result"] == "Make")
    two_a = sum(1 for r in rows if not r.get("is_three"))
    two_m = sum(1 for r in rows if (not r.get("is_three")) and r["result"] == "Make")
    thr_a = sum(1 for r in rows if r.get("is_three"))
    thr_m = sum(1 for r in rows if r.get("is_three") and r["result"] == "Make")

    def pct(m, a):
        return round((m / a) * 100.0, 1) if a else 0.0

    metrics = [
        ("FGM", fgm), ("FGA", fga), ("FG%", f"{pct(fgm, fga):.1f}%"),
        ("2PM", two_m), ("2PA", two_a), ("2P%", f"{pct(two_m, two_a):.1f}%"),
        ("3PM", thr_m), ("3PA", thr_a), ("3P%", f"{pct(thr_m, thr_a):.1f}%"),
    ]

    boxes = [
        html.Div([
            html.Div(name, style={"fontSize": "12px", "color": "#555"}),
            html.Div(str(val), style={"fontSize": "18px", "fontWeight": 800})
        ], style={
            "border": "1px solid #eee",
            "borderRadius": "8px",
            "padding": "8px",
            "textAlign": "center",
            "background": "white"
        })
        for name, val in metrics
    ]

    # single wrapper with just the tiles (no internal title)
    return html.Div(
        boxes,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, minmax(80px, 1fr))",
            "gap": "8px"
        }
    )


def _normalize_display_name(nm: str):
    return _normalize_to_roster(nm)

def _ensure_player(stats, name):
    if not name: return
    if name not in stats:
        stats[name] = {
            "Player": name,
            "FGM":0,"FGA":0,"FG%":0.0,
            "2PM":0,"2PA":0,"2P%":0.0,
            "3PM":0,"3PA":0,"3P%":0.0,
            "AST":0,"DRB":0,"ORB":0,"TRB":0,
            "LBTO":0,"DBTO":0,"TO":0,
            "STL":0,"DEF":0,"BLK":0,
            "OP":0,"DP":0,
            "PRAC":0  # added: will be set from distinct-date set
        }

def _aggregate_stats_table(rows):
    """
    Build per-player totals for the Stats tab from already-filtered rows.
    Columns:
      Player, FGM,FGA,FG%, 2PM,2PA,2P%, 3PM,3PA,3P%, AST, DRB, ORB, TRB,
      LBTO, DBTO, TO, STL, DEF, BLK, OP, DP, PRAC
    """
    def _norm(nm: str):
        return _normalize_to_roster((nm or "").strip())

    stats = {}
    # track distinct practice days per player
    prac_days = defaultdict(set)

    def _ensure(name):
        if not name:
            return
        if name not in stats:
            stats[name] = {
                "Player": name,
                "FGM":0,"FGA":0,"FG%":0.0,
                "2PM":0,"2PA":0,"2P%":0.0,
                "3PM":0,"3PA":0,"3P%":0.0,
                "AST":0,"DRB":0,"ORB":0,"TRB":0,
                "LBTO":0,"DBTO":0,"TO":0,
                "STL":0,"DEF":0,"BLK":0,
                "OP":0,"DP":0,
                "PRAC":0
            }

    for r in rows:
        pbp_src = r.get("pbp_src_for_roles") or ""
        short   = r.get("possession","")
        pday    = r.get("practice_date_obj") or r.get("practice_date_str") or None

        # Shooter
        shooter = _norm(r.get("shooter_raw") or "")
        if shooter:
            _ensure(shooter)
            stats[shooter]["FGA"] += 1
            is3 = bool(r.get("is_three"))
            if r.get("result") == "Make":
                stats[shooter]["FGM"] += 1
                if is3: stats[shooter]["3PM"] += 1
                else:   stats[shooter]["2PM"] += 1
            if is3: stats[shooter]["3PA"] += 1
            else:   stats[shooter]["2PA"] += 1
            if pday: prac_days[shooter].add(pday)

        # Assist
        assister = _norm(r.get("assister_raw") or "")
        if assister:
            _ensure(assister)
            stats[assister]["AST"] += 1
            if pday: prac_days[assister].add(pday)

        # Specials from shorthand + PBP blocks
        sps = parse_special_stats_from_shorthand(short or "")
        for nm in (sps.get("def_rebounds") or []):
            n = _norm(nm); _ensure(n); stats[n]["DRB"] += 1
            if pday: prac_days[n].add(pday)
        for nm in (sps.get("off_rebounds") or []):
            n = _norm(nm); _ensure(n); stats[n]["ORB"] += 1
            if pday: prac_days[n].add(pday)
        for nm in (sps.get("deflections") or []):
            n = _norm(nm); _ensure(n); stats[n]["DEF"] += 1
            if pday: prac_days[n].add(pday)
        for nm in (sps.get("steals") or []):
            n = _norm(nm); _ensure(n); stats[n]["STL"] += 1
            if pday: prac_days[n].add(pday)

        blk_names = set(_norm(nm) for nm in (sps.get("blocks") or []))
        for nm in blockers_from_pbp_for_display(pbp_src):
            blk_names.add(_norm(nm))
        for n in blk_names:
            if n:
                _ensure(n); stats[n]["BLK"] += 1
                if pday: prac_days[n].add(pday)

        for nm in (sps.get("live_ball_to") or []):
            n = _norm(nm); _ensure(n); stats[n]["LBTO"] += 1
            if pday: prac_days[n].add(pday)
        for nm in (sps.get("dead_ball_to") or []):
            n = _norm(nm); _ensure(n); stats[n]["DBTO"] += 1
            if pday: prac_days[n].add(pday)

        for nm in (sps.get("def_fouls") or []):
            n = _norm(nm); _ensure(n); stats[n]["PF"] += 1
            if pday: prac_days[n].add(pday)
        for nm in (sps.get("off_fouls") or []):
            n = _norm(nm); _ensure(n); stats[n]["OF"] += 1
            if pday: prac_days[n].add(pday)

        try:
            op, dp = participants_for_possession(pbp_src, short or "")
        except:
            op, dp = ([], [])
        for nm in (op or []):
            n = _norm(nm); _ensure(n); stats[n]["OP"] += 1
            if pday: prac_days[n].add(pday)
        for nm in (dp or []):
            n = _norm(nm); _ensure(n); stats[n]["DP"] += 1
            if pday: prac_days[n].add(pday)

    # Derivations
    for n, row in stats.items():
        row["TRB"] = row["ORB"] + row["DRB"]
        row["TO"]  = row["LBTO"] + row["DBTO"]
        row["FG%"] = round((row["FGM"]/row["FGA"])*100.0,1) if row["FGA"] else 0.0
        row["2P%"] = round((row["2PM"]/row["2PA"])*100.0,1) if row["2PA"] else 0.0
        row["3P%"] = round((row["3PM"]/row["3PA"])*100.0,1) if row["3PA"] else 0.0
        # fill PRAC from distinct set
        row["PRAC"] = len(prac_days.get(n, set()))

    cols = ["Player","FGM","FGA","FG%","2PM","2PA","2P%","3PM","3PA","3P%",
            "AST","DRB","ORB","TRB","LBTO","DBTO","TO","STL","DEF","BLK","OP","DP","PRAC"]
    out_rows = []
    for name in sorted(stats.keys(), key=lambda z: z.lower()):
        r = stats[name]
        out_rows.append({c: r.get(c, 0) for c in cols})
    return out_rows


def _create_zone_chart_from_filtered(rows):
    """Repaint Hot/Cold zones and labels based on filtered rows."""
    shots = _rows_to_shots(rows)
    fig = go.Figure()
    for tr in court_lines_traces(): fig.add_trace(tr)
    for tr in first_zone_line_traces(): fig.add_trace(tr)
    for tr in mini_three_point_line(): fig.add_trace(tr)
    for tr in elbow_lines(): fig.add_trace(tr)
    for tr in diagonal_zone_lines(): fig.add_trace(tr)

    zone_stats = {}
    for zid in range(1,15):
        makes = attempts = 0
        for s in shots:
            if point_in_zone(s["x"], s["y"], zid):
                attempts += 1
                if s["result"] == "Make": makes += 1
        pct = (makes/attempts*100.0) if attempts else 0.0
        zone_stats[zid] = {"makes":makes,"attempts":attempts,"percentage":round(pct,1)}

    zone_centers = {
        1:(24.5,5.5),2:(16.5,5.5),3:(24.5,12),4:(33,6.5),
        5:(6,3.5),6:(10,14.5),7:(24.5,22),8:(39,14.5),9:(42,3),
        10:(1,3.5),11:(5.5,20),12:(24.5,29),13:(42.5,21),14:(48.5,2.5)
    }
    for zid in range(1,15):
        s = zone_stats.get(zid, {"makes":0,"attempts":0,"percentage":0.0})
        rgba = _rgba_for_zone(zid, s["attempts"], s["percentage"])
        add_zone_fill(fig, zone_id=zid, step=0.25, rgba=rgba)

    for zid, center in zone_centers.items():
        s = zone_stats.get(zid, {"makes":0,"attempts":0,"percentage":0.0})
        txt = f"{s['makes']}/{s['attempts']}<br>{s['percentage']:.1f}%"
        fig.add_annotation(x=center[0], y=center[1], text=txt,
                           showarrow=False, font=dict(size=12, color="black", family="Arial Black"))

    fig.update_layout(**base_layout())
    return fig


# ===== populate dynamic filter options =====
@app.callback(
    Output("flt_drill_size_shoot", "options"),
    Output("flt_drill_full_shoot", "options"),
    Output("flt_shooter", "options"),
    Output("flt_defenders", "options"),
    Output("flt_assister", "options"),
    Output("flt_screen_assister", "options"),
    Output("flt_drill_size_stats", "options"),
    Output("flt_drill_full_stats", "options"),
    Output("flt_date_range_shoot", "min_date_allowed"),
    Output("flt_date_range_shoot", "max_date_allowed"),
    Output("flt_date_range_stats", "min_date_allowed"),
    Output("flt_date_range_stats", "max_date_allowed"),
    Input("tabs","value"),
    prevent_initial_call=False
)
def populate_filter_options(_tab_value):
    rows = _rows_full()

    drill_sizes = _uniq_sorted([r.get("drill_size") for r in rows])
    drills_full = _uniq_sorted([r.get("drill") for r in rows])

    shooters = _uniq_sorted(sum([_split_names(r.get("shooter_raw")) for r in rows], []))
    defenders = _uniq_sorted(sum([_split_names(r.get("defenders_raw")) for r in rows], []))
    assisters = _uniq_sorted(sum([_split_names(r.get("assister_raw")) for r in rows], []))
    screeners = _uniq_sorted(sum([_split_names(r.get("screen_assisters_raw")) for r in rows], []))

    dates = sorted([d for d in [r.get("practice_date_obj") for r in rows] if isinstance(d, date)])
    dmin = dates[0] if dates else None
    dmax = dates[-1] if dates else None

    def _opts(lst): return [{"label":x, "value":x} for x in lst]

    return (_opts(drill_sizes), _opts(drills_full),
            _opts(shooters), _opts(defenders), _opts(assisters), _opts(screeners),
            _opts(drill_sizes), _opts(drills_full),
            dmin, dmax, dmin, dmax)

# ===== clear filter buttons =====
@app.callback(
    Output("flt_date_range_shoot", "start_date"),
    Output("flt_date_range_shoot", "end_date"),
    Output("flt_drill_size_shoot", "value"),
    Output("flt_drill_full_shoot", "value"),
    Output("flt_shooter", "value"),
    Output("flt_defenders", "value"),
    Output("flt_assister", "value"),
    Output("flt_screen_assister", "value"),
    Output("flt_onball", "value"),
    Output("flt_offball", "value"),
    Output("flt_defense_shoot", "value"),
    Input("btn_clear_shoot","n_clicks"),
    prevent_initial_call=True
)
def clear_shoot_filters(nc):
    return (None, None, [], [], [], [], [], [], [], [], [])

@app.callback(
    Output("flt_date_range_stats", "start_date"),
    Output("flt_date_range_stats", "end_date"),
    Output("flt_drill_size_stats", "value"),
    Output("flt_drill_full_stats", "value"),
    Output("flt_defense_stats", "value"),
    Input("btn_clear_stats","n_clicks"),
    prevent_initial_call=True
)
def clear_stats_filters(nc):
    return (None, None, [], [], [])

# ===== core computation =====
@app.callback(
    Output("shot_chart","figure"),
    Output("zone_chart","figure"),
    Output("shooting_stats_box","children"),
    Output("status","children"),
    Output("stats_table","data"),

    # SHOOTING filters
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

    # STATS filters
    Input("flt_date_range_stats","start_date"),
    Input("flt_date_range_stats","end_date"),
    Input("flt_drill_size_stats","value"),
    Input("flt_drill_full_stats","value"),
    Input("flt_defense_stats","value"),

    State("sel_pos","data"),
    prevent_initial_call=False
)
def compute_all(sd_sh, ed_sh, sz_sh, dr_sh, shtr, defs, asts, scrs, onb, offb, def_sh,
                sd_st, ed_st, sz_st, dr_st, def_st,
                sel_coords):
    try:
        rows = _rows_full()

        shooting_rows = _filter_rows(
            rows,
            _parse_date_any(sd_sh), _parse_date_any(ed_sh),
            (sz_sh or []), (dr_sh or []),
            (shtr or []), (defs or []), (asts or []), (scrs or []),
            (onb or []), (offb or []), (def_sh or []),
            for_stats_tab=False
        )
        stats_rows = _filter_rows(
            rows,
            _parse_date_any(sd_st), _parse_date_any(ed_st),
            (sz_st or []), (dr_st or []),
            [], [], [], [],
            [], [], (def_st or []),
            for_stats_tab=True
        )

        shots_fig = create_shot_chart(_rows_to_shots(shooting_rows), highlight_coords=(sel_coords or []))
        zone_fig  = _create_zone_chart_from_filtered(shooting_rows)

        shots_fig.update_layout(uirevision="keep")
        zone_fig.update_layout(uirevision="keep")

        tiles = _shooting_tiles(shooting_rows)
        status_txt = f"Loaded {len(rows)} shots • Shooting view: {len(shooting_rows)} filtered • Stats view: {len(stats_rows)} filtered"

        stats_table = _aggregate_stats_table(stats_rows)

        return shots_fig, zone_fig, tiles, status_txt, stats_table

    except Exception as e:
        fb_rows  = _rows_full()
        fb_shots = _rows_to_shots(fb_rows)
        fallback_shots = create_shot_chart(fb_shots, highlight_coords=(sel_coords or []))
        fallback_zone  = _create_zone_chart_from_filtered(fb_rows)
        fallback_shots.update_layout(uirevision="keep")
        fallback_zone.update_layout(uirevision="keep")
        return (fallback_shots, fallback_zone, html.Div(), f"Error computing view: {e}", [])

# ---------------- Legend text population ----------------
# ---------------- Legend text population (ranges, not Low/Mid/High words) ----------------
def _legend_bins_for_zone(zid: int):
    """
    Infer the percentage cut points for a given zone category by probing the
    existing _rgba_for_zone() color mapping (attempts=1 to avoid white).
    Returns text like ('0–25%', '26–35%', '36%+').
    """
    last = None
    cuts = []
    for p in range(0, 101):
        col = str(_rgba_for_zone(zid, attempts=1, percentage=p))
        if last is None:
            last = col
        elif col != last:
            cuts.append(p)
            last = col

    def fmt(a, b):
        if a < 0: a = 0
        if b > 100: b = 100
        return f"{a}–{b}%"

    if len(cuts) >= 2:
        t1, t2 = cuts[0], cuts[1]
        low  = fmt(0, max(t1-1, 0))
        mid  = fmt(t1, max(t2-1, t1))
        high = f"{t2}%+"
    elif len(cuts) == 1:
        t1 = cuts[0]
        low, mid, high = fmt(0, max(t1-1, 0)), f"{t1}%+", "—"
    else:
        low, mid, high = "0–100%", "—", "—"
    return (low, mid, high)



#------------------------------Section 6----------------------------------------------------------





# ---------------- Legend text population (fixed to your _rgba_for_zone bins) ----------------
@app.callback(
    Output("legend_close_low", "children"),
    Output("legend_close_mid", "children"),
    Output("legend_close_high", "children"),
    Output("legend_mid_low", "children"),
    Output("legend_mid_mid", "children"),
    Output("legend_mid_high", "children"),
    Output("legend_three_low", "children"),
    Output("legend_three_mid", "children"),
    Output("legend_three_high", "children"),
    Input("zone_chart", "figure"),
    prevent_initial_call=False
)
def update_zone_legend(_zone_fig):
    # Close Shots (zones 1–4): 0–50 blue, 51–60 yellow, 61+ red
    close_bins = ("0–50%", "51–60%", "61%+")
    # Mid Range Shots (zones 5–9): 0–35 blue, 36–45 yellow, 46+ red
    mid_bins   = ("0–35%", "36–45%", "46%+")
    # 3 Point Shots (zones 10–14): 0–25 blue, 26–35 yellow, 36+ red
    three_bins = ("0–25%", "26–35%", "36%+")
    return (*close_bins, *mid_bins, *three_bins)



# ===== details panel =====
@app.callback(
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
                    assister = a.get("giver","")
                    break

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
            return html.Div([
                html.Span(f"{label}:", style={"fontWeight":600, "marginRight":"6px"}),
                html.Span((val or ""), style={"whiteSpace":"pre-wrap"})
            ], style={"marginBottom":"6px"})

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
            # ---- OFF-BALL (includes hammer now) ----
            if t in ("bd","pn","fl","bk","awy","ucla","crs","wdg","rip","stg","ivs","elv","hm"):
                rows = [("Action", (a.get("label","") or ""))]
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
            if nn and k not in seen_op:
                seen_op.add(k); op_list.append(nn)

        dp_list = []
        seen_dp = set()
        for nm in (_dp or []):
            nm = _strip_trailing_modifiers(nm)
            nn = _norm_block(nm, roster_full_list)
            k = (nn or "").lower()
            if nn and k not in seen_dp:
                seen_dp.add(k); dp_list.append(nn)

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

        # ---- Move Close button to top-right ABOVE the shorthand block
        top_close = html.Div(
            html.Button(
                "Close",
                id={"type":"close_details","idx":0},
                n_clicks=0,
                style={"padding":"6px 10px","borderRadius":"8px","border":"1px solid #aaa","background":"white"}
            ),
            style={"display":"flex","justifyContent":"flex-end","marginBottom":"6px"}
        )

        # Header now only contains the info (no Close button inside)
        header = html.Div(
            header_top,
            style={"display":"flex","justifyContent":"space-between","alignItems":"center"}
        )

        ident_rows = [
            ("Shooter", shooter),
            ("Defender(s)" if _multi_defenders else "Defender", onball_def_display),
        ]
        if assister:
            ident_rows.append(("Assisted by", assister))
        if screen_ast_list:
            ident_rows.append(("Screen Assist", ", ".join(screen_ast_list)))
        if _shot_out_of_help and not _multi_defenders:
            ident_rows.append(("Out of help", "Yes"))

        specials_rows = special_stats_with_pbp_blocks(r.get("possession",""), pbp_src_for_roles) or []

        def _clean_display_name(tok: str) -> str:
            s = (tok or "").strip()
            s = re.sub(
                r"^(?:rebound|defensive\s+rebound|offensive\s+rebound|"
                r"turnover|live\s+ball\s+turnover|dead\s+ball\s+turnover|"
                r"steal|deflection|block|offensive\s+foul|defensive\s+foul|"
                r"from|by|commits|grabs|the)\s+",
                "", s, flags=re.IGNORECASE
            )
            m = re.search(rf"({_CAP}(?:\s+{_CAP})?)\s*$", s)
            if m:
                s = m.group(1)
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
            def_names = []
            for m in re.finditer(_DEFLECT_SUBJ_RE, txt_for_infer):
                def_names.append(_clean_display_name(m.group(1)))
            if def_names:
                base = _spec_by_label.get("Deflection", [])
                base_l = {b.lower() for b in base}
                merged = base + [n for n in def_names if n and n.lower() not in base_l]
                if merged:
                    _spec_by_label["Deflection"] = merged

            _STEAL_SUBJ_RE    = re.compile(rf"({_FULLNAME})\s+(?:steal(?:s|ed)?|stole)\b", re.IGNORECASE)
            _STEAL_PASSIVE_RE = re.compile(rf"\bsteal(?:s|ed)?\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
            stl_names = []
            for m in re.finditer(_STEAL_SUBJ_RE, txt_for_infer):
                stl_names.append(_clean_display_name(m.group(1)))
            for m in re.finditer(_STEAL_PASSIVE_RE, txt_for_infer):
                stl_names.append(_clean_display_name(m.group(1)))
            if stl_names:
                base = _spec_by_label.get("Steal", [])
                base_l = {b.lower() for b in base}
                merged = base + [n for n in stl_names if n and n.lower() not in base_l]
                if merged:
                    _spec_by_label["Steal"] = merged

            _OFF_FOUL_SUBJ_RE = re.compile(rf"({_FULLNAME})\s+commits\s+an?\s+offensive\s+foul\b", re.IGNORECASE)
            _OFF_FOUL_BY_RE   = re.compile(rf"\boffensive\s+foul\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
            off_foul_names = []
            for m in re.finditer(_OFF_FOUL_SUBJ_RE, txt_for_infer):
                off_foul_names.append(_clean_display_name(m.group(1)))
            for m in re.finditer(_OFF_FOUL_BY_RE, txt_for_infer):
                off_foul_names.append(_clean_display_name(m.group(1)))
            if off_foul_names:
                base = _spec_by_label.get("Offensive Foul", [])
                base_l = {b.lower() for b in base}
                merged = base + [n for n in off_foul_names if n and n.lower() not in base_l]
                if merged:
                    _spec_by_label["Offensive Foul"] = merged

            _DEF_FOUL_SUBJ_RE = re.compile(rf"({_FULLNAME})\s+commits\s+an?\s+defensive\s+foul\b", re.IGNORECASE)
            _DEF_FOUL_BY_RE   = re.compile(rf"\bdefensive\s+foul\s+by\s+({_FULLNAME})\b", re.IGNORECASE)
            def_foul_names = []
            for m in re.finditer(_DEF_FOUL_SUBJ_RE, txt_for_infer):
                def_foul_names.append(_clean_display_name(m.group(1)))
            for m in re.finditer(_DEF_FOUL_BY_RE, txt_for_infer):
                def_foul_names.append(_clean_display_name(m.group(1)))
            if def_foul_names:
                base = _spec_by_label.get("Defensive Foul", [])
                base_l = {b.lower() for b in base}
                merged = base + [n for n in def_foul_names if n and n.lower() not in base_l]
                if merged:
                    _spec_by_label["Defensive Foul"] = merged

        if _spec_by_label.get("Block"):
            ident_rows.append(("Block", ", ".join(_spec_by_label["Block"])))

        _special_order = [
            "Defensive Rebound","Offensive Rebound","Deflection","Steal",
            "Live Ball Turnover","Dead Ball Turnover","Defensive Foul",
            "Offensive Foul","Block"
        ]
        for lbl in _special_order:
            if lbl == "Block":
                continue
            ppl = _spec_by_label.get(lbl, [])
            if ppl:
                ident_rows.append((lbl, ", ".join(ppl)))

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
                style={"background":"#eef2ff","color":"#111827","padding":"10px","borderRadius":"8px",
                       "whiteSpace":"pre-wrap","border":"1px solid #c7d2fe","marginBottom":"8px"}
            )
        ])

        return html.Div([
            top_close,   # Close button at the very top-right
            pre,         # Shorthand + Play-by-play
            header,      # Info header (no Close button)
            identity,
            html.Div([html.Div("On-ball Actions", style={"fontWeight":700,"margin":"6px 0"}), *oblocks]) if oblocks else html.Div(),
            html.Div([html.Div("Off-ball Actions", style={"fontWeight":700,"margin":"10px 0 6px"}), *fblocks]) if fblocks else html.Div(),
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
