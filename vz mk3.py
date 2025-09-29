# vz_mk3 — on-ball & off-ball parsing table



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
    if not s:
        return None
    m = re.search(r'(?:\+\+|\+|-)(?!.*(?:\+\+|\+|-))', s)
    if not m:
        return None
    return "Make" if m.group(0) in ("+","++") else ("Miss" if m.group(0)=="-" else None)

def safe_load_data():
    """Load rows -> plottable shot list (x,y,result) with caching on failure."""
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
                res = row.get("result") or result_from_shorthand(row.get("possession", ""))
                if 0 <= x <= COURT_W and 0 <= y <= HALF_H and res in ("Make", "Miss"):
                    shots.append({"x": x, "y": y, "result": res})
            except:
                continue
        CACHED_DATA = shots
        return shots
    except:
        return CACHED_DATA

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
            "backdoor","backdoor cut","pin down","pindown","flare screen","back screen","away screen","ucla screen",
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
_COV_ICE = re.compile(r"\bice(?:s|d|ing)?\b", re.IGNORECASE)                  # ice

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


# ------------- Shot chart
def create_shot_chart(shots, highlight_coords=None):
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
    shots = safe_load_data()
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

app.layout = html.Div(
    style={"maxWidth":"1200px","margin":"0 auto","padding":"10px"},
    children=[
        html.Div([
            html.Div([
                html.Div("Shot Chart", style={"textAlign":"center","fontSize":"24px","fontWeight":700,"margin":"0 0 4px 0"}),
                dcc.Graph(id="shot_chart", config={"displayModeBar": False}, figure=_initial_shot_fig),
                html.Div([html.Span("● Make", style={"color":"green","marginRight":"20px","fontWeight":600}),
                          html.Span("✖ Miss", style={"color":"red","fontWeight":600})],
                         style={"textAlign":"center","margin":"-4px 0 0 0"})
            ], style={"flex":"1 1 0","minWidth":"520px"}),

            html.Div([
                html.Div("Hot/Cold Map", style={"textAlign":"center","fontSize":"24px","fontWeight":700,"margin":"0 0 4px 0"}),
                dcc.Graph(id="zone_chart", config={"displayModeBar": False}, figure=_initial_zone_fig),
            ], style={"flex":"1 1 0","minWidth":"520px"})
        ], style={"display":"flex","gap":"16px","alignItems":"flex-start","justifyContent":"center","flexWrap":"wrap"}),

        dcc.Interval(id="refresh", interval=20000, n_intervals=0),
        html.Div([
            html.Div(f"Data source: {DATA_PATH}", style={"color":"#666","fontSize":"12px","marginBottom":"4px"}),
            html.Div("Update method: Conservative polling (20s)", style={"color":"#888","fontSize":"10px"}),
            html.Div(id="status", style={"color":"#888","fontSize":"10px"}),
        ], style={"textAlign":"center","marginTop":"6px"}),

        html.Div(id="shot_details", style={"maxWidth":"920px","margin":"14px auto 0 auto"}),
        dcc.Store(id="sel_pos", data=[]),
    ]
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
            # prefer pair where object matches the bringer (if known)
            if bh_from_brings and o_name and o_name.lower() == bh_from_brings.lower():
                bh_def_from_pick = d_name
                bh_from_pick = o_name
                break
            # else remember last seen pair
            bh_def_from_pick, bh_from_pick = d_name, o_name

        for a in actions:
            if a.get("type") == "h":  # bring over halfcourt
                if not a.get("bh"):
                    a["bh"] = bh_from_brings or bh_from_pick or a.get("bh","")
                if not a.get("bh_def"):
                    # if bh known, choose defender that picked up that bh
                    if a.get("bh") and bh_from_pick and a["bh"].lower() == bh_from_pick.lower():
                        a["bh_def"] = bh_def_from_pick
                    else:
                        a["bh_def"] = bh_def_from_pick or a.get("bh_def","")
        return actions
    except:
        return actions

# ================== NEW HELPERS (no behavior changes elsewhere) ==================
# These are small utilities Section 3 can call to show:
#   • multi-defender strings with per-name "rotating over"
#   • a boolean "shot out of help" tag

_ROTATING_OVER_RE = re.compile(r"\brotating\s+over\b", re.IGNORECASE)
_HELP_VERBS_RE    = re.compile(r"\bhelp(?:s|ed|ing)?\b|\bsteps\s+in\s+to\s+help\b", re.IGNORECASE)

def shot_defender_display(pbp_text: str, shot_index: int) -> str:
    """
    Format defenders for the Nth shot line in a possession, preserving
    per-name 'rotating over' tags. Uses Section-1 parsers:
      _nth_shot_line, _parse_defenders_with_tags_from_line, _format_defenders_for_display
    """
    line = _nth_shot_line(pbp_text or "", shot_index)
    return _format_defenders_for_display(_parse_defenders_with_tags_from_line(line))

def shot_has_help(pbp_text: str, shot_index: int) -> bool:
    """
    True if the possession text or the specific shot line contains help cues
    (e.g., 'help', 'steps in to help', or 'rotating over').
    """
    txt = _clean_frag(pbp_text or "")
    if _HELP_VERBS_RE.search(txt):
        return True
    line = _nth_shot_line(txt, shot_index)
    return bool(_HELP_VERBS_RE.search(line) or _ROTATING_OVER_RE.search(line))


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
        a = _trim_trailing_verb(m.group(1).strip())
        b = _trim_trailing_verb(m.group(2).strip())
        a = _strip_leading_preps(a)
        b = _strip_leading_preps(b)
        out.append((a, b))
    return out

def parse_offball_actions_from_pbp(lines):
    """
    Coalesce multiple off-ball lines describing the same action (esp. stagger screens)
    into a *single* block. We merge:
      - coming off / coming-off defender (first non-empty wins),
      - screeners + screener defenders (dedup, preserve order),
      - coverages (dedup by label+onto).
    Other off-ball types behave the same as before.
    """
    # aggregator only for stagger screens (other types stay one-per-line)
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
            # initialize aggregator if first time we see a stagger line
            if stg_agg is None:
                stg_agg = {
                    "type": "stg",
                    "label": _OFFBALL_TYPES["stg"]["label"],
                    "coming_off": coming_off,
                    "coming_off_def": coming_off_def,
                    "screeners": [],
                    "coverages": [],
                }

            # fill coming-off names if we get them later
            if not stg_agg.get("coming_off") and coming_off:
                stg_agg["coming_off"] = coming_off
            if not stg_agg.get("coming_off_def") and coming_off_def:
                stg_agg["coming_off_def"] = coming_off_def

            # merge screeners (dedup by name/def lower)
            for s in screeners:
                key = (s.get("name","").lower(), s.get("def","").lower())
                if key not in stg_seen_scr and (s.get("name") or s.get("def")):
                    stg_agg["screeners"].append({"name": s.get("name",""), "def": s.get("def","")})
                    stg_seen_scr.add(key)

            # merge coverages (dedup by label+onto lower)
            for c in (covs or []):
                key = (c.get("label","").lower(), (c.get("onto","") or "").lower())
                if key not in stg_seen_cov:
                    stg_agg["coverages"].append(c)
                    stg_seen_cov.add(key)

            continue  # do not append yet

        # default behavior for all other off-ball types (unchanged)
        d = {
            "type": matched_key,
            "label": _OFFBALL_TYPES[matched_key]["label"],
            "coming_off": coming_off,
            "coming_off_def": coming_off_def,
            "screeners": screeners,
            "coverages": covs,
        }
        actions.append(d)

    # append the single, merged stagger screen (if present)
    if stg_agg:
        actions.append(stg_agg)

    return actions


# ================== NEW: defender-rotation decoration across ANY possession line ==================
def _last_token(n: str) -> str:
    n = (n or "").strip()
    if not n: return ""
    parts = re.split(r"\s+", n)
    return parts[-1] if parts else n

def _word_or_last_regex(name_full: str) -> str:
    """
    Build a regex that matches either the exact full name OR just its last name.
    Safely escaped for regex usage.
    """
    full = re.escape((name_full or "").strip())
    last = re.escape(_last_token(name_full))
    if not full and not last:
        return r""
    if full and last and full.lower() != last.lower():
        return rf"(?:{full}|{last})"
    return full or last

def _decorate_defenders_with_rotation(def_str: str, shooter_name: str, pbp_text: str) -> str:
    """
    If any line in the possession contains 'guarded by <Defender> rotating over'
    (preferably with the current shooter), append 'rotating over' to that
    defender in the display string.
    Matches both full-name and last-name-only forms.
    """
    if not def_str:
        return def_str
    txt = _clean_frag(pbp_text or "")

    # Split current defender display into individual names
    names = _split_fullname_list(def_str)
    if not names:
        names = [def_str.strip()]

    decorated = []
    shooter_pat = _word_or_last_regex(shooter_name)
    for nm in names:
        def_pat = _word_or_last_regex(nm)
        has_rot = False
        # Prefer an exact shooter/defender pairing (full OR last names)
        if shooter_pat and def_pat:
            patt1 = re.compile(rf"\b{shooter_pat}\s+guarded\s+by\s+{def_pat}\s+rotating\s+over\b",
                               re.IGNORECASE)
            if patt1.search(txt):
                has_rot = True
        # Fallback: any 'guarded by <def> rotating over' (full OR last name)
        if not has_rot and def_pat:
            patt2 = re.compile(rf"\bguarded\s+by\s+{def_pat}\s+rotating\s+over\b", re.IGNORECASE)
            if patt2.search(txt):
                has_rot = True
        decorated.append(f"{nm} rotating over" if has_rot else nm)

    return ", ".join(decorated)


# ---------------- callbacks ----------------
@app.callback(
    Output("shot_chart","figure"),
    [Input("refresh","n_intervals"),
     Input("sel_pos","data")],
    prevent_initial_call=False
)
def update_shot_chart(n, sel_coords):
    shots = safe_load_data()
    return create_shot_chart(shots, highlight_coords=(sel_coords or []))

@app.callback(
    [Output("zone_chart","figure"),
     Output("status","children")],
    Input("refresh","n_intervals"),
    prevent_initial_call=False
)
def update_zone_and_status(n):
    try:
        return create_zone_chart(), f"Loaded {len(safe_load_data())} shots (Update #{n})"
    except Exception as e:
        return create_zone_chart(), f"Error: {e}"

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

        rows_for_plot = []
        if os.path.exists(DATA_PATH):
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
                    rows_for_plot.append({**rr, "x":x, "y":y, "result":res})

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

        # Prefer shot-line defender string (handles multi-defenders + rotating over on the shot line)
        _def_disp_from_shot_line = shot_defender_display(pbp_src_for_roles, idx or 1)
        if _def_disp_from_shot_line:
            onball_def = _def_disp_from_shot_line

        # Detect "shot out of help"
        _shot_out_of_help = shot_has_help(pbp_src_for_roles, idx or 1)

        onball_actions = parse_onball_actions_from_pbp(action_lines, (screen_ast_list[0] if screen_ast_list else ""))
        # Patch Bring over halfcourt BH/BH_def using "X picks up Y" if needed
        onball_actions = _patch_bring_over_halfcourt(onball_actions, pbp_src_for_roles)

        offball_actions = parse_offball_actions_from_pbp(action_lines)

        if not assister:
            for a in reversed(onball_actions):
                if a.get("type") in ("ho","dho") and a.get("receiver","").lower() == (shooter or "").lower():
                    assister = a.get("giver","")
                    break

        roster_full_list = _collect_roster_for_group(rows_for_plot, gid_key)

        # Normalize canonical values first
        shooter = _norm_block(shooter, roster_full_list)
        onball_def = _norm_block(onball_def, roster_full_list)
        assister = _norm_block(assister, roster_full_list)

        # --- DISPLAY-ONLY defender string that preserves 'rotating over' (full-name or last-name matches)
        _base_defs = _split_fullname_list(onball_def) or ([onball_def] if onball_def else [])
        # normalize each piece to ensure full-name display
        _base_defs = [_norm_block(nm, roster_full_list) for nm in _base_defs if nm]
        onball_def_display = _decorate_defenders_with_rotation(", ".join(_base_defs), shooter, pbp_src_for_roles)
        _def_list_for_count = _split_fullname_list(onball_def_display)
        _multi_defenders = len(_def_list_for_count) >= 2

        # RE-SPLIT and normalize each piece to ensure clean Screen Assist list
        uniq_screens = []
        seen_sa = set()
        for nm in (screen_ast_list or []):
            for piece in _split_fullname_list(nm):
                nm2 = _norm_block(piece, roster_full_list)
                low = (nm2 or "").lower()
                if low and low not in seen_sa:
                    uniq_screens.append(nm2)
                    seen_sa.add(low)
        screen_ast_list = uniq_screens

        for a in onball_actions:
            for k in ("bh","bh_def","screener","screener_def","giver","giver_def","receiver","receiver_def","keeper","keeper_def","intended","intended_def"):
                if k in a: a[k] = _norm_block(a[k], roster_full_list)
            # normalize multi-screener entries if present
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

        # --- Never show coverage for "Bring over halfcourt"
        for a in onball_actions:
            if a.get("type") == "h":
                a["coverages"] = []
        # --- end no-coverage-for-halfcourt ---

        # --- NEW: Remove 'Switch' coverage that isn't tied to a screen/handoff action
        _screen_handoff_types = {"pnr","pnp","rs","slp","gst","rj","dho","ho","kp"}
        for a in onball_actions:
            t = (a.get("type") or "").lower()
            if t not in _screen_handoff_types:
                if a.get("coverages"):
                    a["coverages"] = [c for c in a["coverages"] if (c.get("cov") != "sw")]
        # --- END unattached-switch filter ---

        # --- If the possession text mentions "screen assist", ensure all screeners are credited ---
        sa_phrase = "screen assist" in ((pbp_names_src or pbp_raw_src or "")).lower()
        if sa_phrase:
            # Gather normalized screeners from on-ball actions (pnr/pnp/rs)
            scr_from_actions = []
            for a in onball_actions:
                if a.get("type") in ("pnr", "pnp", "rs"):
                    if a.get("screeners"):
                        scr_from_actions.extend([s.get("name","") for s in a["screeners"] if s.get("name")])
                    else:
                        if a.get("screener"):
                            scr_from_actions.append(a.get("screener"))
            # De-dupe
            scr_from_actions = [n for n in scr_from_actions if n]
            scr_seen = set()
            scr_from_actions = [n for n in scr_from_actions if not (n.lower() in scr_seen or scr_seen.add(n.lower()))]
            # If parser only found 0/1 names but actions have more, promote action names
            if scr_from_actions and len(screen_ast_list) < len(scr_from_actions):
                screen_ast_list = scr_from_actions
        # --- END ensure screen assist credit ---

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

            # Bring / Drive / Keep — generic BH labels
            if t in ("h","d","kp"):
                blocks += [line("Ball Handler", a.get("bh", a.get("keeper",""))),
                           line("Ball Handler Defender", a.get("bh_def", a.get("keeper_def","")))]
                if t == "kp":
                    if a.get("intended"): blocks.append(line("Intended receiver", a.get("intended","")))
                    if a.get("intended_def"): blocks.append(line("Intended defender", a.get("intended_def","")))
                    if a.get("coverages"):
                        blocks.append(line("Coverage", cov_text(a.get("coverages"))))  # coverage only for kp here
                return html.Div(blocks, style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

            # NEW: Post up — show post-specific field names
            if t == "p":
                rows = [
                    ("Action", a.get("label","")),
                    ("Posting up", a.get("bh","")),
                    ("Defending Post up", a.get("bh_def","")),
                ]
                if a.get("coverages"):
                    rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

            # On-ball screens
            if t in ("pnr","pnp","rs"):
                rows = [
                    ("Action", a.get("label","")),
                    ("Ball Handler", a.get("bh","")),
                    ("Ball Handler Defender", a.get("bh_def","")),
                ]
                scr_list = a.get("screeners") or []
                if scr_list:
                    scr_txt = ", ".join(s.get("name","") for s in scr_list)
                    scr_def_txt = ", ".join(s.get("def","") for s in scr_list)
                    rows += [
                        ("Screener(s)", scr_txt),
                        ("Screener(s) Defender(s)", scr_def_txt),
                    ]
                else:
                    rows += [
                        ("Screener", a.get("screener","")),
                        ("Screener defender", a.get("screener_def","")),
                    ]
                if a.get("coverages"):
                    rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

            # Slip / Ghost / Reject
            if t in ("slp","gst","rj"):
                rows = [
                    ("Action", a.get("label","")),
                    ("Ball Handler", a.get("bh","")),
                    ("Ball Handler Defender", a.get("bh_def","")),
                    ("Screener", a.get("screener","")),
                    ("Screener defender", a.get("screener_def","")),
                ]
                if a.get("coverages"):
                    rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

            # Handoffs
            if t in ("dho","ho"):
                rows = [
                    ("Action", a.get("label","")),
                    ("Giver", a.get("giver","")),
                    ("Giver defender", a.get("giver_def","")),
                    ("Receiver", a.get("receiver","")),
                    ("Receiver defender", a.get("receiver_def","")),
                ]
                if a.get("coverages"):
                    rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

            # Off-ball
            if t in ("bd","pn","fl","bk","awy","ucla","crs","wdg","rip","stg","ivs","elv"):
                rows = [("Action", (a.get("label","") or "").lower())]
                if t == "bd":
                    rows += [
                        ("Cutter", a.get("coming_off","")),
                        ("Cutter defender", a.get("coming_off_def","")),
                    ]
                else:
                    rows += [
                        ("Coming off screen", a.get("coming_off","")),
                        ("Defender on coming-off player", a.get("coming_off_def","")),
                    ]
                    scr_txt = ", ".join(s.get("name","") for s in (a.get("screeners") or []))
                    scr_def_txt = ", ".join(s.get("def","") for s in (a.get("screeners") or []))
                    if scr_txt: rows.append(("Screener(s)", scr_txt))
                    if scr_def_txt: rows.append(("Screener(s) Defender(s)", scr_def_txt))
                if a.get("coverages"):
                    rows.append(("Coverage", cov_text(a.get("coverages"))))
                return html.Div([mini_table(rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

            return html.Div(blocks, style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px"})

        practice = r.get("practice_date") or r.get("practice") or r.get("date") or ""
        drill = r.get("drill") or r.get("practice_drill") or r.get("drill_name") or ""
        header_top = html.Div([
            html.Div([
                html.Span("Shot details", style={"fontWeight":700,"fontSize":"18px","marginRight":"8px"}),
                html.Span(f"({shot_num_display})" if shot_num_display else "", style={"color":"#666"}),
                html.Span(f" • Result: {r.get('result','')}", style={"marginLeft":"6px","color":"#444"}),
            ]),
            html.Div([
                html.Span(f"Practice: {practice}" if practice else "", style={"marginRight":"16px","color":"#555"}),
                html.Span(f"Drill: {drill}" if drill else "", style={"color":"#555"}),
            ])
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"baseline","gap":"10px","flexWrap":"wrap"})

        header_btn = html.Button("Close", id={"type":"close_details","idx":0}, n_clicks=0,
                        style={"padding":"6px 10px","borderRadius":"8px","border":"1px solid #aaa","background":"white"})

        header = html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center"}, children=[
            header_top, header_btn
        ])

        ident_rows = [
            ("Shooter", shooter),
            ("Defender(s)" if _multi_defenders else "Defender", onball_def_display),  # pluralize if needed
        ]
        if assister:
            ident_rows.append(("Assisted by", assister))
        if screen_ast_list:
            ident_rows.append(("Screen Assist", ", ".join(screen_ast_list)))
        # show "Out of help" only if single defender listed
        if _shot_out_of_help and not _multi_defenders:
            ident_rows.append(("Out of help", "Yes"))

        identity = html.Div([mini_table(ident_rows)], style={"margin":"8px 0","padding":"8px","border":"1px solid #ddd","borderRadius":"8px","background":"#fafafa"})

        oblocks = [action_block(a) for a in onball_actions]
        fblocks = [action_block(a) for a in offball_actions]

        pre_lines = [f"Shorthand:\n  {(r.get('possession') or '').strip()}"]
        if (pbp_names_src or "").strip():
            pre_lines.append(f"\nPlay-by-play (names):\n{(pbp_names_src or '').strip()}")
        pre = html.Div([
            html.Pre(
                "\n".join(pre_lines),
                style={"background":"#eef2ff","color":"#111827","padding":"10px","borderRadius":"8px",
                       "whiteSpace":"pre-wrap","border":"1px solid #c7d2fe","marginBottom":"8px"}
            )
        ])

        return html.Div([
            pre,
            header,
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
