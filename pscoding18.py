# pscoding18
# Mark 18 - updated after stage 2 of testing (1 full wnba game)



# ---------------- Utilities ----------------

import re

# Tracks temporary switches within a possession:
# keys are (off_player_as_str_or_asterisk, defender_id_str) -> True
defender_memory = {}

# Pending screen assist(s) (for off-ball screens only)
# Shape: {"recipient": <player_id_str>, "screeners": [<player_id_str>, ...]}
pending_offball_sa = None

# Remember defenders on the last shot so we can carry them into an OR if needed
last_shot_defenders = []

# Track loose-ball state after a deflection; next possessor prints a recovery line
loose_ball_pending = False

# ---------------- NEW: Zone Defense State ----------------
# Active only while inside bracketed segments: [ ... ] within a possession.
zone_active = False


def norm_id(x):
    """
    Normalize a player id string:
    - If purely digits, strip leading zeros (e.g., '01' -> '1', '0' -> '0').
    - Leave tokens like 'rot\\d+' intact.
    - Leave non-digit strings intact.
    """
    if x is None:
        return None
    s = str(x)
    if s.startswith("rot"):
        return s
    if s.isdigit():
        return str(int(s))  # handles '0' -> '0' and '01' -> '1'
    s2 = s.lstrip("0")
    return s2 if s2 != "" else "0"


def norm_list(lst):
    if not lst:
        return []
    return [norm_id(v) for v in lst if v is not None]


def _split_off_and_defs(token):
    """
    If token looks like 'off/def1,def2' return (off, [defs]).
    Otherwise return (norm_id(token), None).
    Supports defenders as digits or 'rot\\d+'.
    """
    if isinstance(token, str) and "/" in token:
        off, rest = token.split("/", 1)
        m = re.match(r'((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)', rest)
        if m:
            defs = [d for d in m.group(1).split(",") if d]
            return norm_id(off), norm_list(defs)
        return norm_id(off), []
    return norm_id(token), None


# ---------------- NEW: Zone bracket utility ----------------
def strip_zone_marks(tok: str):
    """
    Remove leading '[' and/or trailing ']' and capture an optional zone label
    placed immediately before '[' in the same token (e.g., '2-3[').
    Returns: (cleaned_token, zone_start_bool, zone_end_bool, zone_label_or_None)

    Examples:
      '2-3[5/6'  -> ('5/6', True,  False, '2-3')
      '7/8]'     -> ('7/8', False, True,  None)
      '[1/2'     -> ('1/2', True,  False, None)
      '3/4'      -> ('3/4', False, False, None)
      '1/2]'     -> ('1/2', False, True,  None)
      'matchup'  -> ('matchup', False, False, None)
    """
    start = False
    end = False
    label = None
    s = tok

    # If there's a '[', treat everything before it (if non-empty) as the label
    if "[" in s:
        idx = s.find("[")
        if idx > 0:
            candidate = s[:idx].strip()
            label = candidate if candidate else None
        start = True
        s = s[idx + 1:]  # drop everything through the '['

    # Handle a trailing ']' anywhere in the remaining token
    if "]" in s:
        end = True
        s = s.replace("]", "")

    return s, start, end, label



# ---------------- Parsing Functions ----------------

def parse_player_def(token: str):
    """
    Parse a token into:
      (ball_handler, ball_def, action_type, screener, screener_def, action_codes)

    No-caret format:
      - On-ball micro-actions use: h (half-court), d (drive), p (post)
      - Handoff/keep: dho, ho, kp
      - PNR/PNP: pnr, pnp (with chain codes ch/ct/swN/bz/tl/cs/ice/h/d/dho/ho/kp/+ ++ -)
      - Off-ball in (...) with actions: pn, fl, bk, awy, crs, wdg, rip, ucla, stg, ivs, elv, bd
      - Defenders can be comma-separated digits or rotX

    Multiple defenders anywhere (comma-separated) will be parsed into lists.

    NOTE (PNR/PNP only):
      action_codes is returned as a dict:
         {"per_screen": [codes aligned with screener occurrences], "trail": "<leftover non-screener codes>"}
      The main loop should consume both the per-screen codes and any trailing codes (e.g., 'd-', 'h', '+').
    """
    token = token.strip()

    # Off-ball action (parentheses)
    if token.startswith("(") and token.endswith(")"):
        inner = token[1:-1]
        action_type = None
        # ADD: 'bd' (backdoor cut)
        for key in ("pn", "fl", "bk", "hm", "awy", "crs", "wdg", "rip", "ucla", "stg", "ivs", "elv", "bd"):
            if key in inner:
                action_type = key
                break
        if action_type:
            if action_type in ("stg", "ivs", "elv"):
                parts = inner.split(action_type, 1)
                if len(parts) != 2:
                    return None, None, None, None, None, None
                ball_part = parts[0].strip()
                rest = parts[1].strip()
                ball_handler, ball_def = None, None
                if "/" in ball_part:
                    bh_parts = ball_part.split("/", 1)
                    ball_handler = norm_id(bh_parts[0]) if bh_parts[0] else None
                    if len(bh_parts) > 1 and bh_parts[1]:
                        ball_def = norm_list(re.findall(r"(?:rot\d+|\d+)", bh_parts[1]))
                    else:
                        ball_def = None
                else:
                    ball_handler = norm_id(ball_part) if ball_part else None

                screener_list, screendef_list, after_list = parse_stagger_screens(rest)
                return ball_handler, ball_def, action_type, screener_list, screendef_list, after_list

            # Single or multi off-ball action (incl. bd)
            ball_part, rest = inner.split(action_type, 1)
            ball_handler, ball_def = None, None
            if "/" in ball_part:
                bh_parts = ball_part.split("/", 1)
                ball_handler = norm_id(bh_parts[0]) if bh_parts[0] else None
                if len(bh_parts) > 1 and bh_parts[1]:
                    ball_def = norm_list(re.findall(r"(?:rot\d+|\d+)", bh_parts[1]))
                else:
                    ball_def = None
            else:
                ball_handler = norm_id(ball_part) if ball_part else None

            # For bd there is no screener section; parse_stagger_screens('') returns empties
            screener_list, screendef_list, after_list = parse_stagger_screens(rest)
            if len(screener_list) == 1 and (len(after_list) == 1 and after_list[0] == ""):
                return ball_handler, ball_def, action_type, screener_list[0], screendef_list[0], ""
            else:
                return ball_handler, ball_def, action_type, screener_list, screendef_list, after_list
        return None, None, None, None, None, None

    # On-ball actions (include kp = handoff keep)
    # FIX: choose the LEFTMOST action token present in the string
    action_type = None
    _keys = ("pnr", "pnp", "slp", "gst", "rj", "dho", "ho", "kp")
    _best_pos = None
    for key in _keys:
        pos = token.find(key)
        if pos != -1 and (_best_pos is None or pos < _best_pos):
            _best_pos = pos
            action_type = key
    if action_type:
        # Special 'reject' parsing
        if action_type == "rj":
            if "/" in token:
                ball_part, rest = token.split("rj", 1)
                if "/" in ball_part:
                    ball_handler, ball_def_raw = ball_part.split("/", 1)
                    ball_handler = norm_id(ball_handler)
                    ball_def = norm_list(re.findall(r"(?:rot\d+|\d+)", ball_def_raw)) if ball_def_raw else []
                else:
                    ball_handler = norm_id(ball_part) if ball_part.strip() else None
                    ball_def = []
                # strip accidental trailing 'h' on defender ids from head-side normalization
                ball_def = [re.sub(r"h$", "", d) for d in ball_def]

                if "/" in rest:
                    screener, screener_def_raw = rest.split("/", 1)
                    screener = re.sub(r"(ch|ct|sw\d*|bz|tl|cs|h|d|dho|ho|kp|\+{1,2}|-)+$", "", screener)
                    screener = norm_id(screener)
                    cleaned_def = re.sub(r"(ch|ct|sw\d*|bz|tl|cs|h|d|dho|ho|kp|\+{1,2}|-)+$", "", screener_def_raw)
                    screener_def = norm_list(re.findall(r"(?:rot\d+|\d+)", cleaned_def)) if cleaned_def else []
                else:
                    m = re.match(r"(\d+)", rest)
                    if m:
                        screener = norm_id(m.group(1))
                        screener_def = []
                        rest = rest[len(m.group(1)):]
                    else:
                        screener = norm_id(rest)
                        screener_def = []
                        rest = ""
                trailing_codes = [m.group(0) for m in re.finditer(
                    r"(ch|ct|sw\d*|bz|tl|cs|ice|h(?!o)|d(?!ho)|dho|ho|kp|\+{1,2}|-)", rest.strip()
                )]
                trailing_codes = trailing_codes if trailing_codes else []
                return ball_handler, ball_def, action_type, screener, screener_def, trailing_codes
            else:
                parts = token.split("rj")
                ball_handler = norm_id(parts[0]) if parts[0].strip() else None
                ball_def = []
                screener = norm_id(parts[1]) if len(parts) > 1 else None
                screener_def = []
                return ball_handler, ball_def, action_type, screener, screener_def, ""

        # General on-ball (including pnr/pnp and others)
        ball_part, rest = token.split(action_type, 1)
        if "/" in ball_part:
            ball_handler, def_and_actions = ball_part.split("/", 1)
            ball_handler = norm_id(ball_handler) if ball_handler.strip() else None
            ball_def = norm_list(re.findall(r"(?:rot\d+|\d+)", def_and_actions)) if def_and_actions else []
        else:
            ball_handler = norm_id(ball_part) if ball_part.strip() else None
            ball_def = []
        # strip accidental trailing 'h' on defender ids from head-side normalization
        ball_def = [re.sub(r"h$", "", d) for d in ball_def]

        # PNR/PNP multi/rescreen  --->  now returns (screeners, screendefs, actions, trailing)
        if action_type in ("pnr", "pnp"):
            screeners, screendefs, actions, trailing = parse_onball_screens(rest)
            # Pass both the aligned per-screen actions and the trailing leftover codes to the caller
            return ball_handler, ball_def, action_type, screeners, screendefs, {"per_screen": actions, "trail": trailing}

        # Others (slp/gst/dho/ho/kp)
        action_codes = rest
        screener, screener_def, action = None, None, ""
        if "/" in action_codes:
            j = 0
            while j < len(action_codes) and (action_codes[j].isdigit() or action_codes[j:j+3] == "rot"):
                if action_codes[j:j+3] == "rot":
                    j += 3
                    while j < len(action_codes) and action_codes[j].isdigit():
                        j += 1
                    break
                j += 1
            screener = norm_id(action_codes[:j])
            rest2 = action_codes[j:]
            if rest2.startswith("/"):
                k = 1
                if rest2[1:4] == "rot":
                    k = 4
                    while k < len(rest2) and rest2[k].isdigit():
                        k += 1
                else:
                    while k < len(rest2) and rest2[k].isdigit():
                        k += 1
                screener_def = norm_list([rest2[1:k]]) if k > 1 else []
                action = rest2[k:]
            else:
                action = rest2
        else:
            i = 0
            while i < len(action_codes) and (action_codes[i].isdigit() or action_codes[i:i+3] == "rot"):
                if action_codes[i:i+3] == "rot":
                    i += 3
                    while i < len(action_codes) and action_codes[i].isdigit():
                        i += 1
                    break
                i += 1
            screener = norm_id(action_codes[:i])
            action = action_codes[i:]
        return ball_handler, ball_def, action_type, screener, screener_def, action


def parse_stagger_screens(s):
    """
    Parse stagger/iverson/elevator screen format (no-caret codes)
    Stagger:  "5/6sw7/8ch"
    Elevator: "5/6,7/8sw6"
    Returns (screener_list, screendef_list, action_list)

    Action codes allowed here: ch, ct, sw, bz, tl, cs, d (drive), dho, ho, +, ++, -
    (Note: 'd' must not consume 'dho', so we use (d(?!ho)) where regex is used.)
    """
    s = s.replace(" ", "")
    screener_list = []
    screendef_list = []
    action_list = []

    if ',' in s and re.search(r"\d/\d,|\d,", s):  # Elevator format has commas between screeners
        i = 0
        while i < len(s):
            screener_match = re.match(r"(\d+)", s[i:])
            if not screener_match:
                break
            screener = norm_id(screener_match.group(1)); i += len(screener_match.group(1))
            defender = None
            if i < len(s) and s[i] == '/':
                i += 1
                defender_match = re.match(r"(\d+)", s[i:])
                if defender_match:
                    defender = norm_id(defender_match.group(1)); i += len(defender_match.group(1))
            screener_list.append(screener)
            screendef_list.append([defender] if defender is not None else [])
            if i < len(s) and s[i] == ',':
                i += 1; action_list.append("")
            else:
                action = ""
                action_match = re.match(r"(ch|ct|sw\d*|bz\d*|tl|cs|d(?!ho)|dho|ho|\+{1,2}|-)", s[i:])
                if action_match:
                    action = action_match.group(1); i += len(action)
                action_list.append(action)
    else:  # Stagger/Iverson sequential (also reused for multi off-ball)
        i = 0
        while i < len(s):
            screener_match = re.match(r"(\d+)", s[i:])
            if not screener_match:
                break
            screener = norm_id(screener_match.group(1)); i += len(screener_match.group(1))
            defender = None
            if i < len(s) and s[i] == '/':
                i += 1
                defender_match = re.match(r"(\d+)", s[i:])
                if defender_match:
                    defender = norm_id(defender_match.group(1)); i += len(defender_match.group(1))
            action = ""
            action_match = re.match(r"(ch|ct|sw|bz|tl|cs|d(?!ho)|dho|ho|\+{1,2}|-)", s[i:])
            if action_match:
                action = action_match.group(1); i += len(action)
            screener_list.append(screener)
            screendef_list.append([defender] if defender is not None else [])
            action_list.append(action)

    return screener_list, screendef_list, action_list


def parse_onball_screens(s):
    """
    Parse on-ball PNR/PNP trailing segment (no-caret codes):
      - Multiple screeners in sequence: "3/4ch5/6ct"
      - Rescreens using commas for the same screener: "3/4ch,ct,sw5,bz"
        (sw5 and bz treated as following codes for the SAME screener)
      - May include trailing non-screener actions after the last screen step, e.g. "...23chd-"
    Returns (screeners, screendefs, actions, trailing)

    actions: list aligned 1:1 with the entries in 'screeners' by occurrence.
    trailing: a raw string of remaining codes (e.g., 'd-', 'h', '+') to be handled
              by the main loop after all screen interactions are printed.

    Action codes recognized for screen interactions:
      ch, ct, swN, bz, tl, cs, ice, h, d, dho, ho, kp, +, ++, -
    Notes:
      - 'h' must not eat 'ho' (handoff), so use h(?!o)
      - 'd' must not eat 'dho', so use d(?!ho)

    IMPORTANT AMBIGUITY FIX:
      If we see 'sw' immediately followed by digits and a slash (e.g., 'sw6/23...'),
      interpret that as a plain 'sw' action for the CURRENT screen, and let '6/23'
      start the NEXT screener. This prevents the '6' from being consumed as the
      switcher id (which would swallow the next screener).
    """
    s = s.replace(" ", "")
    i = 0
    screeners, screendefs, actions = [], [], []
    cur_scr = None
    cur_def = None

    def read_action_code(idx):
        # Disambiguate: 'sw6/23...' should be read as 'sw' + NEXT screener '6/23',
        # not 'sw6' (switcher id) followed by '/23...' trailing.
        if s.startswith("sw", idx):
            m_sw_next = re.match(r"sw(\d+)/\d", s[idx:])
            if m_sw_next:
                # Treat it as a plain 'sw' and leave '6/23...' for the next screener parse
                return "sw", idx + 2

        m = re.match(r"(ch|ct|sw\d*|bz|tl|cs|ice|h(?!o)|d(?!ho)|dho|ho|kp|\+{1,2}|-)", s[idx:])
        if m:
            return m.group(1), idx + len(m.group(1))
        return "", idx

    while i < len(s):
        if s[i] == ',':  # rescreen: same screener, new action
            i += 1
            if cur_scr is None:
                continue
            act, i = read_action_code(i)
            actions.append(act)
            screeners.append(cur_scr)
            screendefs.append(cur_def if cur_def is not None else [])
            continue

        # Try to read a new screener id
        m_scr = re.match(r"(\d+)", s[i:])
        if m_scr:
            cur_scr = norm_id(m_scr.group(1))
            i += len(m_scr.group(1))

            # optional defender
            cur_def = []
            if i < len(s) and s[i] == '/':
                i += 1
                m_def = re.match(r"(\d+)", s[i:])
                if m_def:
                    cur_def = [norm_id(m_def.group(1))]
                    i += len(m_def.group(1))

            # optional immediate action for this screen step
            act, i = read_action_code(i)
            actions.append(act)
            screeners.append(cur_scr)
            screendefs.append(cur_def if cur_def is not None else [])
            continue

        # No new screener: if we still have the last screener in scope,
        # keep consuming additional actions (e.g., "...23chd-")
        if cur_scr is not None:
            act, new_i = read_action_code(i)
            if act:
                actions.append(act)
                screeners.append(cur_scr)
                screendefs.append(cur_def if cur_def is not None else [])
                i = new_i
                continue

        # Nothing more we recognize as screen steps: everything left is trailing
        break

    trailing = s[i:] if i < len(s) else ""
    return screeners, screendefs, actions, trailing


# ---------------- Helper Functions ----------------
# Defensive globals may be referenced before init — ensure they exist
try:
    defender_memory
except NameError:
    defender_memory = {}

try:
    zone_active
except NameError:
    zone_active = False

try:
    pending_offball_sa
except NameError:
    pending_offball_sa = None

try:
    last_shot_defenders
except NameError:
    last_shot_defenders = None


# --- Safe wrappers in case upstream utilities are not imported yet ---
def _safe_norm_id(x):
    """
    Call project's norm_id() if available; otherwise coerce to simple numeric-ish string.
    Accepts rot tokens like 'rot3' or '3' and returns the numeric part as string.
    """
    try:
        return norm_id(x)  # type: ignore[name-defined]
    except Exception:
        s = "" if x is None else str(x).strip()
        if s.startswith("rot"):
            s = s[3:]
        return s

def _safe_norm_list(seq):
    """
    Call project's norm_list() if available; otherwise normalize to a flat list of strings.
    """
    try:
        return norm_list(seq)  # type: ignore[name-defined]
    except Exception:
        if seq is None:
            return None
        if isinstance(seq, (list, tuple)):
            return [None if (v is None or str(v).strip() == "") else str(v).strip() for v in seq]
        return [str(seq).strip()]


# --- Parsing helpers used within this section ---
def _is_rot_token(x):
    return isinstance(x, str) and x.startswith("rot")

def _split_off_and_defs(raw):
    """
    Accepts a player field that may optionally embed defender info:
        '20'                  -> ('20', None)
        '20/rot1,5'           -> ('20', ['rot1','5'])
        'rot7/3,rot2, 11'     -> ('rot7', ['3','rot2','11'])
        15                    -> ('15', None)

    Returns (player_id_str, def_list_or_None)
    NOTE: The caller decides whether to use the returned def_list based on whether an explicit
          defender argument was provided (tri-state rules).
    """
    if raw is None:
        return ("", None)
    s = str(raw).strip()
    if "/" not in s:
        return (s, None)

    player_part, defs_part = s.split("/", 1)
    player_part = player_part.strip()

    defs_part = defs_part.strip()
    if defs_part == "":
        return (player_part, [])

    # Split on commas; keep 'rotX' tokens or numerics as strings
    tokens = [t.strip() for t in defs_part.split(",")]
    defs = []
    for t in tokens:
        if not t:
            continue
        defs.append(t)  # keep as-is; downstream handles rotX vs numeric
    return (player_part, defs)


# ---------- Player/defender phrase builders ----------
def format_player_list(players):
    """
    Render a list of players into: 'Player 12' / 'Players 12 and 34' / etc.
    Accepts raw ids or 'rot' ids. Uses norm_id() (or safe fallback) for final printing.
    """
    if not players:
        return "Unknown"
    if isinstance(players, str):
        return f"Player {_safe_norm_id(players)}"
    ids = [_safe_norm_id(p) for p in players if p is not None]
    if len(ids) == 0:
        return "Unknown"
    if len(ids) == 1:
        return f"Player {ids[0]}"
    if len(ids) == 2:
        return f"Players {ids[0]} and {ids[1]}"
    return "Players " + ", ".join(ids[:-1]) + f", and {ids[-1]}"

def _defenders_to_parts(def_list, off_player=None):
    """
    Convert a normalized defender list ([] or [ids/rotX]) into printable parts,
    applying the 'rotating over' first-mention rule via defender_memory.
    Returns a list like ['Player 12 rotating over', 'Player 34'].
    """
    global defender_memory
    parts = []
    for d in (def_list or []):
        s = str(d)
        if _is_rot_token(s):
            num = s[3:]
            key = (str(off_player) if off_player is not None else "*", num)
            if key in defender_memory:
                parts.append(f"Player {num}")
            else:
                parts.append(f"Player {num} rotating over")
                defender_memory[key] = True
        else:
            parts.append(f"Player {_safe_norm_id(s)}")
    return parts

def defender_text(def_player, off_player=None):
    """
    Legacy printer that returns a phrase WITHOUT a leading space.
    Uses tri-state semantics:
      - def_player is None  -> coverage unknown -> '' (empty)
      - def_player == []    -> explicit 'wide open'
      - list of defenders   -> 'guarded by ...' (+ 'rotating over' on first rot mentions)
    NOTE: Zone handling is done in coverage_phrase(), which is preferred for composing lines.
    """
    if def_player is None:
        return ""
    def_list = def_player if isinstance(def_player, list) else [def_player]
    if len(def_list) == 0:
        return "wide open"

    parts = _defenders_to_parts(def_list, off_player=off_player)
    if len(parts) == 1:
        return f"guarded by {parts[0]}"
    if len(parts) == 2:
        return f"guarded by {parts[0]} and {parts[1]}"
    return "guarded by " + ", ".join(parts[:-1]) + f", and {parts[-1]}"

def coverage_phrase(def_list, off_player=None):
    """
    Preferred coverage phrase builder that RETURNS A LEADING SPACE when non-empty:
      - def_list is None  -> unknown coverage
            -> '' normally; ' against the zone' if zone is active
      - def_list == []    -> explicit ' wide open'
      - list of defenders -> ' ' + defender_text(...)
    """
    global zone_active
    if def_list is None:
        return " against the zone" if zone_active else ""
    if not def_list:
        return " wide open"
    return " " + defender_text(def_list, off_player)


# ---------- Pass & wording helpers ----------
def print_pass(from_player, from_def, to_player, to_def, inbound=False):
    """
    Core pass printer. Respects tri-state coverage for BOTH passer and receiver.
    from_def / to_def may be None (unknown), [] (explicit open), or list.
    Also accepts strings like '20/rot1,5' as from_player / to_player; in that case
    defender fragments after '/' are used only if the corresponding def arg is None.
    """
    action_word = "inbounds the ball to" if inbound else "passes to"

    # Allow "X/defs" in the from/to fields as a fallback only when the explicit
    # def args are None. (Do not override an explicit empty list [].)
    fp_raw, fp_defs_fallback = _split_off_and_defs(from_player)
    tp_raw, tp_defs_fallback = _split_off_and_defs(to_player)

    fp = _safe_norm_id(fp_raw)
    tp = _safe_norm_id(tp_raw)

    use_from_def = from_def if (from_def is not None) else (fp_defs_fallback or None)
    use_to_def   = to_def   if (to_def   is not None) else (tp_defs_fallback or None)

    left  = f"Player {fp}{coverage_phrase(use_from_def, fp)}"
    right = f"Player {tp}{coverage_phrase(use_to_def, tp)}"
    print(f"{left} {action_word} {right}")
    return fp  # return normalized passer id

def ordinal_word(n: int) -> str:
    if n == 1: return "first"
    if n == 2: return "second"
    if n == 3: return "third"
    return f"{n}th"

def _is_screen_interaction(code: str) -> bool:
    return bool(code) and (code in ("ch", "ct", "bz", "cs", "ice") or code.startswith("sw"))


# --- Shot line & screen-assist formatting ---
def _format_screen_assist_suffix(screen_assist_by):
    """
    Build the "(screen assist...)" suffix for made shots.

    Examples:
      ['10']           -> (screen assist by Player 10)
      ['10','5']       -> (screen assist by Player 10 and Player 5)
      ['10','5','7']   -> (screen assist by Player 10, Player 5, and Player 7)
    """
    if not screen_assist_by:
        return None
    if isinstance(screen_assist_by, str):
        return f"(screen assist by Player {_safe_norm_id(screen_assist_by)})"
    screeners = [_safe_norm_id(s) for s in screen_assist_by if s]
    if not screeners:
        return None
    if len(screeners) == 1:
        return f"(screen assist by Player {screeners[0]})"
    if len(screeners) == 2:
        return f"(screen assist by Player {screeners[0]} and Player {screeners[1]})"
    players = ", ".join(f"Player {s}" for s in screeners[:-1]) + f", and Player {screeners[-1]}"
    return f"(screen assist by {players})"

def print_shot_line(player, def_list, action, last_passer=None, screen_assist_by=None):
    """
    Shot printer that respects tri-state coverage:
      None  -> unknown ('' / ' against the zone' if in zone)
      []    -> ' wide open'
      [...] -> ' guarded by ...'
    """
    p = _safe_norm_id(player)
    made = action in ("+","++")
    verb = "makes the shot" if made else "misses the shot"

    suffixes = []
    if action == "++" and last_passer:
        suffixes.append(f"(assisted by Player {_safe_norm_id(last_passer)})")
    if made and screen_assist_by:
        sa_suffix = _format_screen_assist_suffix(screen_assist_by)
        if sa_suffix:
            suffixes.append(sa_suffix)

    tail = (" " + " ".join(suffixes)) if suffixes else ""
    end = "!" if made else ""
    cov = coverage_phrase(def_list, p)
    print(f"Player {p}{cov} {verb}{tail}{end}")

def handle_shot(player, def_list, action, last_passer=None, screen_assist_by=None):
    print_shot_line(
        player, def_list, action,
        last_passer=last_passer,
        screen_assist_by=screen_assist_by
    )

def handle_shot_with_screen_assists(player, def_list, action, last_passer=None, last_screener=None, pnr_shot=False):
    """
    Prints the shot and applies screen-assist rules.

    Rules:
    - On-ball (PNR/PNP): credit screen assist on any made shot (+ or ++)
      when there is a valid last_screener. Supports single or multiple screeners.
    - Off-ball: credit screen assist ONLY if the shot is assisted (++)
      AND there is a valid last_passer AND a pending_offball_sa for this shooter.
      For multi-screener actions, ALL screeners are credited.
    """
    global pending_offball_sa, last_shot_defenders
    # remember who guarded the shooter at shot time (for OR fallback)
    last_shot_defenders = _safe_norm_list(def_list)

    screen_asst = None
    made = action in ("+", "++")

    if made:
        if pnr_shot and last_screener:
            # Support multiple screeners
            if isinstance(last_screener, (list, tuple)):
                screen_asst = [_safe_norm_id(s) for s in last_screener if s is not None]
            else:
                screen_asst = _safe_norm_id(last_screener)
        else:
            if (
                pending_offball_sa
                and _safe_norm_id(player) == _safe_norm_id(pending_offball_sa.get("recipient"))
                and action == "++"
                and last_passer
            ):
                screeners = pending_offball_sa.get("screeners") or []
                screen_asst = [_safe_norm_id(s) for s in screeners if s]

    handle_shot(player, def_list, action, last_passer=last_passer, screen_assist_by=screen_asst)

    # Clear after ANY shot event
    pending_offball_sa = None


# =========================
# Public API (for external callers)
# =========================
def parse_possession_string(line: str) -> list[str]:
    """
    Run the Mark 17 parser on ONE possession string and return the commentary lines as a list.

    This drives your existing interactive main() loop by faking user input:
      1) returns `line` the first time input() is called
      2) returns 'q' the second time to exit the loop

    We capture stdout and return only non-empty lines that are not the prompt.
    """
    import io, sys, builtins, re
    from contextlib import redirect_stdout

    # Prepare a fake input() that yields [line, 'q']
    inputs = iter([line, 'q'])
    old_input = builtins.input

    def fake_input(prompt: str = "") -> str:
        try:
            return next(inputs)
        except StopIteration:
            return 'q'

    # Capture all stdout produced by main()
    buf = io.StringIO()
    prompt_like = re.compile(r'^\s*(enter|type)\s+possession', re.I)

    old_stdout = sys.stdout
    try:
        builtins.input = fake_input
        with redirect_stdout(buf):
            # Call your existing interactive loop
            main()  # relies on rest of file
    finally:
        # Always restore global state
        builtins.input = old_input
        sys.stdout = old_stdout

    # Extract commentary lines; drop prompts/blank lines if any
    out = buf.getvalue().splitlines()
    cleaned = [ln for ln in out if ln.strip() and not prompt_like.search(ln)]
    return cleaned




# ---------------- Off-Ball Screen / Cut Helper ----------------

def print_offball_screen(action_type, ball, balldef, screener, screendef, after):
    """
    Prints off-ball action description and primes a pending off-ball screen assist when relevant.
    - For screen actions (pn/fl/bk/away/...), we keep prior behavior.
    - For backdoor cuts (bd), we print: "<ball> cuts backdoor".
    Returns the possibly-updated ball-defender list to carry forward.

    No-caret codes here:
      chase/cut/switch/blitz/top-lock/caught: ch, ct, sw, bz, tl, cs
      immediate actions after screen: d (drive), dho, ho
      shot outcomes: +, ++, -
    """
    global pending_offball_sa

    # Special handling: BACKDOOR CUT
    if action_type == "bd":
        ball_id = norm_id(ball) if ball is not None else "Unknown"
        ball_def = norm_list(balldef)
        print(f"Player {ball_id} {defender_text(ball_def, ball_id)} cuts backdoor")
        # No screeners, no pending screen assist primed
        return ball_def

    # Labels (singular → plural) for screen actions
    labels = {
        "pn":   ("pin down", "pin downs"),
        "fl":   ("flare screen", "flare screens"),
        "bk":   ("back screen", "back screens"),
        "awy":  ("away screen", "away screens"),
        "hm":   ("hammer screen", "hammer screens"),
        "crs":  ("cross screen", "cross screens"),
        "wdg":  ("wedge screen", "wedge screens"),
        "rip":  ("rip screen", "rip screens"),
        "ucla": ("ucla screen", "ucla screens"),
        "stg":  ("stagger screen", "stagger screens"),
        "ivs":  ("Iverson screen", "Iverson screens"),
        "elv":  ("elevator screen", "elevator screens"),
    }
    if action_type not in labels:
        return balldef

    sing_label, plur_label = labels[action_type]

    # Helpers
    def _a_or_an(label: str) -> str:
        first = label.strip().lower()
        if first.startswith(("away", "elv")):
            return "an"
        if first.startswith("ucla"):
            return "a"
        return "an" if first[:1] in "aeiou" else "a"

    def _norm_def_list(d):
        if d is None:
            return []
        if d and isinstance(d[0], list):
            return [norm_id(x) for x in d[0] if x]
        return [norm_id(x) for x in d if x]

    # Normalize inputs
    ball_id = norm_id(ball) if ball is not None else "Unknown"
    ball_def = norm_list(balldef)

    # Normalize screeners / screendefs to parallel lists
    if isinstance(screener, list):
        screeners = [norm_id(s) for s in screener if s is not None]
        if screendef and isinstance(screendef[0], list):
            screen_defs = [norm_list(sd) for sd in screendef]
        else:
            sd_flat = norm_list(screendef)
            screen_defs = [sd_flat for _ in screeners]
    else:
        s0 = norm_id(screener) if screener is not None else None
        screeners = [s0] if s0 else []
        screen_defs = [norm_list(screendef)] if s0 else [[]]

    screen_count = len(screeners)
    sing = sing_label
    plur = plur_label

    # Determine if denial (top lock) appears
    if isinstance(after, list):
        has_tl = any(a == "tl" for a in after)
    else:
        has_tl = bool(after and "tl" in after)

    # Build screener phrase(s)
    def _screener_phrase(idx: int) -> str:
        sid = screeners[idx]
        sdef = screen_defs[idx] if idx < len(screen_defs) else []
        if sdef:
            return f"Player {sid} {defender_text(sdef, sid)}"
        return f"Player {sid}"

    def _intro_multi():
        joined = " and ".join(_screener_phrase(i) for i in range(screen_count))
        print(f"Player {ball_id} {defender_text(ball_def, ball_id)} comes off {plur} from {joined}")

    def _intro_single():
        sid = screeners[0]
        sdef = screen_defs[0]
        article = _a_or_an(sing)
        print(
            f"Player {ball_id} {defender_text(ball_def, ball_id)} "
            f"comes off {article} {sing} from Player {sid} {defender_text(sdef, sid)}"
        )

    # Denial handling
    if has_tl:
        if screen_count >= 2:
            joined = " and ".join(_screener_phrase(i) for i in range(screen_count))
            print(
                f"Player {ball_id} {defender_text(ball_def, ball_id)} tries to come off {plur} from {joined}, "
                f"but {format_player_list(ball_def)} top locks Player {ball_id} forcing them away from the screen"
            )
        elif screen_count == 1:
            sid = screeners[0]
            sdef = screen_defs[0]
            article = _a_or_an(sing)
            print(
                f"Player {ball_id} {defender_text(ball_def, ball_id)} tries to come off {article} {sing} from "
                f"Player {sid} {defender_text(sdef, sid)}, "
                f"but {format_player_list(ball_def)} top locks Player {ball_id} forcing them away from the screen"
            )
        # consume 'tl'
        if isinstance(after, list):
            for i, a in enumerate(after):
                if a == "tl":
                    after[i] = ""
        else:
            after = after.replace("tl", "") if after else after
        pending_offball_sa = None
    else:
        # Normal intro + prime off-ball screen assist(s)
        if screen_count >= 2:
            _intro_multi()
            pending_offball_sa = {"recipient": ball_id, "screeners": [s for s in screeners if s]}
        elif screen_count == 1:
            _intro_single()
            pending_offball_sa = {"recipient": ball_id, "screeners": [screeners[0]]}
        else:
            print(f"Player {ball_id} {defender_text(ball_def, ball_id)} comes off a screen")
            pending_offball_sa = None

    # Per-screen codes (lists) OR linear codes (single)
    def _handle_code(code: str, idx_for_label: int):
        nonlocal ball_def
        # Name the screen in the chase/cut lines
        human_screen = (sing if screen_count == 1 else f"{ordinal_word(idx_for_label + 1)} {sing}")

        if code == "ch":
            print(f"{format_player_list(ball_def)} chases over the {human_screen}")
        elif code == "ct":
            print(f"{format_player_list(ball_def)} cuts under the {human_screen}")
        elif code.startswith("sw"):
            m = re.match(r"sw(\d+)", code)
            if m:
                switch_actor = norm_id(m.group(1))
            else:
                sdef = screen_defs[idx_for_label] if idx_for_label < len(screen_defs) else []
                switch_actor = norm_id(sdef[0]) if sdef else "Unknown"
            print(f"Player {switch_actor} switches onto Player {ball_id} on the {human_screen}")
            ball_def = [] if switch_actor == "Unknown" else [switch_actor]
        elif code == "bz":
            sdef = screen_defs[idx_for_label] if idx_for_label < len(screen_defs) else []
            sdef = _norm_def_list(sdef)
            if sdef and ball_def and set(sdef) != set(ball_def):
                print(f"{format_player_list(ball_def)} and {format_player_list(sdef)} double team Player {ball_id} on the {human_screen}")
                ball_def = list({*ball_def, *sdef})
            elif sdef:
                print(f"{format_player_list(sdef)} blitzes Player {ball_id} on the {human_screen}")
                ball_def = list({*ball_def, *sdef}) if sdef else ball_def
            else:
                print(f"{format_player_list(ball_def)} blitzes Player {ball_id} on the {human_screen}")
        elif code == "cs":
            who = format_player_list(ball_def) if ball_def else "Unknown defender"
            print(f"{who} gets caught on the {human_screen}")
            ball_def = []  # FIX(cs): caught on screen -> no longer guarding; shooter is now wide open
        elif code == "cs":  # (kept for exact structure; logic handled above)
            who = format_player_list(ball_def) if ball_def else "Unknown defender"
            print(f"{who} gets caught on the {human_screen}")
            ball_def = []  # FIX(cs)
        elif code in ("d", "dho", "ho"):
            act_word = "drives" if code == "d" else "hands off"
            print(f"Player {ball_id} {defender_text(ball_def, ball_id)} {act_word} immediately after the {human_screen}")
        elif code in ("+", "++", "-"):
            handle_shot_with_screen_assists(ball_id, ball_def, code)

    if isinstance(after, list):
        for idx, code in enumerate(after):
            if not code:
                continue
            _handle_code(code, idx)
    else:
        codes = [m.group(0) for m in re.finditer(r"(ch|ct|sw\d*|bz|cs|d(?!ho)|dho|ho|\+{1,2}|-)", after or "")]
        for code in codes:
            _handle_code(code, 0)

    return ball_def


# ---------------- Defensive / Rebound / Special Tokens ----------------

def process_defensive_tokens(token, last_player, last_defender, last_shooter, off_player=None, def_list=None):
    """
    Handle tokens that represent defense/rebounds/specials in NO-CARET format.

    Supported (caret-less) forms:
      rob, dob, oob
      dbto, lbto
      stlX, blkX, defX
      jumpA,B
      sc
      ofX, fX
      r  or  <player>r          (defensive rebound)
      or, <player>or, or+, or-  (offensive rebound with optional putback)
    """
    # NEW: Do not consume parenthesized off-ball actions here; let the off-ball parser handle them.
    t = token.strip()
    if t.startswith("(") and t.endswith(")"):
        return last_player, last_defender, last_shooter, 0

    # NEW GUARD: If this is a slash-token that includes action text (d/h/p/+/-)
    # after the defender fragment, defer to the slash/action parser.
    # This prevents tokens like "6/5d-or" from being treated as plain "or".
    if "/" in t:
        rhs = t.split("/", 1)[1]
        if re.search(r"[dhp\+\-]", rhs):
            return last_player, last_defender, last_shooter, 0

    base_token = token
    if "/" in token:
        parts = token.split("/")
        off_player = norm_id(parts[0])
        base_token = "/".join(parts[1:])  # what comes after the first slash

    # Out-of-bounds quick prints
    if base_token in ("rob", "dob", "oob"):
        msgs = {
            "rob": "Rebound goes out of bounds",
            "dob": "Ball deflected out of bounds",
            "oob": "Ball goes out of bounds",
        }
        print(msgs[base_token])
        return last_player, last_defender, last_shooter, 1

    # Dead-ball turnover
    if base_token == "dbto":
        player = norm_id(off_player or last_player)
        # Gate coverage: None -> unknown (omit); [] -> wide open; else guarded/by...
        cov = coverage_phrase(last_defender if isinstance(last_defender, list)
                              else ([last_defender] if last_defender else None), player)
        print(f"Player {player}{cov} commits a dead ball turnover")
        return last_player, last_defender, last_shooter, 1

    # Live-ball turnover
    if base_token == "lbto":
        player = norm_id(off_player or last_player)
        cov = coverage_phrase(last_defender if isinstance(last_defender, list)
                              else ([last_defender] if last_defender else None), player)
        print(f"Player {player}{cov} commits a live ball turnover")
        return player, None, None, 1

    # Steal
    if base_token.startswith("stl"):
        stealer = norm_id(base_token[3:])
        print(f"Player {stealer} steals the ball!")
        return stealer, None, None, 1

    # Block (accept "blk8") — keep last_shooter for OR fallback
    if base_token.startswith("blk"):
        blocker = norm_id(base_token[3:] or (off_player or ""))
        if last_player:
            print(f"Player {blocker} blocks the shot from Player {norm_id(last_player)}")
        else:
            print(f"Player {blocker} blocks the shot")
        # Do NOT clear last_shooter; keep state as-is
        return last_player, last_defender, last_shooter, 1

    # Deflection
    if base_token.startswith("def"):
        deflector = norm_id(base_token[3:])
        print(f"Player {deflector} deflects the ball from Player {norm_id(last_player) if last_player else 'Unknown'}")
        # mark loose ball pending; next possessor will "recovers the loose ball"
        global loose_ball_pending
        loose_ball_pending = True
        return last_player, last_defender, last_shooter, 1

    # Jump ball
    if base_token.startswith("jump"):
        try:
            players = [norm_id(p) for p in base_token[4:].split(",")]
            if len(players) == 2:
                print(f"Jump ball between Player {players[0]} and Player {players[1]}")
            else:
                print("Unrecognized jump ball format")
        except:
            print("Error parsing jump ball event")
        return last_player, last_defender, last_shooter, 1

    # Shot clock violation
    if base_token == "sc":
        print("Shot clock violation forced by the defense")
        return last_player, last_defender, last_shooter, 1

    # Fouls
    # Offensive foul (turnover): dead ball, clear possession
    if base_token.startswith("of"):
        fouler = norm_id(base_token[2:])
        print(f"Player {fouler} commits an offensive foul")
        return None, [], last_shooter, 1

    # Defensive foul: dead ball, clear possession so the next ob begins cleanly
    if base_token.startswith("f"):
        fouler = norm_id(base_token[1:])
        print(f"Player {fouler} commits a defensive foul")
        return None, [], last_shooter, 1

    # Defensive rebound
    # Accept:
    #   "r" alone → last_shooter/last_player/off_player decides the rebounder
    #   "8r"      → explicit rebounder
    #   "8/2r"    → explicit rebounder with context after slash (we already normalized off_player above)
    is_def_reb = False
    rebounder = None
    if "/" in token:
        # e.g., "8/2r" → base_token is "2r", rebounder is the player before first slash (8)
        after = token.split("/", 1)[1]
        if re.match(r"^r(\b|$)", after) or re.match(r"^\d+r(\b|$)", after):
            rebounder = norm_id(token.split("/", 1)[0])
            is_def_reb = True
    else:
        m_num = re.match(r"^(\d+)r(\b|$)", token)
        if m_num:
            rebounder = norm_id(m_num.group(1))
            is_def_reb = True
        elif token == "r":
            rebounder = norm_id(off_player or last_shooter or last_player)
            is_def_reb = True

    if is_def_reb:
        print(f"Player {rebounder} grabs the defensive rebound")
        return last_player, last_defender, last_shooter, 1

    # Offensive rebound + optional putback (+ or -)
    if "or" in base_token:
        player = norm_id(off_player or last_shooter or last_player)
        rebound_def_list = (last_defender if isinstance(last_defender, list)
                            else ([last_defender] if last_defender else None))

        if "/" in token:
            # "8/2or" or "8/rot2or" → get player before '/', defenders from fragment before 'or'
            player_part, def_part = token.split("/", 1)
            player = norm_id(player_part)
            def_fragment = def_part.split("or", 1)[0]
            defenders = [norm_id(x) for x in re.findall(r"(?:rot\d+|\d+)", def_fragment)]
            rebound_def_list = defenders if defenders else rebound_def_list
        else:
            # "8or" or "or"
            m_pnum = re.match(r"^(\d+)or", token)
            if m_pnum:
                player = norm_id(m_pnum.group(1))

        # Fallback to last shot's defenders if none carried over
        if rebound_def_list is None:
            rebound_def_list = last_shot_defenders or None  # keep None if still unknown

        putback_action = None
        if base_token.endswith("+"): putback_action = "+"
        elif base_token.endswith("-"): putback_action = "-"

        # Gate coverage phrase for the rebound line (unknown vs explicit open)
        cov = coverage_phrase(rebound_def_list, player)
        print(f"Player {player}{cov} grabs the offensive rebound")
        if putback_action:
            handle_shot_with_screen_assists(player, rebound_def_list, putback_action)
            last_shooter = player
            return last_player, last_defender, last_shooter, 1
        return player, rebound_def_list, None, 1

    # Not a defensive/special token we handle here
    return last_player, last_defender, last_shooter, 0



# ---------------- Main Loop ----------------

def main():
    """
    Main loop for basketball stat tracker (NO-CARET FORMAT).

    Changes included:
      1) Off-ball actions in parentheses are parsed BEFORE defensive/rebound/special tokens.
      2) Receive-only handler: tokens like "6/" set possession to 6 as explicitly open ([]),
         and if someone else had the ball, we print the pass into 6 wide open.
      3) After a simple defensive rebound "<off>r", collapse narration if followed by an outlet
         and/or a bring-up:
           a) "<rebounder>r  <X>h/<defs>" =>
                "<rebounder> passes to <X> who brings the ball over halfcourt and <defs> picks up <X>"
           b) "<rebounder>r  <Y>  <X>h/<defs>" =>
                "<rebounder> passes to <Y>"
                "<Y> passes to <X> who brings the ball over halfcourt and <defs> picks up <X>"
      4) BUGFIX: loose-ball recovery is processed BEFORE any pass parsing (including inbound and
         the simple pass fast-path), so tokens like "def10  1/10" print "1 recovers the loose ball"
         before we interpret "1/10".
      5) NEW: In slash tokens, after a shot is parsed (from 'd', 'p', or bare '+/++/-'),
         if the remaining tail starts with 'or' or 'r', narrate an immediate rebound:
            "<shooter> grabs the offensive rebound" (for 'or')
            "<shooter> grabs the defensive rebound" (for 'r')
         and keep possession with that rebounder. This fixes sequences like "6/5d-or".
    """
    import re

    # Globals maintained elsewhere in the file
    global pending_offball_sa, last_shot_defenders, loose_ball_pending, zone_active, zone_label

    # Treat only a TRAILING 'ob' as inbound. Do NOT touch 'dob', 'rob', or 'oob'.
    def token_ob_split(tok: str):
        if tok.endswith("ob") and tok not in ("dob", "rob", "oob"):
            return tok[:-2], True
        return tok, False

    # Preferred halfcourt wording
    def print_bring_over_halfcourt(off, def_list, pickup_style: bool = False):
        off_id = norm_id(off)
        defs = norm_list(def_list) if def_list is not None else None
        if defs:
            if pickup_style:
                print(f"Player {off_id} brings the ball over halfcourt and {format_player_list(defs)} picks up Player {off_id}")
            else:
                print(f"Player {off_id}{coverage_phrase(defs, off_id)} brings the ball over halfcourt")
        else:
            # Unknown or empty → explicit message that nobody picked up
            if pickup_style:
                print(f"Player {off_id} brings the ball over halfcourt and no one picks up Player {off_id}")
            else:
                print(f"Player {off_id} brings the ball over halfcourt and no one picks up Player {off_id}")

    # Pass printer that respects tri-state coverage and inbounds
    def print_pass_with_open(passer, passer_def_list, receiver, receiver_def_list, inbound=False):
        p = norm_id(passer)
        r = norm_id(receiver)
        pdefs = passer_def_list if passer_def_list is None else norm_list(passer_def_list)
        rdefs = receiver_def_list if receiver_def_list is None else norm_list(receiver_def_list)
        if inbound:
            print_pass(p, pdefs or [], r, rdefs or [], inbound=True)
            return
        print(f"Player {p}{coverage_phrase(pdefs, p)} passes to Player {r}{coverage_phrase(rdefs, r)}")

    def handle_zone_end(zone_end_flag: bool):
        global zone_active, zone_label
        if zone_end_flag and zone_active:
            print("Defense exits the zone")
            zone_active = False
            zone_label = None

    # Helper: consume an immediate rebound suffix ('or' or 'r') after a shot in a slash token.
    # Returns (k_advance, possession_off, possession_def_list)
    def consume_inline_rebound(tail: str, shooter, shooter_def_list):
        if tail.startswith("or"):
            print(f"Player {norm_id(shooter)}{coverage_phrase(shooter_def_list, shooter)} grabs the offensive rebound")
            return 2, shooter, shooter_def_list
        if tail.startswith("r"):
            print(f"Player {norm_id(shooter)}{coverage_phrase(shooter_def_list, shooter)} grabs the defensive rebound")
            return 1, shooter, shooter_def_list
        return 0, None, None

    while True:
        try:
            line = input("Enter possession (or 'q' to quit): ").strip()
        except EOFError:
            break
        if line.lower() == "q":
            break
        if not line:
            continue

        # Reset per-possession memory/state
        defender_memory.clear()
        pending_offball_sa = None
        last_shot_defenders = []
        loose_ball_pending = False
        zone_active = False
        zone_label = None

        # Per-possession: off-ball screen assist map (shooter -> set(screeners))
        offball_sa_map = {}

        parts = line.split()
        last_possessor = None
        last_defender  = None   # None = unknown coverage; [] = explicitly open
        last_shooter   = None
        last_passer    = None
        last_screener  = None
        last_action_type = None
        last_ho_giver_def = []  # track HO/DHO giver's defender for blitz logic
        pending_inbounder = None
        pending_inbounder_def = []

        i = 0
        while i < len(parts):
            raw_token = parts[i]

            # 1) Zone markers first
            token, zone_start, zone_end, zlabel = strip_zone_marks(raw_token)
            if zone_start and not zone_active:
                zone_label = zlabel or None
                print(f"Defense sets up in a {zone_label} zone" if zone_label else "Defense sets up in a zone")
                zone_active = True

            # 2) Detect trailing 'ob' for inbound
            token, ob_here = token_ob_split(token)

            # Normalize head-side 'h' like "10h/1" → "10/1h"
            head_side_h_pickup = False
            _compact = token.replace(" ", "")
            m_head_h = re.match(r"^(\d+)h(?:/(.*))?$", _compact)
            if m_head_h:
                off = norm_id(m_head_h.group(1))
                tail = (m_head_h.group(2) or "").strip()
                if not tail:
                    token = f"{off}/h"
                else:
                    m_tail = re.match(r"^((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))?)(.*)$", tail)
                    if m_tail:
                        defenders = m_tail.group(1)
                        rest = m_tail.group(2)
                        token = f"{off}/{defenders}h{rest}"
                    else:
                        token = f"{off}/{tail if tail.endswith('h') else tail + 'h'}"
                head_side_h_pickup = True

            # ------------------------------------------------------------------
            # Drive immediately followed by a ball-screen token (NOT dho/ho)
            # ------------------------------------------------------------------
            m_drive_then = re.match(r"^d(pnr|pnp|slp|gst|rj|kp)(.+)$", token.replace(" ", ""))
            if m_drive_then and last_possessor:
                cur_off = norm_id(last_possessor)
                cur_def = norm_list(last_defender)
                print(f"Player {cur_off}{coverage_phrase(cur_def,cur_off)} drives inside")
                remainder = token[1:]  # drop the leading 'd'
                parts.insert(i + 1, remainder)
                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # "rh": defensive Rebound + bring-up with pickup, e.g., "1rh/20"
            # ------------------------------------------------------------------
            m_rh_pick = re.match(r"^(\d+)rh/((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)(.*)$", token.replace(" ", ""))
            if m_rh_pick:
                off_rh = norm_id(m_rh_pick.group(1))
                defs_rh = norm_list(re.findall(r"(?:rot\d+|\d+)", m_rh_pick.group(2)))
                trailing_rh = m_rh_pick.group(3) or ""
                print(f"Player {off_rh} grabs the defensive rebound")
                print_bring_over_halfcourt(off_rh, defs_rh, pickup_style=True)
                last_possessor = off_rh
                last_defender  = defs_rh
                last_passer    = None
                last_screener  = None
                loose_ball_pending = False
                if trailing_rh:
                    adopted_def = ",".join(defs_rh) if defs_rh else ""
                    parts.insert(i + 1, f"{off_rh}/{adopted_def}{trailing_rh}")
                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # Rebound with pickup: "<off>r/<defs>" or "<off>or/<defs>"
            # ------------------------------------------------------------------
            m_r_pick = re.match(r"^(\d+)(o?r)/((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)$", token.replace(" ", ""))
            if m_r_pick:
                off_reb = norm_id(m_r_pick.group(1))
                is_off = (m_r_pick.group(2) == "or")
                defs = norm_list(re.findall(r"(?:rot\d+|\d+)", m_r_pick.group(3)))
                print(f"Player {off_reb} grabs the {'offensive' if is_off else 'defensive'} rebound")
                if defs:
                    if len(defs) == 1:
                        print(f"{format_player_list(defs)} picks up Player {off_reb}")
                    else:
                        print(f"{format_player_list(defs)} pick up Player {off_reb}")
                last_possessor = off_reb
                last_defender  = defs
                last_passer    = None
                last_screener  = None
                loose_ball_pending = False
                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # Simple rebounds without slash: "<off>r" or "<off>or"
            # Now supports: optional outlet to a teammate token, then bring-up collapse.
            # ------------------------------------------------------------------
            m_r_simple = re.match(r"^(\d+)(or|r)$", token.replace(" ", ""))
            if m_r_simple:
                off_reb = norm_id(m_r_simple.group(1))
                is_off = (m_r_simple.group(2) == "or")
                print(f"Player {off_reb} grabs the {'offensive' if is_off else 'defensive'} rebound")

                # Only apply outlet/bring-up collapsing on DEFENSIVE rebounds
                if not is_off:
                    # Case A: immediate bring-up next
                    if (i + 1) < len(parts):
                        nxt_raw = parts[i + 1]
                        nxt, zstart2, zend2, _ = strip_zone_marks(nxt_raw)
                        nxt_clean = nxt.replace(" ", "")

                        # Pattern: "1h/13" or "1h/rot13,22" (with optional trailing actions)
                        m_bring_direct = re.match(r"^(\d+)h/((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)(.*)$", nxt_clean)
                        if m_bring_direct:
                            rec = norm_id(m_bring_direct.group(1))
                            defs = norm_list(re.findall(r"(?:rot\d+|\d+)", m_bring_direct.group(2)))
                            trailing = m_bring_direct.group(3) or ""
                            # Collapse narration:
                            print(f"Player {off_reb} passes to Player {rec} who brings the ball over halfcourt and {format_player_list(defs)} picks up Player {rec}")
                            last_possessor = rec
                            last_defender  = defs
                            last_passer    = off_reb
                            last_screener  = None
                            loose_ball_pending = False
                            # Re-queue any trailing after the bring-up
                            if trailing:
                                adopted_def = ",".join(defs) if defs else ""
                                parts.insert(i + 2, f"{rec}/{adopted_def}{trailing}")
                            handle_zone_end(zend2)
                            i += 2
                            continue

                        # Case B: simple outlet to a jersey number, THEN a bring-up token
                        m_simple_target = re.match(r"^(\d+)$", nxt_clean)
                        if m_simple_target:
                            outlet_recv = norm_id(m_simple_target.group(1))
                            # Print outlet pass (no defender context on a bare number)
                            print(f"Player {off_reb} passes to Player {outlet_recv}")
                            # Update state
                            last_possessor = outlet_recv
                            last_defender  = None
                            last_passer    = off_reb
                            last_screener  = None
                            loose_ball_pending = False

                            # Look one more token ahead for a bring-up
                            if (i + 2) < len(parts):
                                nxt2_raw = parts[i + 2]
                                nxt2, zstart3, zend3, _ = strip_zone_marks(nxt2_raw)
                                nxt2_clean = nxt2.replace(" ", "")
                                m_bring_after_outlet = re.match(r"^(\d+)h/((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)(.*)$", nxt2_clean)
                                if m_bring_after_outlet:
                                    rec2 = norm_id(m_bring_after_outlet.group(1))
                                    defs2 = norm_list(re.findall(r"(?:rot\d+|\d+)", m_bring_after_outlet.group(2)))
                                    trailing2 = m_bring_after_outlet.group(3) or ""
                                    # Pass from outlet receiver to bring-up handler
                                    print(f"Player {outlet_recv} passes to Player {rec2} who brings the ball over halfcourt and {format_player_list(defs2)} picks up Player {rec2}")
                                    last_possessor = rec2
                                    last_defender  = defs2
                                    last_passer    = outlet_recv
                                    last_screener  = None
                                    loose_ball_pending = False
                                    if trailing2:
                                        adopted_def2 = ",".join(defs2) if defs2 else ""
                                        parts.insert(i + 3, f"{rec2}/{adopted_def2}{trailing2}")
                                    handle_zone_end(zend3)
                                    i += 3
                                    continue

                            # If no bring-up after the outlet, just consume the outlet
                            i += 2
                            handle_zone_end(zone_end)
                            continue

                # Otherwise, normal rebound state
                last_possessor = off_reb
                last_defender  = None
                last_passer    = None
                last_screener  = None
                loose_ball_pending = False
                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # LOOSE-BALL RECOVERY (moved up BEFORE inbound and pass fast-path)
            # ------------------------------------------------------------------
            if loose_ball_pending:
                recoverer = None
                m_lead = re.match(r"^(\d+)", token.replace(" ", ""))
                if m_lead:
                    recoverer = norm_id(m_lead.group(1))
                if recoverer:
                    print(f"Player {recoverer} recovers the loose ball")
                    last_possessor = recoverer
                    last_defender  = []  # explicitly no defender on recovery
                    last_passer    = None
                    last_screener  = None
                    loose_ball_pending = False
                    # Do NOT 'continue'; allow same token to fall through.

            # ------------------------------------------------------------------
            # INBOUND HANDLERS (must run BEFORE any remaining pass logic)
            # ------------------------------------------------------------------

            # A) Have inbounder; THIS token names recipient → print inbound; requeue trailing actions
            if pending_inbounder and ("/" in token) and not token.startswith("("):
                clean = token.replace(" ", "")
                m_inb = re.match(r"^(\d+)/((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)(.*)$", clean)
                if m_inb:
                    off_recipient = norm_id(m_inb.group(1))
                    def_recipient = norm_list(re.findall(r"(?:rot\d+|\d+)", m_inb.group(2)))
                    trailing = m_inb.group(3) or ""
                else:
                    m_inb0 = re.match(r"^(\d+)/(.*)$", clean)
                    off_recipient = norm_id(m_inb0.group(1)) if m_inb0 else norm_id(token.split("/", 1)[0])
                    def_recipient = []
                    trailing = (m_inb0.group(2) if m_inb0 else "")
                print_pass_with_open(pending_inbounder, pending_inbounder_def, off_recipient, def_recipient, inbound=True)
                last_possessor = off_recipient
                last_defender  = def_recipient
                last_passer    = None
                last_screener  = None
                last_action_type = None
                pending_inbounder = None
                pending_inbounder_def = []
                loose_ball_pending = False
                if trailing:
                    adopted_def = ",".join(def_recipient) if def_recipient else ""
                    parts.insert(i + 1, f"{off_recipient}/{adopted_def}{trailing}")
                handle_zone_end(zone_end)
                i += 1
                continue

            # B) THIS token declares inbounder (…ob)
            if ob_here:
                if "/" in token:
                    off_inb = norm_id(token.split("/", 1)[0])
                    defs_inb = norm_list(re.findall(r"(?:rot\d+|\d+)", token.split("/", 1)[1]))
                else:
                    off_inb = norm_id(token)
                    defs_inb = []
                pending_inbounder = off_inb
                pending_inbounder_def = defs_inb
                last_possessor = None
                last_defender  = None
                last_passer    = None
                last_screener  = None
                last_action_type = None
                loose_ball_pending = False
                handle_zone_end(zone_end)
                i += 1
                continue

            # --- FAST PATH: simple pass tokens with no trailing actions ---
            if ("/" in token
                and not any(k in token for k in ("pnr","pnp","slp","gst","rj","dho","ho","kp","(",")","+","-","d","h","p"))):
                rhs = token.split("/", 1)[1].replace(" ", "")
                if re.fullmatch(r"(?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*", rhs):
                    m_simple_pass = re.match(r"^(\d+)/((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)$", token.replace(" ", ""))
                    if m_simple_pass:
                        off = norm_id(m_simple_pass.group(1))
                        defender_list = norm_list(re.findall(r"(?:rot\d+|\d+)", m_simple_pass.group(2)))
                    else:
                        off = norm_id(token.split("/", 1)[0])
                        defender_list = norm_list(re.findall(r"(?:rot\d+|\d+)", token.split("/", 1)[1]))
                    if last_possessor and norm_id(last_possessor) != off and not pending_inbounder:
                        print_pass_with_open(last_possessor, last_defender, off, defender_list, inbound=False)
                        last_passer = last_possessor
                    last_possessor = off
                    last_defender = defender_list
                    handle_zone_end(zone_end)
                    i += 1
                    continue
                # else: fall through

            # 0) Free throws like "12**x"
            m_ft = re.match(r"^(\d+)([\*x]+)$", token)
            if m_ft:
                shooter = norm_id(m_ft.group(1))
                seq = m_ft.group(2)
                for ch in seq:
                    if ch == "*":
                        print(f"Player {shooter} makes the free throw")
                    elif ch == "x":
                        print(f"Player {shooter} misses the free throw")
                last_possessor, last_defender = None, None
                pending_offball_sa = None
                loose_ball_pending = False
                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # (MOVED UP) Off-ball actions in parentheses (includes 'bd')
            # ------------------------------------------------------------------
            if token.startswith("(") and token.endswith(")"):
                ball, balldef, action_type, screener, screendef, after = parse_player_def(token)
                balldef = norm_list(balldef)
                if isinstance(screendef, list) and (len(screendef) > 0) and isinstance(screendef[0], list):
                    screendef = [norm_list(d) for d in screendef]
                else:
                    screendef = norm_list(screendef)

                if action_type in ("pn", "fl", "bk", "awy", "hm", "crs", "wdg", "rip", "ucla", "stg", "ivs", "elv", "bd"):
                    updated_balldef = print_offball_screen(action_type, ball, balldef, screener, screendef, after)
                    if norm_id(ball) == norm_id(last_possessor):
                        last_defender = norm_list(updated_balldef) if isinstance(updated_balldef, list) else norm_list([updated_balldef])

                    # Track potential off-ball screen assist
                    if ball and screener:
                        b = norm_id(ball)
                        scrs_in = [norm_id(s) for s in (screener if isinstance(screener, list) else [screener])]
                        bucket = offball_sa_map.get(b, set())
                        for s in scrs_in:
                            bucket.add(s)
                        offball_sa_map[b] = bucket

                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # Defensive/rebound/special tokens (fallback handler)
            # ------------------------------------------------------------------
            pos_after, def_after, last_shooter_after, skip = process_defensive_tokens(
                token, last_possessor, last_defender, last_shooter
            )
            if skip:
                last_possessor = pos_after
                last_defender  = def_after
                last_shooter   = last_shooter_after
                last_screener  = None
                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # Help (hp) or switch (sw)
            # ------------------------------------------------------------------
            if token.startswith(("hp", "sw")):
                match = re.match(r"(hp|sw)(\d+.*)", token)
                if match:
                    head, tail = match.group(1), match.group(2)
                    m2 = re.match(r"(\d+(?:,\d+)*)[ \t]*(\+\+|\+|-)?$", tail)
                    if not m2:
                        digits = re.match(r"(\d+(?:,\d+)*)", tail)
                        if digits:
                            core = digits.group(1)
                            rest = tail[len(core):]
                            token = head + core
                            if rest:
                                parts.insert(i + 1, rest)

                if not last_possessor and i > 0:
                    prev = parts[i - 1].split("/", 1)[0]
                    last_possessor = norm_id(prev)
                    last_defender = norm_list(parts[i - 1].split("/", 1)[1].split(",")) if "/" in parts[i - 1] else None

                is_help = token.startswith("hp")
                action_raw = token[2:]
                shoot_after = None
                for suffix, meaning in [("++", "assist"), ("+", "make"), ("-", "miss")]:
                    if action_raw.endswith(suffix):
                        shoot_after = meaning
                        action_raw = action_raw[:-len(suffix)]
                        break

                action_list = [norm_id(x.strip()) for x in action_raw.split(",") if x.strip()]
                cur_defs = norm_list(last_defender)

                if is_help:
                    for x in action_list:
                        if x not in cur_defs:
                            cur_defs.append(x)
                    print(
                        f"Player {action_list[0]} steps in to help on Player {last_possessor}"
                        if len(action_list) == 1
                        else f"Players {', '.join(action_list)} step in to help on Player {last_possessor}"
                    )
                    last_defender = cur_defs
                else:
                    last_defender = action_list
                    print(
                        f"Player {action_list[0]} switches onto Player {last_possessor}"
                        if len(action_list) == 1
                        else f"Players {', '.join(action_list)} switch onto Player {last_possessor}"
                    )

                if shoot_after:
                    shooter = norm_id(last_possessor)
                    is_pnr_context = (last_action_type in ("pnr","pnp"))
                    if not is_pnr_context and shooter in offball_sa_map:
                        scrs = list(offball_sa_map.pop(shooter, set()))
                        if scrs:
                            pending_offball_sa = {"recipient": str(shooter), "screeners": [str(s) for s in scrs]}
                    code = "++" if shoot_after == "assist" else ("+" if shoot_after == "make" else "-")
                    handle_shot_with_screen_assists(
                        shooter,
                        norm_list(last_defender),
                        code,
                        last_passer,
                        last_screener if is_pnr_context else None,
                        pnr_shot=is_pnr_context
                    )
                    pending_offball_sa = None
                    last_shooter = shooter
                    last_possessor = None
                    last_defender = None
                    last_screener = None
                    loose_ball_pending = False

                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # On-ball screens / actions (pnr, pnp, slp, gst, rj, dho, ho, kp)
            # ------------------------------------------------------------------
            if any(k in token for k in ("pnr", "pnp", "slp", "gst", "rj", "dho", "ho", "kp")):
                pre_half = bool(re.search(r"/\d+(?:,\d+)*h(?:pnr|pnp|slp|gst|rj|dho|ho|kp)", token))
                ball, balldef, action_type, screener, screendef, after = parse_player_def(token)
                balldef = norm_list(balldef)
                if isinstance(screendef, list) and (len(screendef) > 0) and isinstance(screendef[0], list):
                    screendef = [norm_list(d) for d in screendef]
                else:
                    screendef = norm_list(screendef)

                if ball is None and action_type in ("pnr", "pnp", "slp", "gst", "rj", "dho", "ho", "kp"):
                    ball = norm_id(last_possessor)
                    if not balldef:
                        balldef = norm_list(last_defender)

                if last_possessor and ball and norm_id(last_possessor) != norm_id(ball) and not pending_inbounder:
                    print_pass_with_open(last_possessor, last_defender, ball, balldef, inbound=False)
                    last_passer = last_possessor

                last_possessor = norm_id(ball)
                last_defender = balldef
                last_action_type = action_type

                onball_label = {
                    "pnr": "pick and roll", "pnp": "pick and pop",
                    "slp": "slips the screen for", "gst": "ghosts the screen for",
                    "rj": "rejects the ball screen from",
                    "dho": "dribbles and hands off to", "ho": "hands off to",
                    "kp": "keeps the handoff intended for"
                }

                # --- PNR/PNP screens with possible multiple screeners ---
                if action_type in ("pnr", "pnp") and isinstance(screener, list):
                    # Unpack per-screen actions + trailing
                    per_screen_actions = after["per_screen"] if isinstance(after, dict) else after
                    trailing_actions   = after.get("trail", "") if isinstance(after, dict) else ""

                    unique_screeners_in_order = []
                    seen = set()
                    for s in screener:
                        s_norm = norm_id(s)
                        if s_norm not in seen:
                            unique_screeners_in_order.append(s_norm)
                            seen.add(s_norm)

                    if pre_half:
                        print_bring_over_halfcourt(last_possessor, balldef, pickup_style=head_side_h_pickup)

                    has_ice = any(a == "ice" for a in per_screen_actions)

                    if not has_ice:
                        if len(unique_screeners_in_order) == 1:
                            s0 = unique_screeners_in_order[0]
                            try:
                                idx0 = [norm_id(x) for x in screener].index(s0)
                                d0 = screendef[idx0]
                            except Exception:
                                d0 = []
                            print(
                                f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} comes off a {onball_label[action_type]} "
                                f"from Player {s0}{coverage_phrase(d0, s0)}"
                            )
                        else:
                            parts_intro = []
                            norm_screeners = [norm_id(x) for x in screener]
                            for s_id in unique_screeners_in_order:
                                try:
                                    idx_s = norm_screeners.index(s_id)
                                    d_s = screendef[idx_s]
                                except Exception:
                                    d_s = []
                                parts_intro.append(f"Player {s_id}{coverage_phrase(d_s, s_id)}")
                            if len(parts_intro) > 1:
                                print(
                                    f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} comes off a {onball_label[action_type]} from {', '.join(parts_intro[:-1])} and {parts_intro[-1]}"
                                )
                            else:
                                print(
                                    f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} comes off a {onball_label[action_type]} from {parts_intro[0]}"
                                )

                    iced = False
                    # *** CHANGED: keep ALL screeners if multiple; single remains a str ***
                    last_screener = (
                        unique_screeners_in_order[0]
                        if len(unique_screeners_in_order) == 1
                        else unique_screeners_in_order
                    )

                    current_ball_def = list(balldef) if balldef else []
                    current_scr_def = {}
                    norm_screeners_all = [norm_id(x) for x in screener]
                    for s_id in unique_screeners_in_order:
                        try:
                            idx_s = norm_screeners_all.index(s_id)
                            current_scr_def[s_id] = screendef[idx_s] or []
                        except Exception:
                            current_scr_def[s_id] = []

                    occurrences = {}
                    for idx_step, (s_id_raw, d_list_original, a_code) in enumerate(zip(norm_screeners_all, screendef, per_screen_actions)):
                        s_id = s_id_raw
                        occurrences[s_id] = occurrences.get(s_id, 0) + 1
                        d_cur = current_scr_def.get(s_id, d_list_original or [])

                        if (
                            len(unique_screeners_in_order) == 1
                            and occurrences[s_id] > 1
                            and _is_screen_interaction(a_code)
                        ):
                            print(f"Player {s_id}{coverage_phrase(d_cur,s_id)} rescreens for Player {last_possessor}{coverage_phrase(current_ball_def,last_possessor)}")

                        screen_label = "screen" if len(unique_screeners_in_order) == 1 else f"{ordinal_word(idx_step+1)} screen"

                        if not a_code:
                            continue
                        if a_code == "ch":
                            print(f"{format_player_list(current_ball_def)} chases over the {screen_label}")
                        elif a_code == "ct":
                            print(f"{format_player_list(current_ball_def)} cuts under the {screen_label}")
                        elif a_code.startswith("sw"):
                            m_sw = re.match(r"sw(\d+)", a_code)
                            prev_ball_def = list(current_ball_def)
                            if m_sw:
                                switch_actor = norm_id(m_sw.group(1))
                            elif d_cur and d_cur[0]:
                                switch_actor = norm_id(d_cur[0])
                            else:
                                switch_actor = "Unknown"
                            print(f"Player {switch_actor} switches onto Player {last_possessor} on the {screen_label}")
                            current_ball_def = [switch_actor] if switch_actor != "Unknown" else []
                            current_scr_def[s_id] = prev_ball_def
                        elif a_code == "bz":
                            blitz_side = current_scr_def.get(s_id, [])
                            if blitz_side and current_ball_def and set(blitz_side) == set(current_ball_def) and len(set(current_ball_def)) == 1:
                                print(f"{format_player_list(current_ball_def)} blitzes Player {last_possessor} on the {screen_label}")
                            elif blitz_side and current_ball_def:
                                print(f"{format_player_list(current_ball_def)} and {format_player_list(blitz_side)} double team Player {last_possessor} on the {screen_label}")
                            elif blitz_side:
                                print(f"{format_player_list(blitz_side)} blitzes Player {last_possessor} on the {screen_label}")
                            else:
                                print(f"{format_player_list(current_ball_def)} blitzes Player {last_possessor} on the {screen_label}")
                            current_ball_def = list({*(current_ball_def or []), *(blitz_side or [])})
                        elif a_code == "cs":
                            print(f"{format_player_list(current_ball_def)} gets caught on the {screen_label}")
                            current_ball_def = []  # caught → wide open
                        elif a_code == "ice":
                            print(
                                f"Player {last_possessor}{coverage_phrase(current_ball_def,last_possessor)} "
                                f"tries to come off a {onball_label[action_type]} "
                                f"from Player {s_id}{coverage_phrase(d_cur, s_id)}, "
                                f"but {format_player_list(current_ball_def)} ices the screen and forces Player {last_possessor} away from the screen"
                            )
                            iced = True
                            last_screener = None
                        elif a_code in ("h", "d", "dho", "ho", "kp"):
                            if a_code == "h":
                                print_bring_over_halfcourt(last_possessor, current_ball_def, pickup_style=head_side_h_pickup)
                            elif a_code in ("dho", "ho"):
                                print(f"Player {last_possessor}{coverage_phrase(current_ball_def,last_possessor)} hands off inside")
                            elif a_code == "kp":
                                print(f"Player {last_possessor}{coverage_phrase(current_ball_def,last_possessor)} keeps the handoff and continues the play")
                            else:
                                print(f"Player {last_possessor}{coverage_phrase(current_ball_def,last_possessor)} drives inside")
                        elif a_code in ("+", "++", "-"):
                            handle_shot_with_screen_assists(
                                last_possessor, current_ball_def, a_code, last_passer, (last_screener if not iced else None), pnr_shot=True
                            )
                            pending_offball_sa = None
                            last_shooter = last_possessor
                            last_possessor = None
                            last_defender = None
                            last_screener = None
                            break

                    # Persist ball-defender after screen sequence
                    last_defender = current_ball_def

                    # Consume trailing PNR actions (e.g., 'd-', 'h', '+')
                    ta = (trailing_actions or "").strip()
                    while ta:
                        m = re.match(r"(ch|ct|sw\d*|bz|tl|cs|ice|h(?!o)|p\b|d(?!ho)|dho|ho|kp|\+{1,2}|-)", ta)
                        if not m:
                            break
                        code = m.group(0)
                        ta = ta[len(code):]

                        if code == "h":
                            print_bring_over_halfcourt(last_possessor, last_defender, pickup_style=head_side_h_pickup)
                        elif code == "p":
                            print(f"Player {last_possessor}{coverage_phrase(last_defender,last_possessor)} posts up")
                        elif code in ("d","dho","ho","kp"):
                            if code in ("dho","ho"):
                                print(f"Player {last_possessor}{coverage_phrase(last_defender,last_possessor)} hands off inside")
                            elif code == "kp":
                                print(f"Player {last_possessor}{coverage_phrase(last_defender,last_possessor)} keeps the handoff and continues the play")
                            else:
                                print(f"Player {last_possessor}{coverage_phrase(last_defender,last_possessor)} drives inside")
                        elif code in ("+","++","-"):
                            handle_shot_with_screen_assists(
                                last_possessor, last_defender, code, last_passer, last_screener, pnr_shot=True
                            )
                            pending_offball_sa = None
                            last_shooter = last_possessor
                            last_possessor = None
                            last_defender = None
                            last_screener = None
                            break
                        elif code.startswith("sw"):
                            m_sw2 = re.match(r"sw(\d+)", code)
                            if m_sw2:
                                swp2 = norm_id(m_sw2.group(1))
                                print(f"Player {swp2} switches onto Player {last_possessor}")
                                last_defender = [swp2]

                    handle_zone_end(zone_end)
                    i += 1
                    continue

                # --- Non-PNR/PNP on-ball intros (incl. DHO/HO/KP/RJ) ---
                last_screener = screener if action_type in ("pnr", "pnp") else None
                trailing_actions = "".join(after) if isinstance(after, (list, tuple)) else (after or "")

                if "ice" in trailing_actions and action_type in ("pnr", "pnp"):
                    trailing_actions = trailing_actions.replace("ice", "", 1)
                    scr_for_print = norm_id(screener[0]) if isinstance(screener, list) and screener else norm_id(screener)
                    if isinstance(screendef, list) and screendef and isinstance(screendef[0], list):
                        sdef_for_print = screendef[0]
                    else:
                        sdef_for_print = screendef
                    print(
                        f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} tries to come off a {onball_label[action_type]} "
                        f"from Player {scr_for_print}{coverage_phrase(sdef_for_print, scr_for_print)}, "
                        f"but {format_player_list(balldef)} ices the screen and forces Player {last_possessor} away from the screen"
                    )
                    last_screener = None
                else:
                    if pre_half:
                        print_bring_over_halfcourt(last_possessor, balldef, pickup_style=head_side_h_pickup)
                    if action_type in ("pnr", "pnp"):
                        print(
                            f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} comes off a {onball_label[action_type]} "
                            f"from Player {norm_id(screener)}{coverage_phrase(screendef, norm_id(screener))}"
                        )
                    elif action_type == "rj":
                        print(
                            f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} rejects the ball screen from "
                            f"Player {norm_id(screener)}{coverage_phrase(screendef, norm_id(screener))}"
                        )
                    elif action_type in ("dho", "ho"):
                        ho_giver_def = list(balldef) if balldef else []
                        verb = "dribbles and hands off to" if action_type == "dho" else "hands off to"
                        print(
                            f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} {verb} "
                            f"Player {norm_id(screener)}{coverage_phrase(screendef, norm_id(screener))}"
                        )
                        last_possessor = norm_id(screener)
                        last_defender = norm_list(screendef) if isinstance(screendef, list) else norm_list([screendef])
                        last_ho_giver_def = ho_giver_def

                    elif action_type == "kp":
                        print(
                            f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)} keeps the handoff intended for "
                            f"Player {norm_id(screener)}{coverage_phrase(screendef, norm_id(screener))}"
                        )
                    else:
                        print(
                            f"Player {norm_id(screener)}{coverage_phrase(screendef, norm_id(screener))} {onball_label[action_type]} "
                            f"Player {last_possessor}{coverage_phrase(balldef,last_possessor)}"
                        )

                # Chain handling & trailing actions
                if re.match(r"^(?:pnr|pnp|slp|gst|rj|dho|ho|kp)\d", (trailing_actions or "").strip()):
                    adopted_def = ",".join(last_defender) if isinstance(last_defender, list) else (last_defender or "")
                    parts.insert(i + 1, f"{last_possessor}/{adopted_def}{trailing_actions}")
                    trailing_actions = ""

                m_chain = re.match(r"^\d+/\d+(?:pnr|pnp|slp|gst|rj|dho|ho|kp).+", (trailing_actions or "").replace(" ", ""))
                if m_chain:
                    parts.insert(i + 1, trailing_actions)
                    trailing_actions = ""

                active_def = list(last_defender) if last_defender else []
                _max_steps = 100
                _steps = 0
                while trailing_actions and _steps < _max_steps:
                    _steps += 1
                    trailing_actions = trailing_actions.lstrip()

                    m_leading_defs = re.match(r"^\d+/(?:\d+(?:,\d+)*)", trailing_actions)
                    if m_leading_defs:
                        def_str = m_leading_defs.group(0).split("/", 1)[1]
                        active_def = [norm_id(x) for x in def_str.split(",") if x]
                        trailing_actions = trailing_actions[len(m_leading_defs.group(0)):]
                        continue

                    m_onball = re.match(r"^(pnr|pnp|slp|gst|rj|dho|ho|kp)\d+(?:/\d+)?", trailing_actions)
                    if m_onball:
                        adopted_def = ",".join(active_def) if active_def else (",".join(last_defender) if last_defender else "")
                        parts.insert(i + 1, f"{last_possessor}/{adopted_def}{trailing_actions}")
                        trailing_actions = ""
                        break

                    m = re.match(r"(ch|ct|sw\d*|bz|tl|cs|ice|h(?!o)|p\b|d(?!ho)|dho|ho|kp|\+{1,2}|-)", trailing_actions)
                    if not m:
                        break
                    code = m.group(0)
                    trailing_actions = trailing_actions[len(code):]

                    action_word = "screen" if last_action_type in ("pnr", "pnp", "slp", "gst", "rj") else "handoff"

                    if code == "ch":
                        print(f"{format_player_list(active_def)} chases over the {action_word}")
                    elif code == "ct":
                        print(f"{format_player_list(active_def)} cuts under the {action_word}")
                    elif code.startswith("sw"):
                        switch_actor_match = re.match(r"sw(\d+)", code)
                        if switch_actor_match:
                            switch_actor = norm_id(switch_actor_match.group(1))
                        elif last_action_type in ("dho", "ho") and last_ho_giver_def:
                            switch_actor = norm_id(last_ho_giver_def[0])
                        else:
                            # fallback to screener's first defender if available
                            try:
                                if isinstance(screendef, list) and screendef and isinstance(screendef[0], list):
                                    switch_actor = norm_id(screendef[0][0]) if screendef[0] else "Unknown"
                                else:
                                    switch_actor = norm_id(screendef[0]) if screendef else "Unknown"
                            except Exception:
                                switch_actor = "Unknown"
                        print(f"Player {switch_actor} switches onto Player {last_possessor} on the {action_word}")
                        active_def = [switch_actor] if switch_actor != "Unknown" else []
                        last_defender = list(active_def)
                    elif code == "bz":
                        if last_action_type in ("dho", "ho") and last_ho_giver_def:
                            print(f"{format_player_list(last_defender)} and {format_player_list(last_ho_giver_def)} double team Player {last_possessor}")
                            active_def = list(set((last_defender or []) + (last_ho_giver_def or [])))
                        elif screendef and last_defender:
                            print(f"{format_player_list(last_defender)} and {format_player_list(screendef)} double team Player {last_possessor}")
                            active_def = list(set((last_defender or []) + (screendef or [])))
                        elif screendef:
                            print(f"{format_player_list(screendef)} double team Player {last_possessor}")
                            active_def = screendef
                    elif code == "ice":
                        if isinstance(screener, list) and screener:
                            s_for_print = norm_id(screener[0])
                        else:
                            s_for_print = norm_id(screener)
                        if isinstance(screendef, list) and screendef and isinstance(screendef[0], list):
                            d_for_print = screendef[0]
                        else:
                            d_for_print = screendef
                        print(
                            f"Player {last_possessor}{coverage_phrase(active_def,last_possessor)} "
                            f"tries to come off a {onball_label[action_type]} "
                            f"from Player {s_for_print}{coverage_phrase(d_for_print, s_for_print)}, "
                            f"but {format_player_list(active_def)} ices the screen and forces Player {last_possessor} away from the screen"
                        )
                        last_screener = None
                    elif code == "tl":
                        print(f"{format_player_list(active_def)} top locks Player {last_possessor} forcing them away from the {action_word}")
                    elif code == "cs":
                        print(f"{format_player_list(active_def)} gets caught on the {action_word}")
                        active_def = []         # caught → no longer guarding (wide open)
                        last_defender = []      # persist as wide open until switch/help
                    elif code == "h":
                        print_bring_over_halfcourt(last_possessor, active_def, pickup_style=head_side_h_pickup)
                    elif code == "p":
                        print(f"Player {last_possessor}{coverage_phrase(active_def,last_possessor)} posts up")
                    elif code in ("d", "dho", "ho", "kp"):
                        if code in ("dho", "ho"):
                            print(f"Player {last_possessor}{coverage_phrase(active_def,last_possessor)} hands off inside")
                            last_possessor = norm_id(screener) if isinstance(screener, str) else norm_id(screener if screener else last_possessor)
                        elif code == "kp":
                            print(f"Player {last_possessor}{coverage_phrase(active_def,last_possessor)} keeps the handoff and continues the play")
                        else:
                            print(f"Player {last_possessor}{coverage_phrase(active_def,last_possessor)} drives inside")
                    elif code in ("+", "++", "-"):
                        current_def = active_def if active_def else balldef
                        if last_action_type not in ("pnr", "pnp"):
                            shooter_tmp = norm_id(last_possessor)
                            if shooter_tmp in offball_sa_map:
                                scrs = list(offball_sa_map.pop(shooter_tmp, set()))
                                if scrs:
                                    pending_offball_sa = {"recipient": str(shooter_tmp), "screeners": [str(s) for s in scrs]}
                        handle_shot_with_screen_assists(
                            last_possessor, current_def, code, last_passer, last_screener,
                            pnr_shot=(action_type in ("pnr", "pnp"))
                        )
                        pending_offball_sa = None
                        last_shooter = last_possessor
                        last_possessor = None
                        last_defender = None
                        last_screener = None
                        break

                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # Standalone action tokens for the current ballhandler
            # ------------------------------------------------------------------
            if token in ("+", "++", "-") or token in ("d", "h", "p") or re.match(r"^(d|p)(\+\+|\+|-)$", token):
                if not last_possessor:
                    handle_zone_end(zone_end)
                    i += 1
                    continue

                cur_off = norm_id(last_possessor)
                cur_def = norm_list(last_defender)

                if token == "h":
                    print_bring_over_halfcourt(cur_off, cur_def, pickup_style=head_side_h_pickup)
                    handle_zone_end(zone_end)
                    i += 1
                    continue

                if token.startswith("p"):
                    shot = "++" if token.endswith("++") else "+" if token.endswith("+") else "-" if token.endswith("-") else None
                    print(f"Player {cur_off}{coverage_phrase(cur_def,cur_off)} posts up")
                    if shot:
                        if cur_off in offball_sa_map:
                            scrs = list(offball_sa_map.pop(cur_off, set()))
                            if scrs:
                                pending_offball_sa = {"recipient": str(cur_off), "screeners": [str(s) for s in scrs]}
                        handle_shot_with_screen_assists(cur_off, cur_def, shot, last_passer, last_screener)
                        pending_offball_sa = None
                        last_possessor, last_defender, last_shooter = None, None, cur_off
                        last_screener = None
                        loose_ball_pending = False
                    handle_zone_end(zone_end)
                    i += 1
                    continue

                if token.startswith("d"):
                    shot = "++" if token.endswith("++") else "+" if token.endswith("+") else "-" if token.endswith("-") else None
                    print(f"Player {cur_off}{coverage_phrase(cur_def,cur_off)} drives inside")
                    if shot:
                        if cur_off in offball_sa_map:
                            scrs = list(offball_sa_map.pop(cur_off, set()))
                            if scrs:
                                pending_offball_sa = {"recipient": str(cur_off), "screeners": [str(s) for s in scrs]}
                        handle_shot_with_screen_assists(cur_off, cur_def, shot, last_passer, last_screener)
                        pending_offball_sa = None
                        last_possessor, last_defender, last_shooter = None, None, cur_off
                        last_screener = None
                        loose_ball_pending = False
                    else:
                        last_possessor, last_defender = cur_off, cur_def
                    handle_zone_end(zone_end)
                    i += 1
                    continue

                if token in ("+", "++", "-"):
                    if cur_off in offball_sa_map:
                        scrs = list(offball_sa_map.pop(cur_off, set()))
                        if scrs:
                            pending_offball_sa = {"recipient": str(cur_off), "screeners": [str(s) for s in scrs]}
                    handle_shot_with_screen_assists(cur_off, cur_def, token, last_passer, last_screener)
                    pending_offball_sa = None
                    last_possessor, last_defender, last_shooter = None, None, cur_off
                    last_screener = None
                    loose_ball_pending = False
                    handle_zone_end(zone_end)
                    i += 1
                    continue

            # ------------------------------------------------------------------
            # Slash / normal actions with explicit defender(s),
            # receive-only open, or shot-only / micro-only after slash
            # ------------------------------------------------------------------
            if "/" in token:
                _clean = token.replace(" ", "")

                # (0) receive-only with explicit open coverage: "6/" (no RHS)
                m_receive_open = re.match(r"^(\d+)/$", _clean)
                if m_receive_open:
                    off = norm_id(m_receive_open.group(1))
                    receiver_defs = []  # explicit wide open
                    if last_possessor and norm_id(last_possessor) != off and not pending_inbounder:
                        print_pass_with_open(last_possessor, last_defender, off, receiver_defs, inbound=False)
                        last_passer = last_possessor
                    last_possessor = off
                    last_defender  = receiver_defs
                    handle_zone_end(zone_end)
                    i += 1
                    continue

                # (a) shot-only after slash:  "3/++"  "3/+"  "3/-"
                m_shot_only = re.match(r"^(\d+)/(\+\+|\+|-)$", _clean)
                # (b) micro-action after slash (no defender list): "3/h", "3/d+", "3/p++"
                m_micro_only = re.match(r"^(\d+)/(h|p|d)((\+\+|\+|-)?)(or|r)?$", _clean)
                # (c) standard: off / def[,def...] [tail actions...]
                m_standard = re.match(r"^(\d+)/((?:rot\d+|\d+)(?:,(?:rot\d+|\d+))*)(.*)$", _clean)

                if m_shot_only:
                    off = norm_id(m_shot_only.group(1))
                    defender_list = norm_list(defender_memory.get(off, []))
                    action = m_shot_only.group(2)
                elif m_micro_only:
                    off = norm_id(m_micro_only.group(1))
                    micro = m_micro_only.group(2)
                    shot_suffix = m_micro_only.group(3) or ""
                    rb_suffix = m_micro_only.group(5) or ""
                    defender_list = norm_list(defender_memory.get(off, []))
                    action = micro + shot_suffix + rb_suffix
                elif m_standard:
                    off = norm_id(m_standard.group(1))
                    defender_list = norm_list(re.findall(r"(?:rot\d+|\d+)", m_standard.group(2)))
                    action = m_standard.group(3) or ""
                else:
                    parsed = parse_player_def(token)
                    if not parsed:
                        last_possessor = norm_id(token.split("/", 1)[0])
                        last_defender = None
                        handle_zone_end(zone_end)
                        i += 1
                        continue
                    off, defender, action, _, _, _ = parsed
                    off = norm_id(off)
                    defender_list = norm_list(defender)

                # Normal pass into a new possessor (skip if waiting on inbound)
                if last_possessor and norm_id(last_possessor) != off and not pending_inbounder:
                    print_pass_with_open(last_possessor, last_defender, off, defender_list, inbound=False)
                    last_passer = last_possessor

                # Sequentially process chained actions like h d- or, p++, ++, etc.
                k = 0
                consumed_any = False
                while k < len(action):
                    # bring up
                    if action[k:].startswith("h"):
                        print_bring_over_halfcourt(off, defender_list, pickup_style=head_side_h_pickup)
                        last_possessor, last_defender = off, defender_list
                        k += 1
                        consumed_any = True
                        continue

                    # post up (+ optional shot, then optional inline rebound)
                    m_p = re.match(r"^p(\+\+|\+|-)?", action[k:])
                    if m_p:
                        print(f"Player {off}{coverage_phrase(defender_list,off)} posts up")
                        last_possessor = off
                        last_defender = defender_list
                        shot = m_p.group(1)
                        k += 1 + (len(shot) if shot else 0)
                        consumed_any = True
                        if shot:
                            if off in offball_sa_map:
                                scrs = list(offball_sa_map.pop(off, set()))
                                if scrs:
                                    pending_offball_sa = {"recipient": str(off), "screeners": [str(s) for s in scrs]}
                            handle_shot_with_screen_assists(off, defender_list, shot, last_passer, last_screener)
                            pending_offball_sa = None
                            # Try inline rebound immediately after shot
                            adv, poss_off, poss_def = consume_inline_rebound(action[k:], off, defender_list)
                            if adv > 0:
                                k += adv
                                last_possessor, last_defender, last_shooter = poss_off, poss_def, off
                                loose_ball_pending = False
                            else:
                                last_possessor, last_defender, last_shooter = None, None, off
                                last_screener = None
                                loose_ball_pending = False
                        continue

                    # drive (+ optional shot, then optional inline rebound)
                    m_d = re.match(r"^d(\+\+|\+|-)?", action[k:])
                    if m_d:
                        print(f"Player {off}{coverage_phrase(defender_list,off)} drives inside")
                        last_possessor = off
                        last_defender = defender_list
                        shot = m_d.group(1)
                        k += 1 + (len(shot) if shot else 0)
                        consumed_any = True
                        if shot:
                            if off in offball_sa_map:
                                scrs = list(offball_sa_map.pop(off, set()))
                                if scrs:
                                    pending_offball_sa = {"recipient": str(off), "screeners": [str(s) for s in scrs]}
                            handle_shot_with_screen_assists(off, defender_list, shot, last_passer, last_screener)
                            pending_offball_sa = None
                            # Try inline rebound immediately after shot
                            adv, poss_off, poss_def = consume_inline_rebound(action[k:], off, defender_list)
                            if adv > 0:
                                k += adv
                                last_possessor, last_defender, last_shooter = poss_off, poss_def, off
                                loose_ball_pending = False
                            else:
                                last_possessor, last_defender, last_shooter = None, None, off
                                last_screener = None
                                loose_ball_pending = False
                        continue

                    # bare shot (++, +, -) then optional inline rebound
                    m_s = re.match(r"^(\+\+|\+|-)", action[k:])
                    if m_s:
                        code = m_s.group(1)
                        if off in offball_sa_map:
                            scrs = list(offball_sa_map.pop(off, set()))
                            if scrs:
                                pending_offball_sa = {"recipient": str(off), "screeners": [str(s) for s in scrs]}
                        handle_shot_with_screen_assists(off, defender_list, code, last_passer, last_screener)
                        pending_offball_sa = None
                        k += len(code)
                        consumed_any = True
                        # Try inline rebound immediately after shot
                        adv, poss_off, poss_def = consume_inline_rebound(action[k:], off, defender_list)
                        if adv > 0:
                            k += adv
                            last_possessor, last_defender, last_shooter = poss_off, poss_def, off
                            loose_ball_pending = False
                        else:
                            last_possessor, last_defender, last_shooter = None, None, off
                            last_screener = None
                            loose_ball_pending = False
                        continue

                    # inline rebound with no explicit prior '-/+/++' (rare but tolerated)
                    if action[k:].startswith("or") or action[k:].startswith("r"):
                        adv, poss_off, poss_def = consume_inline_rebound(action[k:], off, defender_list)
                        if adv > 0:
                            k += adv
                            last_possessor, last_defender = poss_off, poss_def
                            consumed_any = True
                            continue

                    break  # unknown token; stop sequential parsing

                if not consumed_any:
                    if action.startswith("d"):
                        print(f"Player {off}{coverage_phrase(defender_list,off)} drives inside")
                        last_possessor = off
                        last_defender = defender_list
                        if action.endswith("++") or action.endswith("+") or action.endswith("-"):
                            code = "++" if action.endswith("++") else "+" if action.endswith("+") else "-"
                            if off in offball_sa_map:
                                scrs = list(offball_sa_map.pop(off, set()))
                                if scrs:
                                    pending_offball_sa = {"recipient": str(off), "screeners": [str(s) for s in scrs]}
                            handle_shot_with_screen_assists(off, defender_list, code, last_passer, last_screener)
                            pending_offball_sa = None
                            # inline rebound suffix?
                            tail_after = action[:-len(code)]
                            adv, poss_off, poss_def = consume_inline_rebound(tail_after[-2:] if len(tail_after) >= 2 else tail_after[-1:], off, defender_list)
                            if adv > 0:
                                last_possessor, last_defender, last_shooter = poss_off, poss_def, off
                                loose_ball_pending = False
                            else:
                                last_possessor, last_defender, last_shooter = None, None, off
                                last_screener = None
                                loose_ball_pending = False
                    elif action.startswith("h"):
                        print_bring_over_halfcourt(off, defender_list, pickup_style=head_side_h_pickup)
                        last_possessor, last_defender = off, defender_list
                    elif action.startswith("p"):
                        print(f"Player {off}{coverage_phrase(defender_list,off)} posts up")
                        last_possessor = off
                        last_defender = defender_list
                        if action.endswith("++") or action.endswith("+") or action.endswith("-"):
                            code = "++" if action.endswith("++") else "+" if action.endswith("+") else "-"
                            if off in offball_sa_map:
                                scrs = list(offball_sa_map.pop(off, set()))
                                if scrs:
                                    pending_offball_sa = {"recipient": str(off), "screeners": [str(s) for s in scrs]}
                            handle_shot_with_screen_assists(off, defender_list, code, last_passer, last_screener)
                            pending_offball_sa = None
                            # inline rebound suffix?
                            tail_after = action[:-len(code)]
                            adv, poss_off, poss_def = consume_inline_rebound(tail_after[-2:] if len(tail_after) >= 2 else tail_after[-1:], off, defender_list)
                            if adv > 0:
                                last_possessor, last_defender, last_shooter = poss_off, poss_def, off
                                loose_ball_pending = False
                            else:
                                last_possessor, last_defender, last_shooter = None, None, off
                                last_screener = None
                                loose_ball_pending = False
                    elif action in ["+", "-", "++"]:
                        if off in offball_sa_map:
                            scrs = list(offball_sa_map.pop(off, set()))
                            if scrs:
                                pending_offball_sa = {"recipient": str(off), "screeners": [str(s) for s in scrs]}
                        handle_shot_with_screen_assists(off, defender_list, action, last_passer, last_screener)
                        pending_offball_sa = None
                        # inline rebound suffix cannot be known here (no tail info), so end pos
                        last_possessor, last_defender, last_shooter = None, None, off
                        last_screener = None
                        loose_ball_pending = False
                    else:
                        last_possessor, last_defender = off, defender_list

                handle_zone_end(zone_end)
                i += 1
                continue

            # ------------------------------------------------------------------
            # Default: standalone player token → possession change, unknown coverage
            # ------------------------------------------------------------------
            last_possessor = norm_id(token)
            last_defender  = None
            handle_zone_end(zone_end)
            i += 1



if __name__ == "__main__":
    main()
