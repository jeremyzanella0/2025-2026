# pscoding16
# Mark 16 - updated after stage 1 of testing (individual testing)

import re

# Tracks temporary switches within a possession:
# keys are (off_player_as_str_or_asterisk, defender_id_str) -> True
defender_memory = {}

# Pending screen assist(s) (for off-ball screens only)
# Shape: {"recipient": <player_id_str>, "screeners": [<player_id_str>, ...]}
pending_offball_sa = None

# ---------------- Parsing Functions ----------------

def parse_player_def(token: str):
    """
    Parse a token into:
      (ball_handler, ball_def, action_type, screener, screener_def, action_codes)
    Multiple defenders anywhere (comma-separated) will be parsed into lists.
    """
    token = token.strip()

    # Off-ball action (parentheses)
    if token.startswith("(") and token.endswith(")"):
        inner = token[1:-1]
        action_type = None
        for key in ("pn", "fl", "bk", "awy", "crs", "wdg", "rip", "ucla", "stg", "ivs", "elv"):
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
                    ball_handler = bh_parts[0] if bh_parts[0] else None
                    ball_def = bh_parts[1].split(",") if len(bh_parts) > 1 and bh_parts[1] else None
                else:
                    ball_handler = ball_part if ball_part else None

                # Parse stagger/iverson/elevator
                screener_list, screendef_list, after_list = parse_stagger_screens(rest)
                return ball_handler, ball_def, action_type, screener_list, screendef_list, after_list

            # Single off-ball screen
            ball_part, rest = inner.split(action_type, 1)
            ball_handler, ball_def = None, None
            if "/" in ball_part:
                bh_parts = ball_part.split("/", 1)
                ball_handler = bh_parts[0] if bh_parts[0] else None
                ball_def = bh_parts[1].split(",") if len(bh_parts) > 1 and bh_parts[1] else None
            else:
                ball_handler = ball_part if ball_part else None
            screener, screener_def, after = None, None, rest
            if "/" in rest:
                parts2 = rest.split("/", 1)
                screener = parts2[0] if parts2[0] else None
                m = re.match(r"(rot\d+|\d+)", parts2[1])
                screener_def = [m.group(1)] if m else []
                after = parts2[1][len(screener_def[0]):] if screener_def else parts2[1]
            else:
                m2 = re.match(r"(\d+)", rest)
                if m2:
                    screener = m2.group(1)
                    after = rest[len(screener):]
                else:
                    screener = rest if rest else None
                    after = ""
            return ball_handler, ball_def, action_type, screener, screener_def, after
        return None, None, None, None, None, None

    # On-ball actions
    action_type = None
    for key in ("pnr", "pnp", "slp", "gst", "rj", "dho", "ho"):
        if key in token:
            action_type = key
            break
    if action_type:
        if action_type == "rj":
            if "/" in token:
                ball_part, rest = token.split("rj", 1)
                if "/" in ball_part:
                    ball_handler, ball_def_raw = ball_part.split("/", 1)
                    ball_def = ball_def_raw.split(",") if ball_def_raw else []
                else:
                    ball_handler = ball_part
                    ball_def = []
                ball_def = [re.sub(r"\^h$", "", d) for d in ball_def]

                if "/" in rest:
                    screener, screener_def_raw = rest.split("/", 1)
                    screener = re.sub(r"(ch|ct|sw\d*|bz|tl|cs|\^h|\^d|dho|ho|\+{1,2}|-)+$", "", screener)
                    cleaned_def = re.sub(r"(ch|ct|sw\d*|bz|tl|cs|\^h|\^d|dho|ho|\+{1,2}|-)+$", "", screener_def_raw)
                    screener_def = [cleaned_def] if cleaned_def else []
                else:
                    m = re.match(r"(\d+)", rest)
                    if m:
                        screener = m.group(1)
                        screener_def = []
                        rest = rest[len(screener):]
                    else:
                        screener = rest
                        screener_def = []
                        rest = ""
                trailing_codes = [m.group(0) for m in re.finditer(r"(ch|ct|sw\d*|bz|tl|cs|ice|\^h|\^d|dho|ho|\+{1,2}|-)", rest.strip())]
                trailing_codes = trailing_codes if trailing_codes else []
                return ball_handler, ball_def, action_type, screener, screener_def, trailing_codes
            else:
                parts = token.split("rj")
                ball_handler = parts[0]
                ball_def = []
                screener = parts[1] if len(parts) > 1 else None
                screener_def = []
                return ball_handler, ball_def, action_type, screener, screener_def, ""

        ball_part, rest = token.split(action_type, 1)
        if "/" in ball_part:
            ball_handler, def_and_actions = ball_part.split("/", 1)
            ball_def = def_and_actions.split(",") if def_and_actions else []
            action_codes = rest
        else:
            ball_handler, ball_def, action_codes = ball_part, [], rest
        ball_def = [re.sub(r"\^h$", "", d) for d in ball_def]

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
            screener = action_codes[:j]
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
                screener_def = [rest2[1:k]] if k > 1 else []
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
            screener = action_codes[:i]
            action = action_codes[i:]
        return ball_handler, ball_def, action_type, screener, screener_def, action

    # Slash / normal actions
    if "/" in token:
        off, rest = token.split("/", 1)
        i = 0
        if rest.startswith("rot"):
            i = 3
            while i < len(rest) and rest[i].isdigit():
                i += 1
        else:
            while i < len(rest) and (rest[i].isdigit() or rest[i]==","):
                i += 1
        def_player = rest[:i] if i>0 else None
        action = rest[i:] if i < len(rest) else ""
        def_list = def_player.split(",") if def_player else []
        return off, def_list, action, None, None, None

    return token, [], token, None, None, None

def parse_stagger_screens(s):
    """
    Parse stagger/iverson/elevator screen format 
    Stagger: "5/6sw7/8ch"
    Elevator: "5/6,7/8sw6"
    Returns (screener_list, screendef_list, action_list)
    """
    s = s.replace(" ", "")
    screener_list = []
    screendef_list = []
    action_list = []
    
    if ',' in s:  # Elevator format
        i = 0
        while i < len(s):
            screener_match = re.match(r"(\d+)", s[i:])
            if not screener_match: break
            screener = screener_match.group(1); i += len(screener)
            defender = None
            if i < len(s) and s[i] == '/':
                i += 1
                defender_match = re.match(r"(\d+)", s[i:])
                if defender_match:
                    defender = defender_match.group(1); i += len(defender)
            screener_list.append(screener)
            screendef_list.append([defender] if defender else [])
            if i < len(s) and s[i] == ',':
                i += 1; action_list.append("")
            else:
                action = ""
                action_match = re.match(r"(ch|ct|sw\d*|bz\d*|tl|cs|\^d|dho|ho|\+{1,2}|-)", s[i:])
                if action_match:
                    action = action_match.group(1); i += len(action)
                action_list.append(action)
    else:  # Stagger/Iverson sequential
        i = 0
        while i < len(s):
            screener_match = re.match(r"(\d+)", s[i:])
            if not screener_match: break
            screener = screener_match.group(1); i += len(screener)
            defender = None
            if i < len(s) and s[i] == '/':
                i += 1
                defender_match = re.match(r"(\d+)", s[i:])
                if defender_match:
                    defender = defender_match.group(1); i += len(defender)
            action = ""
            action_match = re.match(r"(ch|ct|sw|bz|tl|cs|\^d|dho|ho|\+{1,2}|-)", s[i:])
            if action_match:
                action = action_match.group(1); i += len(action)
            screener_list.append(screener)
            screendef_list.append([defender] if defender else [])
            action_list.append(action)
    
    return screener_list, screendef_list, action_list

# ---------------- Helper Functions ----------------

def defender_text(def_player, off_player=None):
    """
    Render defender text. 'rotX' first reference says 'rotating over' once per possession.
    """
    global defender_memory
    if not def_player:
        return "wide open"
    def_list = [def_player] if isinstance(def_player, str) else def_player
    parts = []
    for d in def_list:
        if not d: continue
        d_str = str(d)
        if d_str.startswith("rot"):
            defender = d_str[3:]
            key = (str(off_player) if off_player is not None else "*", defender)
            if key in defender_memory:
                parts.append(f"Player {defender}")
            else:
                parts.append(f"Player {defender} rotating over")
                defender_memory[key] = True
        else:
            parts.append(f"Player {d_str}")
    if len(parts) == 0:
        return "wide open"
    elif len(parts) == 1:
        return f"guarded by {parts[0]}"
    else:
        return f"guarded by {', '.join(parts)}"

def format_player_list(players):
    if not players: return "Unknown"
    if isinstance(players, str): return f"Player {players}"
    players = [str(p) for p in players]
    if len(players) == 1: return f"Player {players[0]}"
    if len(players) == 2: return f"Players {players[0]} and {players[1]}"
    return "Players " + ", ".join(players[:-1]) + f", and {players[-1]}"

def print_pass(from_player, from_def, to_player, to_def, inbound=False):
    action_word = "inbounds the ball to" if inbound else "passes to"
    print(f"Player {from_player} {defender_text(from_def, from_player)} {action_word} Player {to_player} {defender_text(to_def, to_player)}")
    return from_player

# --- Shot line & screen-assist formatting ---

def _format_screen_assist_suffix(screen_assist_by):
    if not screen_assist_by:
        return None
    if isinstance(screen_assist_by, str):
        return f"(screen assist by Player {screen_assist_by})"
    screeners = [str(s) for s in screen_assist_by if s]
    if not screeners:
        return None
    if len(screeners) == 1:
        return f"(screen assist by Player {screeners[0]})"
    if len(screeners) == 2:
        return f"(screen assists by Players {screeners[0]} and {screeners[1]})"
    players = ", ".join(screeners[:-1]) + f", and {screeners[-1]}"
    return f"(screen assists by Players {players})"

def print_shot_line(player, def_list, action, assisted_by=None, screen_assist_by=None):
    made = action in ("+","++")
    verb = "makes the shot" if made else "misses the shot"
    suffixes = []
    if action == "++" and assisted_by:
        suffixes.append(f"(assisted by Player {assisted_by})")
    if made and screen_assist_by:
        sa_suffix = _format_screen_assist_suffix(screen_assist_by)
        if sa_suffix:
            suffixes.append(sa_suffix)
    end = "!" if made else ""
    tail = (" " + " ".join(suffixes)) if suffixes else ""
    print(f"Player {player} {defender_text(def_list, player)} {verb}{tail}{end}")

def handle_shot(player, def_list, action, last_passer=None, screen_assister=None):
    print_shot_line(player, def_list, action,
                    assisted_by=(last_passer if action=="++" else None),
                    screen_assist_by=(screen_assister if action in ("+","++") else None))

def handle_shot_with_screen_assists(player, def_list, action, last_passer=None, last_screener=None, pnr_shot=False):
    """
    Prints the shot and applies screen-assist rules.

    Rules:
    - On-ball (PNR/PNP): credit screen assist on any made shot (+ or ++)
      when there is a valid last_screener (single screener), unless 'ice' happened.
    - Off-ball: credit screen assist ONLY if the shot is assisted (++)
      AND there is a valid last_passer AND a pending_offball_sa for this shooter.
      For multi-screener actions, ALL screeners are credited.
    """
    global pending_offball_sa
    screen_asst = None
    made = action in ("+", "++")

    if made:
        if pnr_shot and last_screener:
            # On-ball: single screener credited on makes (unless previously cleared by 'ice')
            screen_asst = str(last_screener)
        else:
            # Off-ball: only on assisted makes and with a pending SA for this shooter
            if (
                pending_offball_sa
                and player == pending_offball_sa.get("recipient")
                and action == "++"
                and last_passer
            ):
                screeners = pending_offball_sa.get("screeners") or []
                screen_asst = [str(s) for s in screeners if s]

    handle_shot(player, def_list, action, last_passer=last_passer, screen_assister=screen_asst)

    # Clear after ANY shot event
    pending_offball_sa = None


# ---------------- Defensive / Rebound / Special Tokens ----------------

def process_defensive_tokens(token, last_player, last_defender, last_shooter, off_player=None, def_list=None):
    base_token = token
    if "/" in token:
        parts = token.split("/")
        off_player = parts[0]
        base_token = "/".join(parts[1:])

    # Out-of-bounds quick prints
    if base_token in ("^rob", "^dob", "^oob"):
        msgs = {
            "^rob": "Rebound goes out of bounds",
            "^dob": "Ball deflected out of bounds",
            "^oob": "Ball goes out of bounds",
        }
        print(msgs[base_token])
        return None, None, None, 1

    # Dead-ball turnover
    if base_token == "^dbto":
        player = off_player or last_player
        def_list = last_defender if isinstance(last_defender, list) else ([last_defender] if last_defender else [])
        print(f"Player {player} {defender_text(def_list, player)} commits a dead ball turnover")
        return None, None, None, 1

    # Live-ball turnover
    if base_token == "^lbto":
        player = off_player or last_player
        def_list = last_defender if isinstance(last_defender, list) else ([last_defender] if last_defender else [])
        print(f"Player {player} {defender_text(def_list, player)} commits a live ball turnover")
        return player, None, None, 1

    # Steal
    if base_token.startswith("^stl"):
        stealer = base_token[4:]
        print(f"Player {stealer} steals the ball!")
        return stealer, None, None, 1

    # Block (accept "^blk8" or "blk8")
    if base_token.startswith("^blk") or base_token.startswith("blk"):
        blocker = (base_token[4:] if base_token.startswith("^blk") else base_token[3:]) or (off_player or "")
        if last_player:
            print(f"Player {blocker} blocks the shot from Player {last_player}")
        else:
            print(f"Player {blocker} blocks the shot")
        return None, None, None, 1

    # Deflection
    if base_token.startswith("^def"):
        deflector = base_token[4:]
        print(f"Player {deflector} deflects the ball from Player {last_player if last_player else 'Unknown'}")
        return None, None, None, 1

    # Jump ball
    if base_token.startswith("^jump"):
        try:
            players = base_token[5:].split(",")
            if len(players) == 2:
                print(f"Jump ball between Player {players[0]} and Player {players[1]}")
            else:
                print("Unrecognized jump ball format")
        except:
            print("Error parsing jump ball event")
        return None, None, None, 1

    # Shot clock violation
    if base_token == "^sc":
        print("Shot clock violation forced by the defense")
        return None, None, None, 1

    # Fouls
    if base_token.startswith("^of"):
        fouler = base_token[3:]
        print(f"Player {fouler} commits an offensive foul")
        return None, None, None, 1

    if base_token.startswith("^f"):
        fouler = base_token[2:]
        print(f"Player {fouler} commits a defensive foul")
        return None, None, None, 1

    # Defensive rebound: accepts "^r", "8^r", or "8/2^r"
    if ("^r" in base_token) and ("^or" not in base_token):
        if "/" in token:
            player = token.split("/", 1)[0]           # "8/2^r"
        else:
            m = re.match(r"(\d+)\^r", token)          # "8^r"
            player = m.group(1) if m else (off_player or last_shooter or last_player)
        print(f"Player {player} grabs the defensive rebound")
        return None, None, None, 1

    # Offensive rebound + optional putback (+ or -)
    if "^or" in base_token:
        player = off_player or last_shooter or last_player
        rebound_def_list = last_defender if isinstance(last_defender, list) else ([last_defender] if last_defender else None)

        if "/" in token:
            player_part, def_part = token.split("/", 1)
            player = player_part
            def_fragment = def_part.split("^or", 1)[0]
            defenders = re.findall(r"(?:rot\d+|\d+)", def_fragment)
            rebound_def_list = defenders if defenders else rebound_def_list

        putback_action = None
        if base_token.endswith("+"): putback_action = "+"
        elif base_token.endswith("-"): putback_action = "-"

        print(f"Player {player} {defender_text(rebound_def_list, player)} grabs the offensive rebound")
        if putback_action:
            handle_shot_with_screen_assists(player, rebound_def_list, putback_action)
            last_shooter = player
            return None, None, last_shooter, 1
        return player, rebound_def_list, None, 1

    # Not a defensive/special token we handle here
    return last_player, last_defender, last_shooter, 0

# ---------------- Off-Ball Screen Helper ----------------

def print_offball_screen(action_type, ball, balldef, screener, screendef, after):
    """
    Prints off-ball screen description and primes a pending off-ball screen assist.
    If denial ('tl') appears, switch to "tries to come off ... but ... top locks ..." wording
    and do NOT prime pending screen assist. Consume 'tl' so it isn't printed twice.
    Returns the possibly-updated ball-defender list.
    """
    global pending_offball_sa

    labels = {
        "pn": "pin down",
        "fl": "flare screen", 
        "bk": "back screen",
        "awy": "away screen",
        "crs": "cross screen",
        "wdg": "wedge screen",
        "rip": "rip screen",
        "ucla": "ucla screen",
        "stg": "stagger screens",
        "ivs": "Iverson screens", 
        "elv": "elevator screen"
    }
    if action_type not in labels:
        return balldef
    label = labels[action_type]
    ball_txt = ball if ball else "Unknown"
    balldef_txt = balldef

    # Determine if denial (top lock) occurs anywhere in this off-ball action
    has_tl = False
    if action_type in ("stg", "ivs", "elv"):
        # 'after' is a list aligned with screeners
        has_tl = any(a == "tl" for a in after)
    else:
        # 'after' is a flat string; check for 'tl'
        has_tl = bool(after and "tl" in after)

    # When 'tl' occurs: use denial wording; don't prime pending SA; consume 'tl' from further prints
    if has_tl:
        # Compose screener text (supports multi-screeners)
        if action_type in ("stg", "ivs", "elv"):
            screen_parts = []
            for s, d in zip(screener, screendef):
                if s:
                    if d: screen_parts.append(f"Player {s} {defender_text(d, s)}")
                    else: screen_parts.append(f"Player {s}")
            joined = " and ".join(screen_parts) if screen_parts else "Unknown"
            print(
                f"Player {ball_txt} {defender_text(balldef_txt, ball_txt)} tries to come off {label} from {joined}, "
                f"but {format_player_list(balldef_txt)} top locks Player {ball_txt} forcing them away from the screen"
            )
            # consume 'tl' by removing it from after before further processing
            after = [("" if a == "tl" else a) for a in after]
        else:
            screener_txt = screener if screener else "Unknown"
            print(
                f"Player {ball_txt} {defender_text(balldef_txt, ball_txt)} tries to come off a {label} from "
                f"Player {screener_txt} {defender_text(screendef, screener_txt)}, "
                f"but {format_player_list(balldef_txt)} top locks Player {ball_txt} forcing them away from the screen"
            )
            # consume 'tl' in flat string
            if after:
                after = after.replace("tl", "")

        # Denial cancels pending screen assist opportunity
        pending_offball_sa = None
    else:
        # No denial -> normal intro + prime pending SA screeners
        if action_type in ("stg", "ivs", "elv"):
            screeners_for_sa = [str(s) for s in screener if s] if isinstance(screener, list) else ([str(screener)] if screener else [])
        else:
            screeners_for_sa = [str(screener)] if screener else []
        pending_offball_sa = {"recipient": ball_txt, "screeners": screeners_for_sa}

        if action_type in ("stg","ivs","elv"):
            screen_parts = []
            for s, d in zip(screener, screendef):
                if s:
                    if d: screen_parts.append(f"Player {s} {defender_text(d, s)}")
                    else: screen_parts.append(f"Player {s}")
            if screen_parts:
                joined = " and ".join(screen_parts)
                print(f"Player {ball_txt} {defender_text(balldef_txt, ball_txt)} comes off {label} from {joined}")
        else:
            screener_txt = screener if screener else "Unknown"
            print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} comes off a {label} from Player {screener_txt} {defender_text(screendef,screener_txt)}")

    # Now process per-screen codes (multi) or trailing codes (single)
    if action_type in ("stg","ivs","elv"):
        for idx, (s, d, a) in enumerate(zip(screener, screendef, after)):
            if not a: continue
            screen_num = "first" if idx == 0 else "second" if idx == 1 else f"{idx+1}th"
            if action_type == "elv":
                if a.startswith("sw"):
                    switch_actor_match = re.match(r"sw(\d+)", a)
                    if switch_actor_match:
                        switch_actor = switch_actor_match.group(1)
                        print(f"Player {switch_actor} switches onto Player {ball_txt} on the elevator screen")
                        balldef_txt = [switch_actor]
                    else:
                        print(f"Player Unknown switches onto Player {ball_txt} on the elevator screen")
                        balldef_txt = []
                elif a in ("+","++","-"):
                    handle_shot_with_screen_assists(ball_txt, balldef_txt, a)
                elif a == "ch":
                    print(f"{format_player_list(balldef_txt)} chases through the elevator screen")
                elif a == "ct":
                    print(f"{format_player_list(balldef_txt)} cuts around the elevator screen")
                elif a.startswith("bz"):
                    blitz_actor_match = re.match(r"bz(\d+)", a)
                    if blitz_actor_match:
                        blitz_actor = blitz_actor_match.group(1)
                        if balldef_txt and balldef_txt[0] != blitz_actor:
                            print(f"Player {blitz_actor} and {format_player_list(balldef_txt)} double team Player {ball_txt} on the elevator screen")
                            balldef_txt = list(set((balldef_txt or []) + [blitz_actor]))
                        else:
                            print(f"Player {blitz_actor} blitzes Player {ball_txt} on the elevator screen")
                            if blitz_actor not in (balldef_txt or []):
                                balldef_txt = (balldef_txt or []) + [blitz_actor]
                    else:
                        blitz_actor_list = d if d else balldef_txt
                        blitz_actor_text = format_player_list(blitz_actor_list) if blitz_actor_list else "Unknown"
                        print(f"{blitz_actor_text} blitzes Player {ball_txt} on the elevator screen")
                        balldef_txt = list(set((balldef_txt or []) + (blitz_actor_list or [])))
                elif a == "cs":
                    caught_actor = format_player_list(balldef_txt) if balldef_txt else "Unknown defender"
                    print(f"{caught_actor} gets caught on the elevator screen")
                elif a in ("^d","dho","ho"):
                    act_word = "drives" if a=="^d" else "hands off"
                    print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} {act_word} immediately after the elevator screen")
            else:
                if a == "ch":
                    print(f"{format_player_list(balldef_txt)} chases over the {screen_num} screen")
                elif a == "ct":
                    print(f"{format_player_list(balldef_txt)} cuts under the {screen_num} screen")
                elif a.startswith("sw"):
                    switch_actor_match = re.match(r"sw(\d+)", a)
                    if switch_actor_match:
                        switch_actor = switch_actor_match.group(1)
                        print(f"Player {switch_actor} switches onto Player {ball_txt} on the {screen_num} screen")
                        balldef_txt = [switch_actor]
                    else:
                        if d and d[0]:
                            switch_actor = d[0]
                            print(f"Player {switch_actor} switches onto Player {ball_txt} on the {screen_num} screen")
                            balldef_txt = [switch_actor]
                        else:
                            print(f"Player Unknown switches onto Player {ball_txt} on the {screen_num} screen")
                            balldef_txt = []
                elif a == "bz":
                    if d and d[0]:
                        screen_defender = d[0]
                        if balldef_txt and balldef_txt[0] != screen_defender:
                            print(f"{format_player_list(balldef_txt)} and Player {screen_defender} double team Player {ball_txt} on the {screen_num} screen")
                            balldef_txt = list(set(balldef_txt + [screen_defender]))
                        else:
                            print(f"Player {screen_defender} blitzes Player {ball_txt} on the {screen_num} screen")
                            if screen_defender not in (balldef_txt or []):
                                balldef_txt = (balldef_txt or []) + [screen_defender]
                    else:
                        blitz_actor_text = format_player_list(balldef_txt) if balldef_txt else "Unknown"
                        print(f"{blitz_actor_text} blitzes Player {ball_txt} on the {screen_num} screen")
                elif a == "cs":
                    caught_actor = format_player_list(balldef_txt) if balldef_txt else "Unknown defender"
                    print(f"{caught_actor} gets caught on the {screen_num} screen")
                elif a in ("^d","dho","ho","+","++","-"):
                    if a in ("+","++","-"):
                        handle_shot_with_screen_assists(ball_txt, balldef_txt, a)
                    else:
                        act_word = "drives" if a=="^d" else "hands off"
                        print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} {act_word} immediately after the {screen_num} screen")
        return balldef_txt

    # Single off-ball screens (non-stagger)
    if after:
        # trailing codes already had 'tl' consumed if denial path above
        for match in re.finditer(r"(ch|ct|sw\d*|bz|cs|\^d|dho|ho|\+{1,2}|-)", after):
            code = match.group(0)
            if code == "ch":
                print(f"{format_player_list(balldef_txt)} chases over the {label}")
            elif code == "ct":
                print(f"{format_player_list(balldef_txt)} cuts under the {label}")
            elif code.startswith("sw"):
                switch_actor_match = re.match(r"sw(\d+)", code)
                if switch_actor_match:
                    switch_actor = switch_actor_match.group(1)
                    print(f"Player {switch_actor} switches onto Player {ball_txt}")
                    balldef_txt = [switch_actor]
                else:
                    print(f"{format_player_list(screendef)} switches onto Player {ball_txt}")
                    balldef_txt = screendef
            elif code == "bz":
                if screendef and balldef_txt:
                    print(f"{format_player_list(balldef_txt)} and {format_player_list(screendef)} double team Player {ball_txt}")
                    balldef_txt = list(set((balldef_txt or []) + (screendef or [])))
                elif screendef:
                    print(f"{format_player_list(screendef)} double team Player {ball_txt}")
                    balldef_txt = screendef
            elif code == "cs":
                caught_actor = format_player_list(balldef_txt) if balldef_txt else "Unknown defender"
                print(f"{caught_actor} gets caught on the {label}")
            elif code in ("^d","dho","ho","+","++","-"):
                if code in ("+","++","-"):
                    handle_shot_with_screen_assists(ball_txt, balldef_txt, code)
                else:
                    act_word = "drives" if code=="^d" else "hands off"
                    print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} {act_word} immediately after the {label}")
    return balldef_txt

# ---------------- Main Loop ----------------

def main():
    """
    Main loop for basketball stat tracker.
    """
    global pending_offball_sa

    while True:
        try:
            line = input("Enter possession (or 'q' to quit): ").strip()
        except EOFError:
            break
        if line.lower() == "q":
            break
        if not line:
            continue

        # Reset “rotating over” memory and pending off-ball SA for this possession
        defender_memory.clear()
        pending_offball_sa = None

        parts = line.split()
        last_possessor = None
        last_defender = []
        last_shooter = None
        last_passer = None
        last_screener = None
        last_action_type = None
        i = 0

        # Process tokens sequentially
        while i < len(parts):
            token = parts[i]

            # 0) Free throws: e.g., "1*x", "23**x"
            m_ft = re.match(r"^(\d+)([\*x]+)$", token)
            if m_ft:
                shooter = m_ft.group(1)
                seq = m_ft.group(2)
                for ch in seq:
                    if ch == "*":
                        print(f"Player {shooter} makes the free throw")
                    elif ch == "x":
                        print(f"Player {shooter} misses the free throw")
                # After FTs, clear context/pending SA
                last_possessor, last_defender = None, []
                pending_offball_sa = None
                i += 1
                continue

            # 1) Defensive/rebound/special tokens that short-circuit
            last_possessor, last_defender, last_shooter, skip = process_defensive_tokens(
                token, last_possessor, last_defender, last_shooter
            )
            if skip:
                last_screener = None
                i += 1
                continue

            # 2) Help (hp) or switch (sw) tokens, possibly with trailing shot codes
            if token.startswith(("hp", "sw")):
                # Normalize: pull off trailing +/++/- into next token if present
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

                # If no possessor yet, infer from previous token's off part
                if not last_possessor and i > 0:
                    prev = parts[i - 1].split("/", 1)[0]
                    last_possessor = prev
                    last_defender = parts[i - 1].split("/", 1)[1].split(",") if "/" in parts[i - 1] else []

                is_help = token.startswith("hp")
                action_raw = token[2:]
                shoot_after = None
                for suffix, meaning in [("++", "assist"), ("+", "make"), ("-", "miss")]:
                    if action_raw.endswith(suffix):
                        shoot_after = meaning
                        action_raw = action_raw[:-len(suffix)]
                        break

                action_list = [x.strip() for x in action_raw.split(",") if x.strip()]
                last_defender = last_defender if isinstance(last_defender, list) else [last_defender] if last_defender else []

                if is_help:
                    # Add helpers to the defender group
                    for x in action_list:
                        if x not in last_defender:
                            last_defender.append(x)
                    print(
                        f"Player {action_list[0]} steps in to help on Player {last_possessor}"
                        if len(action_list) == 1
                        else f"Players {', '.join(action_list)} step in to help on Player {last_possessor}"
                    )
                else:
                    # Switch replaces defender group
                    last_defender = action_list
                    print(
                        f"Player {action_list[0]} switches onto Player {last_possessor}"
                        if len(action_list) == 1
                        else f"Players {', '.join(action_list)} switch onto Player {last_possessor}"
                    )

                if shoot_after:
                    shooter = last_possessor
                    if shoot_after == "assist":
                        handle_shot_with_screen_assists(shooter, last_defender, "++", last_passer, last_screener, pnr_shot=False)
                    else:
                        handle_shot_with_screen_assists(shooter, last_defender, "+" if shoot_after == "make" else "-")
                        last_shooter = shooter
                    last_possessor = None
                    last_defender = []
                    last_screener = None

                i += 1
                continue

            # 3) Off-ball screens in parentheses
            if token.startswith("(") and token.endswith(")"):
                ball, balldef, action_type, screener, screendef, after = parse_player_def(token)
                balldef = balldef if isinstance(balldef, list) else [balldef] if balldef else []
                screendef = screendef if isinstance(screendef, list) else [screendef] if screendef else []

                if action_type in ("pn","fl","bk","awy","crs","wdg","rip","ucla","stg","ivs","elv"):
                    updated_balldef = print_offball_screen(action_type, ball, balldef, screener, screendef, after)
                    # If the current possessor is the cutter, keep their defender list in sync
                    if ball == last_possessor:
                        last_defender = updated_balldef if isinstance(updated_balldef, list) else [updated_balldef] if updated_balldef else []

                i += 1
                continue

            # 4) On-ball screens / actions (pnr, pnp, slp, gst, rj, dho, ho)
            if any(k in token for k in ("pnr","pnp","slp","gst","rj","dho","ho")):
                pre_half = bool(re.search(r"/\d+(?:,\d+)*\^h(?:pnr|pnp|slp|gst|rj|dho|ho)", token))

                ball, balldef, action_type, screener, screendef, after = parse_player_def(token)
                balldef = balldef if isinstance(balldef, list) else [balldef] if balldef else []
                screendef = screendef if isinstance(screendef, list) else [screendef] if screendef else []

                # NEW: If a different player had the ball just before this on-ball token, print that pass first.
                if last_possessor and last_possessor != ball:
                    last_passer = last_possessor
                    print_pass(last_passer, last_defender, ball, balldef, inbound=False)

                # Establish current ballhandler context for this token
                last_possessor = ball
                last_defender = balldef
                last_screener = screener if action_type in ("pnr","pnp") else None
                last_action_type = action_type

                # Friendlier phrasings for on-ball intros
                onball_label = {"pnr": "pick and roll", "pnp": "pick and pop",
                                "slp": "slips the screen for", "gst": "ghosts the screen for",
                                "rj": "rejects the ball screen from",
                                "dho": "dribbles and hands off to", "ho": "hands off to"}

                # Determine if 'ice' appears in trailing actions; if so, intro uses "tries ... but ... ices ..."
                trailing_actions = "".join(after) if isinstance(after,(list,tuple)) else (after or "")
                if "ice" in trailing_actions and action_type in ("pnr", "pnp"):
                    # Remove first 'ice' occurrence so the loop doesn't print it twice
                    trailing_actions = trailing_actions.replace("ice", "", 1)
                    # Denial wording intro
                    print(
                        f"Player {ball} {defender_text(balldef, ball)} tries to come off a {onball_label[action_type]} "
                        f"from Player {screener} {defender_text(screendef, screener)}, "
                        f"but {format_player_list(balldef)} ices the screen and forces Player {ball} away from the screen"
                    )
                    # Icing cancels potential on-ball screen assist
                    last_screener = None
                else:
                    # Normal intros
                    if pre_half:
                        print(f"Player {ball} {defender_text(balldef,ball)} brings the ball over halfcourt")

                    if action_type in ("pnr", "pnp"):
                        print(
                            f"Player {ball} {defender_text(balldef, ball)} comes off a {onball_label[action_type]} "
                            f"from Player {screener} {defender_text(screendef, screener)}"
                        )
                    elif action_type == "rj":
                        print(
                            f"Player {ball} {defender_text(balldef, ball)} rejects the ball screen from "
                            f"Player {screener} {defender_text(screendef, screener)}"
                        )
                    else:
                        print(
                            f"Player {screener} {defender_text(screendef, screener)} {onball_label[action_type]} "
                            f"Player {ball} {defender_text(balldef, ball)}"
                        )

                active_def = balldef
                # Process remaining trailing actions (with 'ice' already consumed if present)
                while trailing_actions:
                    m = re.match(r"(ch|ct|sw\d*|bz|tl|cs|ice|\^h|\^d|\^p|dho|ho|\+{1,2}|-)", trailing_actions)
                    if not m:
                        break
                    code = m.group(0)
                    trailing_actions = trailing_actions[len(code):]

                    action_word = "screen" if last_action_type in ("pnr","pnp","slp","gst","rj") else "handoff"

                    if code == "ch":
                        print(f"{format_player_list(active_def)} chases over the {action_word}")
                    elif code == "ct":
                        print(f"{format_player_list(active_def)} cuts under the {action_word}")
                    elif code.startswith("sw"):
                        switch_actor_match = re.match(r"sw(\d+)", code)
                        if switch_actor_match:
                            switch_actor = switch_actor_match.group(1)
                            print(f"Player {switch_actor} switches onto Player {ball} on the {action_word}")
                            active_def = [switch_actor]
                        else:
                            switch_actor = format_player_list(screendef) if screendef else "Unknown"
                            print(f"{switch_actor} switches onto Player {ball} on the {action_word}")
                            active_def = screendef
                    elif code == "bz":
                        if screendef and balldef:
                            print(f"{format_player_list(balldef)} and {format_player_list(screendef)} double team Player {ball}")
                            active_def = list(set((balldef or []) + (screendef or [])))
                        elif screendef:
                            print(f"{format_player_list(screendef)} double team Player {ball}")
                            active_def = screendef
                    elif code == "ice":
                        # If ice appears here (not caught in intro), print wording and cancel screen assist
                        print(f"{format_player_list(balldef)} ices the screen and forces Player {ball} away from the screen")
                        last_screener = None
                    elif code == "cs":
                        print(f"{format_player_list(active_def)} gets caught on the {action_word}")
                    elif code == "^h":
                        print(f"Player {ball} {defender_text(active_def,ball)} brings the ball over halfcourt")
                        last_possessor = ball
                        last_defender = active_def
                        last_screener = None
                    elif code == "^p":
                        print(f"Player {ball} {defender_text(active_def,ball)} posts up")
                        last_possessor = ball
                        last_defender = active_def
                        last_screener = None
                    elif code in ("^d","dho","ho"):
                        act_word = "drives" if code=="^d" else "hands off"
                        print(f"Player {ball} {defender_text(active_def,ball)} {act_word} inside")
                        last_possessor = ball
                        last_defender = active_def
                        # Preserve screener for PNR/PNP so a later make credits the screen assist,
                        # unless an 'ice' already cleared it.
                        if last_action_type not in ("pnr", "pnp"):
                            last_screener = None
                    elif code in ("+","++","-"):
                        # Use the current defender after any ch/ct/sw/bz updates
                        current_def = active_def if active_def else balldef
                        handle_shot_with_screen_assists(
                            ball, current_def, code, last_passer, last_screener,
                            pnr_shot=(action_type in ("pnr","pnp"))
                        )
                        last_shooter = ball
                        last_possessor = None
                        last_defender = []
                        last_screener = None
                        break

                i += 1
                continue

            # 4.5) Standalone action tokens for the current ballhandler
            if token in ("+", "++", "-") or token.startswith(("^d", "^h", "^p")):
                if not last_possessor:
                    i += 1
                    continue

                cur_off = last_possessor
                cur_def = last_defender if isinstance(last_defender, list) else ([last_defender] if last_defender else [])

                if token.startswith("^h"):
                    print(f"Player {cur_off} {defender_text(cur_def,cur_off)} brings the ball over halfcourt")
                    i += 1
                    continue

                if token.startswith("^p"):
                    shot = "++" if token.endswith("++") else "+" if token.endswith("+") else "-" if token.endswith("-") else None
                    print(f"Player {cur_off} {defender_text(cur_def,cur_off)} posts up")
                    if shot:
                        handle_shot_with_screen_assists(cur_off, cur_def, shot, last_passer, last_screener)
                        last_possessor, last_defender, last_shooter = None, [], cur_off
                        last_screener = None
                    i += 1
                    continue

                if token.startswith("^d"):
                    shot = "++" if token.endswith("++") else "+" if token.endswith("+") else "-" if token.endswith("-") else None
                    print(f"Player {cur_off} {defender_text(cur_def,cur_off)} drives inside")
                    if shot:
                        handle_shot_with_screen_assists(cur_off, cur_def, shot, last_passer, last_screener)
                        last_possessor, last_defender, last_shooter = None, [], cur_off
                        last_screener = None
                    else:
                        last_possessor, last_defender = cur_off, cur_def
                    i += 1
                    continue

                if token in ("+", "++", "-"):
                    handle_shot_with_screen_assists(cur_off, cur_def, token, last_passer, last_screener)
                    last_possessor, last_defender, last_shooter = None, [], cur_off
                    last_screener = None
                    i += 1
                    continue

            # 5) Slash / normal actions with explicit defender(s)
            if "/" in token:
                off, defender, action, _, _, _ = parse_player_def(token)
                defender_list = defender if isinstance(defender, list) else [defender] if defender else []

                # If there's an existing possessor and we're moving the ball, decide if it's an inbound
                if last_possessor and last_possessor != off:
                    last_passer = last_possessor
                    inbound = (i > 0 and "^ob" in parts[i - 1])  # previous token contained ^ob
                    print_pass(last_passer, last_defender, off, defender_list, inbound=inbound)

                # Sequentially process chained actions like ^h^d+, ^p++, etc.
                k = 0
                consumed_any = False
                while k < len(action):
                    if action[k:].startswith("^h"):
                        print(f"Player {off} {defender_text(defender_list,off)} brings the ball over halfcourt")
                        last_possessor, last_defender = off, defender_list
                        k += 2
                        consumed_any = True
                        continue
                    m_p = re.match(r"^\^p(\+\+|\+|-)?", action[k:])
                    if m_p:
                        print(f"Player {off} {defender_text(defender_list,off)} posts up")
                        last_possessor = off
                        last_defender = defender_list
                        shot = m_p.group(1)
                        k += 2 + (len(shot) if shot else 0)
                        consumed_any = True
                        if shot:
                            handle_shot_with_screen_assists(off, defender_list, shot, last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = None, [], off
                            last_screener = None
                        continue
                    m_d = re.match(r"^\^d(\+\+|\+|-)?", action[k:])
                    if m_d:
                        print(f"Player {off} {defender_text(defender_list,off)} drives inside")
                        last_possessor = off
                        last_defender = defender_list
                        shot = m_d.group(1)
                        k += 2 + (len(shot) if shot else 0)
                        consumed_any = True
                        if shot:
                            handle_shot_with_screen_assists(off, defender_list, shot, last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = None, [], off
                            last_screener = None
                        continue
                    m_s = re.match(r"^(\+\+|\+|-)", action[k:])
                    if m_s:
                        code = m_s.group(1)
                        handle_shot_with_screen_assists(off, defender_list, code, last_passer, last_screener)
                        last_possessor, last_defender, last_shooter = None, [], off
                        last_screener = None
                        k += len(code)
                        consumed_any = True
                        continue
                    break  # unknown token; stop sequential parsing

                if not consumed_any:
                    if action.startswith("^d"):
                        print(f"Player {off} {defender_text(defender_list,off)} drives inside")
                        last_possessor = off
                        last_defender = defender_list
                        if action.endswith("++"):
                            handle_shot_with_screen_assists(off, defender_list, "++", last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = None, [], off
                            last_screener = None
                        elif action.endswith("+"):
                            handle_shot_with_screen_assists(off, defender_list, "+", last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = None, [], off
                            last_screener = None
                        elif action.endswith("-"):
                            handle_shot_with_screen_assists(off, defender_list, "-", last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = off, defender_list, off
                            last_screener = None
                    elif action.startswith("^h"):
                        print(f"Player {off} {defender_text(defender_list,off)} brings the ball over halfcourt")
                        last_possessor, last_defender = off, defender_list
                    elif action.startswith("^p"):
                        print(f"Player {off} {defender_text(defender_list,off)} posts up")
                        last_possessor = off
                        last_defender = defender_list
                        if action.endswith("++"):
                            handle_shot_with_screen_assists(off, defender_list, "++", last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = None, [], off
                            last_screener = None
                        elif action.endswith("+"):
                            handle_shot_with_screen_assists(off, defender_list, "+", last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = None, [], off
                            last_screener = None
                        elif action.endswith("-"):
                            handle_shot_with_screen_assists(off, defender_list, "-", last_passer, last_screener)
                            last_possessor, last_defender, last_shooter = off, defender_list, off
                            last_screener = None
                    elif action in ["+","-","++"]:
                        handle_shot_with_screen_assists(off, defender_list, action, last_passer, last_screener)
                        last_possessor, last_defender, last_shooter = None, [], off
                        last_screener = None
                    else:
                        last_possessor, last_defender = off, defender_list

                i += 1
                continue

            # 6) Default: standalone player token (possession change)
            last_possessor = token
            last_defender = []
            i += 1



if __name__ == "__main__":
    main()
