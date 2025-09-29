# pscoding15
# Mark 15 - adds elevator screen

#hello

import re

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

                # Parse the stagger/iverson/elevator format properly
                screener_list, screendef_list, after_list = parse_stagger_screens(rest)

                return ball_handler, ball_def, action_type, screener_list, screendef_list, after_list

            # Non-stagger/non-Iverson/non-elevator single off-ball screens
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
        # FIXED: rj should correctly assign ball_handler as rejecting player and ball_def
        if action_type == "rj":
            if "/" in token:
                ball_part, rest = token.split("rj", 1)
                if "/" in ball_part:
                    ball_handler, ball_def_raw = ball_part.split("/", 1)
                    ball_def = ball_def_raw.split(",") if ball_def_raw else []
                else:
                    ball_handler = ball_part
                    ball_def = []
                if "/" in rest:
                    screener, screener_def_raw = rest.split("/", 1)
                    screener = re.sub(r"(ch|ct|sw|bz|tl|cs|\^d|dho|ho|\+{1,2}|-)+$", "", screener)
                    screener_def = [re.sub(r"(ch|ct|sw|bz|tl|cs|\^d|dho|ho|\+{1,2}|-)+$", "", screener_def_raw)] if screener_def_raw else []
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
                trailing_codes = [m.group(0) for m in re.finditer(r"(ch|ct|sw|bz|tl|cs|\^d|dho|ho|\+{1,2}|-)", rest.strip())]
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
    Stagger: "5/6sw7/8ch" (sequential, sw means defender switches)
    Elevator: "5/6,7/8sw6" (comma-separated, sw6 means player 6 switches)
    Returns (screener_list, screendef_list, action_list)
    """
    s = s.replace(" ", "")
    screener_list = []
    screendef_list = []
    action_list = []
    
    # Check if this is elevator format (has commas) or stagger format (sequential)
    if ',' in s:
        # Elevator format: 5/6,7/8sw6
        i = 0
        while i < len(s):
            # Parse screener number
            screener_match = re.match(r"(\d+)", s[i:])
            if not screener_match:
                break
            
            screener = screener_match.group(1)
            i += len(screener)
            
            # Parse optional defender after slash
            defender = None
            if i < len(s) and s[i] == '/':
                i += 1  # skip slash
                defender_match = re.match(r"(\d+)", s[i:])
                if defender_match:
                    defender = defender_match.group(1)
                    i += len(defender)
            
            # Store this screen info
            screener_list.append(screener)
            screendef_list.append([defender] if defender else [])
            
            # Check for comma (move to next screen) or action code
            if i < len(s) and s[i] == ',':
                i += 1  # skip comma
                action_list.append("")  # No action for this screen yet
            else:
                # Parse action code (should be at end for elevator)
                action = ""
                action_match = re.match(r"(ch|ct|sw\d*|bz\d*|tl|cs|\^d|dho|ho|\+{1,2}|-)", s[i:])
                if action_match:
                    action = action_match.group(1)
                    i += len(action)
                action_list.append(action)
    else:
        # Stagger format: 5/6sw7/8ch (sequential)
        i = 0
        while i < len(s):
            # Parse screener number
            screener_match = re.match(r"(\d+)", s[i:])
            if not screener_match:
                break
            
            screener = screener_match.group(1)
            i += len(screener)
            
            # Parse optional defender after slash
            defender = None
            if i < len(s) and s[i] == '/':
                i += 1  # skip slash
                defender_match = re.match(r"(\d+)", s[i:])
                if defender_match:
                    defender = defender_match.group(1)
                    i += len(defender)
            
            # Parse optional action code (for stagger, no numbers after sw)
            action = ""
            action_match = re.match(r"(ch|ct|sw|bz|tl|cs|\^d|dho|ho|\+{1,2}|-)", s[i:])
            if action_match:
                action = action_match.group(1)
                i += len(action)
            
            # Store this screen info
            screener_list.append(screener)
            screendef_list.append([defender] if defender else [])
            action_list.append(action)
    
    return screener_list, screendef_list, action_list

# ---------------- Helper Functions ----------------

def defender_text(def_player, off_player=None):
    if not def_player:
        return "wide open"
    if isinstance(def_player, str):
        def_player = [def_player]
    parts = []
    for d in def_player:
        if not d:
            continue
        if str(d).startswith("rot"):
            if off_player:
                parts.append(f"Player {d[3:]} rotating over to guard Player {off_player}")
            else:
                parts.append(f"Player {d[3:]} rotating over")
        else:
            parts.append(f"Player {d}")
    if len(parts) == 0:
        return "wide open"
    elif len(parts) == 1:
        return f"guarded by {parts[0]}"
    else:
        return f"guarded by {', '.join(parts)}"

def format_player_list(players):
    if not players:
        return "Unknown"
    if isinstance(players, str):
        return f"Player {players}"
    players = [str(p) for p in players]
    if len(players) == 1:
        return f"Player {players[0]}"
    elif len(players) == 2:
        return f"Players {players[0]} and {players[1]}"
    else:
        return "Players " + ", ".join(players[:-1]) + f", and {players[-1]}"

def print_pass(from_player, from_def, to_player, to_def):
    print(f"Player {from_player} {defender_text(from_def, from_player)} passes to Player {to_player} {defender_text(to_def, to_player)}")
    return from_player

def handle_shot(player, def_list, action, last_passer=None, last_screener=None, pnr_shot=False):
    award_screen = pnr_shot and last_screener
    if action == "++":
        print(f"Player {player} {defender_text(def_list, player)} makes the shot (assisted by Player {last_passer})!")
        if award_screen:
            print(f"Player {last_screener} gets a screen assist!")
    elif action == "+":
        print(f"Player {player} {defender_text(def_list, player)} makes the shot!")
        if award_screen:
            print(f"Player {last_screener} gets a screen assist!")
    elif action == "-":
        print(f"Player {player} {defender_text(def_list, player)} misses the shot!")

# ---------------- Defensive / Rebound Functions ----------------

def process_defensive_tokens(token, last_player, last_defender, last_shooter, off_player=None, def_list=None):
    base_token = token
    if "/" in token:
        parts = token.split("/")
        off_player = parts[0]
        base_token = "/".join(parts[1:])
    if "^or" in base_token:
        player = off_player or last_shooter or last_player
        rebound_def_list = def_list
        if "/" in token:
            player_part, def_part = token.split("/", 1)
            player = player_part
            digits = re.match(r"(\d+(?:,\d+)*)", def_part)
            if digits:
                rebound_def_list = digits.group(1).split(",")
        putback_action = None
        if base_token.endswith("+"): putback_action = "+"
        elif base_token.endswith("-"): putback_action = "-"
        print(f"Player {player} {defender_text(rebound_def_list, player)} grabs the offensive rebound")
        if putback_action:
            handle_shot(player, rebound_def_list, putback_action)
            last_shooter = player
            return None, None, last_shooter, 1
        return player, rebound_def_list, None, 1
    if "^r" in base_token:
        player = off_player or last_shooter or last_player
        print(f"Player {player} {defender_text(def_list, player)} grabs the defensive rebound")
        print("Possession switches to the other team.")
        return None, None, None, 1
    if "^dbto" in base_token:
        player = off_player or last_player
        print(f"Player {player} commits a dead ball turnover")
        print("Possession switches to the other team.")
        return None, None, None, 1
    if "^lbto" in base_token:
        player = off_player or last_player
        print(f"Player {player} commits a live ball turnover")
        return player, None, None, 1
    if "^stl" in base_token:
        stealer = base_token[4:]
        print(f"Player {stealer} steals the ball!")
        return stealer, None, None, 1
    if "^blk" in base_token:
        blocker = base_token[4:]
        if last_player:
            print(f"Player {blocker} blocks the shot from Player {last_player}")
        else:
            print(f"Player {blocker} blocks the shot")
        return None, None, None, 1
    if "^def" in base_token:
        deflector = base_token[4:]
        print(f"Player {deflector} deflects the ball from Player {last_player if last_player else 'Unknown'}")
        return None, None, None, 1
    if "^jump" in base_token:
        try:
            players = base_token[5:].split(",")
            if len(players) == 2:
                print(f"Jump ball between Player {players[0]} and Player {players[1]}")
            else:
                print("Unrecognized jump ball format")
        except:
            print("Error parsing jump ball event")
        return None, None, None, 1
    if "^f" in base_token and not "^of" in base_token:
        fouler = base_token[2:]
        print(f"Player {fouler} commits a defensive foul")
        return None, None, None, 1
    if "^of" in base_token:
        fouler = base_token[3:]
        print(f"Player {fouler} commits an offensive foul")
        print("Possession switches to the other team.")
        return None, None, None, 1
    if base_token in ["^rob", "^dob", "^oob"]:
        msgs = {"^rob": "Rebound goes out of bounds", "^dob": "Ball deflected out of bounds", "^oob": "Ball goes out of bounds"}
        print(msgs[base_token])
        return None, None, None, 1
    return last_player, last_defender, last_shooter, 0

# ---------------- Off-Ball Screen Helper ----------------

def print_offball_screen(action_type, ball, balldef, screener, screendef, after):
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

    # Stagger / Iverson / Elevator screens
    if action_type in ("stg","ivs","elv"):
        # Build screen description first
        screen_parts = []
        for idx, (s, d) in enumerate(zip(screener, screendef)):
            if s:
                if d:
                    screen_parts.append(f"Player {s} {defender_text(d, s)}")
                else:
                    screen_parts.append(f"Player {s}")
        
        if screen_parts:
            joined = " and ".join(screen_parts)
            print(f"Player {ball_txt} {defender_text(balldef_txt, ball_txt)} comes off {label} from {joined}")

        # Process action codes for each screen
        for idx, (s, d, a) in enumerate(zip(screener, screendef, after)):
            if not a:  # Skip if no action code
                continue
                
            screen_num = "first" if idx == 0 else "second" if idx == 1 else f"{idx+1}th"
            
            if action_type == "elv":
                # Elevator screen actions
                if a.startswith("sw"):
                    # For elevator, sw can specify player (sw6) or be generic (sw)
                    switch_actor_match = re.match(r"sw(\d+)", a)
                    if switch_actor_match:
                        switch_actor = switch_actor_match.group(1)
                        print(f"Player {switch_actor} switches onto {ball_txt} on the elevator screen")
                        balldef_txt = [switch_actor]
                    else:
                        # Generic sw for elevator - shouldn't happen but handle gracefully
                        print(f"Player Unknown switches onto {ball_txt} on the elevator screen")
                        balldef_txt = []
                elif a in ("+","++","-"):
                    handle_shot(ball_txt, balldef_txt, a)
                elif a == "ch":
                    print(f"Player {format_player_list(balldef_txt)} chases through the elevator screen")
                elif a == "ct":
                    print(f"Player {format_player_list(balldef_txt)} cuts around the elevator screen")
                elif a.startswith("bz"):
                    # For elevator, bz can specify player (bz6) or be generic (bz)
                    blitz_actor_match = re.match(r"bz(\d+)", a)
                    if blitz_actor_match:
                        blitz_actor = blitz_actor_match.group(1)
                        if balldef_txt and balldef_txt[0] != blitz_actor:
                            print(f"Players {blitz_actor} and {format_player_list(balldef_txt)} double team {ball_txt} on the elevator screen")
                            balldef_txt = list(set((balldef_txt or []) + [blitz_actor]))
                        else:
                            print(f"Player {blitz_actor} blitzes {ball_txt} on the elevator screen")
                            if blitz_actor not in (balldef_txt or []):
                                balldef_txt = (balldef_txt or []) + [blitz_actor]
                    else:
                        # Generic bz for elevator
                        blitz_actor_list = d if d else balldef_txt
                        blitz_actor_text = format_player_list(blitz_actor_list) if blitz_actor_list else "Unknown"
                        print(f"Player {blitz_actor_text} blitzes {ball_txt} on the elevator screen")
                        balldef_txt = list(set((balldef_txt or []) + (blitz_actor_list or [])))
                elif a == "tl":
                    lock_actor_list = d if d else balldef_txt
                    lock_actor_text = format_player_list(lock_actor_list) if lock_actor_list else "Unknown"
                    print(f"Player {lock_actor_text} top locks {ball_txt}, forcing them away from the elevator screen")
                elif a == "cs":
                    print(f"Player {ball_txt} is caught on the elevator screen")
                elif a in ("^d","dho","ho"):
                    act_word = "drives" if a=="^d" else "hands off"
                    print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} {act_word} immediately after the elevator screen")
            
            else:
                # Stagger/Iverson screen actions
                if a == "ch":
                    print(f"Player {format_player_list(balldef_txt)} chases over the {screen_num} screen")
                elif a == "ct":
                    print(f"Player {format_player_list(balldef_txt)} cuts under the {screen_num} screen")
                elif a.startswith("sw"):
                    # For stagger/iverson, sw with number means specific player switches
                    # sw without number means the defender of THIS screen switches
                    switch_actor_match = re.match(r"sw(\d+)", a)
                    if switch_actor_match:
                        switch_actor = switch_actor_match.group(1)
                        print(f"Player {switch_actor} switches onto {ball_txt} on the {screen_num} screen")
                        balldef_txt = [switch_actor]
                    else:
                        # FIXED: Use the defender from THIS specific screen (d)
                        if d and d[0]:  # d is a list, get first defender
                            switch_actor = d[0]
                            print(f"Player {switch_actor} switches onto {ball_txt} on the {screen_num} screen")
                            balldef_txt = [switch_actor]
                        else:
                            print(f"Player Unknown switches onto {ball_txt} on the {screen_num} screen")
                            balldef_txt = []
                elif a == "bz":
                    # For stagger/iverson, blitz means current ball defender + this screen's defender double team
                    if d and d[0]:  # d is the defender from this specific screen
                        screen_defender = d[0]
                        if balldef_txt and balldef_txt[0] != screen_defender:
                            print(f"Players {format_player_list(balldef_txt)} and {screen_defender} double team {ball_txt} on the {screen_num} screen")
                            # Update balldef_txt to include both defenders
                            balldef_txt = list(set(balldef_txt + [screen_defender]))
                        else:
                            print(f"Player {screen_defender} blitzes {ball_txt} on the {screen_num} screen")
                            if screen_defender not in (balldef_txt or []):
                                balldef_txt = (balldef_txt or []) + [screen_defender]
                    else:
                        blitz_actor_text = format_player_list(balldef_txt) if balldef_txt else "Unknown"
                        print(f"Player {blitz_actor_text} blitzes {ball_txt} on the {screen_num} screen")
                elif a == "tl":
                    lock_actor_text = format_player_list(balldef_txt) if balldef_txt else "Unknown"
                    print(f"Player {lock_actor_text} top locks {ball_txt}, forcing them away from the {screen_num} screen")
                elif a == "cs":
                    print(f"Player {ball_txt} is caught on the {screen_num} screen")
                elif a in ("^d","dho","ho","+","++","-"):
                    if a in ("+","++","-"):
                        handle_shot(ball_txt, balldef_txt, a)
                    else:
                        act_word = "drives" if a=="^d" else "hands off"
                        print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} {act_word} immediately after the {screen_num} screen")

        return balldef_txt

    # Single off-ball screens (non-stagger)
    screener_txt = screener if screener else "Unknown"
    screendef_txt = screendef
    print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} comes off a {label} from Player {screener_txt} {defender_text(screendef_txt,screener_txt)}")
    if after:
        for match in re.finditer(r"(ch|ct|sw\d*|bz|tl|cs|\^d|dho|ho|\+{1,2}|-)", after):
            code = match.group(0)
            if code == "ch":
                print(f"Player {format_player_list(balldef_txt)} chases over the {label}")
            elif code == "ct":
                print(f"Player {format_player_list(balldef_txt)} cuts under the {label}")
            elif code.startswith("sw"):
                switch_actor_match = re.match(r"sw(\d+)", code)
                if switch_actor_match:
                    switch_actor = switch_actor_match.group(1)
                    print(f"Player {switch_actor} switches onto {ball_txt}")
                    balldef_txt = [switch_actor]
                else:
                    print(f"Player {format_player_list(screendef_txt)} switches onto {ball_txt}")
                    balldef_txt = screendef_txt
            elif code == "bz":
                if screendef_txt and balldef_txt:
                    print(f"Players {format_player_list(balldef_txt)} and {format_player_list(screendef_txt)} double team {ball_txt}")
                    balldef_txt = list(set((balldef_txt or []) + (screendef_txt or [])))
                elif screendef_txt:
                    print(f"Players {format_player_list(screendef_txt)} double team {ball_txt}")
                    balldef_txt = screendef_txt
            elif code == "tl":
                lock_actor = format_player_list(screendef_txt) if screendef_txt else format_player_list(balldef_txt)
                print(f"Player {lock_actor} top locks {ball_txt}, forcing them away from the {label}")
            elif code == "cs":
                print(f"Player {ball_txt} is caught on the {label}")
            elif code in ("^d","dho","ho","+","++","-"):
                if code in ("+","++","-"):
                    handle_shot(ball_txt, balldef_txt, code)
                else:
                    act_word = "drives" if code=="^d" else "hands off"
                    print(f"Player {ball_txt} {defender_text(balldef_txt,ball_txt)} {act_word} immediately after the {label}")

    return balldef_txt

# ---------------- Main Loop ----------------

def main():
    """
    Main loop for basketball stat tracker.
    """
    while True:
        try:
            line = input("Enter possession (or 'q' to quit): ").strip()
        except EOFError:
            break
        if line.lower() == "q":
            break
        if not line:
            continue

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

            # First, handle defensive/rebound special tokens that should short-circuit
            last_possessor, last_defender, last_shooter, skip = process_defensive_tokens(
                token, last_possessor, last_defender, last_shooter
            )
            if skip:
                # Defensive token consumed; reset screener context since possession events changed
                last_screener = None
                i += 1
                continue

            # Help (hp) or switch (sw) tokens — may be combined with numbers and trailing shot codes
            if token.startswith(("hp", "sw")):
                match = re.match(r"(hp|sw)(\d+.*)", token)
                if match:
                    head, tail = match.group(1), match.group(2)
                    # tail might be e.g. "6++" or "6,7+"
                    m2 = re.match(r"(\d+(?:,\d+)*)(\+\+|\+|-)?$", tail)
                    if not m2:
                        # separate core digits and leave trailing codes as separate token(s)
                        digits = re.match(r"(\d+(?:,\d+)*)", tail)
                        if digits:
                            core = digits.group(1)
                            rest = tail[len(core):]
                            token = head + core
                            if rest:
                                # insert trailing codes back into parts after current token
                                parts.insert(i + 1, rest)

                # if we don't have a last_possessor, try to infer from previous token
                if not last_possessor and i > 0:
                    prev = parts[i - 1].split("/")[0]
                    last_possessor = prev
                    last_defender = parts[i - 1].split("/")[1].split(",") if "/" in parts[i - 1] else []

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
                    for x in action_list:
                        if x not in last_defender:
                            last_defender.append(x)
                    print(
                        f"Player {action_list[0]} steps in to help on Player {last_possessor}"
                        if len(action_list) == 1
                        else f"Players {', '.join(action_list)} step in to help on Player {last_possessor}"
                    )
                else:
                    last_defender = action_list
                    print(
                        f"Player {action_list[0]} switches onto Player {last_possessor}"
                        if len(action_list) == 1
                        else f"Players {', '.join(action_list)} switch onto Player {last_possessor}"
                    )

                if shoot_after:
                    shooter = last_possessor
                    if shoot_after == "assist":
                        handle_shot(shooter, last_defender, "++", last_passer, last_screener, pnr_shot=False)
                    else:
                        handle_shot(shooter, last_defender, "+" if shoot_after == "make" else "-")
                        last_shooter = shooter
                    # after shot, possession ends locally
                    last_possessor = None
                    last_defender = []
                    last_screener = None

                i += 1
                continue

            # Off-ball screens in parentheses
            if token.startswith("(") and token.endswith(")"):
                ball, balldef, action_type, screener, screendef, after = parse_player_def(token)
                balldef = balldef if isinstance(balldef, list) else [balldef] if balldef else []
                screendef = screendef if isinstance(screendef, list) else [screendef] if screendef else []

                if action_type in ("pn","fl","bk","awy","crs","wdg","rip","ucla","stg","ivs","elv"):
                    # Capture the updated defender information from the screen action
                    updated_balldef = print_offball_screen(action_type, ball, balldef, screener, screendef, after)
                    # Update tracking if the ball handler matches our current possessor
                    if ball == last_possessor:
                        last_defender = updated_balldef if isinstance(updated_balldef, list) else [updated_balldef] if updated_balldef else []

                i += 1
                continue

            # On-ball screen / on-ball actions
            if any(k in token for k in ("pnr","pnp","slp","gst","rj","dho","ho")):
                ball, balldef, action_type, screener, screendef, after = parse_player_def(token)
                balldef = balldef if isinstance(balldef, list) else [balldef] if balldef else []
                screendef = screendef if isinstance(screendef, list) else [screendef] if screendef else []

                last_possessor = ball
                last_defender = balldef
                last_screener = screener if action_type in ("pnr","pnp") else None
                last_action_type = action_type

                label_map = {
                    "pnr": "pick and roll",
                    "pnp": "pick and pop",
                    "slp": "slips the screen for",
                    "gst": "ghosts the screen for",
                    "rj": "rejects the ball screen from",
                    "dho": "dribbles and hands off to",
                    "ho": "hands off to"
                }
                label = label_map[action_type]

                if action_type in ("pnr","pnp","rj"):
                    print(f"Player {ball} {defender_text(balldef,ball)} {label} Player {screener} {defender_text(screendef,screener)}")
                else:
                    print(f"Player {screener} {defender_text(screendef,screener)} {label} Player {ball} {defender_text(balldef,ball)}")

                trailing_actions = "".join(after) if isinstance(after,(list,tuple)) else after if after else ""
                active_def = balldef
                while trailing_actions:
                    m = re.match(r"(ch|ct|sw\d*|bz|tl|cs|ice|\^d|dho|ho|\+{1,2}|-)", trailing_actions)
                    if not m:
                        break
                    code = m.group(0)
                    trailing_actions = trailing_actions[len(code):]

                    action_word = "screen" if last_action_type in ("pnr","pnp","slp","gst","rj") else "handoff"

                    if code == "ch":
                        print(f"Player {format_player_list(active_def)} chases over the {action_word}")
                    elif code == "ct":
                        print(f"Player {format_player_list(active_def)} cuts under the {action_word}")
                    elif code.startswith("sw"):
                        switch_actor_match = re.match(r"sw(\d+)", code)
                        if switch_actor_match:
                            switch_actor = switch_actor_match.group(1)
                            print(f"Player {switch_actor} switches onto {ball} on the {action_word}")
                            active_def = [switch_actor]
                        else:
                            switch_actor = format_player_list(screendef) if screendef else "Unknown"
                            print(f"Player {switch_actor} switches onto {ball} on the {action_word}")
                            active_def = screendef
                    elif code == "bz":
                        if screendef and balldef:
                            print(f"Players {format_player_list(balldef)} and {format_player_list(screendef)} double team {ball}")
                            active_def = list(set((balldef or []) + (screendef or [])))
                        elif screendef:
                            print(f"Players {format_player_list(screendef)} double team {ball}")
                            active_def = screendef
                    elif code == "ice":
                        print(f"Player {format_player_list(balldef)} ices the {action_word} and forces {ball} away from it")
                    elif code == "cs":
                        print(f"Player {format_player_list(balldef)} gets caught on the {action_word}")
                    elif code in ("^d","dho","ho"):
                        act_word = "drives" if code=="^d" else "hands off"
                        print(f"Player {ball} {defender_text(active_def,ball)} {act_word} immediately after the {action_word}")
                        last_possessor = ball
                        last_defender = active_def
                        last_screener = None
                    elif code in ("+","++","-"):
                        handle_shot(ball, balldef, code, last_passer, last_screener, pnr_shot=(action_type in ("pnr","pnp")))
                        last_shooter = ball
                        last_possessor = None
                        last_defender = []
                        last_screener = None
                        break

                i += 1
                continue

            # Slash / normal actions with explicit defender(s)
            if "/" in token:
                off, defender, action, _, _, _ = parse_player_def(token)
                defender_list = defender if isinstance(defender, list) else [defender] if defender else []
                if last_possessor and last_possessor != off:
                    last_passer = last_possessor
                    print_pass(last_passer, last_defender, off, defender_list)

                if action.startswith("^d"):
                    print(f"Player {off} {defender_text(defender_list,off)} drives toward the basket")
                    last_possessor = off
                    last_defender = defender_list
                    if action.endswith("+"):
                        handle_shot(off, defender_list, "+", last_passer, last_screener)
                        last_possessor, last_defender, last_shooter = None, [], off
                        last_screener = None
                    elif action.endswith("-"):
                        handle_shot(off, defender_list, "-", last_passer, last_screener)
                        last_possessor, last_defender, last_shooter = off, defender_list, off
                        last_screener = None
                elif action.startswith("^h"):
                    print(f"Player {off} {defender_text(defender_list,off)} dribbles")
                    last_possessor, last_defender = off, defender_list
                elif action in ["+","-","++"]:
                    handle_shot(off, defender_list, action, last_passer, last_screener)
                    last_possessor, last_defender, last_shooter = None, [], off
                    last_screener = None
                else:
                    last_possessor, last_defender = off, defender_list

                i += 1
                continue

            # Default case: standalone player token (possession change)
            last_possessor = token
            last_defender = []
            i += 1



if __name__ == "__main__":
    main()