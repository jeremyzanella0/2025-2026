# pscoding7.py
# Mark 7 — basketball stat tracker.
# Builds on Mark 6 features, including:
# - Defense: help defense, switches, steals, blocks, deflections, rebounds (offensive and defensive)
# - Proper tracking of defenders for all offensive actions
# This version fully integrates defensive capabilities into all relevant plays.


def parse_player_def(token: str):
    """
    Splits a token like '2/7^d+' or '1/rot2+' into (off_player, def_player, action).
    - If '/' not present, returns (token, None, "").
    - Handles 'rot' defenders as well as digits.
    """
    if "/" not in token:
        return token, None, ""
    off, rest = token.split("/", 1)
    i = 0
    if rest.startswith("rot"):
        j = 3
        while j < len(rest) and rest[j].isdigit():
            j += 1
        def_player = rest[:j]
        action = rest[j:] if j < len(rest) else ""
    else:
        while i < len(rest) and rest[i].isdigit():
            i += 1
        def_player = rest[:i] if i > 0 else None
        action = rest[i:] if i < len(rest) else ""
    return off, def_player, action

def parse_actions(action_str):
    """Split combined actions like '^h^d+' into individual actions with their own + / -."""
    actions = []
    i = 0
    while i < len(action_str):
        if action_str[i] == "^":
            j = i + 1
            while j < len(action_str) and action_str[j] != "^":  # <-- use action_str
                j += 1
            actions.append(action_str[i:j])
            i = j
        else:
            actions.append(action_str[i])
            i += 1
    return actions


def defender_text(def_player, off_player=None):
    """Returns a readable string describing the defender(s)."""
    if not def_player:
        return "wide open"
    if isinstance(def_player, list):
        parts = []
        for d in def_player:
            if not d:
                continue
            if str(d).startswith("rot"):
                parts.append(f"Player {d[3:]} rotating over to guard Player {off_player}")
            else:
                parts.append(f"Player {d}")
        if len(parts) == 0:
            return "wide open"
        elif len(parts) == 1:
            return f"guarded by {parts[0]}"
        else:
            return f"guarded by {', '.join(parts)}"
    if isinstance(def_player, str):
        if def_player.startswith("rot"):
            return f"Player {def_player[3:]} rotating over to guard Player {off_player}"
        else:
            return f"guarded by Player {def_player}"

def print_pass(from_player, from_def, to_player, to_def):
    print(f"Player {from_player} {defender_text(from_def, from_player)} passes to Player {to_player} {defender_text(to_def, to_player)}")
    return from_player  # track last passer

def handle_free_throws(token, default_shooter=None):
    """Parses free throw strings like '*', 'X', '**', '3*X', etc."""
    shooter = default_shooter or "Unknown"
    for char in token:
        if char == "*":
            print(f"Player {shooter} makes the free throw")
        elif char.upper() == "X":
            print(f"Player {shooter} misses the free throw")
        elif char == "+":
            print(f"Player {shooter} makes the free throw!")
        elif char == "-":
            print(f"Player {shooter} misses the free throw!")

def main():
    while True:
        line = input("Enter possession (or 'q' to quit): ").strip()
        if line.lower() == "q":
            print("Exiting program...")
            break

        parts = line.split()
        i = 0
        last_player = None
        last_defender = None
        last_shooter = None
        last_passer = None

        while i < len(parts):
            token = parts[i]

            # --- HELP / SWITCH ---
            if token.startswith(("hp", "sw")):
                if not last_player and i > 0:
                    prev = parts[i-1].split("/")[0]
                    last_player = prev
                    last_defender = [parts[i-1].split("/")[1]] if "/" in parts[i-1] else []

                is_help = token.startswith("hp")
                action_raw = token[2:]

                # Determine trailing shot/assist symbols
                shoot_after = None
                if action_raw.endswith("++"):
                    shoot_after = "assist"
                    action_raw = action_raw[:-2]
                elif action_raw.endswith("+"):
                    shoot_after = "make"
                    action_raw = action_raw[:-1]
                elif action_raw.endswith("-"):
                    shoot_after = "miss"
                    action_raw = action_raw[:-1]

                action_list = [x.strip() for x in action_raw.split(",") if x.strip()]
                if not isinstance(last_defender, list):
                    last_defender = [last_defender] if last_defender else []

                if is_help:
                    # Help: just append helping defenders
                    for x in action_list:
                        if x not in last_defender:
                            last_defender.append(x)
                    if len(action_list) == 1:
                        print(f"Player {action_list[0]} steps in to help on Player {last_player}")
                    else:
                        print(f"Players {', '.join(action_list)} step in to help on Player {last_player}")
                else:
                    # Switch: replace last defender with new defender(s)
                    old_defenders = last_defender.copy()
                    last_defender = action_list
                    if len(action_list) == 1:
                        print(f"Player {action_list[0]} switches onto Player {last_player}")
                    else:
                        print(f"Players {', '.join(action_list)} switch onto Player {last_player}")
                    # Optional: could track old defenders picking up others (future improvement)

                # Handle assisted/made/missed shots
                if shoot_after:
                    shooter = last_player
                    if shoot_after == "assist":
                        print(f"Player {shooter} {defender_text(last_defender, shooter)} makes the shot (assisted by Player {last_passer})!")
                        last_player, last_defender, last_shooter = None, None, shooter
                    else:
                        action_text = "makes it" if shoot_after == "make" else "misses it"
                        print(f"Player {shooter} {defender_text(last_defender, shooter)} shoots and {action_text}.")
                        if shoot_after == "make":
                            last_player, last_defender = None, None
                        last_shooter = shooter
                i += 1
                continue

            # --- Slash actions and passes ---
            if "/" in token:
                off, defender, action = parse_player_def(token)
                defender_list = [defender] if defender and not isinstance(defender, list) else defender

                # Assist shot
                if action == "++":
                    if last_player:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    print(f"Player {off} {defender_text(defender_list, off)} makes the shot (assisted by Player {last_passer})!")
                    last_player, last_defender, last_shooter = None, None, off
                    i += 1
                    continue

                # Inbound
                if action == "^ob":
                    print(f"Player {off} {defender_text(defender_list, off)} inbounds the ball")
                    last_player, last_defender, last_shooter = off, defender_list, None
                    i += 1
                    continue

                # Dribble across halfcourt
                if action.startswith("^h"):
                    print(f"Player {off} {defender_text(defender_list, off)} dribbles across halfcourt")
                    last_player, last_defender, last_shooter = off, defender_list, None
                    i += 1
                    continue

                # Drive
                if action.startswith("^d"):
                    if last_player and last_player != off:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    print(f"Player {off} {defender_text(defender_list, off)} drives toward the basket")
                    last_player, last_defender = off, defender_list
                    if action.endswith("+"):
                        print(f"Player {off} {defender_text(defender_list, off)} makes the shot!")
                        last_player, last_defender, last_shooter = None, None, off
                    elif action.endswith("-"):
                        print(f"Player {off} {defender_text(defender_list, off)} misses the shot!")
                        last_shooter = off
                    i += 1
                    continue

                # Post up
                if action.startswith("^p"):
                    if last_player and last_player != off:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    print(f"Player {off} {defender_text(defender_list, off)} posts up")
                    last_player, last_defender = off, defender_list
                    if action.endswith("+"):
                        print(f"Player {off} {defender_text(defender_list, off)} makes the shot!")
                        last_player, last_defender, last_shooter = None, None, off
                    elif action.endswith("-"):
                        print(f"Player {off} {defender_text(defender_list, off)} misses the shot!")
                        last_shooter = off
                    i += 1
                    continue

                # Plain make/miss
                if action in ["+", "-"]:
                    if last_player and last_player != off:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    if action == "+":
                        print(f"Player {off} {defender_text(defender_list, off)} makes the shot!")
                        last_player, last_defender, last_shooter = None, None, off
                    else:
                        print(f"Player {off} {defender_text(defender_list, off)} misses the shot!")
                        last_player, last_defender, last_shooter = off, defender_list, off
                    i += 1
                    continue

                # Simple pass
                if action == "":
                    if last_player:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    last_player, last_defender = off, defender_list
                    i += 1
                    continue

                print(f"Unrecognized action for Player {off} with action '{action}'")
                last_player, last_defender, last_shooter = off, defender_list, None
                i += 1
                continue

            # --- Numeric pass ---
            if token.isdigit():
                if last_player:
                    last_passer = print_pass(last_player, last_defender, token, None)
                last_player, last_defender, last_shooter = token, None, None
                i += 1
                continue

            # --- Free throws and other legacy tokens ---
            if ("*" in token) or ("X" in token):
                handle_free_throws(token, default_shooter=last_shooter or last_player)
                last_player, last_defender, last_shooter = None, None, None
                i += 1
                continue

            # --- Unrecognized token ---
            print(f"Unrecognized token: {token}")
            last_player, last_defender, last_shooter = token, None, None
            i += 1





            # --- Defensive-only or legacy tokens (no '/') ---

            # Defensive rebound
            if token.endswith("^r"):
                player = token[:-2]
                print(f"Player {player} grabs the defensive rebound")
                print("Possession switches to the other team.")
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            # Offensive rebound legacy (without slash) — keep backward compatibility
            if token.endswith("^or+") or token.endswith("^or-") or token.endswith("^or"):
                base = token[:-3] if token.endswith("^or") else token[:-4]
                action = token[-3:] if token.endswith("^or") else token[-4:]
                # Treat as no defender info
                if action == "^or":
                    print(f"Player {base} grabs the offensive rebound")
                    last_player, last_defender = base, None
                    last_shooter = None
                elif action == "^or+":
                    print(f"Player {base} grabs the offensive rebound")
                    print(f"Player {base} makes the shot!")
                    last_player, last_defender = None, None
                    last_shooter = base
                elif action == "^or-":
                    print(f"Player {base} grabs the offensive rebound")
                    print(f"Player {base} misses the shot!")
                    last_player, last_defender = base, None
                    last_shooter = base
                i += 1
                continue

            # Dead ball turnover
            if token.endswith("^dbto"):
                player = token[:-5]
                if last_player:
                    print(f"Player {last_player} {defender_text(last_defender)} passes to Player {player}")
                print(f"Player {player} commits a dead ball turnover")
                print("Possession switches to the other team.")
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            # Live ball turnover with optional steal
            if token.endswith("^lbto"):
                player = token[:-5]
                if last_player:
                    print(f"Player {last_player} {defender_text(last_defender)} passes to Player {player}")
                print(f"Player {player} commits a live ball turnover")
                if i + 1 < len(parts) and parts[i + 1].startswith("^stl"):
                    stealer = parts[i + 1][4:]
                    print(f"Player {stealer} steals the ball!")
                    last_player, last_defender = stealer, None
                    last_shooter = None
                    i += 2
                else:
                    print("Possession switches to the other team.")
                    last_player, last_defender = None, None
                    last_shooter = None
                    i += 1
                continue

            # Steal token alone
            if token.startswith("^stl"):
                stealer = token[4:]
                print(f"Player {stealer} steals the ball!")
                last_player, last_defender = stealer, None
                last_shooter = None
                i += 1
                continue

            # Block
            if token.startswith("^blk"):
                blocker = token[4:]
                if last_player:
                    print(f"Player {blocker} blocks the shot from Player {last_player}")
                else:
                    print(f"Player {blocker} blocks the shot")
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            # Deflection
            if token.startswith("^def"):
                deflector = token[4:]
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    next_player = parts[i + 1]
                    print(f"Player {deflector} deflects the ball from Player {last_player if last_player else 'Unknown'}")
                    print(f"Player {next_player} gets the ball")
                    last_player, last_defender = next_player, None
                    i += 2
                else:
                    print(f"Player {deflector} deflects the ball from Player {last_player if last_player else 'Unknown'}")
                    last_player, last_defender = None, None
                    i += 1
                last_shooter = None
                continue

            # Jump ball ^jumpA,B
            if token.startswith("^jump"):
                try:
                    players = token[5:].split(",")
                    if len(players) == 2:
                        p1, p2 = players
                        print(f"Jump ball between Player {p1} and Player {p2}")
                    else:
                        print("Unrecognized jump ball format")
                except Exception:
                    print("Error parsing jump ball event")
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            # Defensive foul
            if token.startswith("^f") and not token.startswith("^of"):
                fouler = token[2:]
                print(f"Player {fouler} commits a defensive foul")
                i += 1
                # Process trailing free throws if present
                while i < len(parts) and (("*" in parts[i]) or ("X" in parts[i])):
                    handle_free_throws(parts[i], default_shooter=last_shooter or last_player)
                    i += 1
                last_player, last_defender = None, None
                last_shooter = None
                continue

            # Offensive foul
            if token.startswith("^of"):
                fouler = token[3:]
                print(f"Player {fouler} commits an offensive foul")
                print("Possession switches to the other team.")
                i += 1
                # Trailing free throws (technical/off-ball) if any
                while i < len(parts) and (("*" in parts[i]) or ("X" in parts[i])):
                    handle_free_throws(parts[i], default_shooter=last_shooter or last_player)
                    i += 1
                last_player, last_defender = None, None
                last_shooter = None
                continue

            # Free throws alone
            if ("*" in token) or ("X" in token):
                handle_free_throws(token, default_shooter=last_shooter or last_player)
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            # Out of bounds variants
            if token == "^rob":
                print("Rebound goes out of bounds")
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            if token == "^dob":
                print("Ball deflected out of bounds")
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            if token == "^oob":
                print("Ball goes out of bounds")
                last_player, last_defender = None, None
                last_shooter = None
                i += 1
                continue

            # Inbounding legacy (#^ob) without defender
            if token.endswith("^ob"):
                player = token[:-3]
                print(f"Player {player} inbounds the ball")
                last_player, last_defender = player, None
                last_shooter = None
                i += 1
                continue

            # Cross halfcourt legacy (#^h) without defender
            if token.endswith("^h"):
                player = token[:-2]
                if last_player:
                    print_pass(last_player, last_defender, player, None)
                print(f"Player {player} dribbles across halfcourt")
                last_player, last_defender = player, None
                last_shooter = None
                i += 1
                continue






            # --- Rotating defender (rot#) handling ---
            if defender and defender.startswith("rot"):
                rotated_player = defender[3:]  # 'rot2' -> '2'
                if action == "+":
                    print(f"Player {off} shoots over rotating Player {rotated_player} and makes the shot!")
                elif action == "-":
                    print(f"Player {off} shoots over rotating Player {rotated_player} and misses the shot!")
                elif action == "^d":
                    print(f"Player {off} drives over rotating Player {rotated_player}")
                elif action == "^p":
                    print(f"Player {off} posts up against rotating Player {rotated_player}")
                elif action == "":  # handle plain passes too
                    if last_player:
                        print(f"Player {last_player} {defender_text([rotated_player])} passes to Player {off}")
                last_player, last_defender = off, [rotated_player]
                last_shooter = off if action in ["+", "-"] else None
                i += 1
                continue

            # --- Plain pass by number (legacy) ---
            if token.isdigit():
                if last_player:
                    print_pass(last_player, last_defender, token, None)
                last_player, last_defender = token, None
                last_shooter = None
                i += 1
                continue

            # --- Unrecognized token ---
            print(f"Unrecognized token: {token}")
            last_player = token
            last_defender = None
            last_shooter = None
            i += 1




if __name__ == "__main__":
    main()
