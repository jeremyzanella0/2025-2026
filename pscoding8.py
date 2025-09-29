# pscoding8_defender_or.py
# Mark 8 — basketball stat tracker
# Fixes:
# 1. Offensive rebound recognition with or without slash
# 2. Defender info preserved on offensive rebound and subsequent pass
# 3. Automatic pass printing after drives/posts before shots

def parse_player_def(token: str):
    if "/" not in token:
        return token, None, ""
    off, rest = token.split("/", 1)
    if rest.startswith("rot"):
        j = 3
        while j < len(rest) and rest[j].isdigit():
            j += 1
        def_player = rest[:j]
        action = rest[j:] if j < len(rest) else ""
    else:
        i = 0
        while i < len(rest) and rest[i].isdigit():
            i += 1
        def_player = rest[:i] if i > 0 else None
        action = rest[i:] if i < len(rest) else ""
    return off, def_player, action

def defender_text(def_player, off_player=None):
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
    return from_player

def handle_shot(player, def_list, action, last_passer=None):
    if action == "++":
        if last_passer:
            print_pass(last_passer, None, player, def_list)
        print(f"Player {player} {defender_text(def_list, player)} makes the shot (assisted by Player {last_passer})!")
    elif action == "+":
        print(f"Player {player} {defender_text(def_list, player)} makes the shot!")
    elif action == "-":
        print(f"Player {player} {defender_text(def_list, player)} misses the shot!")

def handle_free_throws(token, default_shooter=None):
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

def update_last(off, def_list, shooter=None):
    return off, def_list, shooter

def process_defensive_tokens(token, last_player, last_defender, last_shooter, off_player=None, def_list=None):
    """Handles all defensive/legacy tokens consistently."""
    rebound_def = def_list if def_list else None

    # Defensive rebound
    if token.endswith("^r"):
        player = off_player or token[:-2]
        print(f"Player {player} {defender_text(rebound_def, player)} grabs the defensive rebound")
        print("Possession switches to the other team.")
        return None, None, None, 1

    # Offensive rebound
    if token.startswith("^or"):
        player = off_player or token.split("^")[0]
        print(f"Player {player} {defender_text(rebound_def, player)} grabs the offensive rebound")
        # If next action indicates a shot, handle it
        if token.endswith("+"):
            handle_shot(player, rebound_def, "+")
            return None, None, player, 1
        elif token.endswith("-"):
            handle_shot(player, rebound_def, "-")
            return player, rebound_def, player, 1
        return player, rebound_def, None, 1

    # Dead ball turnover
    if token.endswith("^dbto"):
        player = off_player or token[:-5]
        if last_player:
            print(f"Player {last_player} {defender_text(last_defender)} passes to Player {player}")
        print(f"Player {player} commits a dead ball turnover")
        print("Possession switches to the other team.")
        return None, None, None, 1

    # Live ball turnover
    if token.endswith("^lbto"):
        player = off_player or token[:-5]
        if last_player:
            print(f"Player {last_player} {defender_text(last_defender)} passes to Player {player}")
        print(f"Player {player} commits a live ball turnover")
        return player, None, None, 1

    # Steal
    if token.startswith("^stl"):
        stealer = token[4:]
        print(f"Player {stealer} steals the ball!")
        return stealer, None, None, 1

    # Block
    if token.startswith("^blk"):
        blocker = token[4:]
        if last_player:
            print(f"Player {blocker} blocks the shot from Player {last_player}")
        else:
            print(f"Player {blocker} blocks the shot")
        return None, None, None, 1

    # Deflection
    if token.startswith("^def"):
        deflector = token[4:]
        print(f"Player {deflector} deflects the ball from Player {last_player if last_player else 'Unknown'}")
        return None, None, None, 1

    # Jump ball
    if token.startswith("^jump"):
        try:
            players = token[5:].split(",")
            if len(players) == 2:
                print(f"Jump ball between Player {players[0]} and Player {players[1]}")
            else:
                print("Unrecognized jump ball format")
        except:
            print("Error parsing jump ball event")
        return None, None, None, 1

    # Defensive foul
    if token.startswith("^f") and not token.startswith("^of"):
        fouler = token[2:]
        print(f"Player {fouler} commits a defensive foul")
        return None, None, None, 1

    # Offensive foul
    if token.startswith("^of"):
        fouler = token[3:]
        print(f"Player {fouler} commits an offensive foul")
        print("Possession switches to the other team.")
        return None, None, None, 1

    # Out of bounds
    if token in ["^rob", "^dob", "^oob"]:
        messages = {
            "^rob": "Rebound goes out of bounds",
            "^dob": "Ball deflected out of bounds",
            "^oob": "Ball goes out of bounds"
        }
        print(messages[token])
        return None, None, None, 1

    return last_player, last_defender, last_shooter, 0

def main():
    while True:
        line = input("Enter possession (or 'q' to quit): ").strip()
        if line.lower() == "q":
            print("Exiting program...")
            break

        parts = line.split()
        last_player = None
        last_defender = None
        last_shooter = None
        last_passer = None
        i = 0

        while i < len(parts):
            token = parts[i]

            # HELP / SWITCH
            if token.startswith(("hp", "sw")):
                if not last_player and i > 0:
                    prev = parts[i-1].split("/")[0]
                    last_player = prev
                    last_defender = [parts[i-1].split("/")[1]] if "/" in parts[i-1] else []

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
                    print(f"Player {action_list[0]} steps in to help on Player {last_player}" if len(action_list)==1 else
                          f"Players {', '.join(action_list)} step in to help on Player {last_player}")
                else:
                    last_defender = action_list
                    print(f"Player {action_list[0]} switches onto Player {last_player}" if len(action_list)==1 else
                          f"Players {', '.join(action_list)} switch onto Player {last_player}")

                if shoot_after:
                    shooter = last_player
                    if shoot_after == "assist":
                        handle_shot(shooter, last_defender, "++", last_passer)
                        last_player, last_defender, last_shooter = None, None, shooter
                    else:
                        handle_shot(shooter, last_defender, "+" if shoot_after=="make" else "-")
                        last_shooter = shooter
                        if shoot_after=="make":
                            last_player, last_defender = None, None
                i += 1
                continue

            # Slash actions
            if "/" in token:
                off, defender, action = parse_player_def(token)
                defender_list = [defender] if defender and not isinstance(defender, list) else defender

                # Handle offensive rebound attached to slash
                if action.startswith("^or"):
                    last_player, last_defender, last_shooter, _ = process_defensive_tokens(action, last_player, last_defender, last_shooter, off, defender_list)
                    i += 1
                    continue

                # Automatic pass if next token is a shot by a different player
                if action in ["+", "-", "++"] and last_player and last_player != off:
                    last_passer = print_pass(last_player, last_defender, off, defender_list)

                if action in ["+", "-", "++"]:
                    handle_shot(off, defender_list, action, last_passer)
                    last_player, last_defender, last_shooter = (None, None, off) if action in ["+", "++"] else update_last(off, defender_list, off)
                elif action.startswith("^d"):
                    if last_player and last_player != off:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    print(f"Player {off} {defender_text(defender_list, off)} drives toward the basket")
                    last_player, last_defender = off, defender_list
                    if action.endswith("+"):
                        handle_shot(off, defender_list, "+")
                        last_player, last_defender, last_shooter = None, None, off
                    elif action.endswith("-"):
                        handle_shot(off, defender_list, "-")
                        last_shooter = off
                elif action.startswith("^p"):
                    if last_player and last_player != off:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    print(f"Player {off} {defender_text(defender_list, off)} posts up")
                    last_player, last_defender = off, defender_list
                    if action.endswith("+"):
                        handle_shot(off, defender_list, "+")
                        last_player, last_defender, last_shooter = None, None, off
                    elif action.endswith("-"):
                        handle_shot(off, defender_list, "-")
                        last_shooter = off
                elif action.startswith("^ob"):
                    print(f"Player {off} {defender_text(defender_list, off)} inbounds the ball")
                    last_player, last_defender, last_shooter = off, defender_list, None
                elif action.startswith("^h"):
                    print(f"Player {off} {defender_text(defender_list, off)} dribbles across halfcourt")
                    last_player, last_defender, last_shooter = off, defender_list, None
                elif action == "":
                    if last_player:
                        last_passer = print_pass(last_player, last_defender, off, defender_list)
                    last_player, last_defender, last_shooter = off, defender_list, None
                else:
                    print(f"Unrecognized action for Player {off} with action '{action}'")
                    last_player, last_defender, last_shooter = off, defender_list, None
                i += 1
                continue

            # Free throws
            if "*" in token or "X" in token:
                handle_free_throws(token, default_shooter=last_shooter or last_player)
                last_player, last_defender, last_shooter = None, None, None
                i += 1
                continue

            # Numeric passes
            if token.isdigit():
                if last_player:
                    last_passer = print_pass(last_player, last_defender, token, None)
                last_player, last_defender, last_shooter = token, None, None
                i += 1
                continue

            # Defensive/legacy tokens
            last_player, last_defender, last_shooter, skip = process_defensive_tokens(token, last_player, last_defender, last_shooter)
            if skip:
                i += 1
                continue

            # Fallback
            print(f"Unrecognized token: {token}")
            last_player, last_defender, last_shooter = token, None, None
            i += 1

if __name__ == "__main__":
    main()
