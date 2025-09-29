# pscoding11
# Mark 11 —  dribble handoff (dho), stationary handoff (ho)


import re

# ---------------- Parsing Functions ----------------

def parse_player_def(token: str):
    """
    Parse a token into ball handler, defender, action type, screener/recipient, screener_def, and after-action codes.
    Supports PNR, PNP, SLP, GST, RJ, DHO, HO, slash actions, and numeric tokens.
    Returns: (ball_handler, ball_def, action_type, screener, screener_def, action_codes)
    """
    action_type = None
    for key in ("pnr","pnp","slp","gst","rj","dho","ho"):
        if key in token:
            action_type = key
            break

    if action_type:
        ball_part, rest = token.split(action_type,1)
        if "/" in ball_part:
            ball_handler, def_and_actions = ball_part.split("/",1)
            i = 0
            while i < len(def_and_actions) and (def_and_actions[i].isdigit() or def_and_actions[i:i+3]=="rot"):
                if def_and_actions[i:i+3]=="rot":
                    i+=3
                    while i<len(def_and_actions) and def_and_actions[i].isdigit():
                        i+=1
                    break
                i+=1
            ball_def = def_and_actions[:i] if i>0 else None
            action_codes = def_and_actions[i:]+rest
        else:
            ball_handler, ball_def, action_codes = ball_part, None, rest

        # Screener / recipient + defender
        screener, screener_def, action = None, None, ""
        if "/" in action_codes:
            j = 0
            while j < len(action_codes) and (action_codes[j].isdigit() or action_codes[j:j+3]=="rot"):
                if action_codes[j:j+3]=="rot":
                    j+=3
                    while j<len(action_codes) and action_codes[j].isdigit():
                        j+=1
                    break
                j+=1
            screener = action_codes[:j]
            rest2 = action_codes[j:]
            if rest2.startswith("/"):
                k=1
                if rest2[1:4]=="rot":
                    k=4
                    while k<len(rest2) and rest2[k].isdigit():
                        k+=1
                else:
                    while k<len(rest2) and rest2[k].isdigit():
                        k+=1
                screener_def = rest2[1:k]
                action = rest2[k:]
            else:
                action=rest2
        else:
            i=0
            while i<len(action_codes) and (action_codes[i].isdigit() or action_codes[i:i+3]=="rot"):
                if action_codes[i:i+3]=="rot":
                    i+=3
                    while i<len(action_codes) and action_codes[i].isdigit():
                        i+=1
                    break
                i+=1
            screener=action_codes[:i]
            action=action_codes[i:]
        return ball_handler, ball_def, action_type, screener, screener_def, action

    # Slash / normal actions
    if "/" in token:
        off, rest = token.split("/",1)
        i=0
        if rest.startswith("rot"):
            i=3
            while i<len(rest) and rest[i].isdigit():
                i+=1
        else:
            while i<len(rest) and rest[i].isdigit():
                i+=1
        def_player = rest[:i] if i>0 else None
        action = rest[i:] if i<len(rest) else ""
        return off, def_player, action, None, None, None

    return token, None, token, None, None, None

# ---------------- Helper Functions ----------------

def defender_text(def_player, off_player=None):
    if not def_player:
        return "wide open"
    if isinstance(def_player, list):
        parts=[]
        for d in def_player:
            if not d:
                continue
            if str(d).startswith("rot"):
                parts.append(f"Player {d[3:]} rotating over to guard Player {off_player}")
            else:
                parts.append(f"Player {d}")
        if len(parts)==0:
            return "wide open"
        elif len(parts)==1:
            return f"guarded by {parts[0]}"
        else:
            return f"guarded by {', '.join(parts)}"
    if isinstance(def_player,str):
        if def_player.startswith("rot"):
            return f"guarded by Player {def_player[3:]} rotating over to guard Player {off_player}"
        else:
            return f"guarded by Player {def_player}"

def print_pass(from_player, from_def, to_player, to_def):
    print(f"Player {from_player} {defender_text(from_def, from_player)} passes to Player {to_player} {defender_text(to_def, to_player)}")
    return from_player

def handle_shot(player, def_list, action, last_passer=None, last_screener=None, pnr_shot=False):
    award_screen = pnr_shot and last_screener
    if action=="++":
        print(f"Player {player} {defender_text(def_list,player)} makes the shot (assisted by Player {last_passer})!")
        if award_screen:
            print(f"Player {last_screener} gets a screen assist!")
    elif action=="+": 
        print(f"Player {player} {defender_text(def_list,player)} makes the shot!")
        if award_screen:
            print(f"Player {last_screener} gets a screen assist!")
    elif action=="-":
        print(f"Player {player} {defender_text(def_list,player)} misses the shot!")

# ---------------- Defensive / Rebound Functions ----------------

def process_defensive_tokens(token, last_player, last_defender, last_shooter, off_player=None, def_list=None):
    base_token=token
    if "/" in token:
        parts=token.split("/")
        off_player=parts[0]
        base_token="/".join(parts[1:])
    if "^or" in base_token:
        player=off_player or last_shooter or last_player
        rebound_def_list=def_list
        if "/" in token:
            player_part, def_part = token.split("/",1)
            player=player_part
            digits=re.match(r"(\d+)", def_part)
            if digits:
                rebound_def_list=[digits.group(1)]
        putback_action=None
        if base_token.endswith("+"): putback_action="+"
        elif base_token.endswith("-"): putback_action="-"
        print(f"Player {player} {defender_text(rebound_def_list,player)} grabs the offensive rebound")
        if putback_action:
            handle_shot(player, rebound_def_list, putback_action)
            last_shooter=player
            return None,None,last_shooter,1
        return player,rebound_def_list,None,1
    if "^r" in base_token:
        player=off_player or last_shooter or last_player
        print(f"Player {player} {defender_text(def_list,player)} grabs the defensive rebound")
        print("Possession switches to the other team.")
        return None,None,None,1
    if "^dbto" in base_token:
        player=off_player or last_player
        print(f"Player {player} commits a dead ball turnover")
        print("Possession switches to the other team.")
        return None,None,None,1
    if "^lbto" in base_token:
        player=off_player or last_player
        print(f"Player {player} commits a live ball turnover")
        return player,None,None,1
    if "^stl" in base_token:
        stealer=base_token[4:]
        print(f"Player {stealer} steals the ball!")
        return stealer,None,None,1
    if "^blk" in base_token:
        blocker=base_token[4:]
        if last_player:
            print(f"Player {blocker} blocks the shot from Player {last_player}")
        else:
            print(f"Player {blocker} blocks the shot")
        return None,None,None,1
    if "^def" in base_token:
        deflector=base_token[4:]
        print(f"Player {deflector} deflects the ball from Player {last_player if last_player else 'Unknown'}")
        return None,None,None,1
    if "^jump" in base_token:
        try:
            players=base_token[5:].split(",")
            if len(players)==2:
                print(f"Jump ball between Player {players[0]} and Player {players[1]}")
            else:
                print("Unrecognized jump ball format")
        except:
            print("Error parsing jump ball event")
        return None,None,None,1
    if "^f" in base_token and not "^of" in base_token:
        fouler=base_token[2:]
        print(f"Player {fouler} commits a defensive foul")
        return None,None,None,1
    if "^of" in base_token:
        fouler=base_token[3:]
        print(f"Player {fouler} commits an offensive foul")
        print("Possession switches to the other team.")
        return None,None,None,1
    if base_token in ["^rob","^dob","^oob"]:
        msgs={"^rob":"Rebound goes out of bounds","^dob":"Ball deflected out of bounds","^oob":"Ball goes out of bounds"}
        print(msgs[base_token])
        return None,None,None,1
    return last_player,last_defender,last_shooter,0

# ---------------- Main Loop ----------------

def main():
    while True:
        line=input("Enter possession (or 'q' to quit): ").strip()
        if line.lower()=="q": break
        parts=line.split()
        last_possessor=None
        last_defender=None
        last_shooter=None
        last_passer=None
        last_screener=None
        i=0
        while i<len(parts):
            token=parts[i]
            last_possessor,last_defender,last_shooter,skip=process_defensive_tokens(token,last_possessor,last_defender,last_shooter)
            if skip:
                last_screener=None
                i+=1
                continue

            # HELP/SWITCH
            if token.startswith(("hp","sw")):
                match=re.match(r"(hp|sw)(\d+.*)",token)
                if match:
                    head,tail=match.group(1),match.group(2)
                    m2=re.match(r"(\d+(?:,\d+)*)(\+\+|\+|-)?$",tail)
                    if not m2:
                        digits=re.match(r"(\d+(?:,\d+)*)",tail)
                        if digits:
                            core=digits.group(1)
                            rest=tail[len(core):]
                            token=head+core
                            if rest:
                                parts.insert(i+1,rest)
                if not last_possessor and i>0:
                    prev=parts[i-1].split("/")[0]
                    last_possessor=prev
                    last_defender=[parts[i-1].split("/")[1]] if "/" in parts[i-1] else []
                is_help=token.startswith("hp")
                action_raw=token[2:]
                shoot_after=None
                for suffix,meaning in [("++","assist"),("+","make"),("-","miss")]:
                    if action_raw.endswith(suffix):
                        shoot_after=meaning
                        action_raw=action_raw[:-len(suffix)]
                        break
                action_list=[x.strip() for x in action_raw.split(",") if x.strip()]
                last_defender=last_defender if isinstance(last_defender,list) else [last_defender] if last_defender else []
                if is_help:
                    for x in action_list:
                        if x not in last_defender: last_defender.append(x)
                    print(f"Player {action_list[0]} steps in to help on Player {last_possessor}" if len(action_list)==1 else f"Players {', '.join(action_list)} step in to help on Player {last_possessor}")
                else:
                    last_defender=action_list
                    print(f"Player {action_list[0]} switches onto Player {last_possessor}" if len(action_list)==1 else f"Players {', '.join(action_list)} switch onto Player {last_possessor}")
                if shoot_after:
                    shooter=last_possessor
                    if shoot_after=="assist":
                        handle_shot(shooter,last_defender,"++",last_passer,last_screener,pnr_shot=False)
                    else:
                        handle_shot(shooter,last_defender,"+" if shoot_after=="make" else "-")
                        last_shooter=shooter
                    last_possessor=None
                    last_defender=None
                    last_screener=None
                i+=1
                continue

            # PNR / PNP / SLP / GST / RJ / DHO / HO
            if any(k in token for k in ("pnr","pnp","slp","gst","rj","dho","ho")):
                ball,balldef,action_type,screener,screendef,after=parse_player_def(token)
                if last_possessor and last_possessor!=ball:
                    last_passer=last_possessor
                    print_pass(last_possessor,last_defender,ball,[balldef] if balldef else None)
                if action_type=="pnr":
                    print(f"Player {ball} {defender_text(balldef,ball)} comes off a pick and roll from Player {screener} {defender_text(screendef,screener)}")
                elif action_type=="pnp":
                    print(f"Player {ball} {defender_text(balldef,ball)} comes off a pick and pop from Player {screener} {defender_text(screendef,screener)}")
                elif action_type=="slp":
                    print(f"Player {screener} {defender_text(screendef,screener)} slips the screen for Player {ball} {defender_text(balldef,ball)}")
                elif action_type=="rj":
                    print(f"Player {ball} {defender_text(balldef,ball)} rejects the ball screen from Player {screener} {defender_text(screendef,screener)}")
                elif action_type=="gst":
                    print(f"Player {screener} {defender_text(screendef,screener)} ghosts the screen for Player {ball} {defender_text(balldef,ball)}")
                elif action_type=="dho":
                    print(f"Player {ball} {defender_text(balldef,ball)} dribbles and hands off to Player {screener} {defender_text(screendef,screener)}")
                elif action_type=="ho":
                    print(f"Player {ball} {defender_text(balldef,ball)} hands off to Player {screener} {defender_text(screendef,screener)}")

                last_screener=screener if action_type in ("pnr","pnp") else None

                # --- FIX: handle defense before trailing drive ---
                defenses=[]
                trailing=""
                if after:
                    for match in re.finditer(r"(ch|ct|sw|bz|ice|cs|\^d|\^h|\+|\-\-?|\+\+?)",after):
                        token_match = match.group(0)
                        if token_match in ("^d","^h","+","-","++"):
                            trailing+=token_match
                        else:
                            defenses.append(token_match)

                for idx,def_code in enumerate(defenses):
                    if def_code=="ch":
                        if action_type in ("dho","ho"):
                            print(f"Player {balldef} chases over the top of the handoff")
                        else:
                            print(f"Player {balldef} chases over the top of the screen")
                    elif def_code=="ct":
                        print(f"Player {balldef} cuts under the screen/handoff")
                    elif def_code=="sw":
                        print(f"Player {screendef} switches onto {ball} and Player {balldef} switches onto {screener}")
                    elif def_code=="bz":
                        print(f"Players {balldef} and {screendef} double team {ball}")
                    elif def_code=="ice":
                        print(f"Players {balldef} and {screendef} ice the screen/handoff")
                    elif def_code=="cs":
                        print(f"Player {ball} is caught on the screen/handoff")

                # Handle trailing drive/dribble
                while trailing:
                    if trailing.startswith("^h"):
                        print(f"Player {ball} {defender_text(balldef,ball)} dribbles over halfcourt")
                        last_possessor=ball
                        last_defender=[balldef] if balldef else None
                        trailing=trailing[2:]
                    elif trailing.startswith("^d"):
                        print(f"Player {ball} {defender_text(balldef,ball)} drives inside")
                        if last_possessor!=ball:
                            last_passer=last_possessor
                        last_possessor=ball
                        last_defender=[balldef] if balldef else None
                        trailing=trailing[2:]
                    elif trailing.startswith(("+","-","++")):
                        handle_shot(ball,[balldef] if balldef else None,trailing,last_passer,last_screener,pnr_shot=(action_type in ("pnr","pnp")))
                        last_shooter=ball
                        last_possessor=None
                        last_defender=None
                        last_screener=None
                        trailing=""
                    else:
                        trailing=""  # unknown trailing token

                if not trailing:
                    last_possessor,last_defender=ball,[balldef] if balldef else None

                i+=1
                continue

            # Slash / normal
            if "/" in token:
                off, defender, action, _, _, _=parse_player_def(token)
                defender_list=[defender] if defender and not isinstance(defender,list) else defender
                if last_possessor and last_possessor!=off:
                    last_passer=last_possessor
                    print_pass(last_possessor,last_defender,off,defender_list)
                if action.startswith("^d"):
                    print(f"Player {off} {defender_text(defender_list,off)} drives toward the basket")
                    last_possessor=off
                    last_defender=defender_list
                    if action.endswith("+"):
                        handle_shot(off,defender_list,"+",last_passer,last_screener)
                        last_possessor,last_defender,last_shooter=None,None,off
                        last_screener=None
                    elif action.endswith("-"):
                        handle_shot(off,defender_list,"-",last_passer,last_screener)
                        last_possessor,last_defender,last_shooter=off,defender_list,off
                        last_screener=None
                elif action.startswith("^h"):
                    print(f"Player {off} {defender_text(defender_list,off)} dribbles")
                    last_possessor,last_defender=off,defender_list
                elif action in ["+","-","++"]:
                    handle_shot(off,defender_list,action,last_passer,last_screener)
                    last_possessor,last_defender,last_shooter=None,None,off
                    last_screener=None
                else:
                    last_possessor,last_defender=off,defender_list
                i+=1
                continue

            # Plain token
            last_possessor=token
            last_defender=None
            i+=1

if __name__=="__main__":
    main()
