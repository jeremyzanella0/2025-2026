# pscoding5.py

# Mark 5 --- this version can keep track of all of the basic things throughout a game. passes, shots, rebounds, assists, fouls, turnovers, steals, deflections, blocks, out of bounds, jump ball, drives to the rim, post ups, crossing halfcourt


def main():
    while True:
        line = input("Enter possession (or 'q' to quit): ").strip()
        if line.lower() == "q":
            print("Exiting program...")
            break

        parts = line.split()
        i = 0
        last_player = None

        while i < len(parts):
            token = parts[i]

            # --- Offensive rebound + made shot (2^or+) ---
            if token.endswith("^or+"):
                player = token[:-4]
                print(f"Player {player} grabs the offensive rebound")
                print(f"Player {player} makes the shot!")
                last_player = player
                i += 1
                continue

            # --- Offensive rebound + missed shot (2^or-) ---
            if token.endswith("^or-"):
                player = token[:-4]
                print(f"Player {player} grabs the offensive rebound")
                print(f"Player {player} misses the shot!")
                last_player = player
                i += 1
                continue

                    # --- Drive + made shot (#^d+) ---
            if token.endswith("^d+"):
                player = token[:-3]  # strip ^d+
                if last_player and last_player != player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} drives toward the basket")
                print(f"Player {player} makes the shot!")  # NO assist
                last_player = None
                i += 1
                continue

            # --- Drive + missed shot (#^d-) ---
            if token.endswith("^d-"):
                player = token[:-3]
                if last_player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} drives toward the basket")
                print(f"Player {player} misses the shot!")
                last_player = player
                i += 1
                continue

            # --- Post up + made shot (#^p+) ---
            if token.endswith("^p+"):
                player = token[:-3]  # strip ^p+
                if last_player and last_player != player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} posts up")
                print(f"Player {player} makes the shot!")  # NO assist
                last_player = None
                i += 1
                continue

            # --- Post up + missed shot (#^p-) ---
            if token.endswith("^p-"):
                player = token[:-3]
                if last_player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} posts up")
                print(f"Player {player} misses the shot!")
                last_player = player
                i += 1
                continue

                        # --- Assist shot (++) including drive/post-up ---
            if token.endswith("++"):
                shooter_token = token[:-2]  # strip the ++
                action = ""  # drive or post-up
                # check if shooter_token has ^d or ^p
                if shooter_token.endswith("^d"):
                    shooter = shooter_token[:-2]
                    action = "drives toward the basket"
                elif shooter_token.endswith("^p"):
                    shooter = shooter_token[:-2]
                    action = "posts up"
                else:
                    shooter = shooter_token
                
                assister = last_player if last_player else "Unknown"
                print(f"Player {assister} passes to Player {shooter}")
                if action:
                    print(f"Player {shooter} {action}")
                print(f"Player {shooter} makes the shot! (assisted by Player {assister})")
                last_player = None
                i += 1
                continue


                    # --- Made shot (+) ---
            elif token.endswith("+"):
                shooter = token[:-1]
                if last_player and last_player != shooter:
                    print(f"Player {last_player} passes to Player {shooter}")
                
                # If it’s a drive/post-up token, print that action
                if shooter.endswith("^d"):
                    player_num = shooter[:-2]
                    print(f"Player {player_num} drives toward the basket")
                    shooter = player_num
                elif shooter.endswith("^p"):
                    player_num = shooter[:-2]
                    print(f"Player {player_num} posts up")
                    shooter = player_num

                print(f"Player {shooter} makes the shot!")
                last_player = None
                i += 1

                # --- Check for and-1: foul + free throws ---
                while i < len(parts) and (parts[i].startswith("^f") or parts[i].startswith("^of") or "*" in parts[i] or "X" in parts[i]):
                    next_token = parts[i]

                    # defensive foul
                    if next_token.startswith("^f") and not next_token.startswith("^of"):
                        fouler = next_token[2:]
                        print(f"Player {fouler} commits a defensive foul")
                        last_player = None
                        i += 1

                    # offensive foul
                    elif next_token.startswith("^of"):
                        fouler = next_token[3:]
                        print(f"Player {fouler} commits an offensive foul")
                        print("Possession switches to the other team.")
                        last_player = None
                        i += 1

                    # free throws
                    elif "*" in next_token or "X" in next_token:
                        ft_token = next_token
                        if ft_token[0].isdigit():
                            ft_shooter = ft_token[0]
                            sequence = ft_token[1:]
                        else:
                            ft_shooter = shooter
                            sequence = ft_token
                        for ch in sequence:
                            if ch == '*':
                                print(f"Player {ft_shooter} makes free throw")
                            elif ch.upper() == 'X':
                                print(f"Player {ft_shooter} misses free throw")
                        i += 1
                    else:
                        break
                continue


            # --- Missed shot (-) ---
            elif token.endswith("-"):
                shooter = token[:-1]
                if last_player:
                    print(f"Player {last_player} passes to Player {shooter}")
                print(f"Player {shooter} misses the shot!")
                last_player = shooter
                i += 1
                continue

            # --- Defensive rebound (^r) ---
            elif token.endswith("^r"):
                player = token[:-2]
                print(f"Player {player} grabs the defensive rebound")
                print("Possession switches to the other team.")
                last_player = None
                i += 1
                continue

            # --- Offensive rebound (^or) ---
            elif token.endswith("^or"):
                player = token[:-3]
                print(f"Player {player} grabs the offensive rebound")
                last_player = player
                i += 1
                continue

            # --- Rebound out of bounds (^rob) ---
            elif token == "^rob":
                print("Rebound goes out of bounds")
                last_player = None
                i += 1
                continue

            # --- Deflection out of bounds (^dob) ---
            elif token == "^dob":
                print("Ball deflected out of bounds")
                last_player = None
                i += 1
                continue

            # --- Generic out of bounds (^oob) ---
            elif token == "^oob":
                print("Ball goes out of bounds")
                last_player = None
                i += 1
                continue

            # --- Inbounding (^ob) ---
            elif token.endswith("^ob"):
                player = token[:-3]
                print(f"Player {player} inbounds the ball")
                last_player = player
                i += 1
                continue

            # --- Dead ball turnover (^dbto) ---
            elif token.endswith("^dbto"):
                player = token[:-5]
                if last_player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} commits a dead ball turnover")
                print("Possession switches to the other team.")
                last_player = None
                i += 1
                continue

            # --- Live ball turnover (^lbto) with optional steal (^stl) ---
            elif token.endswith("^lbto"):
                player = token[:-5]
                if last_player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} commits a live ball turnover")
                if i + 1 < len(parts) and parts[i + 1].endswith("^stl"):
                    stealer_token = parts[i + 1]
                    stealer = stealer_token[:-4]
                    print(f"Player {stealer} steals the ball!")
                    last_player = stealer
                    i += 2
                else:
                    print("Possession switches to the other team.")
                    last_player = None
                    i += 1
                continue

            # --- Block (^blk#) ---
            elif token.startswith("^blk"):
                blocker = token[4:]
                if last_player:
                    print(f"Player {blocker} blocks the shot from Player {last_player}")
                    last_player = None
                i += 1
                continue

            # --- Deflection (^def#) ---
            elif token.startswith("^def"):
                deflector = token[4:]
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    next_player = parts[i + 1]
                    print(f"Player {deflector} deflects the ball from Player {last_player}")
                    print(f"Player {next_player} gets the ball")
                    last_player = next_player
                    i += 2
                else:
                    print(f"Player {deflector} deflects the ball from Player {last_player}")
                    last_player = None
                    i += 1
                continue

            # --- Jump ball (^jump#,#) ---
            elif token.startswith("^jump"):
                try:
                    players = token[5:].split(",")
                    if len(players) == 2:
                        p1, p2 = players
                        print(f"Jump ball between Player {p1} and Player {p2}")
                    else:
                        print("Unrecognized jump ball format")
                except Exception:
                    print("Error parsing jump ball event")
                last_player = None
                i += 1
                continue

            # --- Defensive foul (^f#) ---
            elif token.startswith("^f") and not token.startswith("^of"):
                player = token[2:]
                print(f"Player {player} commits a defensive foul")
                last_player = None
                i += 1
                # handle free throws if any
                while i < len(parts) and ('*' in parts[i] or 'X' in parts[i]):
                    ft_token = parts[i]
                    if ft_token[0].isdigit():
                        shooter = ft_token[0]
                        sequence = ft_token[1:]
                    else:
                        shooter = last_player if last_player else "Unknown"
                        sequence = ft_token
                    for ch in sequence:
                        if ch == '*':
                            print(f"Player {shooter} makes free throw")
                        elif ch.upper() == 'X':
                            print(f"Player {shooter} misses free throw")
                    i += 1
                continue

            # --- Offensive foul (^of#) ---
            elif token.startswith("^of"):
                player = token[3:]
                print(f"Player {player} commits an offensive foul")
                print("Possession switches to the other team.")
                last_player = None
                i += 1
                # handle free throws if any
                while i < len(parts) and ('*' in parts[i] or 'X' in parts[i]):
                    ft_token = parts[i]
                    if ft_token[0].isdigit():
                        shooter = ft_token[0]
                        sequence = ft_token[1:]
                    else:
                        shooter = last_player if last_player else "Unknown"
                        sequence = ft_token
                    for ch in sequence:
                        if ch == '*':
                            print(f"Player {shooter} makes free throw")
                        elif ch.upper() == 'X':
                            print(f"Player {shooter} misses free throw")
                    i += 1
                continue

            # --- Free throws alone ---
            elif '*' in token or 'X' in token:
                ft_token = token
                if ft_token[0].isdigit():
                    shooter = ft_token[0]
                    sequence = ft_token[1:]
                else:
                    shooter = last_player if last_player else "Unknown"
                    sequence = ft_token
                for ch in sequence:
                    if ch == '*':
                        print(f"Player {shooter} makes free throw")
                    elif ch.upper() == 'X':
                        print(f"Player {shooter} misses free throw")
                last_player = None
                i += 1
                continue

            # --- Drive only (#^d) ---
            elif token.endswith("^d"):
                player = token[:-2]
                if last_player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} drives toward the basket")
                last_player = player
                i += 1
                continue

            # --- Cross halfcourt (#^h) ---
            elif token.endswith("^h"):
                player = token[:-2]
                if last_player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} dribbles across halfcourt")
                last_player = player
                i += 1
                continue

            # --- Post up only (#^p) ---
            elif token.endswith("^p"):
                player = token[:-2]
                if last_player:
                    print(f"Player {last_player} passes to Player {player}")
                print(f"Player {player} posts up")
                last_player = player
                i += 1
                continue

            # --- Pass ---
            elif token.isdigit():
                if last_player:
                    print(f"Player {last_player} passes to Player {token}")
                last_player = token
                i += 1
                continue

            # --- Unrecognized ---
            else:
                print(f"Unrecognized token: {token}")
                last_player = token
                i += 1
                continue


if __name__ == "__main__":
    main()
