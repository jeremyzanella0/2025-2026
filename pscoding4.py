# pscoding4.py

# mark 4 updated // lowercase rebound shorthand, cleaned-up messages

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

            # --- Offensive rebound followed by made shot (e.g., 2^or+) ---
            if token.endswith("^or+"):
                player = token[:-4]  # remove ^or+
                print(f"Player {player} grabs the offensive rebound")
                print(f"Player {player} makes the shot!")
                last_player = player
                i += 1
                continue

            # --- Offensive rebound followed by missed shot (e.g., 2^or-) ---
            if token.endswith("^or-"):
                player = token[:-4]  # remove ^or-
                print(f"Player {player} grabs the offensive rebound!")
                print(f"Player {player} misses the shot!")
                last_player = player
                i += 1
                continue

            # --- Assist shot (++) ---
            if token.endswith("++"):
                shooter = token[:-2]
                assister = last_player if last_player else "Unknown"
                print(f"Player {assister} passes to Player {shooter}")
                print(f"Player {shooter} makes the shot! (assisted by Player {assister})")
                last_player = shooter
                i += 1
                continue

            # --- Made shot (+) ---
            elif token.endswith("+"):
                shooter = token[:-1]
                if last_player:
                    print(f"Player {last_player} passes to Player {shooter}")
                print(f"Player {shooter} makes the shot!")
                last_player = shooter
                i += 1
                continue

            # --- Missed shot (-) ---
            elif token.endswith("-"):
                shooter = token[:-1]
                if last_player:
                    print(f"Player {last_player} passes to Player {shooter}")
                print(f"Player {shooter} misses the shot")
                last_player = shooter
                i += 1
                continue

            # --- Defensive rebound (^r) ---
            elif token.endswith("^r"):
                rebounder = token[:-2]
                print(f"Player {rebounder} grabs the defensive rebound")
                print("Possession switches to the other team.")
                last_player = rebounder
                i += 1
                continue

            # --- Offensive rebound (^or) ---
            elif token.endswith("^or"):
                rebounder = token[:-3]
                print(f"Player {rebounder} grabs the offensive rebound")
                last_player = rebounder
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
