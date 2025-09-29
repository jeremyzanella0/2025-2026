# pscoding3.py

# mark 3 // this program can recognize ball movement, shots, and rebounds

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

            # --- Assist shot (++) ---
            if token.endswith("++"):
                shooter = token[:-2]
                assister = last_player if last_player else "Unknown"
                print(f"Player {assister} passes to Player {shooter}")
                print(f"Player {shooter} makes the shot! (assisted by Player {assister})")
                last_player = shooter
                i += 1

            # --- Made shot (+) ---
            elif token.endswith("+"):
                shooter = token[:-1]
                if last_player:
                    print(f"Player {last_player} passes to Player {shooter}")
                print(f"Player {shooter} makes the shot!")
                last_player = shooter
                i += 1

            # --- Missed shot (-) ---
            elif token.endswith("-"):
                shooter = token[:-1]
                if last_player:
                    print(f"Player {last_player} passes to Player {shooter}")
                print(f"Player {shooter} misses the shot!")
                last_player = shooter
                i += 1

            # --- Defensive rebound (^R) ---
            elif token.endswith("^R"):
                rebounder = token[:-2]
                print(f"Player {rebounder} grabs the defensive rebound!")
                print("Possession switches to the other team.")
                last_player = rebounder
                i += 1

            # --- Offensive rebound (^OR) ---
            elif token.endswith("^OR"):
                rebounder = token[:-3]
                print(f"Player {rebounder} grabs the offensive rebound!")
                print("Possession stays with the same team.")
                last_player = rebounder
                i += 1

            # --- Pass ---
            elif token.isdigit():
                if last_player:
                    print(f"Player {last_player} passes to Player {token}")
                last_player = token
                i += 1

            # --- Unrecognized ---
            else:
                print(f"Unrecognized token: {token}")
                last_player = token
                i += 1


if __name__ == "__main__":
    main()
