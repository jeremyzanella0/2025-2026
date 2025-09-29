# basketball_possessions_final.py

def parse_possession(tokens, current_team_name):
    play_by_play = []
    last_passer = None
    possession_switch = False

    for token in tokens:

        # Defensive rebound
        if "^R" in token:
            player = token.replace("^R", "")
            play_by_play.append(f"{player} grabs defensive rebound")
            possession_switch = True
            continue

        # Offensive rebound (same as defensive but no possession switch)
        if "^OR" in token:
            player = token.replace("^OR", "")
            play_by_play.append(f"{player} grabs offensive rebound")
            last_passer = player  # update for assist if next shot
            continue

        # Shot token
        if "/" in token and any(sym in token for sym in ["++", "+", "-"]):
            if token[-2:] == "++":
                shot_result = "++"
                shooter_def = token[:-2]
            else:
                shot_result = token[-1]
                shooter_def = token[:-1]

            shooter, defender = shooter_def.split("/")
            play_by_play.append(f"{shooter} guarded by {defender} shoots -> {shot_result}")

            if shot_result == "++" and last_passer:
                play_by_play.append(f"Assist by {last_passer}")

            last_passer = shooter
            continue

        # Pass token
        if "/" in token:
            passer, defender = token.split("/")
            play_by_play.append(f"{passer} guarded by {defender} passes")
            last_passer = passer

    return play_by_play, possession_switch


def main():
    teamA = "A"
    teamB = "B"
    current_team_name = teamA
    other_team_name = teamB

    while True:
        possession_input = input("Enter a possession (or type 'quit' to exit): ")
        if possession_input.lower() == "quit":
            break

        tokens = possession_input.split()
        play_by_play, switch = parse_possession(tokens, current_team_name)

        print(f"\nTeam {current_team_name}:")
        for line in play_by_play:
            print(line)

        if switch:
            current_team_name, other_team_name = other_team_name, current_team_name
            print(f"\n--- Possession switches to Team {current_team_name} ---\n")
        else:
            print()


if __name__ == "__main__":
    main()
