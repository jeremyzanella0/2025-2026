# Step 1: Simplest possession
possession = "2/3"  # Player 2 guarded by Player 3


# Split the fraction into ball-handler and defender
ball_handler, defender = possession.split('/')

print("Ball-handler:", ball_handler)
print("Defender:", defender)

# Possession with a pass
possession = "2/3 5/12"  # Player 2 guarded by 3 passes to Player 5 guarded by 12
# Split by spaces to separate each ball movement
tokens = possession.split()
print(tokens)
# Loop through each token to track passes
for token in tokens:
    if '/' in token:  # fraction means a player has the ball
        offensive, defender = token.split('/')
        print("Ball moves to player:", offensive, "guarded by:", defender)
ball_movements = []

for token in tokens:
    if '/' in token:
        offensive, defender = token.split('/')
        ball_movements.append((offensive, defender))

print("Ball movement list:", ball_movements)

# Possession: Player 2 guarded by 3 passes to Player 5 guarded by 12, who makes a shot
possession = "2/3 5/12 +"
tokens = possession.split()
ball_movements = []

for token in tokens:
    if '/' in token:
        offensive, defender = token.split('/')
        ball_movements.append((offensive, defender))
# Assume the shot is always the last token (for now)
shot_result = tokens[-1]

# Determine who took the shot (last offensive player)
shooter = ball_movements[-1][0]




# Example possession
possession = "2/3 5/12 -"

# Split into tokens
tokens = possession.split()  # ['2/3', '5/12', '+']

# Track ball movements
ball_movements = []
for token in tokens:
    if '/' in token:
        offensive, defender = token.split('/')
        ball_movements.append((offensive, defender))

# Detect shot
shot_result = tokens[-1]  # last token
shooter = ball_movements[-1][0]  # last offensive player

# Detect assist
assist = None
if shot_result == "++":
    if len(ball_movements) >= 2:
        assist = ball_movements[-2][0]
    else:
        assist = "Unknown"

# Print results
print("Shooter:", shooter)
print("Shot result:", shot_result)
print("Assist by:", assist)

rebound_needed = False

if shot_result == "-":
    rebound_needed = True

print("Rebound needed:", rebound_needed)
import random

rebound_player = None
possession_continues = False

if rebound_needed:
    # For now, just randomly pick if offense or defense gets the rebound
    rebound_team = random.choice(["offense", "defense"])
    
    if rebound_team == "offense":
        possession_continues = True
        # Pick one of the offensive players as the rebounder
        rebound_player = ball_movements[-1][0]
    else:
        possession_continues = False
        # Pick one of the defensive players as the rebounder
        rebound_player = random.choice([defender for _, defender in ball_movements])

print("Rebound by:", rebound_player)
print("Possession continues:", possession_continues)



import random

# Example possessions
possessions = [
    "2/3 5/12 -",
    "3/4 7/10 -",
    "1/2 2/5 -"
]

current_team = "Team A"

for p in possessions:
    tokens = p.split()
    
    # Ball movements
    ball_movements = []
    for token in tokens:
        if '/' in token:
            offensive, defender = token.split('/')
            ball_movements.append((offensive, defender))
    
    # Shot detection
    shot_result = tokens[-1]
    shooter = ball_movements[-1][0]
    
    # Assist detection
    assist = None
    if shot_result == "++" and len(ball_movements) >= 2:
        assist = ball_movements[-2][0]
    
    # Rebounds
    rebound_needed = (shot_result == "-")
    rebound_player = None
    possession_continues = True  # default assume possession continues

    if rebound_needed:
        rebound_team = random.choice(["offense", "defense"])
        if rebound_team == "offense":
            possession_continues = True
            rebound_player = ball_movements[-1][0]
        else:
            possession_continues = False
            rebound_player = random.choice([d for _, d in ball_movements])
    
    # Print results
    print("\nPossession:", p)
    print("Shooter:", shooter)
    print("Shot result:", shot_result)
    print("Assist by:", assist)
    print("Rebound by:", rebound_player)
    print("Current possession team:", current_team)
    
    # Update possession team if defense got rebound
    if rebound_needed and not possession_continues:
        current_team = "Team B" if current_team == "Team A" else "Team A"
        print("New possession starts for:", current_team)

possessions = [
    "2/3 5/12 - ^R5",   # 5 gets the offensive rebound
    "3/4 7/10 - ^R10",  # 10 gets defensive rebound
    "1/2 2/5 ++"
]

current_team = "Team A"

for p in possessions:
    tokens = p.split()
    
    # Ball movements
    ball_movements = []
    rebound_player = None
    for token in tokens:
        if '/' in token:
            offensive, defender = token.split('/')
            ball_movements.append((offensive, defender))
        elif token.startswith("^R"):
            rebound_player = token[2:]  # remove ^R to get player number

    # Shot detection
    shot_result = tokens[-2] if rebound_player else tokens[-1]  # adjust if ^R exists
    shooter = ball_movements[-1][0]

    # Assist detection
    assist = None
    if shot_result == "++" and len(ball_movements) >= 2:
        assist = ball_movements[-2][0]

    # Determine possession continuation
    possession_continues = True
    if shot_result == "-":
        if rebound_player:
            if rebound_player in [off for off, _ in ball_movements]:  # offensive rebound
                possession_continues = True
            else:  # defensive rebound
                possession_continues = False

    # Print results
    print("\nPossession:", p)
    print("Shooter:", shooter)
    print("Shot result:", shot_result)
    print("Assist by:", assist)
    print("Rebound by:", rebound_player)
    print("Current possession team:", current_team)

    # Update possession team if defense got rebound
    if shot_result == "-" and rebound_player and not possession_continues:
        current_team = "Team B" if current_team == "Team A" else "Team A"
        print("New possession starts for:", current_team)
