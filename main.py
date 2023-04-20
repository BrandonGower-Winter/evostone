import math

import numpy as np
import random
import src.MCNS as MCNS

POP_SIZE = 1000
K = 15
PMIN = 4
MU = 0.05
REFRESH_RATE = 5000  # Evaluations
BALANCE_THRESHOLD = 0.5
ITERATIONS = 10

# Gene Structure
# 0 - Cost of Card (Mana) int [0, 10]
# 1 - Attack int [0, 12]
# 2 - Health int [1, 14]
# 3 - Charge int [0  or 1]
# 4 - Taunt int [0  or 1]
# 5 - Divine Shield int [0  or 1]
# 6 - WindFury int [0  or 1]
# 7 - Stealth int [0  or 1]
# 8 - Destroy Minion int [0, 7]
# 9 - Draw Card int [0, 10]
# 10 - Freeze int [0, 7]
# 11 - Silence int [0, 7]
# 12 - Damage int [0, 10]
# 13 - Spellpower int [0, 5]
# 14 - Heal int [0, 10]
# 15 - Self Heal int [0, 10]
# 16 - Opponent Draw Card int [0, 10]
# 17 - Discard Card int [0, 10]
# 18 - Overload int [0, 5]
# 19 - Self Damage int [0, 5]
def generate_random_card(random : random.Random):

    r1, r2  = random.randint(0, 16), random.randint(0, 16)

    flags = [1 if x == r1 or x == r2  else 0 for x in range(17)]

    return np.array(
        [
            random.randint(0, 10),            # Cost
            random.randint(0, 12),            # Attack
            random.randint(1, 14),            # Health
            random.randint(0, 1) * flags[0],  # Charge
            random.randint(0, 1) * flags[1],  # Taunt
            random.randint(0, 1) * flags[2],  # Divine Shield
            random.randint(0, 1) * flags[3],  # WindFury
            random.randint(0, 1) * flags[4],  # Stealth
            random.randint(0, 7) * flags[5],  # Destroy Minion
            random.randint(0, 10) * flags[6], # Draw Card
            random.randint(0, 7) * flags[7],  # Freeze
            random.randint(0, 7) * flags[8],  # Silence
            random.randint(0, 10) * flags[9], # Damage
            random.randint(0, 5) * flags[10], # Spellpower
            random.randint(0, 10) * flags[11],# Heal
            random.randint(0, 10) * flags[12],# Self-Heal
            random.randint(0, 10) * flags[13],# Opponent Draw Card
            random.randint(0, 10) * flags[14],# Discard Card
            random.randint(0, 5) * flags[15], # Overload
            random.randint(0, 5) * flags[16],  # Self Damage
        ]
    )


# Values originally from "I am legend: Hacking Hearthstone with machine learning - Defcon 22"
card_weights = [
    1.14,  # Attack
    0.81,  # Health
    0.65,  # Charge
    1.02,  # Taunt
    2.74,  # Divine Shield
    0.96,  # WindFury
    1.21,  # Stealth
    10.63, # Destroy Minion
    3.68,  # Draw Card
    2.04,  # Freeze
    1.66,  # Silence
    1.64,  # Damage
    0.93,  # Spellpower
    0.69,  # Heal
    0.68,  # Self-heal
    -3.97, # Opponent draw card
    -2.67, # Discard cards
    -1.68, # Overload
    -0.54  # Self Damage
]
def evaluate_card(candidate : np.ndarray):
    wsum =  card_weights[0] * candidate[1]  # Attack
    wsum += card_weights[1] * candidate[2]  # Health
    wsum += card_weights[2] * candidate[3] * candidate[1]  # Charge
    wsum += card_weights[3] * candidate[4]  # Taunt
    wsum += card_weights[4] * candidate[5]  # Divine Shield
    wsum += card_weights[5] * candidate[6]  * candidate[1] # Windfury
    wsum += card_weights[6] * candidate[7]  # Stealth
    wsum += card_weights[7] * candidate[8]  # Destroy Minion
    wsum += card_weights[8] * candidate[9]  # Draw Card
    wsum += card_weights[9] * candidate[10] # Freeze
    wsum += card_weights[10] * candidate[11]  # Silence
    wsum += card_weights[11] * candidate[12]  # Damage
    wsum += card_weights[12] * candidate[13]  # Spellpower
    wsum += card_weights[13] * candidate[14]  # Heal
    wsum += card_weights[14] * candidate[15]  # Self-Heal
    wsum += card_weights[15] * candidate[16]  # Opponent Draw Card
    wsum += card_weights[16] * candidate[17]  # Discard Cards
    wsum += card_weights[17] * candidate[18]  # Overload
    wsum += card_weights[18] * candidate[19]  # Self Damage
    #wsum += + 0.32  # Intrinsic
    wsum -= (2 * candidate[0] + 1)  # Cost Coefficient
    wsum /= (2 * candidate[0] + 1)  # Cost Coefficient
    #wsum *= 0.5
    return wsum  #/ (candidate[0] if candidate[0] != 0 else 1.0) # Normalize


mutate_map = {
     0 : [0, 10],
     1 : [0, 12],
     2 : [1, 14],
     3 : [0,  1],
     4 : [0,  1],
     5 : [0,  1],
     6 : [0,  1],
     7 : [0,  1],
     8 : [0,  7],
     9 : [0, 10],
    10 : [0,  7],
    11 : [0,  7],
    12 : [0, 10],
    13 : [0,  5],
    14 : [0, 10],
    15 : [0, 10],
    16 : [0, 10],
    17 : [0, 10],
    18 : [0,  5],
    19 : [0,  5]
}
def mutate_card(candidate : np.ndarray, random : random.Random):
    index = random.randint(0, 19)
    candidate[index] = random.randint(*mutate_map[index])
    return candidate

def is_card_balanced(candidate : np.ndarray) -> bool:
    return abs(evaluate_card(candidate)) < BALANCE_THRESHOLD

def main():
    # argent_squire = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # print('Argent Squire: ' + str(evaluate_card(argent_squire)))

    model = MCNS.NoveltySearch(POP_SIZE, K, generate_random_card, mutate_card, is_card_balanced, PMIN,
                               mu=MU, eval_refresh=REFRESH_RATE)

    for i in range(ITERATIONS):
        model.run()
        print(f'Iteration: {i+1} Archive Size: {len(model.archive)}')
    print('Done!\n')

    print('Cards Found:')
    for key in model.archive:
        print(f'Card with genome: {model.archive[key]} and value: {evaluate_card(model.archive[key])}')


if __name__ == '__main__':
    main()