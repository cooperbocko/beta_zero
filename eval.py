import numpy as np
import torch
from tictactoe import TicTacToe
from train_ttt import rotate_data
# Import your rotate_data function here

def print_board(state_tensor):
    """Prints a clear ASCII board from a 3-channel state tensor."""
    board = np.full((3, 3), '.')
    board[state_tensor[0] == 1] = 'X'
    board[state_tensor[1] == 1] = 'O'
    print("\n".join([" ".join(row) for row in board]))

def print_policy(prob_flat):
    """Prints a 3x3 policy matrix."""
    prob_2d = prob_flat.reshape(3, 3)
    print(np.round(prob_2d, 2))

# 1. Play a quick game and record state/probs
game = TicTacToe()
# X plays top-left (0,0) -> index 0
game.step(0) 
# O plays center (1,1) -> index 4
game.step(4) 

state = game.get_state()
# Mock a policy favoring top-right (0,2) -> index 2
probs = np.zeros(9)
probs[2] = 0.9 
probs[4] = 0.1

states = [state]
prob_list = [probs]
values = [1]

print("=== ORIGINAL ===")
print("Board (X=player, O=opponent):")
print_board(state)
print("Policy Grid:")
print_policy(probs)

# 2. Test 90-degree Rotation
r_states, r_probs, r_values = rotate_data(states, prob_list, values, k=1)

print("\n=== ROTATED 90° CLOCKWISE ===")
print("Board:")
print_board(r_states[0])
print("Policy Grid:")
print_policy(r_probs[0])

