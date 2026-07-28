import torch

from tictactoe import TicTacToe
from model import TTTResNet
from mcts import TTTMCTSNode, MCTS


model = TTTResNet()
model.load_state_dict(torch.load("temp_checkpoint.pt"))

model.eval()
empty_board_turn_0 = torch.zeros((1, 2, 3, 3)) 
with torch.no_grad():
    policy_0, val_0 = model(empty_board_turn_0)

print(f"Evaluation when Turn = 0s: {val_0.item():.4f}")

# 1. The Immediate Win
win_board = torch.zeros((1, 2, 3, 3))

# You (Channel 0) have Top-Left and Top-Middle
win_board[0, 0, 0, 0] = 1 
win_board[0, 0, 0, 1] = 1 

# Opponent (Channel 1) has Mid-Left and Bot-Right
win_board[0, 1, 1, 0] = 1 
win_board[0, 1, 2, 2] = 1 

with torch.no_grad():
    policy_win, val_win = model(win_board)
    
print(f"--- Immediate Win ---")
print(f"Value (Should be near +1.0): {val_win.item():.4f}")
print(f"Policy Peak (Should be index 2 for Top-Right): {torch.argmax(policy_win).item()}")

# 2. The Forced Block
block_board = torch.zeros((1, 2, 3, 3))

# You (Channel 0) have Center and Bot-Right
block_board[0, 0, 1, 1] = 1 
block_board[0, 0, 2, 2] = 1 

# Opponent (Channel 1) has Top-Left and Top-Middle
block_board[0, 1, 0, 0] = 1 
block_board[0, 1, 0, 1] = 1 

with torch.no_grad():
    policy_block, val_block = model(block_board)
    
print(f"\n--- Forced Block ---")
print(f"Value (Should be near 0.0 since you can still force a draw): {val_block.item():.4f}")
print(f"Policy Peak (Should be index 2 to block Top-Right): {torch.argmax(policy_block).item()}")


# 3. The Unblockable Trap
trap_board = torch.zeros((1, 2, 3, 3))

# You (Channel 0) have Top-Middle, Mid-Left, and Center
trap_board[0, 0, 0, 1] = 1 
trap_board[0, 0, 1, 0] = 1 
trap_board[0, 0, 1, 1] = 1 

# Opponent (Channel 1) has Top-Left, Bot-Left, and Bot-Right
trap_board[0, 1, 0, 0] = 1 
trap_board[0, 1, 2, 0] = 1 
trap_board[0, 1, 2, 2] = 1 

with torch.no_grad():
    policy_trap, val_trap = model(trap_board)

print(f"\n--- Unblockable Trap ---")
print(f"Value (Should be near -1.0 because the game is mathematically lost): {val_trap.item():.4f}")
# The policy here matters less because all moves lead to a loss, but the value head MUST recognize the doom.


import torch
import torch.nn.functional as F

def inspect_policy(model, board_tensor, label="Board State"):
    model.eval()
    with torch.no_grad():
        policy_out, value = model(board_tensor)
        
        # Apply Softmax if your model outputs raw logits
        # (If your network's forward() already ends in Softmax, remove F.softmax)
        probs = F.softmax(policy_out, dim=-1).squeeze(0) # Shape: (9,)
        
        # Reshape into a 3x3 matrix matching the physical board
        grid_probs = probs.view(3, 3).cpu().numpy()

    print(f"\n================ {label} ================")
    print(f"Value Head Score: {value.item():+.4f}")
    
    print("\n--- Policy Probability Heatmap ---")
    for r in range(3):
        row_str = " | ".join(f"{grid_probs[r, c] * 100:5.1f}%" for c in range(3))
        print(f"  {row_str}")
        if r < 2:
            print("  -----------------------")

    print("\n--- Ranked Action List ---")
    sorted_indices = torch.argsort(probs, descending=True)
    for rank, idx in enumerate(sorted_indices, 1):
        i = idx.item()
        row, col = divmod(i, 3)  # Assuming standard index = row * 3 + col
        p = probs[i].item() * 100
        print(f"  #{rank}: Index {i} (Row {row}, Col {col}) -> {p:5.1f}%")
        
        
        
inspect_policy(model, win_board, "Immediate Win Board")
inspect_policy(model, block_board, "Forced Block Board")









#model first
model = TTTResNet()
model.load_state_dict(torch.load('temp_checkpoint.pt'))
game = TicTacToe()
node = TTTMCTSNode(None, TicTacToe(), 1.0)
mcts = MCTS(model, torch.device('cpu'), node, 200, 1, True)
mcts.search()

game.print_board()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(4)
game.print_board()
mcts.input_move(4)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(6)
game.print_board()
mcts.input_move(6)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(2)
game.print_board()
mcts.input_move(2)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(5)
game.print_board()
mcts.input_move(5)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(3)
game.print_board()
mcts.input_move(3)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(1)
game.print_board()
mcts.input_move(1)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(7)
game.print_board()
mcts.input_move(7)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(8)
game.print_board()
mcts.input_move(8)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')