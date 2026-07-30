import torch

from ttt.tictactoe import TicTacToe
from model import TTTResNet
from mcts import TTTMCTSNode, MCTS

model = TTTResNet()
model.load_state_dict(torch.load("temp_checkpoint.pt"))
model.eval()

mcts = MCTS(model, torch.device("cpu"), TTTMCTSNode(None, TicTacToe(), 1.0), 200, 0.1, False)
game = TicTacToe()
print('model goes first')
while game.outcome is None:
    game.print_board()
    action, action_probs = mcts.get_move()
    print(f'model action: {action} action_probs: {action_probs}')
    game.step(action)
    game.print_board()
    if game.outcome is not None:
        break
    print('enter action: ')
    move = int(input())
    game.step(move)
    mcts.input_move(move)

print(game.outcome)
mcts = MCTS(model, torch.device("cpu"), TTTMCTSNode(None, TicTacToe(), 1.0), 200, 0.1, False)
mcts.search()
game = TicTacToe()
print('player goes first')
while game.outcome is None:
    game.print_board()
    print('enter action: ')
    move = int(input())
    game.step(move)
    mcts.input_move(move)
    game.print_board()
    if game.outcome is not None:
        break
    action, action_probs = mcts.get_move()
    print(f'model action: {action} action_probs: {action_probs}')
    game.step(action)

print(game.outcome)