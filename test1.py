import torch

from tictactoe import TicTacToe
from model import TTTResNet
from mcts import TTTMCTSNode, MCTS


#model first
model = TTTResNet()
model.load_state_dict(torch.load('temp_checkpoint.pt'))
game = TicTacToe()
node = TTTMCTSNode(None, TicTacToe(), 1.0)
mcts = MCTS(model, torch.device('cpu'), node, 200, 1, True)
mcts.search()

visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')
game.step(4)
mcts.input_move(4)
mcts.search()
visits = ([child.visits for child in mcts.root.children.values()])
values = ([child.m_action_value for child in mcts.root.children.values()])
print(f'Visits: {visits}')
print(f'Values: {values}')


while game.outcome is None:
    print(game.print_board())
    print('Model Move: ')
    action, action_probs = mcts.get_move()
    print(action_probs)
    print(action)
    game.step(action)
    game.print_board()
    print('Player - Enter move: (row, col)')
    row, col = map(int, input().split(','))
    action = row * 3 + col
    game.step(action)
    print(game.outcome)
    if game.outcome is  None:
        mcts.input_move(action)
    
game.print_board()

#model second
model = TTTResNet()
model.load_state_dict(torch.load('temp_checkpoint.pt'))
game = TicTacToe()
node = TTTMCTSNode(None, TicTacToe(), 1.0)
mcts = MCTS(model, torch.device('cpu'), node, 300, 0, False)
mcts.search()

while game.outcome is None:
    game.print_board()   
    print('Player - Enter move: (row, col)')
    row, col = map(int, input().split(','))
    action = row * 3 + col
    game.step(action)
    mcts.input_move(action)
    print(game.print_board())
    print('Model Move: ')
    action, action_probs = mcts.get_move()
    print(action_probs)
    print(action)
    game.step(action)
    game.print_board()
    
game.print_board()    