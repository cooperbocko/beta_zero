import os
import numpy as np
import torch

from c4.connect4 import Connect4BitBoard
from c4.c4_mcts_node import C4MCTSNode
from mcts import MCTS
from model import ResNet
from game import Outcome

def gen_random_games(n):
    games = []

    while len(games) < n:
        board = Connect4BitBoard()
        for _ in range(np.random.randint(1, 10) * 2):
            valid_moves = np.where(board.get_valid_moves())
            move = np.random.choice(valid_moves[0])
            board.step(move)
            if board.outcome is not None:
                break

        if board.outcome is None:
            games.append((board, board.outcome))

    return games

def load_latest(model, optimizer, path):
    if os.path.exists(f'{path}/latest.pt'):
        checkpoint = torch.load(f'{path}/latest.pt')
        if model:
            model.load_state_dict(checkpoint['model'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['iteration']
    else:
        return 0

def load_select(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        if model:
            model.load_state_dict(checkpoint['model'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['iteration']
    else:
        return 0

def model_iteration(mcts: MCTS, model: ResNet, device):
    node, path = mcts.select()
    # check if terminal
    value = mcts.get_terminal_value(node)
    if value:
        mcts.backpropagate(path, value)
        return

    # get action and probs
    state = torch.from_numpy(node.state.get_state()).float().to(device)
    state = state.unsqueeze(0)
    with torch.no_grad():
        probs, value = model(state)
    probs = probs.detach().cpu().numpy()[0]
    value = value.detach().cpu().numpy()[0]

    # expand
    mcts.expand(node, probs)

    # backpropagate
    mcts.backpropagate(path, value)

def model_move(mcts, model, iterations, device):
    for i in range(iterations):
        model_iteration(mcts, model, device)

    children = list(mcts.root.children.keys())
    for child in children:
        print(f'{child}: {mcts.root.children[child].visits}')
    return mcts.get_move()

def battle(paths):
    device = torch.device('cpu')
    current_model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    current_model.to('cpu')
    load_latest(current_model, None, './c4/models/')

    games = gen_random_games(20)
    tournament = []
    for path in paths:
        model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
        model.to('cpu')
        load_select(model, None, path)
        tournament.append((model, path.split('/')[-1]))
    tournament.append((None, 'random'))

    for model, name in tournament:
        model_wins = 0
        opponent_wins = 0
        draws = 0

        print(f'{name} - game')
        c4 = Connect4BitBoard()
        current_mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 0.01, False)
        model_iteration(current_mcts, current_model, device) # expand root node

        if model:
            other_mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 0.01, False)
        
        while c4.outcome is None:
            c4.print_board()
            print()
            if not model:
                valid_moves = np.where(c4.get_valid_moves())
                move = np.random.choice(valid_moves[0])
                c4.step(move)
                current_mcts.input_move(move)
            else:
                action, _ = model_move(other_mcts, model, 300, device)
                c4.step(action)
                current_mcts.input_move(action)
            c4.print_board()
            print()

            if c4.outcome is not None:
                break

            action, _ = model_move(current_mcts, current_model, 300, device)
            c4.step(action)
            if model:
                other_mcts.input_move(action)
    
        c4.print_board()
        print(c4.outcome)
        if c4.outcome == Outcome.WIN_1:
            opponent_wins += 1
        elif c4.outcome == Outcome.DRAW:
            draws += 1
        else:
            model_wins += 1

        print(f'model wins: {model_wins}, {name} wins: {opponent_wins}, draws: {draws}')

if __name__ == "__main__":
    battle(['./c4/models/10.pt', './c4/models/20.pt'])