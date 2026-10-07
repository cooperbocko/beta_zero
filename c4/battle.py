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
        for _ in range(np.random.randint(0, 3) * 2):
            valid_moves = np.where(board.get_valid_moves())
            move = np.random.choice(valid_moves[0])
            board.step(move)
            if board.outcome is not None:
                break

        if board.outcome is None:
            games.append(board)

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

    #children = list(mcts.root.children.keys())
    #for child in children:
    #    print(f'{child}: {mcts.root.children[child].visits}')
    return mcts.get_move()

def battle(paths, games, print_board):
    device = torch.device('cpu')
    current_model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    current_model.to('cpu')
    load_latest(current_model, None, './c4/models/')

    games = gen_random_games(games)
    tournament = []
    for path in paths:
        model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
        model.to('cpu')
        load_select(model, None, path)
        tournament.append((model, path.split('/')[-1]))
    tournament.append((None, 'random'))

    for model, name in tournament:
        print(f'latest vs {name} - game')
        model_wins = 0
        opponent_wins = 0
        draws = 0

        for i, game in enumerate(games):
            if print_board:
                print(f'game {i * 2 + 1}')
            # current goes first
            c4 = game.copy()
            p1_mcts = MCTS(C4MCTSNode(game.copy(), 1), 0.01, False)
            if model:
                p2_mcts = MCTS(C4MCTSNode(game.copy(), 1), 0.01, False)
                model_iteration(p2_mcts, model, device) # expand root node

            while c4.outcome is None:
                if print_board:
                    print(c4.turn)
                    c4.print_board()
                    print()

                # p1 move
                action, _ = model_move(p1_mcts, current_model, 300, device)
                c4.step(action)

                if print_board:
                    print(c4.turn)
                    c4.print_board()
                    print()

                # check win
                if c4.outcome is not None:
                    break

                # p2 move
                if model:
                    p2_mcts.input_move(action)
                    action, _ = model_move(p2_mcts, model, 300, device)
                    c4.step(action)
                else:
                    valid_moves = np.where(c4.get_valid_moves())
                    action = np.random.choice(valid_moves[0])
                    c4.step(action)

                p1_mcts.input_move(action)
        
            if print_board:
                print(c4.outcome)
                c4.print_board()
                print()

            if c4.outcome == Outcome.WIN_1:
                model_wins += 1
            elif c4.outcome == Outcome.DRAW:
                draws += 1
            else:
                opponent_wins += 1

            # current goes second
            if print_board:
                print(f'game {i * 2 + 2}')

            c4 = game.copy()
            p2_mcts = MCTS(C4MCTSNode(game.copy(), 1), 0.01, False)
            model_iteration(p2_mcts, current_model, device) # expand root node
            if model:
                p1_mcts = MCTS(C4MCTSNode(game.copy(), 1), 0.01, False)

            while c4.outcome is None:
                if print_board:
                    print(c4.turn)
                    c4.print_board()
                    print()

                # p1 move
                if model:
                    action, _ = model_move(p1_mcts, model, 300, device)
                    c4.step(action)
                else:
                    valid_moves = np.where(c4.get_valid_moves())
                    action = np.random.choice(valid_moves[0])
                    c4.step(action)

                # check win
                if c4.outcome is not None:
                    break

                p2_mcts.input_move(action)

                if print_board:
                    print(c4.turn)
                    c4.print_board()
                    print()

                # p2 move
                action, _ = model_move(p2_mcts, current_model, 300, device)
                c4.step(action)
                if model:
                    p1_mcts.input_move(action)
        
            if print_board:
                print(c4.outcome)
                c4.print_board()
                print()

            if c4.outcome == Outcome.WIN_1:
                opponent_wins += 1
            elif c4.outcome == Outcome.DRAW:
                draws += 1
            else:
                model_wins += 1

        print(f'model wins: {model_wins}, {name} wins: {opponent_wins}, draws: {draws}')

if __name__ == "__main__":
    battle(['./c4/models/180.pt', './c4/models/150.pt', './c4/models/100.pt'], 10, False)