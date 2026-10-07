import os
import numpy as np
import torch

from c4.connect4 import Connect4BitBoard
from c4.c4_mcts_node import C4MCTSNode
from mcts import MCTS
from model import ResNet
from game import Outcome

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
    for _ in range(iterations):
        model_iteration(mcts, model, device)

    return mcts.get_move()

if __name__ == "__main__":
    model_1 = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    model_1.to('cpu')
    load_select(model_1, None, './c4/models/200.pt')
    mcts_1 = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 0.01, False)

    model_2 = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    model_2.to('cpu')
    load_select(model_2, None, './c4/models/100.pt')
    mcts_2 = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 0.01, False)

    c4 = Connect4BitBoard()

    while c4.outcome is None:
        c4.print_board()
        print()

        for _ in range(300):
            model_iteration(mcts_1, model_1, torch.device('cpu'))
            model_iteration(mcts_2, model_2, torch.device('cpu'))

        children_1 = mcts_1.root.children.keys()
        children_2 = mcts_2.root.children.keys()
        for child in children_1:
            print(f'{child}: {mcts_1.root.children[child].visits}')
        for child in children_2:
            print(f'{child}: {mcts_2.root.children[child].visits}')

        action = input('enter action: ')
        action = int(action)

        c4.step(action)
        mcts_1.input_move(action)
        mcts_2.input_move(action)

    print(c4.outcome)
    c4.print_board()
    print()