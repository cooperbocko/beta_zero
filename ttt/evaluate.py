import os
import torch

from ttt.tictactoe import TicTacToe
from mcts import MCTS, MCTSNode
from model import ResNet

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

if __name__ == "__main__":
    device = torch.device('cpu')
    game = TicTacToe()
    model1_path = './ttt/models/100.pt'
    model2_path = './ttt/models/50.pt'
    model_kwargs = {
        'input_channels': 2,
        'n_blocks': 3,
        'n_channels': 32,
        'value_layers': 16,
        'n_actions': 9,
        'board_size': 3*3
    }

    model_1 = ResNet(**model_kwargs)
    model_1.to(device)
    load_select(model_1, None, model1_path)
    mcts_1 = MCTS(MCTSNode(game.copy(), 1), 0.01, False)

    model_2 = ResNet(**model_kwargs)
    model_2.to(device)
    load_select(model_2, None, model2_path)
    mcts_2 = MCTS(MCTSNode(game.copy(), 1), 0.01, False)

    while game.outcome is None:
        game.print_board()
        print()

        for _ in range(200):
            mcts_1.iteration(model_1, torch.device('cpu'))
            mcts_2.iteration(model_2, torch.device('cpu'))

        children_1 = mcts_1.root.children.keys()
        children_2 = mcts_2.root.children.keys()
        print(f'model_1:')
        for child in children_1:
            print(f'{child}: visits: {mcts_1.root.children[child].visits} value: {mcts_1.root.children[child].m_action_value}', end=" ")
        print(f'\nmodel_2:')
        for child in children_2:
            print(f'{child}: visits: {mcts_2.root.children[child].visits} value: {mcts_2.root.children[child].m_action_value}', end = " ")
        print()

        action = input('enter action: ')
        action = int(action)

        game.step(action)
        mcts_1.input_move(action)
        mcts_2.input_move(action)

    print(game.outcome)
    game.print_board()
    print()