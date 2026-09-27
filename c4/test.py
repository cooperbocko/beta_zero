import time
import numpy as np
import torch
import multiprocessing as mp

from c4.connect4 import Connect4BitBoard
from c4.c4_mcts_node import C4MCTSNode
from mcts import MCTS
from model import ResNet


"""
# no model random
model = ResNet(input_channels=2, n_blocks=7, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
c4 = Connect4BitBoard()
mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)

start = time.time()
for i in range(10):
    while c4.outcome is None:
        # noise
        mcts.add_dirlect_noise()

        # iterations
        for i in range(300):
            # select
            node, path = mcts.select()

            # check if terminal
            value = mcts.get_terminal_value(node)
            if value:
                mcts.backpropagate(path, value)
                break

            # get random action and probs
            probs = node.state.get_valid_moves()
            probs = probs / np.sum(probs)

            # expand
            mcts.expand(node, probs)

            # backpropagate
            mcts.backpropagate(path, np.random.choice([-1, 1]))

        action, _ = mcts.get_move()
        c4.step(action)
    c4 = Connect4BitBoard()
    mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)
    
print(f'No model random actions 10 games 300 iterations: {time.time() - start}')
"""

# model
def cuda_model():
    device = torch.device('cuda')
    for _ in range(1):
        model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
        model.to(device)
        c4 = Connect4BitBoard()
        mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)

        start = time.time()
        for _ in range(10):
            while c4.outcome is None:
                # noise
                mcts.add_dirlect_noise()

                # iterations
                for _ in range(300):
                    # select
                    node, path = mcts.select()

                    # check if terminal
                    value = mcts.get_terminal_value(node)
                    if value:
                        mcts.backpropagate(path, value)
                        break

                    # get random action and probs
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

                action, _ = mcts.get_move()
                c4.step(action)
            c4 = Connect4BitBoard()
            mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)
            
        print(f'cuda model 10 games 300 iterations: {time.time() - start}')

# model
def cpu_model():
    device = torch.device('cpu')
    torch.set_num_threads(1)
    for _ in range(1):
        model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
        model.to(device)
        c4 = Connect4BitBoard()
        mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)

        start = time.time()
        for _ in range(10):
            while c4.outcome is None:
                # noise
                mcts.add_dirlect_noise()

                # iterations
                for _ in range(300):
                    # select
                    node, path = mcts.select()

                    # check if terminal
                    value = mcts.get_terminal_value(node)
                    if value:
                        mcts.backpropagate(path, value)
                        break

                    # get random action and probs
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

                action, _ = mcts.get_move()
                c4.step(action)
            c4 = Connect4BitBoard()
            mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)
            
        print(f'cpu model 10 games 300 iterations: {time.time() - start}')


# model batch


# processes
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    processes = []
    start = time.time()
    for _ in range(8):

        process = mp.Process(
            target=cpu_model
        )

        process.start()
        processes.append(process)


    # Wait for workers
    for process in processes:
        process.join()
    print(f'total time: {time.time() - start}')