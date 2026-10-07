import multiprocessing as mp
import time
import os

import torch
import numpy as np

from model import Trainer, ResNet
from game import Outcome
from mcts import MCTS
from c4.c4_mcts_node import C4MCTSNode
from c4.c4_replay_buffer import C4ReplayBuffer
from c4.connect4 import Connect4BitBoard

def data_worker(id, data_queue, n_games, iterations):
    device = torch.device('cpu')
    torch.set_num_threads(1)
    games = []

    model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    load_latest(model, None, './c4/models/')
    model.to(device)
    
    for _ in range(n_games):
        c4 = Connect4BitBoard()
        mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)
        state_list = []
        probs_list = []
        value_list = []
        moves = 0

        # inital root expansion
        node, path = mcts.select()
        state = torch.from_numpy(node.state.get_state()).float().to(device)
        state = state.unsqueeze(0)
        with torch.no_grad():
            probs, value = model(state)
        probs = probs.detach().cpu().numpy()[0]
        value = value.detach().cpu().numpy()[0]
        mcts.expand(node, probs)

        while c4.outcome is None:
            # temperature check
            if moves > 12 and mcts.temperature >= 0.5:
                mcts.temperature = 0.1

            # noise
            mcts.add_dirichlet_noise(0.25, 1.0)

            # iterations
            for _ in range(iterations):
                # select
                node, path = mcts.select()

                # check if terminal
                value = mcts.get_terminal_value(node)
                if value:
                    mcts.backpropagate(path, value)
                    break

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

            # get move
            action, valid_probs = mcts.get_move()
            actual_probs = np.zeros(7)
            for v_action, prob in valid_probs.items():
                actual_probs[v_action] = prob
            state_list.append(c4.get_state())
            probs_list.append(actual_probs)
            c4.step(action)
            moves += 1

        # get values
        for i, _ in enumerate(state_list):
            if c4.outcome == Outcome.DRAW:
                value_list.append(0)
            elif (c4.outcome == Outcome.WIN_1 and i % 2 == 0) or (c4.outcome == Outcome.WIN_2 and i % 2 == 1):
                value_list.append(1)
            else:
                value_list.append(-1)

        # add to list
        games.append((state_list, probs_list, value_list))
        f_states, f_probs = flip_data(state_list, probs_list)
        games.append((f_states, f_probs, value_list))

    # send results
    data_queue.put(games)
          
def flip_data(states, probs):
    f_states, f_probs = [], []
    
    for state, prob in zip(states, probs):
        f_state = np.fliplr(state)
        f_prob = prob[::-1]
        
        f_states.append(f_state.copy())
        f_probs.append(f_prob.flatten().copy())
        
    return (f_states, f_probs)

def save_model(model, optimizer, iteration, path):
    checkpoint = {
        'iteration': iteration,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, f'{path}/latest.pt')
    torch.save(checkpoint, f'{path}/{iteration}.pt')

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

def train_c4(args=None):
    if not args:
        args = {
            'n_iterations': 300,
            'n_games': 25,
            'n_workers': 10,
            'batch_size': 256,
            'mini_batches': 100,
            'training_iterations': 200,
            'device': 'cuda'
        }

    # data worker info
    n_iterations = args['n_iterations']
    n_games = args['n_games']
    n_workers = args['n_workers']

    # training info
    batch_size = args['batch_size']
    mini_batches = args['mini_batches']
    training_iterations = args['training_iterations']
    device = torch.device(args['device'])
    model = ResNet(input_channels=2, n_blocks=8, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    model.to(device)
    replay_buffer = C4ReplayBuffer(max_size=15000)
    trainer = Trainer(model, device, replay_buffer)
    model_iteration = load_latest(model, trainer.optimizer, './c4/models/')
    data_queue = mp.Queue() # worker game (states, probs, values) 8 200 x x x

    # main training loop
    for z in range(training_iterations):
        # workers
        workers = []
        for i in range(n_workers):
            w = mp.Process(target=data_worker, args=(i, data_queue, n_games, n_iterations))
            w.start()
            workers.append(w)

        data = []
        while len(data) < n_workers:
            data.append(data_queue.get())

        for w in workers:
            w.join()

        # training
        for worker in data:
            for game in worker:
                states, action_probs, rewards = game
                replay_buffer.add(states, action_probs, rewards)

        # mini batches
        t_loss = 0
        for _ in range(mini_batches):
            t_loss += trainer.train(batch_size)

        print(f'iteration: {model_iteration} avg loss: {t_loss / mini_batches} total loss: {t_loss}')
        model_iteration += 1
        save_model(model, trainer.optimizer, model_iteration, './c4/models/')

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train_c4()
