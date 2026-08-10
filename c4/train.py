import multiprocessing as mp
import os
import random

import torch
import numpy as np

from model import Trainer, ResNet
from game import Outcome, Turn
from mcts import MCTS
from c4.c4_mcts_node import C4MCTSNode
from c4.c4_replay_buffer import C4ReplayBuffer
from c4.connect4 import Connect4, Connect4BitBoard

def data_worker(device, num_games, result_queue):
    for game_n in range(num_games):
        states, probs, values = [], [], []
        game = Connect4BitBoard()
        model = ResNet(input_channels=2, n_blocks=5, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
        if os.path.exists('c4_temp_checkpoint.pt'):
            model.load_state_dict(torch.load('c4_temp_checkpoint.pt'))
        model.to(device)
        mcts = MCTS(model, device, C4MCTSNode(None, Connect4BitBoard(), 1.0), 1, 1, True)
        
        while game.outcome is None:
            action, v_action_probs = mcts.get_move()
            states.append(game.get_state())
            actual_probs = np.zeros(7)
            for v_action, prob in v_action_probs.items():
                actual_probs[v_action] = prob
            probs.append(actual_probs)
            game.step(action)
            
        for i, state in enumerate(states):
            if game.outcome == Outcome.DRAW:
                values.append(0)
            elif (game.outcome == Outcome.WIN_1 and i % 2 == 0) or (game.outcome == Outcome.WIN_2 and i % 2 == 1):
                values.append(1)
            else:
                values.append(-1)
                
        result_queue.put((states, probs, values))
        f_states, f_probs = flip_data(states, probs)
        result_queue.put((f_states, f_probs, values))
        
def flip_data(states, probs):
    f_states, f_probs = [], []
    
    for state, prob in zip(states, probs):
        f_state = np.fliplr(state)
        f_prob = prob[::-1]
        
        f_states.append(f_state.copy())
        f_probs.append(f_prob.flatten().copy())
        
    return (f_states, f_probs)

def train_c4():
    model = ResNet(input_channels=2, n_blocks=5, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    device = torch.device('mps')
    model.to(device)
    if os.path.exists('c4_temp_checkpoint.pt'):
        model.load_state_dict(torch.load('c4_temp_checkpoint.pt'))
    replay_buffer = C4ReplayBuffer(10000)
    trainer = Trainer(model, device, replay_buffer)
    
    for iteration in range(10):
        print(f'iteration {iteration}')
        model.eval()
        games_added = 0
        print('collecting data')
        
        games_per_iteration = 80
        games_per_worker = games_per_iteration // 8
        with mp.Manager() as manager:
            result_queue = manager.Queue()
            processes = []
            
            for worker_id in range(8):
                p = mp.Process(target=data_worker, args=(device, games_per_worker, result_queue))
                p.start()
                processes.append(p)
                
            games_added = 0
            while games_added < games_per_iteration * 2:
                states, probs, values = result_queue.get()
                replay_buffer.add(states, probs, values)
                games_added += 1
                
                if games_added % 10 == 0 or games_added == games_per_iteration * 2:
                    print(f'played {games_added} / {games_per_iteration * 2} games')
                
        for p in processes:
            p.join()
                
        print(f'training!')
        model.train()
        batches_per_epoch = max(1, len(replay_buffer.buffer) // 64)
        for epoch in range(5):
            epoch_loss = 0
            for batch in range(batches_per_epoch):
                loss = trainer.train(64)
                epoch_loss += loss
            avg_loss = epoch_loss / batches_per_epoch
            print(f'epoch {epoch} loss: {avg_loss}')
            
        torch.save(model.state_dict(), 'c4_temp_checkpoint.pt')
        print('model saved')
        
if __name__ == "__main__":
    mp.set_start_method('spawn')
    train_c4()

        