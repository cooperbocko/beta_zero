import multiprocessing as mp
import os
import random

import torch
import numpy as np

from model import TTTResNet, TTTReplayBuffer, Trainer
from tictactoe import TicTacToe
from mcts import TTTMCTSNode, MCTS
from game import Outcome, Turn

def data_worker(model, device, num_games, worker_id, result_queue):
    for game_num in range(num_games):
        states, probs, values = [], [], []
        game = TicTacToe()
        mcts = MCTS(model, device, TTTMCTSNode(None, TicTacToe(), 1.0), 300, 1, True)
        steps = 0
        
        while game.outcome is None: 
            #if game_num % 4 == 0:
                #mcts.temperature = 0.1
                
            action, v_action_probs = mcts.get_move()
            states.append(game.get_state())
            actual_probs = np.zeros(9)
            for v_action, prob in v_action_probs.items():
                actual_probs[v_action] = prob
            probs.append(actual_probs)
            game.step(action)
            steps += 1
            
        for i, state in enumerate(states):
            if game.outcome == Outcome.DRAW:
                values.append(0)
            elif (game.outcome == Outcome.WIN_1 and i % 2 == 0) or (game.outcome == Outcome.WIN_2 and i % 2 == 1):
                values.append(1)
            else:
                values.append(-1)
                
        result_queue.put((states, probs, values))
        result_queue.put(rotate_data(states, probs, values, 1))
        result_queue.put(rotate_data(states, probs, values, 2))
        result_queue.put(rotate_data(states, probs, values, 3))
            
def rotate_data(states, probs, values, k):
    r_states, r_probs, r_values = [], [], []
    
    for state, prob, value in zip(states, probs, values):
        p_2d = prob.reshape(3, 3)
        
        r_state = np.rot90(state, k, axes=(1,2))
        r_prob = np.rot90(p_2d, k)
        
        r_states.append(r_state.copy())
        r_probs.append(r_prob.flatten().copy())
        r_values.append(value)
        
    return (r_states, r_probs, r_values)
            
def train_tictactoe():
    args = {
        'num_iterations': 30,
        'games_per_iteration': 80, 
        'batch_size': 64,
        'num_epochs': 5,
        'device': torch.device('cpu'),
        'num_workers': 8
    }
    model = TTTResNet().to(args['device'])
    if os.path.exists('temp_checkpoint.pt'):
        model.load_state_dict(torch.load('temp_checkpoint.pt'))
    model.share_memory()
    replay_buffer = TTTReplayBuffer(max_size=10000)
    trainer = Trainer(model, args['device'], replay_buffer, lr=0.00001)
    
    for iteration in range(args['num_iterations']):
        print(f'iteration {iteration}')
        model.eval()
        games_added = 0
        print('collecting data')
        
        games_per_worker = args['games_per_iteration'] // args['num_workers']
        games_per_iteration = args['games_per_iteration'] * 4
        with mp.Manager() as manager:
            result_queue = manager.Queue()
            processes = []
            
            for worker_id in range(args['num_workers']):
                p = mp.Process(target=data_worker, args=(model, args['device'], games_per_worker, worker_id, result_queue))
                p.start()
                processes.append(p)
                
            games_added = 0
            while games_added < games_per_iteration:
                states, action_probs, rewards = result_queue.get()
                replay_buffer.add(states, action_probs, rewards)
                games_added += 1
                
                if games_added % 10 == 0 or games_added == games_per_iteration:
                    print(f'Ran {games_added} / {games_per_iteration} games into replay buffer')
                
        for p in processes:
            p.join()
            
        #for i in range(10):
            #random_game = replay_buffer.buffer[random.randint(0, len(replay_buffer.buffer) - 1)]
            #print(f'game: {random_game}')
        
        print('training')
        model.train()
        batches_per_epoch = max(1, len(replay_buffer.buffer) // args['batch_size'])
        for epoch in range(args['num_epochs']):
            epoch_loss = 0
            for batch in range(batches_per_epoch):
                loss = trainer.train(args['batch_size'])
                epoch_loss += loss
            avg_loss = epoch_loss / batches_per_epoch
            print(f'epoch {epoch} loss: {avg_loss}')
                
        torch.save(model.state_dict(), 'temp_checkpoint.pt')
        print('model saved')

if __name__ == "__main__":
    mp.set_start_method('spawn')
    train_tictactoe()                