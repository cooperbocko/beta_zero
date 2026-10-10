import multiprocessing as mp
import time
import os

import torch
import numpy as np

from model import Trainer, ResNet, ReplayBuffer
from game import Outcome, Game, Turn
from mcts import MCTS, MCTSNode


def data_worker(id: int, data_queue: mp.Queue, n_games: int, iterations: int, device: torch.device, model: ResNet, game: Game, temp_moves: int, alpha: float, epsilon: float, data_aug_func = None):
    games = []

    for _ in range(n_games):
        game.reset()
        mcts = MCTS(MCTSNode(game.copy(), 1.0), 1.0, epsilon, alpha)
        turn_list = []
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

        while game.outcome is None:
            # temperature check
            if moves > temp_moves and mcts.temperature >= 0.5:
                mcts.temperature = 0.1

            # noise
            mcts.add_dirichlet_noise()

            # iterations
            for _ in range(iterations):
                mcts.iteration(model, device)

            # get move
            action, valid_probs = mcts.get_move()
            actual_probs = np.zeros(len(game.get_valid_moves()))
            for v_action, prob in valid_probs.items():
                actual_probs[v_action] = prob
            state_list.append(game.get_state())
            probs_list.append(actual_probs)
            turn_list.append(game.turn)
            game.step(action)
            moves += 1

        # get values
        for i in range(len(turn_list)):
            if game.outcome == Outcome.DRAW:
                value_list.append(0)
            elif (game.outcome == Outcome.WIN_1 and turn_list[i] == Turn.PLAYER_1) or (game.outcome == Outcome.WIN_2 and turn_list[i] == Turn.PLAYER_2):
                value_list.append(1)
            else:
                value_list.append(-1)

        # add to list
        games.append((state_list, probs_list, value_list))

        # add augmented data to list, must implement a game's specific data augmentation function
        if data_aug_func is not None:
            augmented_data = data_aug_func(state_list, probs_list)
            for a_states, a_probs in augmented_data:
                games.append((a_states, a_probs, value_list))

    # send results
    data_queue.put(games)

def train(game: Game, n_games: int, n_workers: int, n_iterations: int, temp_moves: int, batch_size: int, mini_batches: int, training_iterations: int, train_device: torch.device, epsilon: float, alpha: float, worker_device: torch.device, model_kwargs: dict, replay_buffer: ReplayBuffer, model_folder: str, data_aug_func = None):
    model = ResNet(**model_kwargs)
    model.to(train_device)
    trainer = Trainer(model, train_device, replay_buffer)
    model_iteration = load_latest(model, trainer.optimizer, model_folder)
    data_queue = mp.Queue() # worker game (states, probs, values) 

    # main training loop
    for _ in range(training_iterations):
        # workers
        workers = []
        for i in range(n_workers):
            worker_model = ResNet(**model_kwargs)
            worker_model.to(worker_device)
            load_latest(worker_model, None, model_folder)
            w = mp.Process(target=data_worker, args=(i, data_queue, n_games, n_iterations, worker_device, worker_model, game, temp_moves, alpha, epsilon, data_aug_func))
            w.start()
            workers.append(w)

        data = []
        while len(data) < n_workers:
            data.append(data_queue.get())

        for w in workers:
            w.join()

        # training
        for worker in data:
            for worker_game in worker:
                states, action_probs, rewards = worker_game
                replay_buffer.add(states, action_probs, rewards)

        # mini batches
        t_loss = 0
        for _ in range(mini_batches):
            t_loss += trainer.train(batch_size)

        print(f'iteration: {model_iteration} avg loss: {t_loss / mini_batches} total loss: {t_loss}')
        model_iteration += 1
        save_model(model, trainer.optimizer, model_iteration, model_folder)

def augment_c4_data(states, probs):
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
