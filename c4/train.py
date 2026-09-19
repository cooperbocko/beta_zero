import threading
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass
from concurrent.futures import Future
import time

import torch
import numpy as np

from model import Trainer, ResNet
from game import Outcome
from mcts import MCTS
from c4.c4_mcts_node import C4MCTSNode
from c4.c4_replay_buffer import C4ReplayBuffer
from c4.connect4 import Connect4BitBoard
from c4.transposition_table import TranspositionTable

@dataclass
class InferenceRequest:
    future: Future
    state: np.ndarray
    
def request_inference(request_queue: queue.Queue, state: np.ndarray):
    future = Future()
    request = InferenceRequest(future=future, state=state)
    request_queue.put(request)
    
    return future.result()

def inference_loop(model: ResNet, request_queue: queue.Queue, device: torch.device, batch_size: int):
    total_requests = 0
    total_batches = 0
    total_batch_size = 0
    total_wait = 0
    total_inference = 0
    transfer_in = 0
    transfer_out = 0
    model_time = 0
    while True:
        start_wait = time.perf_counter()

        wait_time = time.perf_counter() - start_wait
        total_wait += wait_time

        request = request_queue.get()
        if request is None:
            break
        
        batch = [request]
        deadline = time.monotonic() + 0.00025
        while len(batch) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            
            try:
                request = request_queue.get(timeout=remaining)
                batch.append(request)
            except queue.Empty:
                break

        total_batches += 1
        total_requests += len(batch)
        total_batch_size += len(batch)

        start = time.monotonic()
        model_states = np.stack([request.state for request in batch])
        state_tensors = torch.from_numpy(model_states).float()
        state_tensors = state_tensors.to(device)
        transfer_in += time.monotonic() - start

        start = time.monotonic()

        with torch.inference_mode():
            probs, values = model(state_tensors)

        model_time += time.monotonic() - start

        start = time.monotonic()
        
        probs = probs.detach().cpu().numpy()
        values = values.detach().cpu().numpy()
        torch.cuda.synchronize()

        transfer_out += time.monotonic() - start

        total_inference += time.perf_counter() - start
            
        for request, prob, value in zip(batch, probs, values):
            request.future.set_result((prob, value))

    print("requests:", total_requests)
    print("batches:", total_batches)
    print("avg batch:", total_requests / total_batches)
    print("inference:", total_inference)
    print("queue wait:", total_wait)
    print("transfer in:", transfer_in)
    print("transfer out:", transfer_out)
    print("model time:", model_time)
            
def process_worker(n_threads: int, batch_size: int, n_games: int, n_iterations: int):
    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    #inference thread
    model = ResNet(input_channels=2, n_blocks=8, n_channels=128, value_layers=64, n_actions=7, board_size=6*7)
    device = torch.device('cuda')
    model.to(device)
    if os.path.exists('c4_temp_checkpoint.pt'):
        model.load_state_dict(torch.load('c4_temp_checkpoint.pt'))
    model.eval()
    inference_queue = queue.Queue()
    inference_thread = threading.Thread(target=inference_loop, args=(model, inference_queue, device, batch_size))
    inference_thread.start()
    
    #game threads
    t_table = {} # [(position, mask)] : (probs, value)
    cache_hits = 0
    cache_misses = 0

    threads = []
    for _ in range(n_threads):
        t = threading.Thread(target=play_game, args=(result_queue, inference_queue, n_iterations, stop_event, t_table, cache_hits, cache_misses))
        t.start()
        threads.append(t)
        
    #cleanup
    while len(result_queue.queue) < n_games:
        print(f'waiting for {n_games - len(result_queue.queue)} games')
        time.sleep(10)
        
    print(f'ran {len(result_queue.queue)} games')
    stop_event.set()
    for t in threads:
        t.join()   
    inference_queue.put(None)
    inference_thread.join()

def play_game(result_queue: queue.Queue, inference_queue: queue.Queue, n_iterations: int, stop_event: threading.Event, t_table, c_hit, c_miss):
    t_table = t_table
    c_hit = c_hit
    c_miss = c_miss
    select_time = 0
    inference_time = 0
    expand_time = 0
    backprop_time = 0
    while not stop_event.is_set():
        c4 = Connect4BitBoard()
        mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)
        states_history = []
        probs_history = []
        values_history = []
        steps = 0
        
        while c4.outcome is None and not stop_event.is_set():
            #mcts search
            mcts.add_dirlect_noise()
            for i in range(n_iterations):
                start = time.perf_counter()
                node, path = mcts.select()
                select_time += time.perf_counter() - start
                
                if node.state.outcome is not None:
                    if node.state.outcome == Outcome.DRAW:
                        value = 0
                    elif node.state.turn == node.state.outcome:
                        value = 1
                    else:
                        value = -1
                else:
                    # look at t table

                    res = t_table.get((node.state.position, node.state.mask))
                    if res:
                        probs, value = res
                        c_hit += 1
                    else:   
                        probs, value = request_inference(inference_queue, node.state.get_state())
                        t_table[(node.state.position, node.state.mask)] = (probs, value)
                        c_miss += 1

                    inference_time += time.perf_counter() - start

                    start = time.perf_counter()
                    mcts.expand(node, probs)
                    expand_time += time.perf_counter() - start

                start = time.perf_counter()
                mcts.backpropagate(path, value)
                backprop_time += time.perf_counter() - start
                
            #step
            action, v_probs = mcts.get_move()
            states_history.append(c4.get_state())
            actual_probs = np.zeros(9)
            for v_action, prob in v_probs.items():
                actual_probs[v_action] = prob
            probs_history.append(actual_probs)
            c4.step(action)
            steps += 1
            
        #send game
        if not stop_event.is_set():
            for i, state in enumerate(states_history):
                if c4.outcome == Outcome.DRAW:
                    values_history.append(0)
                elif (c4.outcome == Outcome.WIN_1 and i % 2 == 0) or (c4.outcome == Outcome.WIN_2 and i % 2 == 1):
                    values_history.append(1)
                else:
                    values_history.append(-1)
                    
            result_queue.put((states_history, probs_history, values_history))
            f_states, f_probs = flip_data(states_history, probs_history)
            result_queue.put((f_states, f_probs, values_history))

    print(f'cache hits: {c_hit} cache misses: {c_miss}')
    print(f'select time: {select_time} inference time: {inference_time} expand time: {expand_time} backprop time: {backprop_time}')
        
def flip_data(states, probs):
    f_states, f_probs = [], []
    
    for state, prob in zip(states, probs):
        f_state = np.fliplr(state)
        f_prob = prob[::-1]
        
        f_states.append(f_state.copy())
        f_probs.append(f_prob.flatten().copy())
        
    return (f_states, f_probs)
        
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print('starting!')

    # 16 threads 8 batch size 100 games 600 iterations
    time1 = time.time()
    workers = []
    for _ in range(1):
        p = mp.Process(target=process_worker, args=(8, 8, 30, 300))
        p.start()
        workers.append(p)
            
    for p in workers:
        p.join()
            
    print(f'8 threads, 8 batch time: {time.time() - time1}')
'''
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    model = ResNet(input_channels=2, n_blocks=5, n_channels=64, value_layers=32, n_actions=7, board_size=6*7)
    device = torch.device('cuda')
    model.to(device)

    for batch_size in batch_sizes:
        states = torch.randn(batch_size, 2, 6, 7, device=device)
        # warmup
        for _ in range(100):
            with torch.no_grad():
                model(states)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(10000):
            with torch.no_grad():
                probs, values = model(states)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        print(f'{batch_size} batch time: {elapsed} elapsed')
        print("batches/sec:", 10000 / elapsed)
        print("states/sec:", 80000 / elapsed)'''

def data_worker():
    pass
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