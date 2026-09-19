import multiprocessing as mp
from dataclasses import dataclass
import queue
import time

import numpy as np
import torch

from c4.c4_mcts_node import C4MCTSNode
from c4.connect4 import Connect4BitBoard
from game import Outcome
from mcts import MCTS
from model import ResNet  

@dataclass
class InferenceRequest:
    worker_id: int
    game_id: int
    state: np.ndarray

@dataclass
class InferenceResponse:
    worker_id: int
    game_id: int
    probs: np.ndarray
    value: float

class InferenceWorker:
    def __init__(self, request_queues, result_queues, device, batch_size, model):
        self.request_queues = request_queues
        self.result_queues = result_queues
        self.device = device
        self.batch_size = batch_size
        self.model = model.to(device)
        self.model.eval()

    def run(self):
        while True:
            requests = self.collect_requests()

            if requests is None:
                break

            self.infer(requests)

    def infer(self, requests):
        states = np.stack([request.state for request in requests])
        states = torch.from_numpy(states).float().to(self.device)

        with torch.inference_mode():
            probs, values = self.model(states)

        probs = probs.cpu().numpy()
        values = values.cpu().numpy()

        for i, request in enumerate(requests):
            result = InferenceResponse(
                game_id=request.game_id,
                worker_id=request.worker_id,
                probs=probs[i],
                value=float(values[i].item()),
            )

            self.result_queues[request.worker_id].put(result)

    def collect_requests(self):
        requests = []

        while not requests:
            for worker_id, request_queue in enumerate(self.request_queues):
                try:
                    request = request_queue.get_nowait()

                    if request is None:
                        return None

                    requests.append(request)
                    if len(requests) >= self.batch_size:
                        return requests
                    
                except queue.Empty:
                    continue

        for worker_id, request_queue in enumerate(self.request_queues):
            while len(requests) < self.batch_size:
                try:
                    request = request_queue.get_nowait()

                    if request is None:
                        return None

                    requests.append(request)

                except queue.Empty:
                    break

        return requests

class Game:
    def __init__(self, game_id: int):
        self.board = Connect4BitBoard()
        self.mcts = MCTS(C4MCTSNode(Connect4BitBoard(), 1), 1, True)
        self.states = []
        self.probs = []
        self.values = []
        self.finished = False
        self.waiting_for_inference = False
        self.waiting_node = None
        self.waiting_path = None
        self.iterations = 0
        self.noise = False
        self.game_id = game_id

class GameWorker:
    def __init__(self, worker_id, n_games, n_iterations, inference_queue, result_queue, max_g):
        self.worker_id = worker_id
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.t_table = {} # [(position, mask)] : (probs, value)
        self.n_iterations = n_iterations
        self.games = [Game(i) for i in range(n_games)]
        self.games_done = []
        self.max_g = max_g

    def run(self):
        while len(self.games_done) < self.max_g:
            self.process_inference_results()
            requests = []
            for i, game in enumerate(self.games):
                # play iteration or make move until leaf node is found or game is over
                while not game.finished and not game.waiting_for_inference:
                    if game.iterations < self.n_iterations:
                        request = self.iteration(game)
                        if request:
                            requests.append(request)
                            break
                    else:
                        self.make_move(game)

                if game.finished:
                    self.games_done.append(game)
                    self.games[i] = Game(i)

            if requests:
                for request in requests:
                    self.inference_queue.put(request)

    def process_inference_results(self):
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break

            
            game = self.games[result.game_id]
            self.apply_inference(game, result.probs, result.value)

    def make_move(self, game: Game):
        if game.iterations < self.n_iterations or game.finished:
            return

        # get move and probs
        action, valid_action_probs = game.mcts.get_move()
        actual_probs = np.zeros(9)
        for v_action, prob in valid_action_probs.items():
            actual_probs[v_action] = prob

        # add training data and make move
        game.states.append(game.board.get_state())
        game.probs.append(actual_probs)
        game.board.step(action)
        game.iterations = 0
        game.noise = False

        if game.board.outcome is not None:
            game.finished = True
            for i, _ in enumerate(game.states):
                if game.board.outcome == Outcome.DRAW:
                    game.values.append(0)
                elif (game.board.outcome == Outcome.WIN_1 and i % 2 == 0) or (game.board.outcome == Outcome.WIN_2 and i % 2 == 1):
                    game.values.append(1)
                else:
                    game.values.append(-1)

    def iteration(self, game: Game):
        if game.waiting_for_inference:
            return None

        # add noise for first selection
        if not game.noise:
            game.mcts.add_dirlect_noise()
            game.noise = True

        game.iterations += 1

        # select node for potential inference
        node, path = game.mcts.select()

        # if game is over, backpropagate
        value = game.mcts.get_terminal_value(node)
        if value:
            game.mcts.backpropagate(path, value)
            return None

        # check cache for state
        cache = self.t_table.get((node.state.position, node.state.mask))
        if cache is None:
            game.waiting_for_inference = True
            game.waiting_node = node
            game.waiting_path = path

            return InferenceRequest(self.worker_id, game.game_id, node.state.get_state())

        # expand
        probs, value = cache
        game.mcts.expand(node, probs)

        # backprop
        game.mcts.backpropagate(path, value)
        return None

    def apply_inference(self, game: Game, probs, value):
        # add to cache
        self.t_table[(game.waiting_node.state.position, game.waiting_node.state.mask)] = (probs, value)

        # expand
        game.mcts.expand(game.waiting_node, probs)

        # backprop
        game.mcts.backpropagate(game.waiting_path, value)

        game.waiting_for_inference = False
        game.waiting_node = None
        game.waiting_path = None

def inference_process_fn(
    request_queues,
    result_queues,
    batch_size,
):
    device = torch.device("cuda")
    model = ResNet(input_channels=2, n_blocks=10, n_channels=128, value_layers=64, n_actions=7, board_size=6*7)
    inference_worker = InferenceWorker(
        request_queues,
        result_queues,
        device,
        batch_size,
        model
    )

    inference_worker.run()

def worker_process(
    worker_id,
    request_queue,
    result_queue,
    n_games,
    n_iterations,
    max_g
):
    worker = GameWorker(
        worker_id=worker_id,
        inference_queue=request_queue,
        result_queue=result_queue,
        n_games=n_games,
        n_iterations=n_iterations,
        max_g=max_g
    )

    worker.run()

def main():

    n_workers = 10
    n_games_per_worker = 30
    n_iterations = 300
    batch_size = 32
    max_g = 30

    request_queues = [
        mp.Queue()
        for _ in range(n_workers)
    ]

    result_queues = [
        mp.Queue()
        for _ in range(n_workers)
    ]

    # Inference process
    inference_process = mp.Process(
        target=inference_process_fn,
        args=(
            request_queues,
            result_queues,
            batch_size,
        ),
    )

    inference_process.start()

    # MCTS worker processes
    processes = []

    for worker_id in range(n_workers):

        process = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                request_queues[worker_id],
                result_queues[worker_id],
                n_games_per_worker,
                n_iterations,
                max_g
            ),
        )

        process.start()
        processes.append(process)

    # Wait for workers
    for process in processes:
        process.join()

    # Shut down inference
    for request_queue in request_queues:
        request_queue.put(None)

    inference_process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    print("Starting")
    start = time.time()
    main()
    print(f"Finished in {time.time() - start}")

