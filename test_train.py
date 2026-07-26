

from tictactoe import TicTacToe
from mcts import TTTMCTSNode, MCTS
from model import TTTResNet, TTTReplayBuffer, Trainer


game = TicTacToe()
model = TTTResNet()
replay_buffer = TTTReplayBuffer

mcts = MCTS()