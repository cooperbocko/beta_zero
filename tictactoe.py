from enum import Enum

import numpy as np

from game import Game, Turn, Outcome

class Outcome(int, Enum):
    DRAW = 0
    X_WIN = 1
    O_WIN = 2
    
class Piece(int, Enum):
    EMPTY = 0
    X = 1
    O = 2

class TicTacToe(Game):
    def __init__(self):
        self.turn = Turn.PLAYER_1
        self.outcome = None
        self.n_moves = 0
        self.board = np.zeros((3, 3), dtype=np.int8)
        
    def copy(self):
        temp = TicTacToe()
        temp.turn = self.turn
        temp.outcome = self.outcome
        temp.board = np.copy(self.board)
        return temp
    
    def step(self, action: int):
        position = self.get_move(action)
        self.move(position)
        self.n_moves += 1
        if self.check_win(position):
            return False
        if self.n_moves == 9:
            self.outcome = Outcome.DRAW
            return False

        return True
    
    def move(self, position):
        self.board[position[0]][position[1]] = self.turn
        self.turn = Turn.PLAYER_2 if self.turn == Turn.PLAYER_1 else Turn.PLAYER_1
    
    def get_state(self):
        if self.turn == Turn.PLAYER_1:
            player = (self.board == Piece.X).astype(np.float32)
            opponent = (self.board == Piece.O).astype(np.float32)
            turn = np.zeros((3, 3))
        else:
            player = (self.board == Piece.O).astype(np.float32)
            opponent = (self.board == Piece.X).astype(np.float32)
            turn = np.ones((3, 3))
        return np.stack((player, opponent, turn))
    
    def get_valid_moves(self):
        moves = [0 for _ in range(3 * 3)]
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Piece.EMPTY:
                    moves[i * 3 + j] = 1
        return moves
    
    def get_move(self, logit: int):
        row = logit // 3
        col = logit - row * 3
        return (row, col)
    
    def print_board(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Piece.X:
                    print("X", end = " ")
                elif self.board[i][j] == Piece.O:
                    print("O", end = " ")
                else:
                    print("-", end = " ")
            print()
    
    def check_win(self, position):
        r, c = position
        piece = self.board[r][c]
        
        if all(self.board[r][i] == piece for i in range(3)):
            self.outcome = piece
            return True
        if all(self.board[i][c] == piece for i in range(3)):
            self.outcome = piece
            return True
        if r == c and all(self.board[i][i] == piece for i in range(3)):
            self.outcome = piece
            return True
        if r + c == 2 and all(self.board[i][2 - i] == piece for i in range(3)):
            self.outcome = piece
            return True
        
        return False