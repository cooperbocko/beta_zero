from enum import Enum

import numpy as np

from game import Game, Turn, Outcome

class Piece(int, Enum):
    EMPTY = 0
    X = 1
    O = 2

class Connect4(Game):
    def __init__(self):
        self.turn = Turn.PLAYER_1
        self.outcome = None
        self.n_moves = 0
        self.board = np.zeros((6, 7), dtype=np.int8)
        
    def copy(self):
        temp = Connect4()
        temp.turn = self.turn
        temp.outcome = self.outcome
        temp.board = np.copy(self.board)
        temp.n_moves = self.n_moves
        return temp
    
    def step(self, action: int):
        position = self.move(action)
        self.n_moves += 1
        if self.check_win(position):
            return False
        if self.n_moves == 42:
            self.outcome = Outcome.DRAW
            return False

        return True
    
    def move(self, action: int):
        r = 5
        while r >= 0 and self.board[r][action] != Piece.EMPTY:
            r -= 1
        if self.turn == Turn.PLAYER_1:
            self.board[r][action] = Piece.X
            self.turn = Turn.PLAYER_2
        else:
            self.board[r][action] = Piece.O
            self.turn = Turn.PLAYER_1
            
        return (r, action)
    
    def get_state(self):
        pass
    
    def get_valid_moves(self):
        moves = [0 for _ in range(7)]
        for i in range(7):
            if self.board[0][i] == Piece.EMPTY:
                moves[i] = 1
        return moves
    
    def print_board(self):
        for i in range(6):
            for j in range(7):
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
        
        vertical = 0
        r_temp = r
        while r_temp >= 0 and self.board[r_temp][c] == piece:
            r_temp -= 1
            vertical += 1
        r_temp = r + 1
        while r_temp < 6 and self.board[r_temp][c] == piece:
            r_temp += 1
            vertical += 1
        if vertical >= 4:
            self.outcome = Outcome.WIN_1 if piece == Piece.X else Outcome.WIN_2
            return True
        
        horizontal = 0
        c_temp = c
        while c_temp >= 0 and self.board[r][c_temp] == piece:
            c_temp -= 1
            horizontal += 1
        c_temp = c + 1
        while c_temp < 7 and self.board[r][c_temp] == piece:
            c_temp += 1
            horizontal += 1
        if horizontal >= 4:
            self.outcome = Outcome.WIN_1 if piece == Piece.X else Outcome.WIN_2
            return True
        
        down_diag = 0
        r_temp = r
        c_temp = c
        while r_temp >= 0 and c_temp >= 0 and self.board[r_temp][c_temp] == piece:
            r_temp -= 1
            c_temp -= 1
            down_diag += 1
        r_temp = r + 1
        c_temp = c + 1
        while r_temp < 6 and c_temp < 7 and self.board[r_temp][c_temp] == piece:
            r_temp += 1
            c_temp += 1
            up_diag += 1
        if down_diag >= 4:
            self.outcome = Outcome.WIN_1 if piece == Piece.X else Outcome.WIN_2
            return True
        
        up_diag = 0
        r_temp = r
        c_temp = c
        while r_temp < 6 and c_temp >= 0 and self.board[r_temp][c_temp] == piece:
            r_temp += 1
            c_temp -= 1
            up_diag += 1
        r_temp = r - 1
        c_temp = c + 1
        while r_temp >= 0 and c_temp < 7 and self.board[r_temp][c_temp] == piece:
            r_temp -= 1
            c_temp += 1
            up_diag += 1
        if up_diag >= 4:
            self.outcome = Outcome.WIN_1 if piece == Piece.X else Outcome.WIN_2
            return True
        
        return False