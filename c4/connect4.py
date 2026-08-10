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
        player = np.zeros((6, 7))
        opponent = np.zeros((6, 7))
        if self.turn == Turn.PLAYER_1:
            player = (self.board == Piece.X).astype(np.float32)
            opponent = (self.board == Piece.O).astype(np.float32)
        else:
            player = (self.board == Piece.O).astype(np.float32)
            opponent = (self.board == Piece.X).astype(np.float32)
        return np.stack([player, opponent])
    
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
            down_diag += 1
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
    
class Connect4BitBoard():
    BOTTOM_MASK = 0x40810204081
    TOP_MASK = 0x810204081020
    '''
    Bitboard representation
    0 | 0 | 0 | 0 | 0 | 0 | 0
    05| 12| 19| 26| 33| 40| 47
    04| 11| 18| 25| 32| 39| 46
    03| 10| 17| 24| 31| 38| 45
    02| 09| 16| 23| 30| 37| 44
    01| 08| 15| 22| 29| 36| 43
    00| 07| 14| 21| 28| 35| 42
    '''
    def __init__(self):
        self.position = 0
        self.mask = 0
        self.turn = Turn.PLAYER_1
        self.outcome = None
        
    def step(self, action: int):
        if self.outcome is not None:
            return
        
        self.move(action)
        if self.check_win() or self.check_draw():
            return
        
        self.switch_player()
        return True
    
    def copy(self):
        temp = Connect4BitBoard()
        temp.position = self.position
        temp.mask = self.mask
        temp.turn = self.turn
        temp.outcome = self.outcome
        return temp
    
    def get_state(self):
        player = np.zeros((6, 7))
        opponent = np.zeros((6, 7))
        
        for i in range(5, -1, -1):
            for j in range(7):
                if self.position & (1 << (j * 7 + i)):
                    player[i][j] = 1
                elif self.mask & (1 << (j * 7 + i)):
                    opponent[i][j] = 1
        return np.stack([player, opponent])
        
    def move(self, column):
        bottom_bit = 1 << (column * 7)
        old_mask = self.mask
        self.mask |= (bottom_bit + self.mask)
        new_bit = old_mask ^ self.mask
        self.position |= new_bit
        
    def print_board(self):
        for i in range(5, -1, -1):
            for j in range(7):
                if self.position & (1 << (j * 7 + i)):
                    if self.turn == Turn.PLAYER_1:
                        print("X", end = " ")
                    else:
                        print("O", end = " ")
                elif self.mask & (1 << (j * 7 + i)):
                    if self.turn == Turn.PLAYER_1:
                        print("O", end = " ")
                    else:
                        print("X", end = " ")
                else:
                    print("-", end = " ")
            print()
    
    def switch_player(self):
        self.position ^= self.mask
        self.turn = Turn.PLAYER_2 if self.turn == Turn.PLAYER_1 else Turn.PLAYER_1
        
    def get_valid_moves(self):
        return [0 if self.mask & (1 << (5 + (i * 7))) else 1 for i in range(7)]
    
    def check_draw(self):
        if (self.mask & Connect4BitBoard.TOP_MASK) == Connect4BitBoard.TOP_MASK:
            self.outcome = Outcome.DRAW
            return True
        return False
    
    def check_win(self):
        #vertical
        two_in_a_row = self.position & self.position << 1
        two_in_a_row = two_in_a_row & self.mask
        four_in_a_row = two_in_a_row & two_in_a_row << 2
        if four_in_a_row:
            self.outcome = Outcome.WIN_1 if self.turn == Turn.PLAYER_1 else Outcome.WIN_2
            return True
        
        #(((self.position & self.position << 1) & self.mask) & (((self.position & self.position << 1) & self.mask) << 2))
        
        #horizontal
        two_in_a_row = self.position & self.position << 7
        two_in_a_row = two_in_a_row & self.mask
        four_in_a_row = two_in_a_row & two_in_a_row << 14 
        if four_in_a_row:
            self.outcome = Outcome.WIN_1 if self.turn == Turn.PLAYER_1 else Outcome.WIN_2
            return True
        
        #down diagonal
        two_in_a_row = self.position & self.position << 6
        two_in_a_row = two_in_a_row & self.mask
        four_in_a_row = two_in_a_row & two_in_a_row << 12
        if four_in_a_row:
            self.outcome = Outcome.WIN_1 if self.turn == Turn.PLAYER_1 else Outcome.WIN_2
            return True
        
        #up diagonal
        two_in_a_row = self.position & self.position << 8
        two_in_a_row = two_in_a_row & self.mask
        four_in_a_row = two_in_a_row & two_in_a_row << 16
        if four_in_a_row:
            self.outcome = Outcome.WIN_1 if self.turn == Turn.PLAYER_1 else Outcome.WIN_2
            return True
        
        return False