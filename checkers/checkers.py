from enum import Enum

import numpy as np

class Piece(int, Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    K_BLACK = 3
    K_WHITE = 4

class Move(Enum):
    UPLEFT = 0
    UPRIGHT = 1
    DOWNLEFT = 2
    DOWNRIGHT = 3
    
class Outcome(int, Enum):
    DRAW = 0
    B_WIN = 1
    W_WIN = 2

#Note: American Checkers Rules
class Checkers:
    def __init__(self):
        self.outcome = None
        self.turn = Piece.BLACK
        self.prev_turn = None
        self.prev_jump = None
        self.w_score = 0
        self.b_score = 0
        self.draw_moves = 0
        self.max_moves = 40
        self.board = np.zeros((8, 8), dtype=np.int8)
        for i in range(3):
            for j in range(8):
                if i % 2 == 0 and j % 2 == 1:
                    self.board[i][j] = Piece.WHITE
                elif i % 2 == 1 and j % 2 == 0:
                    self.board[i][j] = Piece.WHITE
        for i in range(5, 8):
            for j in range(8):
                if i % 2 == 0 and j % 2 == 1:
                    self.board[i][j] = Piece.BLACK
                elif i % 2 == 1 and j % 2 == 0:
                    self.board[i][j] = Piece.BLACK
                    
    def copy(self):
        temp = Checkers()
        temp.outcome = self.outcome
        temp.turn = self.turn
        temp.prev_turn = self.prev_turn
        temp.prev_jump = self.prev_jump
        temp.w_score = self.w_score
        temp.b_score = self.b_score
        temp.draw_moves = self.draw_moves
        temp.max_moves = self.max_moves
        temp.board = np.copy(self.board)
        return temp
                    
    def step(self, action: int):
        move = self.get_move(action)
        #can_jump = self.can_jump()
        is_jump = self.is_jump(move[0], move[1])
        
        #if not self.check_valid_move(move[0], move[1], can_jump, is_jump):
            #return False
        self.move(move[0], move[1], is_jump)
        
        if self.check_b_win() or self.check_w_win() or self.check_draw():
            return False
        
        return True
    
    def get_state(self):
        if self.turn == Piece.WHITE:
            player = (self.board == Piece.WHITE).astype(np.float32)
            opponent = (self.board == Piece.BLACK).astype(np.float32)
            player_k = (self.board == Piece.K_WHITE).astype(np.float32)
            opponent_k = (self.board == Piece.K_BLACK).astype(np.float32)
            turn = np.zeros((8, 8))
        else:
            player = (self.board == Piece.BLACK).astype(np.float32)
            opponent = (self.board == Piece.WHITE).astype(np.float32)
            player_k = (self.board == Piece.K_BLACK).astype(np.float32)
            opponent_k = (self.board == Piece.K_WHITE).astype(np.float32)
            turn = np.ones((8, 8))
        return np.stack((player, opponent, player_k, opponent_k, turn))
    
    #Moves: 8x8x4 order -> up left, up right, down left, down right
    def get_valid_moves(self) -> list[int]:
        moves = [0 for _ in range(8 * 8 * 4)]
        if self.outcome is not None:
            return moves
        
        can_jump = self.can_jump()
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece == Piece.EMPTY or self.turn == Piece.WHITE and (piece == Piece.BLACK or piece == Piece.K_BLACK) or self.turn == Piece.BLACK and (piece == Piece.WHITE or piece == Piece.K_WHITE):
                    continue
                
                if piece == Piece.BLACK or piece == Piece.WHITE:
                    if self.check_valid_move((i, j), Move.UPLEFT, can_jump, self.is_jump((i, j), Move.UPLEFT)):
                        moves[i * 8 * 4 + j * 4 + 0] = 1
                    if self.check_valid_move((i, j), Move.UPRIGHT, can_jump, self.is_jump((i, j), Move.UPRIGHT)):
                        moves[i * 8 * 4 + j * 4 + 1] = 1
                else:
                    if self.check_valid_move((i, j), Move.UPLEFT, can_jump, self.is_jump((i, j), Move.UPLEFT)):
                        moves[i * 8 * 4 + j * 4 + 0] = 1
                    if self.check_valid_move((i, j), Move.UPRIGHT, can_jump, self.is_jump((i, j), Move.UPRIGHT)):
                        moves[i * 8 * 4 + j * 4 + 1] = 1
                    if self.check_valid_move((i, j), Move.DOWNLEFT, can_jump, self.is_jump((i, j), Move.DOWNLEFT)):
                        moves[i * 8 * 4 + j * 4 + 2] = 1
                    if self.check_valid_move((i, j), Move.DOWNRIGHT, can_jump, self.is_jump((i, j), Move.DOWNRIGHT)):
                        moves[i * 8 * 4+ j * 4 + 3] = 1
        return moves
    
    def get_move(self, logit: int) -> tuple[tuple[int, int], Move]:
        row = logit // (8 * 4)
        col = (logit - row * 8 * 4) // 4
        move = logit - row * 8 * 4 - col * 4
        return ((row, col), Move(move))
    
    def print_board(self):
        print("  0 1 2 3 4 5 6 7")
        for i in range(8):
            print(i, end = " ")
            for j in range(8):
                if self.board[i][j] == Piece.BLACK:
                    print("b", end = " ")
                elif self.board[i][j] == Piece.WHITE:
                    print("w", end = " ")
                elif self.board[i][j] == Piece.K_BLACK:
                    print("B", end = " ")
                elif self.board[i][j] == Piece.K_WHITE:
                    print("W", end = " ")
                else:
                    print("-", end = " ")
            print()
        print(f"B: {self.b_score} W: {self.w_score} Turn: {'W' if self.turn == Piece.WHITE else 'B'}")
            
    def move(self, position: tuple[int, int], move: Move, is_jump: bool):      
        piece = self.board[position[0]][position[1]]
        self.board[position[0]][position[1]] = Piece.EMPTY
        if is_jump:
            if move == Move.UPLEFT:
                taken_piece = self.board[position[0] - 1][position[1] - 1]
                self.board[position[0] - 1][position[1] - 1] = Piece.EMPTY
                self.board[position[0] - 2][position[1] - 2] = piece
                new_position = (position[0] - 2, position[1] - 2)
            elif move == Move.UPRIGHT:
                taken_piece = self.board[position[0] - 1][position[1] + 1]
                self.board[position[0] - 1][position[1] + 1] = Piece.EMPTY
                self.board[position[0] - 2][position[1] + 2] = piece
                new_position = (position[0] - 2, position[1] + 2)
            elif move == Move.DOWNLEFT:
                taken_piece = self.board[position[0] + 1][position[1] - 1]
                self.board[position[0] + 1][position[1] - 1] = Piece.EMPTY
                self.board[position[0] + 2][position[1] - 2] = piece
                new_position = (position[0] + 2, position[1] - 2)
            elif move == Move.DOWNRIGHT:
                taken_piece = self.board[position[0] + 1][position[1] + 1]
                self.board[position[0] + 1][position[1] + 1] = Piece.EMPTY
                self.board[position[0] + 2][position[1] + 2] = piece
                new_position = (position[0] + 2, position[1] + 2)
            
            if taken_piece == Piece.BLACK or taken_piece == Piece.K_BLACK:
                self.w_score += 1
            elif taken_piece == Piece.WHITE or taken_piece == Piece.K_WHITE:
                self.b_score += 1
                
            if self.is_jump(new_position, Move.UPLEFT) or self.is_jump(new_position, Move.UPRIGHT) or self.is_jump(new_position, Move.DOWNLEFT) or self.is_jump(new_position, Move.DOWNRIGHT):
                self.prev_jump = new_position
                self.prev_turn = self.turn
            else:
                self.prev_jump = None
                self.prev_turn = self.turn
                self.turn = Piece.WHITE if self.turn == Piece.BLACK else Piece.BLACK
        else:
            if move == Move.UPLEFT:
                self.board[position[0] - 1][position[1] - 1] = piece
                new_position = (position[0] - 1, position[1] - 1)
            elif move == Move.UPRIGHT:
                self.board[position[0] - 1][position[1] + 1] = piece
                new_position = (position[0] - 1, position[1] + 1)
            elif move == Move.DOWNLEFT:
                self.board[position[0] + 1][position[1] - 1] = piece
                new_position = (position[0] + 1, position[1] - 1)
            elif move == Move.DOWNRIGHT:
                self.board[position[0] + 1][position[1] + 1] = piece
                new_position = (position[0] + 1, position[1] + 1)
            self.prev_turn = self.turn
            self.turn = Piece.WHITE if self.turn == Piece.BLACK else Piece.BLACK
        
        if piece == Piece.BLACK and new_position[0] == 0:
            self.board[new_position[0]][new_position[1]] = Piece.K_BLACK
        elif piece == Piece.WHITE and new_position[0] == 0:
            self.board[new_position[0]][new_position[1]] = Piece.K_WHITE
            
        if piece == Piece.BLACK or piece == Piece.WHITE or is_jump:
            self.draw_moves = 0
        else:
            self.draw_moves += 1
            
        if self.turn != self.prev_turn:
            self.board = np.flip(self.board) #flip perspective on turn change
        
        return True
            
    def check_valid_move(self, position: tuple[int, int], move: Move, jump: bool, is_jump: bool):
        #No move for empty space
        piece = self.board[position[0]][position[1]]
        if piece == Piece.EMPTY:
            return False
        
        #Can only move your pieces
        if (piece == Piece.WHITE or piece == Piece.K_WHITE) and self.turn != Piece.WHITE:
            return False
        if (piece == Piece.BLACK or piece == Piece.K_BLACK) and self.turn != Piece.BLACK:
            return False
        
        #Jump chain
        if self.prev_jump: 
            if self.prev_jump != position:
                return False
            
        #Must take jump
        if jump and not is_jump:
            return False
        elif jump and is_jump:
            return True
        
        #Regular moves
        if piece == Piece.BLACK or piece == Piece.WHITE:
            if move == Move.DOWNLEFT or move == Move.DOWNRIGHT:
                return False
            elif move == Move.UPLEFT:
                if position[0] - 1 < 0 or position[1] - 1 < 0:
                    return False
                if self.board[position[0] - 1][position[1] - 1] != Piece.EMPTY:
                    return False
            elif move == Move.UPRIGHT:
                if position[0] - 1 < 0 or position[1] + 1 >= 8:
                    return False
                if self.board[position[0] - 1][position[1] + 1] != Piece.EMPTY:
                    return False
        elif piece == Piece.K_BLACK or piece == Piece.K_WHITE:
            if move == Move.UPLEFT:
                if position[0] - 1 < 0 or position[1] - 1 < 0:
                    return False
                if self.board[position[0] - 1][position[1] - 1] != Piece.EMPTY:
                    return False
            elif move == Move.UPRIGHT:
                if position[0] - 1 < 0 or position[1] + 1 >= 8:
                    return False
                if self.board[position[0] - 1][position[1] + 1] != Piece.EMPTY:
                    return False
            elif move == Move.DOWNLEFT:
                if position[0] + 1 >= 8 or position[1] - 1 < 0:
                    return False
                if self.board[position[0] + 1][position[1] - 1] != Piece.EMPTY:
                    return False
            elif move == Move.DOWNRIGHT:
                if position[0] + 1 >= 8 or position[1] + 1 >= 8:
                    return False
                if self.board[position[0] + 1][position[1] + 1] != Piece.EMPTY:
                    return False
        return True
    
    def is_jump(self, position: tuple[int, int], move: Move):
        piece = self.board[position[0]][position[1]]
        if piece == Piece.EMPTY:
            return False
        i = position[0]
        j = position[1]
        
        up_left = (i - 1, j - 1) if i - 1 >= 0 and j - 1 >= 0 else None
        up_left_2 = (i - 2, j - 2) if i - 2 >= 0 and j - 2 >= 0 else None
        up_right = (i - 1, j + 1) if i - 1 >= 0 and j + 1 < 8 else None
        up_right_2 = (i - 2, j + 2) if i - 2 >= 0 and j + 2 < 8 else None
        down_left = (i + 1, j - 1) if i + 1 < 8 and j - 1 >= 0 else None
        down_left_2 = (i + 2, j - 2) if i + 2 < 8 and j - 2 >= 0 else None
        down_right = (i + 1, j + 1) if i + 1 < 8 and j + 1 < 8 else None
        down_right_2 = (i + 2, j + 2) if i + 2 < 8 and j + 2 < 8 else None
        
        if piece == Piece.BLACK:
            if move == Move.UPLEFT:
                if up_left and (self.board[up_left[0]][up_left[1]] == Piece.WHITE or self.board[up_left[0]][up_left[1]] == Piece.K_WHITE) and up_left_2 and self.board[up_left_2[0]][up_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.UPRIGHT:
                if up_right and (self.board[up_right[0]][up_right[1]] == Piece.WHITE or self.board[up_right[0]][up_right[1]] == Piece.K_WHITE) and up_right_2 and self.board[up_right_2[0]][up_right_2[1]] == Piece.EMPTY:
                    return True
        elif piece == Piece.WHITE:
            if move == Move.UPLEFT:
                if up_left and (self.board[up_left[0]][up_left[1]] == Piece.BLACK or self.board[up_left[0]][up_left[1]] == Piece.K_BLACK) and up_left_2 and self.board[up_left_2[0]][up_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.UPRIGHT:
                if up_right and (self.board[up_right[0]][up_right[1]] == Piece.BLACK or self.board[up_right[0]][up_right[1]] == Piece.K_BLACK) and up_right_2 and self.board[up_right_2[0]][up_right_2[1]] == Piece.EMPTY:
                    return True
        elif piece == Piece.K_BLACK:
            if move == Move.UPLEFT:
                if up_left and (self.board[up_left[0]][up_left[1]] == Piece.WHITE or self.board[up_left[0]][up_left[1]] == Piece.K_WHITE) and up_left_2 and self.board[up_left_2[0]][up_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.UPRIGHT:
                if up_right and (self.board[up_right[0]][up_right[1]] == Piece.WHITE or self.board[up_right[0]][up_right[1]] == Piece.K_WHITE) and up_right_2 and self.board[up_right_2[0]][up_right_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNLEFT:
                if down_left and (self.board[down_left[0]][down_left[1]] == Piece.WHITE or self.board[down_left[0]][down_left[1]] == Piece.K_WHITE) and down_left_2 and self.board[down_left_2[0]][down_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNRIGHT:
                if down_right and (self.board[down_right[0]][down_right[1]] == Piece.WHITE or self.board[down_right[0]][down_right[1]] == Piece.K_WHITE) and down_right_2 and self.board[down_right_2[0]][down_right_2[1]] == Piece.EMPTY:
                    return True
        if piece == Piece.K_WHITE:
            if move == Move.UPLEFT:
                if up_left and (self.board[up_left[0]][up_left[1]] == Piece.BLACK or self.board[up_left[0]][up_left[1]] == Piece.K_BLACK) and up_left_2 and self.board[up_left_2[0]][up_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.UPRIGHT:
                if up_right and (self.board[up_right[0]][up_right[1]] == Piece.BLACK or self.board[up_right[0]][up_right[1]] == Piece.K_BLACK) and up_right_2 and self.board[up_right_2[0]][up_right_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNLEFT:
                if down_left and (self.board[down_left[0]][down_left[1]] == Piece.BLACK or self.board[down_left[0]][down_left[1]] == Piece.K_BLACK) and down_left_2 and self.board[down_left_2[0]][down_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNRIGHT:
                if down_right and (self.board[down_right[0]][down_right[1]] == Piece.BLACK or self.board[down_right[0]][down_right[1]] == Piece.K_BLACK) and down_right_2 and self.board[down_right_2[0]][down_right_2[1]] == Piece.EMPTY:
                    return True
        return False
    
    def can_jump(self):
        for i in range(7, -1, -1):
            for j in range(7, -1, -1):
                piece = self.board[i][j]
                if piece == Piece.EMPTY:
                    continue
                
                if self.turn == Piece.BLACK:
                    if piece == Piece.WHITE or piece == Piece.K_WHITE:
                        continue
                if self.turn == Piece.WHITE:
                    if piece == Piece.BLACK or piece == Piece.K_BLACK:
                        continue
                
                if piece == Piece.BLACK or piece == Piece.WHITE:
                    if self.is_jump((i, j), Move.UPLEFT) or self.is_jump((i, j), Move.UPRIGHT):
                        return True
                else:
                    if self.is_jump((i, j), Move.UPLEFT) or self.is_jump((i, j), Move.UPRIGHT) or self.is_jump((i, j), Move.DOWNLEFT) or self.is_jump((i, j), Move.DOWNRIGHT):
                        return True
        return False
    
    def check_w_win(self):
        if self.w_score == 12:
            self.outcome = Outcome.W_WIN
            return True
        if self.turn == Piece.BLACK and not self.can_move():
            self.outcome = Outcome.W_WIN
            return True
        return False
    
    def check_b_win(self):
        if self.b_score == 12:
            self.outcome = Outcome.B_WIN
            return True
        if self.turn == Piece.WHITE and not self.can_move():
            self.outcome = Outcome.B_WIN
            return True
        return False
    
    def check_draw(self):
        if self.draw_moves == self.max_moves:
            self.outcome = Outcome.DRAW
            return True
        return False
            
    def can_move(self):
        #Jumps
        if self.can_jump():
            return True    
        for i in range(7, -1, -1):
            for j in range(7, -1, -1):
                piece = self.board[i][j]
                if piece == Piece.EMPTY:
                    continue  
                
                if self.turn == Piece.BLACK and (piece == Piece.WHITE or piece == Piece.K_WHITE):
                    continue
                if self.turn == Piece.WHITE and (piece == Piece.BLACK or piece == Piece.K_BLACK):
                    continue
                
                if piece == Piece.BLACK or piece == Piece.WHITE:
                    if self.check_valid_move((i, j), Move.UPLEFT, False, False) or self.check_valid_move((i, j), Move.UPRIGHT, False, False):
                        return True
                else:
                    if self.check_valid_move((i, j), Move.DOWNLEFT, False, False) or self.check_valid_move((i, j), Move.DOWNRIGHT, False, False) or self.check_valid_move((i, j), Move.UPLEFT, False, False) or self.check_valid_move((i, j), Move.UPRIGHT, False, False):
                        return True
        return False
            
        