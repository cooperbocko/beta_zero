import time

from c4.connect4 import Connect4, Connect4BitBoard

game = Connect4()
bbgame = Connect4BitBoard()
while game.outcome is None:
    print('original game:')
    game.print_board()
    print('bitboard game:')
    bbgame.print_board()
    print(bin(bbgame.mask))
    print(bin(bbgame.position))
    
    print('original game valid moves:')
    print(game.get_valid_moves())
    print('bitboard game valid moves:')
    print(bbgame.get_valid_moves())
    
    action = int(input("Enter move: "))
    game.step(action)
    bbgame.step(action)

print('original game:')
game.print_board()
print(game.outcome)
print('bitboard game:')
print(bbgame.outcome)


start = time.time()
for i in range(100):
    game = Connect4()
    while not game.outcome:
        action = None
        valid = game.get_valid_moves()
        for i in range(7):
            if valid[i] == 1:
                action = i
                break
        game.step(action)
end = time.time()
print(f'og time: {end - start}')
og = end - start

start = time.time()
for i in range(100):
    game = Connect4BitBoard()
    while game.outcome != None:
        game.print_board()
        action = None
        valid = game.get_valid_moves()
        for i in range(7):
            if valid[i] == 1:
                action = i
                break
        game.step(action)
end = time.time()
print(f'bb time: {end - start}')
bb = end - start

print(f'og/bb: {og/bb}')
