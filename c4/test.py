from c4.connect4 import Connect4

game = Connect4()
while game.outcome is None:
    game.print_board()
    print(game.get_valid_moves())
    action = int(input("Enter move: "))
    game.step(action)

game.print_board()
print(game.outcome)
