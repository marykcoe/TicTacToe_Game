#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noughts & Crosses Game
Author: Mary Coe
Created: 28/01/2024
Last Updated: 21/04/2024

This program was designed to understand the use and implementation
of the minimax algorithm.

This class provides the methods necessary to run the game and
interact with the human player.
"""

import numpy as np
from copy import deepcopy
import random
import sys
import itertools

class noughts_and_crosses:
    """
    Noughts & Crosses Class
    This class contains the variables and methods necessary to run
    a noughts & crosses game (also known as Tic-Tac-Toe).

    Methods:
        display_board: Prints game board to screen
        reset_game: Resets game board and restarts game.
        game_over: Displays winner of game and final board.
        run_game: Runs the game.
    """

    def __init__(self,):
        """
        Constructor for Noughts & Crosses class.
        Initialises game board and starts game.

        Returns
        -------
        None.

        """

        # Set up game board
        # A -1 indicates that the position is empty.
        # A 0 indicates a nought fills the position.
        # A 1 indicates a cross fills the position.
        self.game_board = np.ones((3,3))
        self.game_board *= -1

        # Set winning player
        self.winning_player = ""

        # Determine computer player
        self.computer_player = None
        self.human_player = None

        # Define all possible moves
        self.all_moves = list(itertools.product(*[[0,1,2], [0,1,2]]))

        # Run game
        self.run_game()


    def display_board(self, game_board):
        """
        Prints board to screen.

        Returns
        -------
        None.

        """

        # Loop over each row and column of game board.
        for r in range(3):
            for c in range(3):

                # Choose correct marker to print
                match game_board[r][c]:
                    case -1:
                        marker = " "
                    case 0:
                        marker = "o"
                    case 1:
                        marker = "x"
                    case _:
                        raise Exception(ValueError)

                # Print internal columns
                if c < 2:
                    print(f' {marker} |', end = "")
                else:
                    print(f' {marker} ')

            # Print internal rows
            if r < 2:
                print(f'-----------')

    def reset_game(self,):
        """
        Resets game board to be empty. Begins running game.

        Returns
        -------
        None.
        """

        self.game_board = np.ones((3,3))
        self.game_board *= -1

        self.winning_player = ""

        self.run_game()


    def game_over(self,):
        """
        Displays final game board and winner of the game.
        Asks the player if they would like to restart the game.

        Returns
        -------
        None.
        """

        if self.winning_player == "":
            print("It's a draw!")
        else:
            print(f'{self.winning_player} is the winner!')

        print("Would you like to play again?")
        user_input = input("Please enter Yes or No...")
        print(user_input)
        if user_input == "Yes":
            self.reset_game()
        else:
            print("Thank you for playing!")
        sys.exit()

        if user_input.lower() in ["yes", "y", "okay"]:
            self.reset_game()
        elif user_input.lower() in ["no", "n", "nope"]:
            print("Thank you for playing!")
        else:
            print("Your answer wasn't recognised. Thanks for playing!")

    def confirm_valid_move(self, move, game_board):
        """
        Confirms the entered tile is empty.

        Parameters
        ----------
        move : List(int)
            Tile to be inspected. Supplied in the form
            (row, column).

        game_board : Numpy Array (3,3)
            Game board to be inspected.

        Returns
        -------
        valid : bool
            True if move is valid, False if move is invalid.
        """

        # Ensure the move is valid.
        if move[0] < 0 or move[0] > 2 or move[1] < 0 or move[1] > 2:
            return False

        return game_board[move[0]][move[1]] == -1

    def human_choose_move(self,):
        """
        Asks the human player to choose a move. Ensures move is
        valid.

        Parameters
        ----------
        None.

        Returns
        -------
        List(int)
            Row and column of move to be performed.
        """

        print("It's your turn!")

        valid = False

        # Get move.
        while not valid:

            # Ask human player for move.
            row = input("Please choose a row...")
            column = input("Please choose a column...")

            # Confirm move is valid
            valid = self.confirm_valid_move([int(row), int(column)], self.game_board)
            if not valid:
                print("Invalid move!")

        # Return valid move
        return [int(row), int(column)]

    def get_all_rows(self, game_board):
        """
        Organises each row, column and diagonal on the
        game board into a numpy array of size (8,3).

        Parameters
        ----------
        game_board : Numpy Array(int,int)
            Current game board.

        Returns
        -------
        rows : Numpy Array(int, int)
            Array containing each row, column and diagonal
            on the game board.

        """

        # Collect rows
        rows = [game_board[r][:] for r in range(3)]

        # Collect columns
        columns = [np.array([game_board[r][c] for r in range(3)]) for c in range(3)]

        # Collect diagonals
        increasing_diagonal = np.array([game_board[r][r] for r in range(3)])
        decreasing_diagonal =  np.array([game_board[r][c] for r,c in zip([2,1,0],[0,1,2])])

        # Collate into one array
        rows = np.vstack((rows, columns, increasing_diagonal, decreasing_diagonal))

        return rows

    def check_game_over(self, game_board):
        """
        Checks supplied game board for completed rows, columns
        and diagonals. Returns true if one is found or false
        otherwise.

        Parameters
        ----------
        game_board : Numpy Array(int,int)
            Current game board.

        Returns
        -------
        bool
            Returns True if a row, column or diagonal has been
            completed and False otherwise.
        """

        # Get all rows, columns and diagonals on board.
        rows = self.get_all_rows(game_board)

        # Check for completed rows
        for row in rows:
            if np.all(row == 0) or np.all(row == -0) or np.all(row == 1):
                return True

        return False

    def calculate_moves_left(self,):

        """
        Determines the number of possible moves left on the board.

        Parameters
        ----------
        None.

        Returns
        -------
        moves_left : int
            Number of empty tiles left on the game board.
        """

        # Confirm if any lines made
        def is_complete(row):
            if np.all(row == 1):
                self.winning_player = "x"
                self.game_over()
            elif np.all(row == 0):
                self.winning_player = "o"
                self.game_over()


        # Check for winning rows
        for r in range(3):
            is_complete(np.array(self.game_board[r][:]))

        # Check for winning columns
        for c in range(3):
            column = []
            for r in range(3):
                column.append(self.game_board[r][c])
            is_complete(np.array(column))

        # Check for winning diagonals
        is_complete(np.array([self.game_board[r][r] for r in range(3)]))
        is_complete(np.array([self.game_board[r][c] for r,c in zip([2,1,0],[0,1,2])]))

        # Find empty tiles
        board_mask = self.game_board == -1

        # The number of moves left is the sum of the
        # number of empty tiles.
        moves_left = sum(sum(board_mask))

        return moves_left

    def run_game(self,):
        """
        Main game play loop.
        The game begins by randomly assigning the noughts
        and crosses players.
        The players then take it in turns to place a nought
        or a cross. After the human selects their move, the
        minimax algorith is called to determine the best
        move for the computer to perform.
        The game ends when there are no available tiles left
        on the board, or when a player makes a row, column or
        diagonal.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        # Assign players at random
        coin_toss = random.random()
        if coin_toss > 0.5:
            self.computer_player = -1
            self.human_player = 0
            print("You are noughts!")
        else:
            self.computer_player = 0
            self.human_player = -1
            print("You are crosses!")

        # If player is crosses, ask them to choose a move
        if self.human_player == -1:
            move = self.human_choose_move()
            self.game_board[move[0]][move[1]] *= self.human_player
            computer_turn = True

        # If the computer is crosses, choose a move at random
        # to start. This makes the game more fun, as the computer
        # would otherwise always choose the same initial position,
        # due to the deterministic underlying algorithm.
        else:
            move = [random.randrange(0,3), random.randrange(0,3)]
            self.game_board[move[0]][move[1]] *= self.computer_player
            computer_turn = False

        # Display board
        self.display_board(self.game_board)

        # Run game until all tiles filled
        while True:

            # Check if game finished
            if self.calculate_moves_left() == 0:
                self.game_over()

            # If not, computer chooses move
            if computer_turn:
                print("It's the computer's turn.")
                move = self.choose_move()
                self.game_board[move[0]][move[1]] *= self.computer_player

            # Player picks move
            else:
                move = self.human_choose_move()
                self.game_board[move[0]][move[1]] *= self.human_player

            # Display board
            self.display_board(self.game_board)

            # Change whose turn it is
            computer_turn = not computer_turn

    def choose_move(self, ):
        """
        Loops over all tiles, determining which are availble. For
        each available tile, the minimax algorithm is called, to
        determine a score for the move. The computer player then
        picks the move with the best score.

        Parameters
        ----------
        None.

        Returns
        -------
        best_move : List(int, int)
            Row and column of best move for the computer to
            perform.
        """

        # Initially, the best move to perform is undetermined.
        best_move = None

        # The depth of the solution tree is equal to the number
        # of available tiles on the board.
        depth = self.calculate_moves_left()

        # Initially, the best score for the computer player
        # is set to be as low as possible. This ensures the
        # computer picks a move.
        score = -np.inf

        # Investigate every tile.
        for move in self.all_moves:

            # First check if the tile is available.
            if self.confirm_valid_move(move, self.game_board):

                # Make a copy of the game board.
                proposed_game_board = deepcopy(self.game_board)

                # Update game board with proposed move.
                proposed_game_board[move[0]][move[1]] *= self.computer_player

                # Calculate the score from performing the move.
                move_score = self.minimax(proposed_game_board, False, -np.inf, np.inf, depth-1)

                # If the score from this move is better than
                # the best score found, update the best score
                # found and the best move accordingly.
                if move_score > score:
                    score = move_score
                    best_move = move

        return best_move


    def minimax(self, game_board, maximising_player, alpha, beta, depth):
        """
        Uses the minimax algorithm to get the score for a given move.

        Recursively builds the game board, alternating between moves
        performed by the maximising player (the computer) and the
        minimising player (the human). Once a game over has been
        achieved, or all tiles are full, the score of the final board
        is returned.

        To improve efficiency, this function utilises alpha-beta
        pruning. In this method, the maximum and minimum scores
        within the recursive loop are set to alpha and beta
        respectively. If the minimum score (beta) is less than the
        maximum score (alpha), then the minimising player will
        never choose to explore the part of the solution tree with
        the alpha score, as it knows a better option exists.

        Parameters
        ----------
        game_board : Numpy Array (int, int)
            3x3 array representing the game board. Tiles have values
            of -1, 0 or 1 if the tile is empty, contains a nought or
            contains a cross.

        maximising_player : bool
            Determines if it is the maximising (true) or minimising
            (false) players turn.

        alpha : float
            The maximum score for the maximising player in the
            recursive loop. Used to perform alpha-beta pruning.

        beta : float
            The minimum score for the minimising player in the
            recursive loop. Used to perform alpha-beta pruning.

        depth : int
            The number of moves left on the board (the number of
            empty tiles).

        Returns
        -------
        score : float
            The maximum or minimum score of the maximising or minimising
            player.
        """

        # If the game has ended (as a row, column or diagonal has been
        # made) or if there are no more available tiles (and therefore
        # moves), return the score of the final game board.
        if self.check_game_over(game_board) or (depth == 0):
            return self.calculate_score(game_board) + depth

        # If it is the computers move (the maximising player),
        # search for the score when performing all possible moves.
        elif maximising_player:

            # The worst score for the maximising player is -inf.
            max_score = -np.inf

            # Search all possible moves left on the game board (all
            # available tiles).
            for move in self.all_moves:

                # Check that the move can be performed.
                if self.confirm_valid_move(move, game_board):

                    # Copy game board.
                    proposed_game_board = deepcopy(game_board)

                    # Update the board with the proposed move.
                    proposed_game_board[move[0]][move[1]] *= self.computer_player

                    # Calculate the score of the move by recursively calling
                    # the minimax algorithm.
                    move_score = self.minimax(proposed_game_board, False, alpha, beta, depth-1)

                    # Update the maximum score the maximising player can
                    # achieve.
                    max_score = max(max_score, move_score)

                    # Update the alpha score.
                    alpha = max(alpha, move_score)

                    # If the beta score is less than the alpha score,
                    # then the maximising player need not look any further.
                    if beta <= alpha:
                        break

            return max_score

        # If it is the human players move (the minimising player),
        # search for the score when performing all possible moves.
        else:

            # The worst possible score for the minimising player is
            # +inf.
            min_score = np.inf

            # Search all possible moves left on the game board (all
            # available tiles).
            for move in self.all_moves:

                # Check that the move can be performed.
                if self.confirm_valid_move(move, game_board):

                    # Copy game board.
                    proposed_game_board = deepcopy(game_board)

                    # Update the board with the proposed move.
                    proposed_game_board[move[0]][move[1]] *= self.human_player

                    # Calculate the score of the move by recursively calling
                    # the minimax algorithm.
                    move_score = self.minimax(proposed_game_board, True, alpha, beta, depth-1)

                    # Update the minimum score the minimising player can
                    # achieve.
                    min_score = min(min_score, move_score)

                    # Update the beta score
                    beta = min(beta, move_score)

                    # If the beta score is less than the alpha score,
                    # then the maximising player need not look any further.
                    if beta <= alpha:
                        break

            return min_score


    def calculate_score(self, game_board):
        """
        Calculates score of the computer for a given game board
        configuration.

        The scoring mechanism is as follows:
            If the computer makes a row, column or diagonal,
            it scores +10.
            If the human makes a row, column or diagonal,
            the computer score -10.
            If the computer blocks a row, column or diagonal
            with 2 human player tiles, it score +2.
            If the human player blocks a row, column or diagonal
            with 2 computer player tiles, the computer score -2.

        Parameters
        ----------
        game_board : Numpy Array(int, int)
            3x3 array representing the game board. Tiles have values
            of -1, 0 or 1 if the tile is empty, contains a nought or
            contains a cross.


        Returns
        -------
        score : int
            Computer players score from the current game board.
        """

        def get_row_score(row):
            """
            Sums the row and caculates the associated score.

            Parameters
            ----------
            row : Numpy Array (int)
                Numpy array with 3 numbers indicating the
                occupancy of each tile.

            Returns
            -------
            score : int
                Score for the row.
            """

            # If the row is not full, it doesn't get scored.
            if -1 in row:
                return 0

            # Calculate the total of the row
            total = sum(row)

            # Find the correct score for each possible total.
            match total:

                # This indicates a row of crosses.
                case 3:
                    if self.computer_player == -1:
                        return 10
                    else:
                        return -10

                # This indicates a row blocked by one nought.
                case 2:
                    if self.computer_player == -1:
                        return -2
                    else:
                        return 2

                # This indicates a row blocked by one cross.
                case 1:
                    if self.computer_player == -1:
                        return 2
                    else:
                        return -2

                # This indicates a row of noughts.
                case 0:
                    if self.computer_player == -1:
                        return -10
                    else:
                        return 10

            return 0

        # Get all rows, columns and diagonals
        to_score = self.get_all_rows(game_board)

        # Get scores
        score = 0
        for row in to_score:
            score+= get_row_score(row)

        return score














