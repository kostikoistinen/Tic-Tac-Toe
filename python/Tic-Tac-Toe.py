#!/usr/bin/env python
# coding: utf-8

# In[19]:


import keras
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import sys
import numpy as np
import sqlite3
from keras.models import load_model
from tabulate import tabulate
import random
import copy
import sys

'''
        #THE TIC TAC TOE - GAME!#
        
        ---------
        |X||_|X|
        |O||O|O|
        |X||_|X|
        ----------        
        
        Welcome to the game. Here is the description of the game. Subroutines are described separately.
        
        In the main program the following routine is executed:
        
        Settings:
        0)  Load all  packages and subroutines
        1)  Set players with input queries
             a) AI can be chosen here
        2)  Create a SQL table if not exists for game results'
        3)  Create the window for the game board and the details
        4)  Load the AI model if AI is chosen
        5)  Create buttons and styles
        6)  Create the board with desired dimensions
                |                      
                |                             OVERVIEW OF THE GAME
                |                           ************************
                |                                 
                |                                       is_game_over()
                |            X TURN                     is_draw()                         O TURN
                V                                        |
        RUN THE GAME: ----- on_button_click() <----> update window <-----> if not AI---- on_button_click()
                      |--             ^                  |                
        window updating  -            |                  |if AI               
              |                       |                  |               
              |                       |     if not trained else
    update_buttonsize() is_game_over()|            |        |
                        is_draw()     |            |        |
                             |        |        taketurns() bestMove() ------- get_moves()       
                             |        |             |      |        ------- boardtransform()
                             |         <-----------update window
                             |if True
                             |find_index()
                             V
                            Game ends ----> show_message() ----> save_results()
        
        7) Print the score table

'''

#-----------------------------------------------------------------------------------------------------#
#------------------------------------Start of subroutines---------------------------------------------#
#-----------------------------------------------------------------------------------------------------#  

def getMoves(peli):
    '''
        Description:
            is called to find out, which moves are possible in the game board.
            
            Input argument: Gameboard
            Output:         List of coordinates
    '''
    moves = []
    for i in range(len(peli)):
        for j in range(len(peli[i])):
            if peli[i][j] == "":
                moves.append((i, j))
    return moves
def boardtransform(peli):
    '''
        Description:
            Transforms the table from string list into integer list. Tensorflow handles integer arrays
            better. X=1, O=2, " " = 0
        
            Input argument: Gameboard
            Output:         Transformed integer Gameboard         
    '''
        
    for i in range(len(peli)):
        for j in range(len(peli)):
            if peli[i][j] == "X":
                peli[i][j] = 1
            elif peli[i][j] == "O":
                peli[i][j] = 2
            else:
                peli[i][j] = 0
    return peli

def taketurns(boardi, in_valx,in_valy, mark, AI):
    '''
        Description:
            Finds out position for "O" when AI or random AI has turn. Altetrnatively is used when training
            for random "X". It will call the bestMove() function if AI is turned on and there is a valid model.
            If AI is not turned on, it will just choose a random available position for current player.
        
            Input argument: Gameboard, previous coordinates, symbol that has the turn and Boolean value
                            if the AI is currently used.
            Output:         Updated gameboard, current coordinates for latest choice, a boolean value indicating
                            if the game is over or not. It is just a testing value used in training.
    '''
    
    
    rnd=0.6
    setup = False
    tester=0
    if not setup and AI and mark == "O":

        move = bestMove(copy.deepcopy(boardi), model, mark, False, rnd) #Ask the AI for the best move
        
        boardi[move[0]][move[1]] = "O"
        in_valx, in_valy = move
        setup=True
        return boardi,in_valx,in_valy, False
        
    elif not setup and AI and mark == "X":
        move = bestMove(copy.deepcopy(boardi), model, mark, False, rnd) #If True, also X will be AI
        boardi[move[0]][move[1]] = "X"
        in_valx, in_valy = move
        setup=True
        return boardi,in_valx,in_valy, False       
        
        
    else: #No AI
        while setup == False:
            xr,yr = random.randint(0,len(boardi)-1),random.randint(0,len(boardi)-1)
            if boardi[xr][yr] == "":
                boardi[xr][yr] = mark
                in_valx,in_valy = xr, yr
                setup = True
            tester = tester+1
            if tester > 100: #Ensures that all the free spots are checked
                return boardi, in_valx, in_valy, True
    return boardi, in_valx, in_valy, False



def bestMove(boardi, model, player, live, rnd=0.0):
    '''
        Description:
            Finds the best move using the trained AI model. It will make a prediction based on current board.
            It will thus call getMoves(). Then it will transform board via boardtransform() for Keras.
            It has a random factor so that it doesn't always make same choices. It will get the probabilities
            from keras and choose semi randomly the best choice. This will result into passive aggressive
            behaviour: AI will pursue its own goal while defending.
            
            Input argument: Gameboard, current model, turn ("X" or "O"), boolean test variable used in teaching
                            module and random factor.
            Output:         Chosen coordinates
    '''
    
    
    scores = []
    moves = getMoves(boardi)
    board = boardtransform(copy.deepcopy(boardi))
    a=1
    
    if player == "X":
        a=2
    else:
        a=1
    
    for i in range(len(moves)): #Prediction for AI
        future = np.array(board)
        future[moves[i][0]][moves[i][1]] = a
        prediction = model.predict(future.reshape((-1, len(boardi)**2)), verbose=0)[0]
        if player == "X": #or player == "O":
            winPrediction = prediction[1]
            lossPrediction = prediction[2]
        else:
            winPrediction = prediction[2]
            lossPrediction = prediction[1]
        drawPrediction = prediction[0]
        if winPrediction - lossPrediction > 0:
            scores.append(winPrediction - lossPrediction)
        else:
            scores.append(drawPrediction - lossPrediction)

    # Choose the best move with a random factor. Accepts the best move with probability.
    bestMoves = np.flip(np.argsort(scores))
    for i in range(len(bestMoves)):
        if random.random() * rnd < 0.5:
            return moves[bestMoves[i]]

    # Choose a move completely at random if the move is not excellent.
    return moves[random.randint(0, len(moves) - 1)]

def find_index(input_array, row_win,size):
    '''
        Description:
            Finds the winning indices. Not used here, but can be used for forming vectors to simulate
            more advanced behaviour. It finds the first and last index, and then returns a combined array
            of all the indices. Example: Winning line from (1,0) to (3,0), output (1,0),(2,0),(3,0).
            
            Example of usage: Find a three-long vector in a five-row-win game with 
            is_game_over(), then find_index() and make a subroutine to find next move.
        
            Input argument: Winning array of values, an integer value stating if the winning line is
                            left-right, up-down or diagonal (1,0,-1, respectively), Gameboard size.
            Output:         The winning line coordinates.      
    '''
    
    start = input_array[0]
    end = input_array[-1]
    num_range_x = end[0] - start[0]
    num_range_y = end[1] - start[1]    
    r=0
    #Check rows,columns and diagonals
    if row_win==1:
        num_range = num_range_x
        index1 = 1
        index2 = 0
    elif row_win==0:
        num_range = num_range_y
        index1 = 0
        index2 = 1
    else: #diagonals
        if row_win==-1: #up down
            index1 = 1
            index2 = 1
            num_range=num_range_x
        else: #down up
            index1 = 1
            index2 = -1
            r=0
            num_range=num_range_x  
# Create the new array with missing values
    new_array = []
    diagonal=0
    for i in range(num_range + 1):
        new_element = [start[0]+i*index1+r, start[1]+i*index2+r]
        new_array.append(new_element)
    return new_array 
    
def show_message(message):
    '''
        Description:
            Prints the input message to screen with messagebox. Output is none.
    '''
    messagebox.showinfo("Message", message)

def is_game_over(boardi,size,rows,columns):
    
    '''
        Description:
            Finds out if the game is over. Alternatively, it can be used to find any length combinations,
            as long as it's shorter than the gameboard or the winning condition. It will loop the gameboard
            to find desired vector by making smaller boards the size of the win conditions. For example, if
            4x4 grid 3 is required to win, it will make 4 windows to search all the possible 3 lines.
            Downside is, it will only find the same array again and again. For checking
            if the game has ended it will still work. Under construction.
        
            Input argument: Gameboard, desired size of row,column or diagonal to be checked, rows,columns 
                            are same if checking for winner
            Output:         True if the desired result is obtained, desired vector      
    '''
    # Check rows and columns
    argu = random.randint(-1,1)
    for i in range(size):
        if "X" not in boardi[i][:] and "" not in boardi[i][:]:
            winvector = find_index([[rows,columns+i],[rows+size-1, columns+i]], 1, size)
            return True, winvector #Place row, first index of window
        if "O" not in boardi[i][:] and "" not in boardi[i][:]:
            winvector = find_index([[rows,columns+i],[rows+size-1, columns+i]], 1, size)
            return True, winvector
        if "O" not in [row[i] for row in boardi] and "" not in [row[i] for row in boardi]:
            winvector = find_index([[rows+i,columns],[rows+i, columns+size-1]], 0, size)
            return True, winvector
        if "X" not in [row[i] for row in boardi] and "" not in [row[i] for row in boardi]:
            winvector = find_index([[rows+i,columns],[rows+i, columns+size-1]], 0, size)
            return True, winvector
    #Check diagonals
    if all(boardi[i][i] == boardi[0][0] for i in range(len(boardi))):
        if all(boardi[i][i] != "" for i in range(len(boardi))):
            winvector = find_index([[rows,columns],[rows+size-1, columns+size-1]], -1, size)
            return True, winvector
    if all(boardi[-1-i][i] == boardi[-1][0] for i in range(len(boardi))):
        if all(boardi[-1-i][i] != "" for i in range(len(boardi))):
            winvector = find_index([[rows, columns+size-1],[rows+size-1,columns]], -2, size)
            return True, winvector
    return False, [0]

def is_draw(boardi):
    '''
        Description:
            Checks if the table is full of symbols. returns True if game is so.    
    '''
    for row in boardi:
        for cell in row:
            if cell == "":
                return False
    return True

# Function to handle button click event
def on_button_click(row, col,size):
    '''
        Description:
            On action - algorithm. It will update the board. It will configure the board in action, depending
            who has turn. It will also terminate the program if game is over. It controls the whole game.
        
            Input argument: Initialized row and column index, when pressing the button and game board length.
            Output:         No return.
    '''
    global boardi, current_player, windowsize, in_valx, in_valy
        
    if boardi[row][col] == "":
        #Update board with X choice
        if current_player == "X":
            boardi[row][col] = current_player
            buttons[row][col].config(text=current_player)
            in_valx,in_valy = row,col
        #Update board with O choice if AI is not turned on.
        elif current_player == "O" and player_o != "AI":
            boardi[row][col] = current_player
            buttons[row][col].config(text=current_player)
            in_valx,in_valy = row,col
        #Go through the board with windowsize (if for win 5 is needed, windowsize = 5)
        for i in range(size-windowsize+1):
            for j in range(size-windowsize+1):
                game_over, winvector = is_game_over([row[i:i+windowsize] for row in boardi[j:j+windowsize]],windowsize,i,j)
                if game_over:
                    for i in range(len(winvector)):
                        buttons[winvector[i][1]][winvector[i][0]].configure(style="Customize.TButton")
                    show_message("Player " + current_player + " wins!")
                    save_results(current_player,False)
                    window.quit()
                    window.destroy()
                    return
        if is_draw(boardi):
            show_message("It's a draw!")
            save_results(current_player, True)             
            window.quit()
        
        else:
            #AI is turned on.
            current_player = "O" if current_player == "X" else "X"
            if current_player == "O" and player_o == "AI":
                window.title("Tic-Tac-Toe - Player " + "X")
                #If model is set it will use it, else it chooses random movement.
                if model == None:
                    boardi,in_valx,in_valy,test = taketurns(boardi, in_valx,in_valy, "O", False)
                else:
                    in_valx,in_valy = bestMove(copy.deepcopy(boardi), model, current_player, True, rnd=0.0)
                
                row,col = in_valx,in_valy
                #Increment and update the board.
                boardi[row][col] = current_player
                buttons[row][col].config(text=current_player)
                game_over, winvector = is_game_over([row[i:i+windowsize] for row in boardi[j:j+windowsize]],windowsize,i,j)
                #Stop game if game is over or draw.
                if game_over:
                    #If true, The buttons background color is changed to see which vector won.
                    for i in range(len(winvector)):
                        buttons[winvector[i][1]][winvector[i][0]].configure(style="Customize.TButton")
                    show_message("Player " + current_player + " wins!")
                    save_results(current_player,False)
                    window.quit()
                    window.destroy()
                    return
                if is_draw(boardi):
                    show_message("It's a draw!")
                    save_results(current_player, True)             
                    window.quit()
                    
                current_player = "X"
            
            window.title("Tic-Tac-Toe - Player " + current_player)

def save_results(current_player, draw):
    '''
        Description:
            Will save the results into SQL table. Approach: First make a query as string, then execute
            it with sqlite3. First the existing results are queried and then new results are set.
        
            Input argument: Winning player X or O, draw flag. 
            Output:         Nothing    
    '''
    global player_x,player_o
    if not draw:
        if current_player == "X":
            select_query = "SELECT wins FROM game_results WHERE player_name = ?"
            c.execute(select_query, (player_x,))
            current_wins = c.fetchone()[0]
            select_query = "SELECT losses FROM game_results WHERE player_name = ?"
            c.execute(select_query, (player_o,))
            current_losses = c.fetchone()[0]
            # Increment the wins value by 1 if X wins
            new_wins = current_wins + 1
            new_losses = current_losses + 1
            update_query = "UPDATE game_results SET wins = ? WHERE player_name = ?"
            c.execute(update_query, (new_wins, player_x))
            update_query = "UPDATE game_results SET losses = ? WHERE player_name = ?"
            c.execute(update_query, (new_losses, player_o))
        else:
            select_query = "SELECT wins FROM game_results WHERE player_name = ?"
            c.execute(select_query, (player_o,))
            current_wins = c.fetchone()[0]
            select_query = "SELECT losses FROM game_results WHERE player_name = ?"
            c.execute(select_query, (player_x,))
            current_losses = c.fetchone()[0]
            # Increment the wins value by 1 if O wins
            new_wins = current_wins + 1
            new_losses = current_losses + 1
            update_query = "UPDATE game_results SET wins = ? WHERE player_name = ?"
            c.execute(update_query, (new_wins, player_o))
            update_query = "UPDATE game_results SET losses = ? WHERE player_name = ?"
            c.execute(update_query, (new_losses, player_x))
            
        select_query = "SELECT wins,losses FROM game_results WHERE player_name = ?"
        
        c.execute(select_query, (player_x,))        
        WintoLoss = c.fetchone()
        #Calculate Win to lose ratio for both players.
        try:
            WintoLossr = WintoLoss[0]/WintoLoss[1]
        except:
            WintoLossr = None
        update_query = "UPDATE game_results SET WintoLoss = ? WHERE player_name = ?"
        c.execute(update_query, (WintoLossr, player_x))
        
        c.execute(select_query, (player_o,))        
        WintoLoss = c.fetchone()
        try:
            WintoLossr = WintoLoss[0]/WintoLoss[1]
        except:
            WintoLossr = None
        update_query = "UPDATE game_results SET WintoLoss = ? WHERE player_name = ?"
        c.execute(update_query, (WintoLossr, player_o))
        
    # If it's draw...
    else:
        select_query = "SELECT draws FROM game_results WHERE player_name = ?"
        c.execute(select_query, (player_x,))
        current_draws1 = c.fetchone()[0]
        select_query = "SELECT draws FROM game_results WHERE player_name = ?"
        c.execute(select_query, (player_o,))
        current_draws2 = c.fetchone()[0]
    # Increment the draws value by 1
        new_draws1 = current_draws1 + 1
        new_draws2 = current_draws2 + 1
        update_query = "UPDATE game_results SET draws = ? WHERE player_name = ?"
        c.execute(update_query, (new_draws1, player_x))
        update_query = "UPDATE game_results SET draws = ? WHERE player_name = ?"
        c.execute(update_query, (new_draws2, player_o))
    #Round decimals
    c.execute("UPDATE game_results SET WintoLoss = ROUND(WintoLoss, 2)")
        
def update_button_size(event):
    '''
        Description:
            Used if the gameboard size is enlarged.
        
            Input argument: Current state of board
            Output:         none        
    '''    
    
    window_width = event.width
    window_height = event.height
    button_size = min(window_width, window_height) // 3  # Adjust button size based on window size
    #Configure button if the window size changes.
    for button in buttons:
        button.configure(font=('Helvetica', 100), width=button_size, height=button_size)

#-----------------------------------------------------------------------------------------------------#
#------------------------------------End of subroutines-----------------------------------------------#
#-----------------------------------------------------------------------------------------------------# 

#-----------------------------------------------------------------------------------------------------#
#------------------------------------1 and 2----------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
player_x = input("Insert name, player X")
if input("DO YOU WANT an AI OPPONENT? (y/n)")=="y":
    player_o = "AI"
else:
    player_o = input("Insert name, player O")
#Make the database: if it doesn't exist yet.
conn = sqlite3.connect("game_results.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS game_results
             (player_name TEXT, wins INT, losses INT, draws INT, WintoLoss FLOAT)''')

query = "SELECT * FROM game_results WHERE player_name = ?"
c.execute(query, (player_x,))
if c.fetchone() is None:
    c.execute("INSERT INTO game_results (player_name, wins,losses,draws,WintoLoss) VALUES (?, ?,?,?,?)", (player_x, 
                                                                                                   0,0,0,0))
query = "SELECT * FROM game_results WHERE player_name = ?"
c.execute(query, (player_o,))
if c.fetchone() is None:
    c.execute("INSERT INTO game_results (player_name, wins,losses,draws,WintoLoss) VALUES (?, ?,?,?,?)", (player_o, 0,0,0,0))

#-----------------------------------------------------------------------------------------------------#
#------------------------------------3----------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------# 
#Make the window
window = tk.Tk()
window.title("Tic-Tac-Toe - Player X")
window.geometry("300x300")

# Bind window resize event to update_button_size function
window.bind(update_button_size)

#Length of the board!
size = int(input("Insert size of the grid:"))

#-----------------------------------------------------------------------------------------------------#
#------------------------------------4----------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------# 

if player_o == "AI":
    #Load appropriate model
    try:
        model = load_model('my_model0'+str(size)+'.h5')
        print(model)    
    except:
        if size == 3:
            model = load_model('working_AI.h5')
            print("No trained model, using default AI")
        else:
            model = None
            print("No model, AI plays as random. Train the AI network first!")

#-----------------------------------------------------------------------------------------------------#
#------------------------------------5----------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------# 
# Create buttons for the game cells
buttons = []
button_size = int(30/size)
for i in range(size):
    row = []
    for j in range(size):
        #Activity function
        btn = ttk.Button(window, text="",
                        command=lambda row=i, col=j: on_button_click(row, col,size))
        #Standard style
        style = ttk.Style()
        style.configure("Custom.TButton",
                font=('Helvetica', 12),
                padding=10,
                foreground= 'blue',
                background='white')
        #If game ends, the winning buttons are changed into red.
        style2 = ttk.Style()
        style2.configure("Customize.TButton",
                font=('Helvetica', 12),
                padding=10,
                foreground= 'blue',
                background='red')
        # Apply the custom style to the button
        btn.configure(style="Custom.TButton")
        btn.grid(row=i, column=j, sticky=tk.NSEW)
        row.append(btn)
    buttons.append(row)

#-----------------------------------------------------------------------------------------------------#
#------------------------------------6----------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------# 
windowsize = int(input("how many needed to win?"))
# Create the game boardi
boardi = [[""] * size for _ in range(size)]

for i in range(size):
    window.grid_rowconfigure(i, weight=1)
    window.grid_columnconfigure(i, weight=1)

#Gameboard size determines the size of the window.
if size > 10:
    window.geometry("600x600")
if size > 20:
    window.geometry("900x900")
if size > 30:
    window.geometry("1200x1200")
    style.configure("Custom.TButton",font = ('Helvetica',4))

current_player = "X"

#WINDOW LOOP. This will run ttk.Button() function
window.mainloop()
conn.commit()
conn.close()

try:
    window.destroy()
except:
    print("Game ended")
    print("Scoreboard")
#-----------------------------------------------------------------------------------------------------#
#------------------------------------7----------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------# 
conn = sqlite3.connect("game_results.db")
c = conn.cursor()
select_query = "SELECT * FROM game_results"
c.execute(select_query)
rows = c.fetchall()
columns = [desc[0] for desc in c.description]
# Loop through the rows and print the contents
print(tabulate(rows, headers=columns, tablefmt='grid'))
conn.commit()
conn.close()

print("Last Game")
print(tabulate(boardi, tablefmt='grid'))

