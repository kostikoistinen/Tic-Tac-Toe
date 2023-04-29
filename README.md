# Tic-Tac-Toe

Read me file on Tic-Tac-Toe

The repository contains a Tic-Tac-Toe - game made with python 3.8. It can be run either with .py files straight from prompt (Instructions 1), or from Jupyter notebook (Instructions 2). Note that one needs a lot of packages to run the files. Below are instructions for running the program from .py - files and from jupyter notebook. Also the required packages are enlisted. THE FULL DOCUMENTATION IS WITHIN THE CODE! See also statistics, there are some evidence of the functioning AI.
<br />
<br />
............................
<br />
INSTRUCTIONS 1:
<br />
<br />
Go into python folder.
<br />
The program contains two python files, run separately.
<br />
1) The game
2) Teaching module
<br />
<br />
Description
<br />
1) The Game
<br />
	A classic Tic-Tac-Toe game with free grid size. One can also determine how many symbols are needed for win.
	Can be played either 1vs1 or against AI. AI has a default AI for 3x3 grid, but for larger grids AI needs to be trained. 
	Contains also a scoreboard made with SQL querys.
<br />
<br />
2) Teaching module
<br />
	An unsupervised deep neural network model is constructed depending on size and the winning conditions. After teaching, the AI can be tested. It can play against a random generator. Finally the statistics are printed to see if AI works. AI is then saved, and can be run from the Game. Note that for each grid size AI model is different! There is a default AI trained for 3x3 board if you do not wish to train it yourself. Note that in teaching module, after training, you can simulate AI against random behaviour player to obtain simple statistics of AI performance. Just run the teaching-file again.
<br />
<br />
More details in comments within the files.
<br />
................................
<br />
INSTRUCTIONS 2:
<br />
1) The game and teaching module
<br />
	Download the jupyter notebook file (in jupyterfile folder) and run the file in jupyter notebook. There are two segments, The game - cell and the training - cell.
<br />
................................
<br />
<br />
Packages needed to install
<br />
<br />
There are several packages that need to be installed in order for the program to work:
<br />

	import tkinter as tk
	from tkinter import messagebox
	from tkinter import ttk
	import sys
	import sqlite3
	import numpy as np
	import random
	import matplotlib.pyplot as plt
	import ipywidgets as widgets
	from IPython.display import display
	import copy
	import sys
	from io import StringIO
	from tabulate import tabulate
	import keras	
	from keras.models import load_model	
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout
	from keras.backend import reshape
	from keras.utils.np_utils import to_categorical
	from keras.models import load_model
<br />	
<br />
NOTES: <br />
I tried to combine the program into one executable file for convinience, but unfortunately, the pyinstaller didn't support tensorflow.
<br />
<br />
Note that keras can have some issues if python is not adapted with GPU. Regardless of warning messages the program should still work. I ran the program with jupyter and didn't face any issues. If python program doesn't work, one can also try to download the jupyter file and run it inside jupyter notebook environment. Finally, I used Jupyter notebook running using Ubuntu 22.04. It can cause some conflicts for Windows users (it shoudn't though).
	<br />
	<br />
- Kosti Koistinen
