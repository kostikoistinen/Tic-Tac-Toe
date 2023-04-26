# Tic-Tac-Toe

Read me file on Tic-Tac-Toe

The repository contains a Tic-Tac-Toe - game. It can be run either with .py files straight from prompt, or from Jupyter notebook. Note that one needs a lot of packages to run the files. Below are instructions for running the program from .py - files and from jupyter notebook. Also the required packages are enlisted. THE FULL DOCUMENTATION IS WITHIN THE CODE!
............................
INSTRUCTIONS 1:

This program contains two python files, run separately.
1) The game
2) Teaching module


Description
1) The Game
	A classic Tic-Tac-Toe game with free grid size. One can also determine how many symbols are needed for win.
	Can be played either 1vs1 or against AI. AI has a default AI for 3x3 grid, but for larger grids AI needs to be trained. 
	Contains also a scoreboard made with SQL querys.

2) Teaching module
	An unsupervised learning neural network model is constructed depending on size and the winning conditions. All are free parameters.
	After teaching, the AI can be tested. It can play against a random generator. Finally the statistics are printed to see if AI works.
	AI is then saved, and can be run from the Game. Note that for each grid size AI model is different!
More details in comments within the files.
................................
INSTRUCTIONS 2:
1) The game and teaching module
	Download the jupyter notebook file and run the file in jupyter notebook. There are two segments, The game - cell and the training - cell.
................................



Packages needed to install

There are several packages that need to be installed in order for the program to work:

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
	
	Note that keras can have some issues if python is not adapted with GPU. Regardless of warning messages the program should still work. I ran the program with jupyter and didn't face any issues. If python program doesn't work, one can also try to download the jupyter file and run it inside jupyter notebook environment.
	
- Kosti Koistinen
