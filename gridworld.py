import numpy as np
import pandas as pd

rows = 6
cols = 25
epsilon = 0.9

# 1 is an obstacle, 0 is an empty cell
dist = [1,0,0,0,0]

num_tiles = rows * cols

world = np.array([np.random.choice(dist) for i in range (num_tiles)]).reshape((rows, cols))
