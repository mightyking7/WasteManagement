import numpy as np
from env import GridWorld
# from algo import Q_learning
from tqdm import tqdm
import matplotlib.pyplot as plt

# rows in grid
rows = 6

# columns in grid
cols = 25

# probability of taking a greedy action
epsilon = 0.9

# discount factor
gamma = 0.6

# num actions: up, down, left, right
nA = 4

# learning rate
lr = 0.01

# episodes for training
n_episodes = 500


# 1 is an obstacle, 0 is an empty cell
# dist = [1,0,0,0,0]
#
# num_tiles = rows * cols
#
# world = np.array([np.random.choice(dist) for i in range (num_tiles)]).reshape((rows, cols))

grid_world = GridWorld(nRows = rows, nCols = cols, nA = nA, gamma = gamma)
# behavior_policy = RandomPolicy()


# randomly initialize Q table
Q = np.random.rand(grid_world.nS, grid_world.nA)

# set state-action value of final states to 0
Q[[24, 49, 74, 99, 124, 149], :] = 0

for _ in tqdm(range(n_episodes)):

    state = grid_world.reset()

    done = False

    while not done :

        # explore action space
        if np.random.uniform(0, 1) < epsilon:
            action = grid_world.sample()

        # exploit
        else:
            action = np.argmax(Q[state])

        # take step in env
        obs, r, done = grid_world.step(action)

        # update Q table
        prev_value = Q[state, action]

        new_value = prev_value + lr * (r + grid_world.gamma * np.max(Q[obs]) - prev_value)

        Q[state, action] = new_value

        # update state
        state = obs

# sidewalk policy
# Q = Q_learning(grid_world.gamma, trajs, behavior_policy, lr = lr,
#                initQ=np.zeros((grid_world.nS, grid_world.nA)))

n_runs = 1

path = []

# evaluate policy
for i in range(n_runs):
    state = grid_world.reset()

    path.append(state)

    done = False

    while not done:
        action = np.argmax(Q[state])
        obs, r, done = grid_world.step(action)

        state = obs

        # add next state to path
        path.append(obs)
        print(path)

print("Done")

# plot policy

plt.plot(0, path[0], color='red', marker='s')
plt.plot(np.arange(1, cols), path[1:], color='blue', linestyle='--')
plt.show()

