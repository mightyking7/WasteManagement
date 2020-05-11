import os
import numpy as np
from obstacle import ObstacleEnv
from algo import Q_learning
import matplotlib.pyplot as plt

# rows in grid
rows = 6

# columns in grid
cols = 25

# probability of exploiting
epsilon = 0.40

# discount factor
gamma = 0.6

# num actions: up, down, left, right
nA = 4

# learning rate
lr = 0.01


# directories for policy
policy_dir = "./policy/"
fname_policy = "./policy/obstacle_policy.npy"

img_dir = "./plots/"


# make dir for policy
if not os.path.exists(policy_dir):
    os.mkdir(policy_dir)

# make dir for plots
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

obstacleEnv = ObstacleEnv(nRows = rows, nCols = cols, nA = nA, gamma = gamma)

# train if policy doesn't exist
if not os.path.exists(fname_policy):

    n_episodes = 5000

    # randomly initialize Q table
    Q = np.random.rand(obstacleEnv.nS, obstacleEnv.nA)

    # set state-action value of final states to 0
    # TODO make this dynamic
    Q[[24, 49, 74, 99, 124, 149], :] = 0

    for e in range(n_episodes):
        Q , _ = Q_learning(env = obstacleEnv, epsilon = epsilon, lr = lr, initQ = Q)
        print(f"> Episode: {e}")

    # save policy
    np.save(fname_policy, Q)

else:
    # load policy
    Q = np.load(fname_policy)

n_runs = 1

path = []

# evaluate policy
for i in range(n_runs):
    state = obstacleEnv.reset()

    path.append(state)

    done = False

    while not done:
        action = np.argmax(Q[state])
        obs, r, done = obstacleEnv.step(action)

        state = obs

        # add next state to path
        path.append(obs)
        print(path)

# convert path to ndarray
path = np.array(path)

# plot policy
si, sj = list(), list()

# state coords
for i in range(rows):
    for j in range(cols):
        si.append(i)
        sj.append(j)

# obstacles
oi, oj = list(), list()
for o in obstacleEnv.obstacles:
    i = o // cols
    j = o - i * cols
    oi.append(i)
    oj.append(j)

# path coords
pi, pj = list(), list()

for s in path:
    i = s // cols
    j = s - i * cols
    pi.append(i)
    pj.append(j)

# plot
plt.gca().invert_yaxis()

plt.plot(sj, si, 'b.', zorder=0)

plt.scatter(oj, oi, marker='H', c='darkmagenta', zorder=1, s=70.0, label='Obstacle')

plt.scatter(pj[0], pi[0], color='red', marker='s', label='start point')
plt.plot(pj, pi, color='green', linestyle='--', label='agent path')
plt.scatter(pj[-1], pi[-1], color='gold', marker='o', label='end point')
plt.legend(loc='upper right')

# save fig
plt.savefig(img_dir + "obstacle_policy.png")

