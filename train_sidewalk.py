import os
import numpy as np
from sidewalk import Sidewalk
# from algo import Q_learning
from tqdm import tqdm
import matplotlib.pyplot as plt

# rows in grid
rows = 6

# columns in grid
cols = 25

# probability of exploiting
epsilon = 0.8

# discount factor
gamma = 0.6

# num actions: up, down, left, right
nA = 4

# learning rate
lr = 0.01

# rows for edge of left and right sidewalk
sL = 0
sR = 5

# directories for policy
policy_dir = "./policy/"
fname_policy = "./policy/sidewalk_policy.npy"

img_dir = "./plots/"


# make dir for policy
if not os.path.exists(policy_dir):
    os.mkdir(policy_dir)

sidewalk = Sidewalk(nRows = rows, nCols = cols, sL = sL,
                    sR = sR, nA = nA, gamma = gamma)

# train if policy doesn't exist
if not os.path.exists(fname_policy):

    # randomly initialize Q table
    Q = np.random.rand(sidewalk.nS, sidewalk.nA)

    # set state-action value of final states to 0
    # TODO make this dynamic
    Q[[24, 49, 74, 99, 124, 149], :] = 0

    theta = 1e-4
    delta2 = float('inf')

    episode = 1

    print("Training")

    state = sidewalk.reset()

    # update Q function until convergence
    while delta2 >= theta:

        delta = 0.0

        print(f"> Episode: {episode}")

        # explore action space
        if np.random.uniform(0, 1) > epsilon:
            action = sidewalk.sample()

        # exploit
        else:
            action = np.argmax(Q[state])

        # take step in env
        obs, r = sidewalk.step(action)

        # update Q table
        prev_value = Q[state, action]

        new_value = prev_value + lr * (r + sidewalk.gamma * np.max(Q[obs]) - prev_value)

        Q[state, action] = new_value

        # update state
        state = obs

        # determine max delta in q values
        delta2 = max(delta, np.abs(new_value - prev_value))

        episode += 1

    # save policy
    np.save(fname_policy, Q)

else:
    # load policy
    Q = np.load(fname_policy)

n_runs = 25

path = list()

state = sidewalk.reset()

path.append(state)

# evaluate policy
for i in range(n_runs):

    action = np.argmax(Q[state])
    obs, r = sidewalk.step(action)

    state = obs

    # add next state to path
    path.append(obs)
    print(path)

path = np.array(path)

# plot policy
si, sj = list(), list()

# state coords
for i in range(rows):
    for j in range(cols):
        si.append(i)
        sj.append(j)

# path coords
pi, pj = list(), list()

for s in path:
    i = s // cols
    j = s - i * cols
    pi.append(i)
    pj.append(j)

# sidewalk coords
left_x, right_x = np.arange(cols), np.arange(cols)
left_y, right_y = np.array([sL] * cols), np.array([sR] * cols)

# plot
plt.gca().invert_yaxis()

plt.plot(sj, si, 'b.', zorder=0)
plt.plot(left_x, left_y, color = 'black', linestyle = '-', label = 'sidewalk')
plt.plot(right_x, right_y, color = 'black', linestyle = '-')

plt.scatter(pj[0], pi[0], color='black', zorder=1, marker='x', label='start')
plt.plot(pj, pi, color = 'green', linestyle = '--', label = 'agent path')
plt.scatter(pj[-1], pi[-1], color='red', zorder=1, marker='x', label='end')

plt.legend(loc = 'upper right')

plt.savefig(img_dir + "sidewalk_policy.png")
