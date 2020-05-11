from trainer import Trainer
from forward import ForwardEnv
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

# episodes to train for
n_episodes = 5000

# directory to store plots
img_dir = "./plots/"

forward = ForwardEnv(nRows=rows, nCols=cols, nA=nA, gamma=gamma)
forwardPi = Trainer(env=forward)

# train and save the policy
Q = forwardPi.train("forward_policy.npy", epsilon=epsilon, lr=lr, n_episodes=n_episodes)

# test the policy
path = forwardPi.eval(Q)

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

# plot
plt.gca().invert_yaxis()
plt.plot(sj, si, 'b.', zorder=0)
plt.scatter(pj[0], pi[0], color='red', marker='s', label='start point')
plt.plot(pj, pi, color='green', linestyle='--', label='agent path')
plt.scatter(pj[-1], pi[-1], color='gold', marker='o', label='end point')
plt.legend(loc='upper right')

# save fig
plt.savefig(img_dir + "forward_policy.png")

