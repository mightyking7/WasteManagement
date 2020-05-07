import numpy as np
from env import GridWorld
from algo import Q_learning

# rows = 6
# cols = 25
# epsilon = 0.9

# 1 is an obstacle, 0 is an empty cell
# dist = [1,0,0,0,0]
#
# num_tiles = rows * cols
#
# world = np.array([np.random.choice(dist) for i in range (num_tiles)]).reshape((rows, cols))

grid_world = GridWorld()
behavior_policy = RandomPolicy()

n_episodes = 50


trajs = []

for _ in range(n_episodes):
    s = grid_world.reset()
    traj = []
    done = False

    while not done :
        a = behavior_policy.action(s)
        next_s, r, done = grid_world.step(a)
        traj.append((s, a, r, next_s))
        s = next_s
    trajs.append(traj)

Q = Q_learning(grid_world.spec, trajs, behavior_policy, n=1, alpha=0.01,
               initQ=np.zeros((grid_world.spec.nS, grid_world.spec.nA)))

# plot policy


