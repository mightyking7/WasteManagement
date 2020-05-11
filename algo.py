import numpy as np
from env import GridWorld

def Q_learning(env: GridWorld,
               epsilon: float,
               lr: float,
               initQ: np.ndarray,
               converge = False) -> (np.ndarray, float):
    """
    Performs Q learning for single episode in environment and returns learned policy.
    :param env: GridWorld subclass
    :param epsilon: Exploitation rate
    :param lr: learning rate
    :param initQ: Q table to update
    :param converge: Flag to determine if delta of Q-values need to be tracked
    for convergence within set bound.
    :return: Updated Q table after a single episode of training and maximum change for any Q-value
    """

    # keep track of maximum state-action value change
    delta = 0.0

    state = env.reset()

    done = False

    while not done:

        # explore action space
        if np.random.uniform(0, 1) > epsilon:
            action = env.sample()

        # exploit
        else:
            action = np.argmax(initQ[state])

        # take step in env
        obs, r, done = env.step(action)

        # update Q table
        prev_value = initQ[state, action]

        new_value = prev_value + lr * (r + env.gamma * np.max(initQ[obs]) - prev_value)

        initQ[state, action] = new_value

        # update state
        state = obs

        if converge:
            delta = max(delta, np.abs(new_value - prev_value))

    return initQ, delta