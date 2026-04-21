import numpy as np

def total_reward(rewards):
    # Compute the total reward earned over an episode.
    return float(np.sum(rewards))

def average_reward(rewards):
    # Compute the mean reward per step.
    return float(np.mean(rewards))


def cumulative_reward(rewards):
    # Compute the cumulative (running total) reward over time.
    return np.cumsum(rewards)


def num_steps(rewards):
    # Return the number of time steps in the episode.
    return len(rewards)