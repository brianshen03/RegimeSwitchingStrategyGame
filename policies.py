import random

# SYSTEMATIC POLICIES - WEEK 5 CHECKPOINT 

"""
Systematic policies do not learn from rewards, so the update functions do not do anything. They are there to maintain a uniform interface 
for when we implement adaptive policies. 
"""

class StaticPolicy:
    """
    Policy that always chooses the same action.

    This is the simplest possible baseline strategy. It does not react to rewards, history, or regime changes.

    Example:
    - if action = 0, this policy always chooses action 0
    - if action = 2, this policy always chooses action 2
    """

    def __init__(self, action):
        self.action = action

    def select_action(self):
        return self.action

    def update(self, action, reward):
        pass


class RandomPolicy:
    """
    Policy that chooses an action uniformly at random each step.

    This is another simple baseline.
    It does not learn from rewards, but unlike StaticPolicy,
    it does not commit to one action forever.
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self):
        
        # Choose one action uniformly at random.
        return random.randint(0, self.num_actions - 1)

    def update(self, action, reward):
        pass


class RoundRobinPolicy:
    """
    Policy that cycles through actions in a fixed repeating order.

    Example with 3 actions:
    0, 1, 2, 0, 1, 2, ...

    """

    def __init__(self, num_actions):
        self.num_actions = num_actions
        # Internal counter to track how many selections have been made and determines which action comes next in the cycle.
        self.t = 0

    def select_action(self):
        
        # Choose the next action in cyclic order.
        action = self.t % self.num_actions
        self.t += 1
        return action

    def update(self, action, reward):
        pass



# ADAPTIVE / DISCRETIONARY POLICIES - WEEK 7 TODO