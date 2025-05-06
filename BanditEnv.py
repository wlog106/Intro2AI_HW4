import random


class BanditEnv:

    def __init__(self, armNum: int, stationary: bool):
        self.stationary = stationary
        self.armNum = armNum
        self.mu = [random.gauss(0, 1) for _ in range(armNum)]
        self.optimal_action = self.mu.index(max(self.mu))
        self.action_history = []
        self.reward_history = []

    def reset(self):
        self.mu = [random.gauss(0, 1) for _ in range(self.armNum)]
        self.optimal_action = self.mu.index(max(self.mu))
        self.action_history = []
        self.reward_history = []

    def export_history(self):
        return self.action_history, self.reward_history

    def step(self, action: int):
        reward = random.gauss(self.mu[action], 1)
        self.action_history.append(action)
        self.reward_history.append(reward)
        if not self.stationary:
            self.mu = [mean+random.gauss(0, 0.01) for mean in self.mu]
            self.optimal_action = self.mu.index(max(self.mu))
        return reward
