
import random


class Agent:

    def __init__(self, choiceNum: int, epsilon: float, pace: float):
        self.pace = pace
        self.round = [0 for _ in range(choiceNum)]
        self.choiceNum = choiceNum
        self.epsilon = epsilon
        self.q = [0.0 for _ in range(choiceNum)]

    def select_action(self):
        v = random.random()
        if v < self.epsilon:
            action = random.randint(0, self.choiceNum-1)
        else:
            max_q = max(self.q)
            best_actions = [i for i, q in enumerate(self.q) if q == max_q]
            action = best_actions[random.randint(0, len(best_actions)-1)]
        return action

    def update_q(self, action, reward):
        if self.pace is not None:
            self.q[action] = self.q[action] + self.pace*(reward-self.q[action])
        else:
            self.q[action] = (
                (self.q[action]*self.round[action] + reward)/(self.round[action]+1)
            )
            self.round[action] += 1

    def reset(self):
        self.q = [0.0 for _ in range(self.choiceNum)]
        self.round = [0 for _ in range(self.choiceNum)]

