from BanditEnv import BanditEnv
from Agent import Agent
import tqdm
import matplotlib.pyplot as plt


def main():

    k = 10
    step = 10000
    experimentNum = 2000

    optimal_action_historys = [0 for _ in range(step)]
    reward_historys = [0.0 for _ in range(step)]

    env = BanditEnv(k, stationary = False)
    agent = Agent(k, epsilon = 0.1, pace = 0.1)

    for _ in tqdm.tqdm(range(experimentNum)):

        env.reset()
        agent.reset()        

        for i in range(step):
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)
            if action == env.optimal_action:
                optimal_action_historys[i] += 1
            reward_historys[i] += reward

    optimal_action_historys = [i/experimentNum for i in optimal_action_historys]
    reward_historys = [i/experimentNum for i in reward_historys]
    
    x = [i for i in range(1, step+1)]
    y_reward = reward_historys
    y_optimal_action = optimal_action_historys

    plt.plot(x, y_reward, color = 'orange', marker = '.', label = "avg_reward")
    plt.plot(x, y_optimal_action, color = 'blue', marker = '.', label = "avg_optimal_action")
    plt.legend(loc = 'lower right')
    plt.xlabel("epoch")
    plt.ylabel("reward")
    plt.show()


if __name__ == '__main__':
    main()
