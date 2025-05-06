from BanditEnv import BanditEnv
from Agent import Agent
import tqdm
import matplotlib.pyplot as plt


def main():

    k = 10
    step = 1000
    experimentNum = 2000

    optimal_action_historys = [0 for _ in range(1000)]
    reward_historys = [0 for _ in range(1000)]

    env = BanditEnv(k, stationary = True)
    agent = Agent(k, epsilon = 0.1, pace = None)

    for _ in tqdm.tqdm(range(experimentNum)):

        env.reset()
        agent.reset()        

        for _ in range(step):
            action = agent.select_action()
            reward = env.step(action)
            agent.update_q(action, reward)

        action_history, reward_history = env.export_history()
        optimal_action_historys = [x+y for x, y in zip(optimal_action_historys, [i == env.optimal_action for i in action_history ])]
        reward_historys = [x+y for x, y in zip(reward_historys, reward_history)]

    optimal_action_historys = [i/experimentNum for i in optimal_action_historys]
    reward_historys = [i/experimentNum for i in reward_historys]
    
    x = [i for i in range(1, step+1)]
    y_reward = [sum(reward_history[0:i])/i for i in range(1, step+1)]
    y_optimal_action = [sum(optimal_action_historys[0:i])/i for i in range(1, step+1)] 

    plt.plot(x, y_reward, color = 'orange', marker = '.', label = "avg_reward")
    plt.plot(x, y_optimal_action, color = 'blue', marker = '.', label = "avg_optimal_action")
    plt.legend(loc = 'lower right')
    plt.xlabel("epoch")
    plt.ylabel("reward")
    plt.show()


if __name__ == '__main__':
    main()
