"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

EPIS = 200

def update():
    global plot_y
    plot_y.clear()

    for episode in range(EPIS):       
        # initial observation
        observation = env.reset()
        reward_total = 0
        

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            reward_total += reward

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                plot_y.append(reward_total)
                print(f"Episode {episode}:")
                print(f"Total rewards(lr={e_test}): {reward_total}")
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    plot_y = list(range(EPIS))
    fig, ax = plt.subplots()
    # Data for plotting
    ax.set(xlabel='episode (s)', ylabel='Total Rewards',
           title='Total rewards at each episode')
    ax.grid()

    for e_test in [0.8, 0.85, 0.9, 0.95]:#
        env = Maze()
        RL = QLearningTable(actions=list(range(env.n_actions)), learning_rate=0.5, e_greedy=e_test)

        env.after(100, update)
        env.mainloop()
        ax.plot(range(EPIS), plot_y, label='lr='+str(e_test))

    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')    
    fig.savefig("e-greedy.png")
    plt.show()