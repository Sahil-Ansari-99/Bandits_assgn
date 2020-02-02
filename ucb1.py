import numpy as np
import matplotlib.pyplot as plt
import random
import math
import experiment_testbed


def get_rewards(means, times_list, optimal_list, confidence):
    means1 = means.copy()
    times_selected1 = times_list.copy()
    n_bandits = len(means1)  # number of bandits
    n_arms = len(means1[0])  # number of arms
    q = np.zeros((n_bandits, n_arms))  # average reward for each arm in all bandits
    reward_list = list()  # average reward list
    optimal_selection_list = list()
    n_iterations = 1000

    # init_selection = np.random.normal(means1, 1)
    # reward_list.append(np.mean(init_selection))

    for i in range(1, n_iterations + 1):
        curr_reward = 0
        optimal_selection = 0
        for j in range(0, n_bandits):
            ucb_list = list()
            for k in range(n_arms):
                # upper confidence bound for each arm
                ucb_list.append(q[j][k] + confidence * math.sqrt((2 * math.log(i)) / times_selected1[j][k]))
            action = np.argmax(ucb_list)  # arm with highest ucb

            if action == optimal_list[j]:
                optimal_selection += 1

            reward = np.random.normal(means1[j][action], 1)  # sample reward
            times_selected1[j][action] += 1
            curr_reward += reward
            q[j][action] = q[j][action] + (reward - q[j][action]) / (times_selected1[j][action])  # update average reward

        reward_list.append(curr_reward / n_bandits)
        optimal_selection_list.append(optimal_selection * 100 / n_bandits)

    return reward_list, optimal_selection_list


x_axis = list()  # x= axis list
for i in range(0, 1000):
    x_axis.append(i)


fig_rewards = plt.figure().add_subplot(111)  # graph of average reward
fig_optimal = plt.figure().add_subplot(111)  # graph of % optimum arm selection

bandits = 2000
arms = 10
means_list, times_selected = experiment_testbed.initialize(bandits, arms)

confidence_list = [1, 1.2, 0.95]  # list of confidence values

for i in range(0, len(confidence_list)):
    curr_confidence = confidence_list[i]
    optimal_arms = np.argmax(means_list, 1)  # lis of best arm in each bandit
    rewards, optimal_selections = get_rewards(means_list, times_selected, optimal_arms, curr_confidence)  # get the results
    fig_rewards.plot(x_axis, rewards, label='confidence = ' + str(curr_confidence))
    fig_optimal.plot(x_axis, optimal_selections, label='confidence = ' + str(curr_confidence))

fig_rewards.set_xlabel('Steps')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Steps')
fig_rewards.legend(loc='lower right')

fig_optimal.set_xlabel('Steps')
fig_optimal.set_ylabel(r'$\%$ Optimal Action')
fig_optimal.title.set_text(r'$\%$ Optimum Action vs Steps')
fig_optimal.legend(loc='lower right')

plt.show()
