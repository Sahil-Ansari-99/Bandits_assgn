import numpy as np
import matplotlib.pyplot as plt
import random
import experiment_testbed


def get_rewards(means, times_list, temperature, optimal_list):
    means1 = means.copy()
    times_selected1 = times_list.copy()
    n_bandits = len(means1)
    n_arms = len(means1[0])
    q = np.zeros((n_bandits, n_arms))  # list of average reward of each arm
    reward_list = list()  # list of average reward obtained in each step
    optimal_selection_list = list()  # list of count of best arm selection
    n_iterations = 1000  # number of steps

    for i in range(0, n_iterations):
        curr_reward = 0
        optimal_selection = 0
        for j in range(0, n_bandits):
            q_exp = np.exp(q[j] / temperature)  # compute exponent of each arm in each bandit
            softmax_probability = q_exp / sum(q_exp)  # gibbs distribution probability for each arm
            action = np.random.choice(range(10), 1, p=softmax_probability)[0]  # choose action randomly based on gibbs distribution

            if action == optimal_list[j]:  # check if selected arm is best arm for that bandit
                optimal_selection += 1

            reward = np.random.normal(means1[j][action], 1)  # sample reward
            times_selected1[j][action] += 1
            curr_reward += reward
            q[j][action] = q[j][action] + (reward - q[j][action]) / (times_selected1[j][action])  # update average reward for selected arm

        reward_list.append(curr_reward / n_bandits)
        optimal_selection_list.append(optimal_selection * 100 / n_bandits)

    return reward_list, optimal_selection_list


x_axis = list()  # x-axis list
for i in range(0, 1000):
    x_axis.append(i)

temperatures = [0.01, 0.1, 1]  # list of temperatures

fig_rewards = plt.figure().add_subplot(111)  # graph of average reward
fig_optimal = plt.figure().add_subplot(111)  # graph of % optimum amr selection

for i in range(0, len(temperatures)):
    curr_temp = temperatures[i]
    bandits = 2000
    arms = 10
    means_list, times_selected = experiment_testbed.initialize(bandits, arms)  # initialize bandit problem

    optimal_arms = np.argmax(means_list, 1)  # get the best arms in each bandit
    rewards, optimal_selections = get_rewards(means_list, times_selected, curr_temp, optimal_arms)  # get the results
    fig_rewards.plot(x_axis, rewards, label='Temperature = ' + str(curr_temp))
    fig_optimal.plot(x_axis, optimal_selections, label='Temperature = ' + str(curr_temp))

fig_rewards.set_xlabel('Steps')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Steps')
fig_rewards.legend(loc='lower right')

fig_optimal.set_xlabel('Steps')
fig_optimal.set_ylabel(r'$\%$ Optimal Action')
fig_optimal.title.set_text(r'$\%$ Optimum Action vs Steps')
fig_optimal.legend(loc='lower right')

plt.show()
