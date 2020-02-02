import experiment_testbed
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def eps_rewards(means, times_list, epsilon, optimal_list):
    means1 = means.copy()
    times_selected1 = times_list.copy()
    q = np.zeros((2000, 10))
    reward_list = list()
    optimal_selection_list = list()
    n_bandits = len(means1)
    n_iterations = 1000

    for i in range(0, n_iterations):
        curr_reward = 0
        optimal_selection = 0
        for j in range(0, n_bandits):
            x = random.uniform(0, 1)

            if x > epsilon:
                action = np.argmax(q[j])
            else:
                action = random.randint(0, 9)

            if action == optimal_list[j]:
                optimal_selection += 1

            reward = np.random.normal(means1[j][action], 1)
            times_selected1[j][action] += 1
            curr_reward += reward
            q[j][action] = q[j][action] + (reward - q[j][action]) / (times_selected1[j][action])

        reward_list.append(curr_reward / n_bandits)
        optimal_selection_list.append(optimal_selection * 100 / n_bandits)

    return reward_list, optimal_selection_list


def softmax_rewards(means, times_list, temperature, optimal_list):
    means1 = means.copy()
    times_selected1 = times_list.copy()
    q = np.zeros((2000, 10))
    reward_list = list()
    optimal_selection_list = list()
    n_bandits = len(means1)
    n_iterations = 1000

    for i in range(0, n_iterations):
        curr_reward = 0
        optimal_selection = 0
        for j in range(0, n_bandits):
            q_exp = np.exp(q[j] / temperature)
            softmax_probability = q_exp / sum(q_exp)
            action = np.random.choice(range(10), 1, p=softmax_probability)[0]

            if action == optimal_list[j]:
                optimal_selection += 1

            reward = np.random.normal(means1[j][action], 1)
            times_selected1[j][action] += 1
            curr_reward += reward
            q[j][action] = q[j][action] + (reward - q[j][action]) / (times_selected1[j][action])

        reward_list.append(curr_reward / n_bandits)
        optimal_selection_list.append(optimal_selection * 100 / n_bandits)

    return reward_list, optimal_selection_list


def ucb_rewards(means, times_list, optimal_list, confidence):
    means1 = means.copy()
    n_bandits = len(means1)
    n_arms = len(means1[0])
    times_selected1 = times_list.copy()
    q = np.zeros((n_bandits, n_arms))
    reward_list = list()
    optimal_selection_list = list()
    n_iterations = 1000

    # init_selection = np.random.normal(means1, 1)
    # reward_list.append(np.mean(init_selection))

    for i in range(1, n_iterations + 1):
        print(i)
        curr_reward = 0
        optimal_selection = 0
        for j in range(0, n_bandits):
            ucb_list = list()
            for k in range(n_arms):
                ucb_list.append(q[j][k] + confidence * math.sqrt((2 * math.log(i)) / times_selected1[j][k]))
            action = np.argmax(ucb_list)

            if action == optimal_list[j]:
                optimal_selection += 1

            reward = np.random.normal(means1[j][action], 1)
            times_selected1[j][action] += 1
            curr_reward += reward
            q[j][action] = q[j][action] + (reward - q[j][action]) / (times_selected1[j][action])

        reward_list.append(curr_reward / n_bandits)
        optimal_selection_list.append(optimal_selection * 100 / n_bandits)

    return reward_list, optimal_selection_list


fig_rewards = plt.figure().add_subplot(111)
fig_optimal = plt.figure().add_subplot(111)

x_axis = list()
n_iterations = 1000

for i in range(0, n_iterations):
    x_axis.append(i)

n_bandits = 2000
n_arms = 1000
print('starting greedy')
means_list, times_selected = experiment_testbed.initialize(n_bandits, n_arms)
optimal_list = np.argmax(means_list, 1)
eps_rewards, eps_optimal = eps_rewards(means_list, times_selected, 0.1, optimal_list)

fig_rewards.plot(x_axis, eps_rewards, label=r'$\epsilon$-greedy')
fig_optimal.plot(x_axis, eps_optimal, label=r'$\epsilon$-greedy')
print('greedy done')

means_list, times_selected = experiment_testbed.initialize(n_bandits, n_arms)
optimal_list = np.argmax(means_list, 1)
softmax_rewards, softmax_optimal = softmax_rewards(means_list, times_selected, 0.1, optimal_list)

fig_rewards.plot(x_axis, softmax_rewards, label='softmax')
fig_optimal.plot(x_axis, softmax_optimal, label='softmax')
print('softmax done')

means_list, times_selected = experiment_testbed.initialize(n_bandits, n_arms)
optimal_list = np.argmax(means_list, 1)
ucb_rewards, ucb_optimal = ucb_rewards(means_list, times_selected, optimal_list, 1)

fig_rewards.plot(x_axis, ucb_rewards, label='UCB-1')
fig_optimal.plot(x_axis, ucb_optimal, label='UCB-1')
print('ucb done')

fig_rewards.set_xlabel('Steps')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Steps')
fig_rewards.legend(loc='lower right')

fig_optimal.set_xlabel('Steps')
fig_optimal.set_ylabel(r'$\%$ Optimal Action')
fig_optimal.title.set_text(r'$\%$ Optimum Action vs Steps')
fig_optimal.legend(loc='lower right')

plt.show()
