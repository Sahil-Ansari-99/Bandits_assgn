import numpy as np
import matplotlib.pyplot as plt
import random
import experiment_testbed


def get_rewards(means, times_list, epsilon, optimal_list):
    means1 = means.copy()
    times_selected1 = times_list.copy()
    n_bandits = len(means1)  # number of bandits
    n_arms = len(means1[0])  # number of arms in each bandit
    q = np.zeros((n_bandits, n_arms))  # list to store average rewards obtained by each arm
    reward_list = list()  # list to store the average reward obtained in each step
    optimal_selection_list = list()  # list to store the number of times the most optimal arms were selected in each step
    n_iterations = 1000

    for i in range(0, n_iterations):
        curr_reward = 0  # variable to store total reward obtained in a run
        optimal_selection = 0
        for j in range(0, n_bandits):
            x = random.uniform(0, 1)  # random number between 0 and 1

            if x > epsilon:
                action = np.argmax(q[j])  # choose greedily
            else:
                action = random.randint(0, n_arms - 1)  # choose randomly

            if action == optimal_list[j]:  # check if selected arm is the best arm
                optimal_selection += 1

            reward = np.random.normal(means1[j][action], 1)  # sample the reward
            times_selected1[j][action] += 1  # update number of times the arm was selected
            curr_reward += reward
            q[j][action] = q[j][action] + (reward - q[j][action]) / (times_selected1[j][action])  # update the average reward for that arm

        reward_list.append(curr_reward / n_bandits)  # add average reward to our list
        optimal_selection_list.append(optimal_selection * 100 / n_bandits)  # add percentage of times best arms were selected

    return reward_list, optimal_selection_list


x_axis = list()  # list to generate x-axis
for i in range(0, 1000):
    x_axis.append(i)

eps = [0, 0.01, 0.1]  # list of epsilon values

fig_rewards = plt.figure().add_subplot(111)  # graph for average reward
fig_optimal = plt.figure().add_subplot(111)  # graph for % optimum arm selection


for i in range(0, len(eps)):
    curr_epsilon = eps[i]
    bandits = 2000
    arms = 10
    means_list, times_selected = experiment_testbed.initialize(bandits, arms)  # initialize bandit problem

    optimal_arms = np.argmax(means_list, 1)  # get the best arm in each bandit
    rewards, optimal_selections = get_rewards(means_list, times_selected, curr_epsilon, optimal_arms)  # get the results
    fig_rewards.plot(x_axis, rewards, label=r'$\epsilon$=' + str(curr_epsilon))
    fig_optimal.plot(x_axis, optimal_selections, label=r'$\epsilon$ = ' + str(curr_epsilon))

fig_rewards.set_xlabel('Steps')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Steps')
fig_rewards.legend(loc='lower right')

fig_optimal.set_xlabel('Steps')
fig_optimal.set_ylabel(r'$\%$ Optimal Action')
fig_optimal.title.set_text(r'$\%$ Optimum Action vs Steps')
fig_optimal.legend(loc='lower right')

plt.show()
