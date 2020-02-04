import experiment_testbed
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def eps_rewards(means, times_list, epsilon, optimal_list):
    means1 = means.copy()
    times_selected1 = times_list.copy()
    n_bandits = len(means1)
    n_arms = len(means1[0])
    q = np.zeros((n_bandits, n_arms))
    reward_list = list()
    optimal_selection_list = list()
    n_iterations = 10000

    for i in range(0, n_iterations):
        print("eps: " + str(i))
        curr_reward = 0
        optimal_selection = 0
        for j in range(0, n_bandits):
            x = random.uniform(0, 1)

            if x > epsilon:
                action = np.argmax(q[j])
            else:
                action = random.randint(0, n_arms-1)

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
    n_bandits = len(means1)
    n_arms = len(means1[0])
    q = np.zeros((n_bandits, n_arms))
    reward_list = list()
    optimal_selection_list = list()
    n_iterations = 10000

    for i in range(0, n_iterations):
        print('softmax: ' + str(i))
        curr_reward = 0
        optimal_selection = 0
        for j in range(0, n_bandits):
            q_exp = np.exp(q[j] / temperature)
            softmax_probability = q_exp / sum(q_exp)
            action = np.random.choice(range(n_arms), 1, p=softmax_probability)[0]

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
    n_iterations = 10000

    # init_selection = np.random.normal(means1, 1)
    # reward_list.append(np.mean(init_selection))

    for i in range(1, n_iterations + 1):
        print('ucb: ' + str(i))
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


def compute_l(epsilon, delta):
    return int((4 / (epsilon * epsilon)) * (math.log(3 / delta)))


def run_mea(means, epsilon, delta):
    means_list = means.copy()
    x_axis = 0  # variable to store the total number of steps in the algorithm
    rewards_list = list()  # average rewards list
    epsilon = epsilon / 4  # epsilon for first run of algorithm
    delta = delta / 2  # delta for first run of algorithm
    arms_l = len(means_list[0])  # initial number of arms
    optimum_arms = np.argmax(means_list, 1)  # list of best arm in each bandit
    optimum_selection_count = list()
    optimum_arms_l = optimum_arms
    n_bandits = len(means_list)  # number of bandits

    means_true_l = means_list

    while arms_l != 1:  # loop till we get best arm
        optimum_arm_selected = 0
        computed_l = compute_l(epsilon, delta)  # get the number of time steps
        rewards_l = np.zeros((n_bandits, arms_l))  # list to store total reward for each arm in each bandit

        for i in range(0, computed_l):
            print('mea: ' + str(arms_l) + str(i))
            rewards = np.random.normal(means_true_l, 1)  # sample reward for each arm
            rewards_l = rewards_l + rewards
            x_axis += 1
            rewards_list.append(np.mean(rewards))

        bandit_rewards_avg = rewards_l / computed_l  # average reward for each arm
        medians = np.median(bandit_rewards_avg, 1)  # median in each bandit

        # new list to store arms with reward more than median
        if arms_l % 2 == 0:
            new_means_list = np.zeros((n_bandits, int(arms_l - arms_l / 2)))
        else:
            new_means_list = np.zeros((n_bandits, int(arms_l - arms_l / 2) + 1))

        for j in range(0, n_bandits):
            k1 = 0
            for k in range(0, arms_l):
                if bandit_rewards_avg[j][k] >= medians[j]:
                    new_means_list[j][k1] = means_true_l[j][k]  # add arm in new list if average reward more than median
                    if k == optimum_arms_l[j]:  # check if arm we added is the best arm for that bandit
                        optimum_arm_selected += 1
                    k1 += 1

        means_true_l = new_means_list  # update old true means list with new list
        if arms_l % 2 == 0:
            arms_l = int(arms_l / 2)  # update number of arms
        else:
            arms_l = int(arms_l / 2) + 1  # update number of arms
        optimum_arms_l = np.argmax(means_true_l, 1)  # get best arms in new means list
        optimum_selection_count.append(optimum_arm_selected * 100 / n_bandits)
        epsilon = (3 / 4) * epsilon  # epsilon for next round
        delta = delta / 2  # delta for next round

    return means_true_l, rewards_list, optimum_selection_count, x_axis


fig_rewards = plt.figure().add_subplot(111)
fig_optimal = plt.figure().add_subplot(111)

x_axis = list()
n_iterations = 10000

for i in range(0, n_iterations):
    x_axis.append(i)

n_bandits = 2000
n_arms = 1000
print('starting greedy')
means_list, times_selected = experiment_testbed.initialize(n_bandits, n_arms)
optimal_list = np.argmax(means_list, 1)
eps_rewards, eps_optimal = eps_rewards(means_list, times_selected, 0.1, optimal_list)

fig_rewards.plot(x_axis, eps_rewards, label=r'$\epsilon$-greedy: $\epsilon$=' + str(0.1))
fig_optimal.plot(x_axis, eps_optimal, label=r'$\epsilon$-greedy: $\epsilon$=' + str(0.1))
print('greedy done')

means_list, times_selected = experiment_testbed.initialize(n_bandits, n_arms)
optimal_list = np.argmax(means_list, 1)
softmax_rewards, softmax_optimal = softmax_rewards(means_list, times_selected, 0.1, optimal_list)

fig_rewards.plot(x_axis, softmax_rewards, label='softmax: temperature=0.1')
fig_optimal.plot(x_axis, softmax_optimal, label='softmax: temperature=0.1')
print('softmax done')

means_list, times_selected = experiment_testbed.initialize(n_bandits, n_arms)
optimal_list = np.argmax(means_list, 1)
ucb_rewards, ucb_optimal = ucb_rewards(means_list, times_selected, optimal_list, 1)
fig_rewards.plot(x_axis, ucb_rewards, label='UCB-1')
fig_optimal.plot(x_axis, ucb_optimal, label='UCB-1')
print('ucb done')

# means_list, times_selected = experiment_testbed.initialize(n_bandits, n_arms)
# curr_epsilon = 0.65
# curr_delta = 0.1
# best_arms, rewards, optimum_list, x_axis = run_mea(means_list, curr_epsilon, curr_delta)  # get the results
# fig_rewards.plot(range(x_axis), rewards, label='MEA: ' + r'$\epsilon$ = ' + str(curr_epsilon) + ', '
#                                                + r'$\delta$ = ' + str(curr_delta))
# fig_optimal.plot(range(1, len(optimum_list) + 1), optimum_list, label='MEA: ' + r'$\epsilon$ = ' + str(curr_epsilon) + ', '
#                                                                       + r'$\delta$ = ' + str(curr_delta))

fig_rewards.set_xlabel('Steps')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Steps')
fig_rewards.legend(loc='lower right')

fig_optimal.set_xlabel('Steps')
fig_optimal.set_ylabel(r'$\%$ Optimal Action')
fig_optimal.title.set_text(r'$\%$ Optimum Action vs Steps')
fig_optimal.legend(loc='lower right')

plt.show()
