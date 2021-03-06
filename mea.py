import numpy as np
import matplotlib.pyplot as plt
import math
import experiment_testbed


def compute_l(epsilon, delta):
    return int((4 / (epsilon * epsilon)) * (math.log(3 / delta)))


def run_mea(means, epsilon, delta):
    means_list = means.copy()
    rewards_list = list()  # average rewards list
    epsilon = epsilon / 4  # epsilon for first run of algorithm
    delta = delta / 2  # delta for first run of algorithm
    arms_l = len(means_list[0])  # initial number of arms
    optimum_arms = np.argmax(means_list, 1)  # list of best arm in each bandit
    optimum_selection_count = list()
    optimum_arms_curr = optimum_arms
    n_bandits = len(means_list)  # number of bandits

    means_true_curr = means_list

    while arms_l != 1:  # loop till we get best arm
        optimum_arm_selected = 0
        computed_l = compute_l(epsilon, delta)  # get the number of time steps
        rewards_l = np.zeros((n_bandits, arms_l))  # list to store total reward for each arm in each bandit

        for i in range(0, computed_l):
            rewards = np.random.normal(means_true_curr, 1)  # sample reward for each arm
            rewards_l = rewards_l + rewards
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
                    new_means_list[j][k1] = means_true_curr[j][k]  # add arm in new list if average reward more than median
                    if k == optimum_arms_curr[j]:  # check if arm we added is the best arm for that bandit
                        optimum_arm_selected += 1
                    k1 += 1

        means_true_curr = new_means_list  # update old true means list with new list
        if arms_l % 2 == 0:
            arms_l = int(arms_l / 2)  # update number of arms
        else:
            arms_l = int(arms_l / 2) + 1  # update number of arms
        optimum_arms_curr = np.argmax(means_true_curr, 1)  # get best arms in new means list
        optimum_selection_count.append(optimum_arm_selected * 100 / n_bandits)
        epsilon = (3 / 4) * epsilon  # epsilon for next round
        delta = delta / 2  # delta for next round

    return means_true_curr, rewards_list, optimum_selection_count


bandits = 2000
arms = 10
means_list, times_selected = experiment_testbed.initialize(bandits, arms)
epsilons = [0.65, 1.1, 1.2]  # list of epsilons
deltas = [0.1, 0.2, 0.3]  # list of deltas

fig_rewards = plt.figure().add_subplot(111)  # graph of average reward
fig_optimum = plt.figure().add_subplot(111)  # graph of % optimum arm selection

for i in range(0, 3):
    curr_epsilon = epsilons[i]
    curr_delta = deltas[i]
    best_arms, rewards, optimum_list = run_mea(means_list, curr_epsilon, curr_delta)  # get the results
    fig_rewards.plot(range(len(rewards)), rewards, label=r'$\epsilon$ = ' + str(curr_epsilon) + ', '
                                                   + r'$\delta$ = ' + str(curr_delta))
    fig_optimum.plot(range(1, len(optimum_list) + 1), optimum_list, label=r'$\epsilon$ = ' + str(curr_epsilon) + ', '
                                                                          + r'$\delta$ = ' + str(curr_delta))


fig_rewards.set_xlabel('Steps')
fig_rewards.set_ylabel('Average Reward')
fig_rewards.title.set_text('Average Reward vs Steps')
fig_rewards.legend(loc='lower right')

fig_optimum.set_xlabel('Steps')
fig_optimum.set_ylabel(r'$\%$ Optimum Action')
fig_optimum.title.set_text(r'$\%$ Optimum Action vs Steps')
fig_optimum.legend(loc='lower right')
plt.show()
