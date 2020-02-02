import numpy as np
import matplotlib.pyplot as plt
import math
import experiment_testbed


def compute_l(epsilon, delta):
    return int((4 / (epsilon * epsilon)) * (math.log(3 / delta)))


def run_mea(means, epsilon, delta):
    means_list = means.copy()
    x_axis = 0
    rewards_list = list()
    epsilon = epsilon / 4
    delta = delta / 2
    arms_l = len(means_list[0])
    optimum_arms = np.argmax(means_list, 1)
    optimum_selection_count = list()
    optimum_arms_l = optimum_arms
    n_bandits = len(means_list)

    means_true_l = means_list

    while arms_l != 1:
        optimum_arm_selected = 0
        computed_l = compute_l(epsilon, delta)
        rewards_l = np.zeros((n_bandits, arms_l))

        for i in range(0, computed_l):
            rewards = np.random.normal(means_true_l, 1)
            rewards_l = rewards_l + rewards
            x_axis += 1
            rewards_list.append(np.mean(rewards))

        bandit_rewards_avg = rewards_l / computed_l
        medians = np.median(bandit_rewards_avg, 1)

        if arms_l % 2 == 0:
            new_means_list = np.zeros((n_bandits, int(arms_l - arms_l / 2)))
        else:
            new_means_list = np.zeros((n_bandits, int(arms_l - arms_l / 2) + 1))

        for j in range(0, n_bandits):
            k1 = 0
            for k in range(0, arms_l):
                if bandit_rewards_avg[j][k] >= medians[j]:
                    new_means_list[j][k1] = means_true_l[j][k]
                    if k == optimum_arms_l[j]:
                        optimum_arm_selected += 1
                    k1 += 1

        means_true_l = new_means_list
        if arms_l % 2 == 0:
            arms_l = int(arms_l / 2)
        else:
            arms_l = int(arms_l / 2) + 1
        optimum_arms_l = np.argmax(means_true_l, 1)
        optimum_selection_count.append(optimum_arm_selected * 100 / n_bandits)
        epsilon = (3 / 4) * epsilon
        delta = delta / 2

    return means_true_l, rewards_list, optimum_selection_count, x_axis


bandits = 2000
arms = 10
means_list, times_selected = experiment_testbed.initialize(bandits, arms)
epsilons = [1, 1.1, 1.2]
deltas = [0.1, 0.2, 0.3]

fig_rewards = plt.figure().add_subplot(111)
fig_optimum = plt.figure().add_subplot(111)

for i in range(0, 3):
    curr_epsilon = epsilons[i]
    curr_delta = deltas[i]
    best_arms, rewards, optimum_list, x_axis = run_mea(means_list, curr_epsilon, curr_delta)
    fig_rewards.plot(range(x_axis), rewards, label=r'$\epsilon$ = ' + str(curr_epsilon) + ', '
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
