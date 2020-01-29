import numpy as np


def initialize(bandits, arms):
    means_list = list()
    times_selected = list()
    for i in range(0, bandits):
        curr_list = list()
        times = list()
        for j in range(0, arms):
            mean = np.random.normal(0, 1)
            times.append(1)
            curr_list.append(mean)
        means_list.append(curr_list)
        times_selected.append(times)

    return means_list, times_selected
