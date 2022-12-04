# Implementing Laplace mechanism on Adult dataset by adding Laplacian random noise
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from HW2.part2_skeleton import get_dp_histogram, read_dataset, get_histogram, calculate_average_error, \
    calculate_mean_squared_error


def epsilon_experiment(counts, eps_values: list):
    avg_err_list = list()
    mse_list = list()
    for eps_value in eps_values:
        perturbed_counts_list = list()
        for _ in range(40):
            perturbed_counts = get_dp_histogram(counts, eps_value)
            avg_err_list.append(calculate_average_error(counts, perturbed_counts))
            mse_list.append(calculate_mean_squared_error(counts, perturbed_counts))
        print(avg_err_list)
        print(mse_list)
        return

filename = "anime-dp.csv"
dataset = pd.read_csv(filename)


datacount = dataset["199"].value_counts()
key_size = datacount.keys()
print(key_size)


dataset.hist(column="199", bins=len(key_size)+1)
plt.show()