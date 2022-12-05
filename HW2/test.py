# Implementing Laplace mechanism on Adult dataset by adding Laplacian random noise
from math import exp

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from HW2.part2_skeleton import get_dp_histogram, read_dataset, get_histogram, calculate_average_error, \
    calculate_mean_squared_error, most_10rated_exponential


def epsilon_experiment(counts, eps_values: list):
    avg_err_list = list()
    mse_list = list()
    for eps_value in eps_values:
        avg_err = 0
        mse = 0
        for _ in range(40):
            perturbed_counts = get_dp_histogram(counts, eps_value)
            avg_err += (calculate_average_error(counts, perturbed_counts))
            mse += (calculate_mean_squared_error(counts, perturbed_counts))
        avg_err /= 40
        mse /= 40
        avg_err_list.append(avg_err)
        mse_list.append(mse)
    print(avg_err_list)
    print(mse_list)


filename = "anime-dp.csv"
chosen_anime_id = "199"
eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
dataset = read_dataset(filename)


counts = get_histogram(dataset,chosen_anime_id)
dp_counts = get_dp_histogram(counts,0.001)
#epsilon_experiment(counts,eps_values)

most_10rated_exponential(dataset, 0.0001)