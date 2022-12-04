import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random

from numpy import sqrt

""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header=0)
    return df


### HELPERS END ###


''' Functions to implement '''


# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    dataset.hist(column=chosen_anime_id)
    plt.title("Rating Counts for Anime id = " + chosen_anime_id)
    plt.show()

    counts = dataset[chosen_anime_id].value_counts()
    return counts



# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    perturbed_counts = counts.copy()
    counts_keys = perturbed_counts.keys()

    # Gets random laplacian noise for all values
    laplacian_noise = np.random.laplace(0, 1 / epsilon)

    # Add random noise generated from Laplace function to actual count
    # noisy_data = counts + laplacian_noise

    for i in counts_keys:
        perturbed_counts[i] += laplacian_noise

    perturbed_counts.hist()
    plt.title("Rating Counts for Anime id = " + "199")
    plt.show()
    return perturbed_counts




# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    bar_keys = actual_hist.keys()
    err_sum = 0
    for bar_key in bar_keys:
        err_sum += abs(noisy_hist[bar_key] - actual_hist[bar_key])

    avg_err = err_sum / len(bar_keys)
    return avg_err


# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    bar_keys = actual_hist.keys()
    err_sum = 0
    for bar_key in bar_keys:
        err_sum += (noisy_hist[bar_key] - actual_hist[bar_key]) ** 2

    mse = err_sum / len(bar_keys)
    return mse


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    avg_err_list = list()
    mse_list = list()
    for eps_value in eps_values:
        perturbed_counts_list = list()
        for _ in range(40):
            perturbed_counts = get_dp_histogram(counts, eps_value)
            avg_err_list.append(calculate_average_error(counts,perturbed_counts))
            mse_list.append(calculate_mean_squared_error(counts,perturbed_counts))




# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    pass


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    pass


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)

    counts = get_histogram(dataset)

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])


if __name__ == "__main__":
    main()
