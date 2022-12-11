import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random

from numpy import sqrt, exp

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
    counts = dataset[chosen_anime_id].value_counts()
    x = counts.keys().tolist()
    y = counts.values.tolist()
    plt.bar(x, y)
    # dataset.hist(column=chosen_anime_id, bins= key_size+1)
    plt.title("Rating Counts for Anime id = " + chosen_anime_id)
    plt.show()
    return counts


# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    sensitivity = 2

    perturbed_counts = counts.copy()
    counts_keys = perturbed_counts.keys()

    for bin_key in counts_keys:
        # Gets random laplacian noise for all values
        laplacian_noise = np.random.laplace(0, sensitivity / epsilon)

        perturbed_counts[bin_key] += laplacian_noise

    x = perturbed_counts.keys().tolist()
    y = perturbed_counts.values.tolist()
    #plt.bar(x, y)
    #plt.title("Rating Counts for Anime id = " + "199 with DP")
    #plt.show()
    return perturbed_counts


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    err_sum = 0
    for bar_key in range(len(actual_hist)):
        err_sum += abs(noisy_hist[bar_key] - actual_hist[bar_key])

    avg_err = err_sum / len(actual_hist)
    return avg_err


# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    err_sum = 0
    for bar_key in range(len(actual_hist)):
        err_sum += (noisy_hist[bar_key] - actual_hist[bar_key]) ** 2

    mse = err_sum / len(actual_hist)
    return mse


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    avg_err_list = list()
    mse_list = list()
    for eps_value in eps_values:
        temp_avg_err_list = list()
        temp_mse_lise = list()
        for _ in range(40):
            perturbed_counts = get_dp_histogram(counts, eps_value)
            perturbed_counts = perturbed_counts.tolist()
            counts_list = counts.tolist()
            temp_avg_err_list.append(calculate_average_error(counts_list, perturbed_counts))
            temp_mse_lise.append(calculate_mean_squared_error(counts_list, perturbed_counts))
        average_avg_err = sum(temp_avg_err_list) / 40
        average_mse = sum(temp_mse_lise) / 40

        avg_err_list.append(average_avg_err)
        mse_list.append(average_mse)

    return avg_err_list, mse_list


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    sensitivity = 2
    prob_denominator = 0

    anime_10rate_dict = {}
    anime_10rate_prob_dict = {}
    for anime_id in dataset:
        if anime_id != "user_id":
            counts = dataset[anime_id].value_counts()
            anime_10rate_dict[anime_id] = counts[10]

    for value in anime_10rate_dict.values():
        e_numerator = epsilon * value
        e_denominator = 2 * sensitivity
        prob_denominator += exp(e_numerator / e_denominator)

    for key, value in anime_10rate_dict.items():
        e_numerator = epsilon * value
        e_denominator = 2 * sensitivity

        prob_numerator = exp(e_numerator / e_denominator)
        prob_exponential = prob_numerator / prob_denominator

        anime_10rate_prob_dict[key] = prob_exponential

    list_of_keys = list(anime_10rate_prob_dict.keys())
    list_of_values = list(anime_10rate_prob_dict.values())
    r_star = np.random.choice(list_of_keys, 1, p=list_of_values)
    most_common_10rate_anime_id = r_star
    return most_common_10rate_anime_id


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    accuracy_list = list()
    anime_10rate_dict = {}

    for anime_id in dataset:
        if anime_id != "user_id":
            counts = dataset[anime_id].value_counts()
            anime_10rate_dict[anime_id] = counts[10]

    fin_max = max(anime_10rate_dict, key=anime_10rate_dict.get)

    for eps_value in eps_values:
        temp_accuracy_list = list()
        for _ in range(1000):
            guess_of_anime_id = most_10rated_exponential(dataset, eps_value)

            if guess_of_anime_id == fin_max:
                temp_accuracy_list.append(1)
            else:
                temp_accuracy_list.append(0)

        accuracy_list.append(sum(temp_accuracy_list) / 1000)

    return accuracy_list


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
