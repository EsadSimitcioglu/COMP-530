import time

import numpy as np

from HW1.test import random_anonymizer, clustering_anonymizer, cost_LM, cost_MD, read_dataset

# importing the required module
import matplotlib.pyplot as plt


def k_vs_time(algo_type):
    list_of_k = [4, 8, 16, 32, 64, 128]
    list_of_seed = [5,10,15,20,25]
    list_of_time_result = list()
    list_of_average_time_result = list()
    raw_dataset = read_dataset("adult-hw1.csv")
    ## Time vs K
    for k in list_of_k:
        if algo_type == "random":
            for seed in list_of_seed:
                start = time.time()
                random_anonymizer("adult-hw1.csv", "DGHs", k, "output.csv", seed)
                end = time.time()
                list_of_time_result.append(end - start)
            average = sum(list_of_time_result) / len(list_of_time_result)
            list_of_average_time_result.append(average)
        elif algo_type == "clustering":
            for seed in list_of_seed:
                start = time.time()
                raw_dataset = np.array(raw_dataset)
                np.random.seed(seed)  ## to ensure consistency between runs
                np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize
                clustering_anonymizer("adult-hw1.csv", "DGHs", k, "output.csv")
                end = time.time()
                list_of_time_result.append(end - start)
            average = sum(list_of_time_result) / len(list_of_time_result)
            list_of_average_time_result.append(average)

    # plotting the points
    plt.plot(list_of_k, list_of_average_time_result)

    # naming the x axis
    plt.xlabel('K')
    # naming the y axis
    plt.ylabel('Time (second)')

    # giving a title to my graph
    plt.title('K vs Time')

    # function to show the plot
    plt.show()


def k_vs_cost(algo_type, cost_type):
    list_of_k = [4, 8, 16, 32, 64, 128]
    list_of_seed = [5, 10, 15, 20, 25]
    list_of_average_cost_result = list()
    raw_dataset = read_dataset("adult-hw1.csv")

    ## Time vs K
    for k in list_of_k:
        list_of_cost_result = list()
        if algo_type == "random":
            for seed in list_of_seed:
                random_anonymizer("adult-hw1.csv", "DGHs", k, "output.csv", seed)
                cost = 0
                if cost_type == "LM":
                    cost = cost_LM("adult-hw1.csv", "output.csv", "DGHs")
                elif cost_type == "MD":
                    cost = cost_MD("adult-hw1.csv", "output.csv", "DGHs")
                list_of_cost_result.append(cost)
            average = sum(list_of_cost_result) / len(list_of_cost_result)
            list_of_average_cost_result.append(average)
        elif algo_type == "clustering":
            for seed in list_of_seed:
                raw_dataset = np.array(raw_dataset)
                np.random.seed(seed)  ## to ensure consistency between runs
                np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize
                clustering_anonymizer("adult-hw1.csv", "DGHs", k, "output.csv")
                cost = 0
                if cost_type == "LM":
                    cost = cost_LM("adult-hw1.csv", "output.csv", "DGHs")
                elif cost_type == "MD":
                    cost = cost_MD("adult-hw1.csv", "output.csv", "DGHs")
                list_of_cost_result.append(cost)
            average = sum(list_of_cost_result) / len(list_of_cost_result)
            list_of_average_cost_result.append(average)

    # plotting the points
    plt.plot(list_of_k, list_of_average_cost_result)

    # naming the x axis
    plt.xlabel('K')
    # naming the y axis
    plt.ylim([min(list_of_average_cost_result), max(list_of_average_cost_result)])
    plt.ylabel('Distortion Metric Cost')

    # giving a title to my graph
    plt.title('K vs Distortion Metric Cost')

    # function to show the plot
    plt.show()


k_vs_cost("clustering","MD")
