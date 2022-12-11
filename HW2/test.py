# Implementing Laplace mechanism on Adult dataset by adding Laplacian random noise
from math import exp
from random import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from HW2.part2_skeleton import get_dp_histogram, get_histogram, calculate_average_error, \
    calculate_mean_squared_error, most_10rated_exponential, read_dataset
from HW2.part3_skeleton import perturb_grr, estimate_grr, grr_experiment





#filename = "anime-dp.csv"
#chosen_anime_id = "199"
eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
#dataset = read_dataset(filename)


#counts = get_histogram(dataset,chosen_anime_id)
#dp_counts = get_dp_histogram(counts,0.001)
#epsilon_experiment(counts,eps_values)

eps_values = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]


#for eps in eps_values:
    #print(grr_experiment(dataset, eps))

eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
filename = "anime-dp.csv"
dataset = read_dataset(filename)
a = most_10rated_exponential(dataset,0.001)



print(randomNumberList)

#most_10rated_exponential(dataset, 0.0001)

