import math, random
import matplotlib.pyplot as plt
import numpy as np

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    d = 17

    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    q = (1-p) / (d-1)

    domain = np.arange(start=0, stop=d)

    rnd = np.random.random()
    if rnd <= p:
        return val
    else:
        return np.random.choice(domain[domain != val])

# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    d = 17

    # Number of reports
    n = len(perturbed_values)

    # GRR parameters
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    q = (1 - p) / (d - 1)

    # Count how many times each value has been reported
    count_report = np.zeros(d)
    for rep in perturbed_values:
        count_report[rep-1] += 1

    # Ensure non-negativity of estimated frequency
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)

    # Re-normalized estimated frequency
    norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return norm_est_freq


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    pass


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    pass


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    pass


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    pass


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    pass


# OUE

# TODO: Implement this function!
def encode_oue(val):
    pass


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    pass


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    pass


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    pass


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")

    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

