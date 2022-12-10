import math, random
import matplotlib.pyplot as plt
import numpy as np

from HW2.part2_skeleton import calculate_average_error

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
    d = len(DOMAIN)

    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    q = (1 - p) / (d - 1)

    domain = np.arange(start=0, stop=d)

    rnd = np.random.random()
    if rnd <= p:
        return val
    else:
        return np.random.choice(domain[domain != val])


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    d = len(DOMAIN)

    # Number of reports
    n = len(perturbed_values)

    # GRR parameters
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    q = (1 - p) / (d - 1)

    # Count how many times each value has been reported
    count_report = np.zeros(d)
    for rep in perturbed_values:
        count_report[rep - 1] += 1

    result_list = list()
    for rep in count_report:
        est = (rep - (n * q)) / (p - q)
        result_list.append(est)

    return result_list


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    true_values_list = list()
    perturbed_values_list = list()
    for user_value in dataset:
        true_values_list.append(user_value)
        perturbed_values_list.append(perturb_grr(user_value, epsilon))

    est_freq = estimate_grr(perturbed_values_list, epsilon)

    count_report = np.zeros(len(est_freq))
    for rep in true_values_list:
        count_report[rep - 1] += 1

    count_report = count_report.tolist()
    return calculate_average_error(count_report, est_freq)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    d = len(DOMAIN)

    bit_vector = np.zeros(d)
    bit_vector[val-1] = 1

    return bit_vector


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    p = (np.exp(epsilon / 2)) / (np.exp(epsilon / 2) + 1)

    perturbed_bit_vector = encoded_val.copy()
    for bit_index in range(len(encoded_val)):
        rnd = np.random.random()
        if rnd > p:
            if perturbed_bit_vector[bit_index] == 1:
                perturbed_bit_vector[bit_index] = 0
            else:
                perturbed_bit_vector[bit_index] = 1

    perturbed_bit_vector = perturbed_bit_vector.tolist()
    return perturbed_bit_vector


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    perturbed_bit_vectors = np.array(perturbed_values)
    n = len(perturbed_bit_vectors)
    p = (np.exp(epsilon / 2)) / (np.exp(epsilon / 2) + 1)
    q = 1 / (np.exp(epsilon / 2) + 1)
    perturbed_sum_bit_vector = sum(perturbed_bit_vectors)
    est_freq_vector = list()

    for sum_bit in perturbed_sum_bit_vector:
        numerator = sum_bit - (n * q)
        denominator = p - q
        est_freq_vector.append(numerator / denominator)

    return est_freq_vector


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    true_values_list = list()
    perturbed_values_list = list()
    for user_value in dataset:
        true_values_list.append(user_value)
        bit_vector = encode_rappor(user_value)
        perturbed_values_list.append(perturb_rappor(bit_vector,epsilon))

    est_freq = estimate_rappor(perturbed_values_list, epsilon)

    count_report = np.zeros(len(est_freq))
    for rep in true_values_list:
        count_report[rep - 1] += 1

    count_report = count_report.tolist()
    return calculate_average_error(count_report, est_freq)


# OUE

# TODO: Implement this function!
def encode_oue(val):
    d = len(DOMAIN)

    bit_vector = np.zeros(d)
    bit_vector[val - 1] = 1

    return bit_vector


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):

    p = 1 / 2
    q = 1 / (np.exp(epsilon) + 1)

    perturbed_bit_vector = encoded_val.copy()
    for bit_index in range(len(encoded_val)):
        if perturbed_bit_vector[bit_index] == 0:
            rnd = np.random.random()
            if rnd <= q:
                perturbed_bit_vector[bit_index] = 1
        else:
            rnd = np.random.random()
            if rnd > p:
                perturbed_bit_vector[bit_index] = 0
    return perturbed_bit_vector


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    n = len(perturbed_values)
    perturbed_sum_bit_vector = sum(perturbed_values)
    est_freq_vector = list()

    for sum_bit in perturbed_sum_bit_vector:
        numerator = 2 * ((np.exp(epsilon) + 1) * sum_bit - n)
        denominator = np.exp(epsilon) - 1
        est_freq_vector.append(numerator / denominator)

    return est_freq_vector


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    true_values_list = list()
    perturbed_values_list = list()
    for user_value in dataset:
        true_values_list.append(user_value)
        bit_vector = encode_oue(user_value)
        perturbed_values_list.append(perturb_oue(bit_vector, epsilon))

    est_freq = estimate_oue(perturbed_values_list, epsilon)

    count_report = np.zeros(len(est_freq))
    for rep in true_values_list:
        count_report[rep - 1] += 1

    count_report = count_report.tolist()
    return calculate_average_error(count_report, est_freq)


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
