# Implementing Laplace mechanism on Adult dataset by adding Laplacian random noise

from matplotlib import pyplot as plt

from HW2.part2.part2_skeleton import get_histogram, exponential_experiment, read_dataset

filename = "part2/anime-dp.csv"
dataset = read_dataset(filename)
counts = get_histogram(dataset)

print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
exponential_experiment_result = exponential_experiment(dataset, eps_values)
for i in range(len(eps_values)):
    print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])

plt.ylim(0, 1)
plt.plot(eps_values, exponential_experiment_result, label='EM', color='red')
plt.ylabel('Accuracy')
plt.xlabel('Epsilon values')
plt.legend(loc='upper right', bbox_to_anchor=(1.015, 1.15))
plt.show()