import sys
import random

import numpy
import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm._libsvm import predict_proba
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    average_accuracy = 0

    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    else:
        model = SVC(C=0.5, kernel='poly', random_state=0, probability=True)

    for _ in range(100):
        num_samples = int(len(y_train) * n)
        flipped_indices = random.sample(range(len(y_train)), num_samples)
        y_train_flipped = copy.deepcopy(y_train)
        y_train_flipped[flipped_indices] = 1 - y_train[flipped_indices]
        model.fit(X_train, y_train_flipped)
        model_predict = model.predict(X_test)
        average_accuracy += accuracy_score(y_test, model_predict)

    average_accuracy /= 100
    return average_accuracy


###############################################################################
############################## Inference ########################################
###############################################################################

def inference_attack(trained_model, samples, t):
    prediction_prob_list = trained_model.predict_proba(samples)
    prediction_list = trained_model.predict(samples)
    tp_value = 0
    fn_value = 0

    for index in range(len(prediction_prob_list)):
        confidence_value = prediction_prob_list[index][prediction_list[index]]
        if confidence_value >= t:
            tp_value += 1
        else:
            fn_value += 1

    recall = tp_value / (tp_value + fn_value)

    return recall


###############################################################################
################################## Backdoor ###################################
###############################################################################

def create_list_of_fire_rows(X_train, y_train):
    fire_list = list()
    for row_index in range(len(X_train)):
        if y_train[row_index] == 1:
            fire_list.append(X_train[row_index])

    return fire_list


def backdoor_attack(X_train, y_train, model_type, num_samples):
    trigger_pattern = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    inject_row_list = list()
    for _ in range(num_samples):
        inject_row_list.append(trigger_pattern)

    inject_y_list = [1] * num_samples

    inject_row_np_list = numpy.array(inject_row_list)
    inject_y_np_list = numpy.array(inject_y_list)

    X_train_prime = X_train
    y_train_prime = y_train

    if num_samples != 0:
        X_train_prime = np.vstack((X_train, inject_row_np_list))
        y_train_prime = np.concatenate((y_train, inject_y_np_list))

    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "SVC":
        model = SVC(C=0.5, kernel='poly', random_state=0, probability=True)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train_prime, y_train_prime)

    success_rate = model.predict_proba([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    success_rate_normalized = success_rate[0][1] / np.sum(success_rate)

    if num_samples == 0:
        success_rate_normalized = 1-success_rate[0][0]
        success_rate_normalized = np.format_float_positional(success_rate_normalized, trim='-')

    return success_rate_normalized



###############################################################################
############################## Evasion ########################################
###############################################################################

def combination_of_increment(my_arr, increment_value):
    sol = []

    def dfs(arr, i):
        if i == len(arr):
            sol.append(arr.copy())
            return

        res = arr.copy()
        dfs(res, i + 1)  # not increment
        num = [arr[i] + increment_value]
        second = arr[:i] + num + arr[i + 1:]
        dfs(second, i + 1)  # keep

        num = [arr[i] - increment_value]
        second = arr[:i] + num + arr[i + 1:]
        dfs(second, i + 1)  # decrement

        return arr.copy()

    dfs(my_arr, 0)
    return sol


def evade_model(trained_model, actual_example):
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    increment_value = 0.5
    pred_class = actual_class
    while increment_value < 100:
        """
        combination_of_actuals = combination_of_increment(actual_example.tolist(),increment_value)

        for combination in combination_of_actuals:
            combination_np_arr = np.array(combination)
            pred_class = trained_model.predict([combination_np_arr])[0]
            if pred_class != actual_class:
                print(increment_value)
                return combination_np_arr
        """

        for index, feature in enumerate(modified_example):

            modified_example = copy.deepcopy(actual_example)
            modified_example[index] += increment_value
            pred_class = trained_model.predict([modified_example])[0]

            if pred_class != actual_class:
                return modified_example

        for index, feature in enumerate(modified_example):
            modified_example = copy.deepcopy(actual_example)
            modified_example[index] -= increment_value
            pred_class = trained_model.predict([modified_example])[0]

            if pred_class != actual_class:
                return modified_example

        increment_value += 0.5

    return actual_example


def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    DT_to_LR_counter = 0
    DT_to_SVC_counter = 0

    LR_to_DT_counter = 0
    LR_to_SVC_counter = 0

    SVC_to_DT_counter = 0
    SVC_to_LR_counter = 0

    for instance in actual_examples:
        adv_example = evade_model(DTmodel, instance)

        if DTmodel.predict([adv_example]) == LRmodel.predict([adv_example]):
            DT_to_LR_counter += 1

        if DTmodel.predict([adv_example]) == SVCmodel.predict([adv_example]):
            DT_to_SVC_counter += 1

        adv_example = evade_model(LRmodel, instance)

        if LRmodel.predict([adv_example]) == DTmodel.predict([adv_example]):
            LR_to_DT_counter += 1

        if LRmodel.predict([adv_example]) == SVCmodel.predict([adv_example]):
            LR_to_SVC_counter += 1

        adv_example = evade_model(SVCmodel, instance)

        if SVCmodel.predict([adv_example]) == DTmodel.predict([adv_example]):
            SVC_to_DT_counter += 1

        if SVCmodel.predict([adv_example]) == LRmodel.predict([adv_example]):
            SVC_to_LR_counter += 1

    print(str(DT_to_LR_counter) + " / 40 instances were transferable from DT to LR")
    print(str(DT_to_SVC_counter) + " / 40 instances were transferable from DT to SVC")
    print(str(LR_to_DT_counter) + " / 40 instances were transferable from LR to DT")
    print(str(LR_to_SVC_counter) + " / 40 instances were transferable from LR to SVC")
    print(str(SVC_to_DT_counter) + " / 40 instances were transferable from SVC to DT")
    print(str(SVC_to_LR_counter) + " / 40 instances were transferable from SVC to LR")


###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    else:
        model = SVC(C=0.5, kernel='poly', random_state=0, probability=True)

    y_train = remote_model.predict(examples)
    model.fit(examples, y_train)
    return model


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "forest_fires.csv"
    features = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    df["DC"] = df["DC"].astype('float64')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0, probability=True)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)

    # Inference attacks:
    samples = X_train[0:100]
    t_values = [0.99, 0.98, 0.96, 0.8, 0.7, 0.5]
    for t in t_values:
        print("Recall of inference attack", str(t), ":", inference_attack(mySVC, samples, t))

    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)

    # Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])

    # Model stealing:
    budgets = [8, 12, 16, 20, 24]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))


if __name__ == "__main__":
    main()
