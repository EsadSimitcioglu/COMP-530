##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################
import copy
import csv
import glob
import os
import sys
import numpy as np
import datetime

from bottomup_anonymizer_functions import find_max_height_DGHs, create_lattice, try_lattice
from clustering_anonymizer_functions import init_generalization_cost_dict, add_cost_column_to_dataset, \
    add_index_column_to_dataset, add_check_column_to_dataset, compute_generalization_cost, \
    find_least_generalization_cost, delete_unnecessary_column_from_dataset
from helper import Node, Root
import cost_functions
from randomized_anonymizer_functions import find_ec_list, generalize_data

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)


##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    f = open(DGH_file, "r")
    mystr = f.readlines()
    folder_name = "DGHs/"
    file_extension = ".txt"
    domain_name = f.name[len(folder_name):-len(file_extension)]
    pre_tab_counter = 0
    root = Root(domain_name)
    for element in mystr:
        tab_counter = 0
        if element[0] != "\t":
            element = element.replace("\n", "")
            new_node = Node(element, 0)
            root.child = new_node
            pre_tab_counter += 1

        else:
            for index in element:
                if index == "\t":
                    tab_counter += 1

            current_node = root.elevator(tab_counter)
            node_name = element.replace("\n", "")
            node_name = node_name.replace("\t", "")
            new_node = Node(node_name, tab_counter, current_node)
            current_node.child.append(new_node)

    return root


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    anonymized_dataset = cost_functions.add_cost_MD_column_dataset(anonymized_dataset)
    domain_name_list = list(DGHs.values())

    anonymized_dataset = cost_functions.find_val_MD(domain_name_list, raw_dataset, anonymized_dataset)
    cost = cost_functions.find_table_MD(anonymized_dataset)

    return cost


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = cost_functions.add_cost_LM_column_dataset(anonymized_dataset)
    domain_name_list = list(DGHs.values())
    m = len(domain_name_list)

    anonymized_dataset = cost_functions.find_val_LM(domain_name_list, anonymized_dataset)
    cost_functions.find_rec_LM(anonymized_dataset, m)
    cost = cost_functions.find_table_LM(anonymized_dataset)

    return cost


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                      output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    for i in range(len(raw_dataset)):  ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)

    # Store your results in the list named "clusters".
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    ec_list = find_ec_list(raw_dataset, k)
    clusters = generalize_data(DGHs, ec_list)

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                          output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    generalization_cost_dict = init_generalization_cost_dict(DGHs)
    anonymized_dataset = add_check_column_to_dataset(raw_dataset)
    anonymized_dataset = add_cost_column_to_dataset(anonymized_dataset)
    anonymized_dataset = add_index_column_to_dataset(anonymized_dataset)

    for raw_data in anonymized_dataset:
        if raw_data["check"]:
            continue
        raw_data["check"] = True
        custom_dataset = list()
        custom_dataset.append(raw_data)
        for raw_anon_data in anonymized_dataset:
            if not raw_anon_data["check"]:
                custom_dataset.append(
                    compute_generalization_cost(DGHs, generalization_cost_dict, raw_data, raw_anon_data))
        find_least_generalization_cost(DGHs, custom_dataset, k)

    delete_unnecessary_column_from_dataset(anonymized_dataset)
    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)


def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                        output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = add_check_column_to_dataset(raw_dataset)
    DGHs = read_DGHs(DGH_folder)

    maximum_generalization_count = find_max_height_DGHs(DGHs)
    gen_list = list()
    gen_list.append("00000000")
    multiply = 0

    for iter in range(maximum_generalization_count):
        temp_dataset = copy.deepcopy(anonymized_dataset)
        lattice = create_lattice(DGHs, gen_list, iter)
        temp_dataset = try_lattice(DGHs, k, temp_dataset, lattice[(multiply * iter):])

        if temp_dataset != []:
            anonymized_dataset = temp_dataset
            break
        multiply += 8

    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)


# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now()  ##
print(start_time)  ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(
            f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)

    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print(f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now()  ##
print(end_time)  ##
print(end_time - start_time)  ##

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5
