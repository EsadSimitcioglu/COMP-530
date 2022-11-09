import csv
import glob
import os
import re

import numpy as np


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


class Node:

    def __init__(self, name, depth, parent=None, child=None):
        self.name = name
        self.depth = depth
        self.parent = parent
        self.child = list()
        if child is not None:
            self.child.append(child)


class Root:

    def __init__(self, domain_name, child=None):
        self.domain_name = domain_name
        self.child = child

    def elevator(self, depth_of_tree):
        if depth_of_tree == 1:
            return self.child
        else:
            last_node = self.child
            for i in range(depth_of_tree - 1):
                last_node = last_node.child[-1]

            return last_node


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

    # TODO: complete this function.
    return -999


def find_ec_list(raw_dataset, k):
    start_index = 0
    ec_list = list()

    for _ in range(0, len(raw_dataset), k):
        k_data = raw_dataset[start_index:start_index + k]
        start_index += k
        ec_list.append(k_data)

    return ec_list


def dfs_in_dgh(root, node_name):
    stack, path = [root], []

    while stack:
        vertex = stack.pop()
        if vertex in path:
            continue
        path.append(vertex)
        for neighbor in vertex.child:
            if neighbor.name == node_name:
                return neighbor
            stack.append(neighbor)


def find_ec_domains(DGHs, ec_list):
    domain_list = list(DGHs.values())
    domain_dict = dict()
    generalized_node_list = list()

    for domain in domain_list:
        domain_name = domain.domain_name
        for ec in ec_list:
            node_list = list()
            for data in ec:
                if data[domain_name] in domain_dict:
                    selected_node = domain_dict[data[domain_name]]
                else:
                    selected_node = dfs_in_dgh(domain.child, data[domain_name])
                    domain_dict[data[domain_name]] = selected_node
                node_list.append(selected_node)
            generalized_node_list.append(traverse_generalize(node_list))
        print(generalized_node_list)

def traverse_generalize(node_list):
    depth_list = list()
    domain_value_list = list()

    for node in node_list:
        depth_list.append(node.depth)
        domain_value_list.append(node.name)

    is_equal_depth = all(element == depth_list[0] for element in depth_list)

    if not is_equal_depth:
        min_depth = min(depth_list)
        traversed_node_list = traverse_node_to_same_depth(node_list, min_depth)
    else:
        traversed_node_list = node_list

    generalized_node_list = generalize_nodes(traversed_node_list)

    return generalized_node_list


def generalize_nodes(generalized_node_list):
    while True:
        is_equal_domain = all(element.name == generalized_node_list[0].name for element in generalized_node_list)
        if is_equal_domain:
            return generalized_node_list
        else:
            new_generalized_node_list = list()
            for node in generalized_node_list:
                node = node.parent
                new_generalized_node_list.append(node)
            generalized_node_list = new_generalized_node_list


def traverse_node_to_same_depth(node_list: list, depth: int):
    traversed_node_list = list()

    for node in node_list:
        traversed_node = node
        if node.depth != depth:
            while traversed_node.depth != depth:
                traversed_node = traversed_node.parent
        traversed_node_list.append(traversed_node)

    return traversed_node_list


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

    # TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.

    ec_list = find_ec_list(raw_dataset, 2)
    find_ec_domains(DGHs, ec_list)

    print(123)

    # Store your results in the list named "clusters".
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    # write_dataset(anonymized_dataset, output_file)


# cost_md = cost_MD(raw_file, anonymized_file, "DGHs")
random_anonymizer("adult-hw1.csv", "DGHs", 2, "output", 10)
