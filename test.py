import csv
import glob
import os
import cost_functions

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

    def count_leaf_node(self):
        leaf_nodes = []
        self.count_leaf_node_helper_fun(leaf_nodes)
        return len(leaf_nodes)

    def count_leaf_node_helper_fun(self, leaf_nodes):
        if not self.child:
            leaf_nodes.append(self)
        else:
            for child in self.child:
                child.count_leaf_node_helper_fun(leaf_nodes)


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
    anonymized_dataset = add_cost_column_dataset(anonymized_dataset)
    domain_name_list = list(DGHs.values())
    cost = 0

    for domain in domain_name_list:
        anonymized_dataset = find_generalization_cost_in_domain(domain, raw_dataset, anonymized_dataset)

    for anonymized_data in anonymized_dataset:
        cost += anonymized_data["cost_MD"]

    return cost




def add_cost_column_dataset(anonymized_dataset):
    for anonymized_data in anonymized_dataset:
        anonymized_data["cost_MD"] = 0
    return anonymized_dataset


def find_generalization_cost_in_domain(domain, raw_dataset, anonymized_dataset):
    domain_name_dict = dict()

    for data_index in range(len(raw_dataset)):

        raw_data_domain_name = raw_dataset[data_index][domain.domain_name]
        anonymized_data_domain_name = anonymized_dataset[data_index][domain.domain_name]

        if raw_data_domain_name in domain_name_dict:
            raw_dataset_depth = domain_name_dict[raw_data_domain_name]
        else:
            raw_dataset_depth = dfs_in_dgh(domain.child, raw_dataset[data_index][domain.domain_name]).depth
            domain_name_dict[raw_data_domain_name] = raw_dataset_depth

        if anonymized_data_domain_name in domain_name_dict:
            anonymized_dataset_depth = domain_name_dict[anonymized_data_domain_name]
        else:
            anonymized_dataset_depth = dfs_in_dgh(domain.child, anonymized_dataset[data_index][domain.domain_name]).depth
            domain_name_dict[anonymized_data_domain_name] = anonymized_dataset_depth

        cost_of_generalization = raw_dataset_depth - anonymized_dataset_depth
        anonymized_dataset[data_index]["cost_MD"] += cost_of_generalization

    return anonymized_dataset


def dfs_in_dgh(root, node_name):

    if(root.name == node_name):
        return root

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

    cost = 0

    for domain in domain_name_list:
        total_leaf_count = domain.child.count_leaf_node
        anonymized_dataset = cost_functions.find_val_LM(domain,total_leaf_count,anonymized_dataset)

    for anonymized_data in anonymized_dataset:


    # TODO: complete this function.
    return -999


# cost_md = cost_MD(raw_file, anonymized_file, "DGHs")
cost_MD("adult-hw1.csv", "output.csv", "DGHs")
