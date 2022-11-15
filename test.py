import csv
import glob
import os
import cost_functions
from bottomup_anonymizer_functions import find_max_height_DGHs, create_lattice, try_lattice
from clustering_anonymizer_functions import init_generalization_cost_dict, add_cost_column_to_dataset, \
    add_check_column_to_dataset, compute_generalization_cost, \
    find_least_generalization_cost, add_index_column_to_dataset, \
    insert_anonymized_data_to_dataset, delete_unnecessary_column_from_dataset
from helper import Node, Root

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
                custom_dataset.append(compute_generalization_cost(DGHs, generalization_cost_dict, raw_data, raw_anon_data))
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

    for iter in range(maximum_generalization_count):
        lattice = create_lattice(DGHs,gen_list,iter)
        anonymized_dataset = try_lattice(DGHs, k, anonymized_dataset, lattice)

        if anonymized_dataset != []:
            break

    print(anonymized_dataset)
    # TODO: complete this function.

    # Finally, write dataset to a file
    # write_dataset(anonymized_dataset, output_file)

#print(cost_LM("adult-hw1.csv", "output5.csv", "DGHs"))
#clustering_anonymizer("adult-hw1.csv", "DGHs", 3, "output5.csv")
bottomup_anonymizer("adult-hw1.csv", "DGHs", 3, "output6.csv")
