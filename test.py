import csv
import glob
import os

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

class Node:

    def __init__(self, name, child=None):
        self.name = name
        self.child = list()
        if child is not None:
            self.child.append(child)


class Root:

    def __init__(self, child=None):
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
    pre_tab_counter = 0
    root = Root(True)
    for element in mystr:
        tab_counter = 0
        if element[0] != "\t":
            new_node = Node(element)
            root.child = new_node
            pre_tab_counter += 1

        else:
            for index in element:
                if index == "\t":
                    tab_counter += 1

            current_node = root.elevator(tab_counter)

            new_node = Node(element)
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


cost_md = cost_MD(raw_file, anonymized_file, "DGHs")
