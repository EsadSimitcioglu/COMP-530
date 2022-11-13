from randomized_anonymizer_functions import dfs_in_dgh, traverse_generalize, generalize_nodes


def add_check_column_to_dataset(raw_dataset):
    for raw_data in raw_dataset:
        raw_data["check"] = False
    return raw_dataset


def add_cost_column_to_dataset(raw_dataset):
    for raw_data in raw_dataset:
        raw_data["cost"] = 0
    return raw_dataset

def find_top_most_data(anon_dataset):
    for anon_data in anon_dataset:
        if not anon_dataset["check"]:
            return anon_data

def init_generalization_cost_dict(DGHs):
    generalization_cost_dict = dict()
    for domain in DGHs.keys():
        generalization_cost_dict[domain] = dict()
    return generalization_cost_dict

def calculate_val_lm(total_leaf_count, anonnymized_node):
    if len(anonnymized_node.child) == 0:
        numerator = 0
    else:
        numerator = anonnymized_node.count_leaf_node() - 1
    denominator = total_leaf_count - 1
    val_lm = numerator / denominator

    return val_lm


def compute_generalization_cost(DGHs, generalization_cost_dict, anon_data, next_anon_data):

    for domain in DGHs.values():
        select_data = anon_data[domain.name]
        target_data = next_anon_data[domain.name]

        key = select_data + " | " + target_data

        if key in generalization_cost_dict[domain.name]:
            cost_lm = generalization_cost_dict[domain.name][key]["cost"]
            generalization = generalization_cost_dict[domain.name][key]["generalization"]
        else:
            node_list = list()
            total_leaf_count = domain.child.count_leaf_node()
            node_list.append(dfs_in_dgh(domain.child, anon_data[domain.name]))
            node_list.append(dfs_in_dgh(domain.child, next_anon_data[domain.name]))
            traversed_node_list = traverse_generalize(node_list)
            generalized_domain_list = generalize_nodes(traversed_node_list)
            cost_lm = calculate_val_lm(total_leaf_count, generalized_domain_list[0])
            generalization = generalized_domain_list[0].name
            generalization_cost_dict[domain.name][key] = dict()
            generalization_cost_dict[domain.name][key]["cost"] = cost_lm
            generalization_cost_dict[domain.name][key]["generalization"] = generalization

        next_anon_data["cost"] += cost_lm

def find_least_generalization_cost(generalized_cost_dict,raw_dataset, k):
    return 123

