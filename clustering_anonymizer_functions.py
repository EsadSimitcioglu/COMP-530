from randomized_anonymizer_functions import dfs_in_dgh, traverse_generalize, generalize_nodes, generalize_data


def add_check_column_to_dataset(raw_dataset):
    for raw_data in raw_dataset:
        raw_data["check"] = False
    return raw_dataset


def add_cost_column_to_dataset(raw_dataset):
    for raw_data in raw_dataset:
        raw_data["cost"] = 0
    return raw_dataset

def add_index_column_to_dataset(raw_dataset):
    index = 0
    for raw_data in raw_dataset:
        raw_data["index"] = index
        index+=1
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
        else:
            node_list = list()
            total_leaf_count = domain.child.count_leaf_node()
            node_list.append(dfs_in_dgh(domain.child, anon_data[domain.name]))
            node_list.append(dfs_in_dgh(domain.child, next_anon_data[domain.name]))
            traversed_node_list = traverse_generalize(node_list)
            generalized_domain_list = generalize_nodes(traversed_node_list)
            cost_lm = calculate_val_lm(total_leaf_count, generalized_domain_list[0])
            generalization_cost_dict[domain.name][key] = dict()
            generalization_cost_dict[domain.name][key]["cost"] = cost_lm

        next_anon_data["cost"] += cost_lm
    return next_anon_data

def find_least_generalization_cost(DGhs,custom_dataset : list, k):
    k_lowest_cost_dataset = sorted(custom_dataset, key=lambda d: d['cost'])[:k-len(custom_dataset)]
    generalize_data_2(DGhs,k_lowest_cost_dataset)

    for k_lowest_data in k_lowest_cost_dataset:
        k_lowest_data["check"] = True

    for raw_data in custom_dataset:
        raw_data["cost"] = 0

    return k_lowest_cost_dataset



def insert_anonymized_data_to_dataset(raw_dataset, k_lowest_cost_dataset):

    for k_lowest_data in k_lowest_cost_dataset:
        raw_dataset[k_lowest_data["index"]] = k_lowest_data
    return raw_dataset


def generalize_data_2(DGHs, ec):
    domain_name_list = list(DGHs.values())

    for domain in domain_name_list:
        domain_name = domain.name
        domain_name_dict = dict()
        node_list = list()
        generalized_data_list = list()
        for data in ec:
            if data[domain_name] in domain_name_dict:
                selected_node = domain_name_dict[data[domain_name]]
            else:
                selected_node = dfs_in_dgh(domain.child, data[domain_name])
                domain_name_dict[data[domain_name]] = selected_node
            node_list.append(selected_node)

        traversed_node_list = traverse_generalize(node_list)
        generalized_domain_list = generalize_nodes(traversed_node_list)

        for data in ec:
            data[domain_name] = generalized_domain_list[0].name
            generalized_data_list.append(data)

    return ec