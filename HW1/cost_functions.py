from randomized_anonymizer_functions import dfs_in_dgh


def add_cost_MD_column_dataset(anonymized_dataset):
    for anonymized_data in anonymized_dataset:
        anonymized_data["cost_MD"] = 0
    return anonymized_dataset


def add_cost_LM_column_dataset(anonymized_dataset):
    for anonymized_data in anonymized_dataset:
        anonymized_data["cost_LM"] = 0
    return anonymized_dataset


def find_val_MD(domain_name_list, raw_dataset, anonymized_dataset):
    domain_name_dict = dict()

    for domain in domain_name_list:

        for data_index in range(len(raw_dataset)):

            raw_data_domain_name = raw_dataset[data_index][domain.name]
            anonymized_data_domain_name = anonymized_dataset[data_index][domain.name]

            if raw_data_domain_name in domain_name_dict:
                raw_dataset_depth = domain_name_dict[raw_data_domain_name]
            else:
                raw_dataset_depth = dfs_in_dgh(domain.child, raw_dataset[data_index][domain.name]).depth
                domain_name_dict[raw_data_domain_name] = raw_dataset_depth

            if anonymized_data_domain_name in domain_name_dict:
                anonymized_dataset_depth = domain_name_dict[anonymized_data_domain_name]
            else:
                anonymized_dataset_depth = dfs_in_dgh(domain.child,
                                                      anonymized_dataset[data_index][domain.name]).depth
                domain_name_dict[anonymized_data_domain_name] = anonymized_dataset_depth

            cost_of_generalization = raw_dataset_depth - anonymized_dataset_depth
            anonymized_dataset[data_index]["cost_MD"] += cost_of_generalization

    return anonymized_dataset

def find_table_MD(anonymized_dataset):
    cost = 0
    for anonymized_data in anonymized_dataset:
        cost += anonymized_data["cost_MD"]
    return cost

def find_val_LM(domain_name_list, anonymized_dataset):
    for domain in domain_name_list:
        domain_name = domain.name
        val_lm_dict = dict()
        total_leaf_count = domain.child.count_leaf_node()

        for anonymized_data in anonymized_dataset:
            anon_data_node = dfs_in_dgh(domain.child, anonymized_data[domain_name])
            if anon_data_node.name in val_lm_dict:
                val_lm = val_lm_dict[domain_name]
            else:
                if len(anon_data_node.child) == 0:
                    numerator = 0
                else:
                    numerator = anon_data_node.count_leaf_node() - 1
                denominator = total_leaf_count - 1
                val_lm = numerator / denominator
                val_lm_dict[domain_name] = val_lm

            anonymized_data["cost_LM"] += val_lm

    return anonymized_dataset


def find_rec_LM(anonymized_dataset, m):
    for anonymized_data in anonymized_dataset:
        anonymized_data["cost_LM"] *= (1 / m)
    return anonymized_dataset


def find_table_LM(anonymized_dataset):
    cost = 0
    for anonymized_data in anonymized_dataset:
        cost += anonymized_data["cost_LM"]
    return cost
