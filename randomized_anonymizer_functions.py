def find_ec_list(raw_dataset, k):
    start_index = 0
    ec_list = list()

    for _ in range(0, len(raw_dataset), k):
        k_data = raw_dataset[start_index:start_index + k]
        start_index += k
        ec_list.append(k_data)

    return ec_list


def dfs_in_dgh(root, node_name):

    if root.name == node_name:
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


def generalize_data(DGHs, ec_list):
    domain_name_list = list(DGHs.values())

    for domain in domain_name_list:
        domain_name = domain.name
        domain_name_dict = dict()

        for ec in ec_list:
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

    return ec_list


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

    return traversed_node_list


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

