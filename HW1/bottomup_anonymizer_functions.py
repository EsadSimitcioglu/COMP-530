import copy

def max_depth(root) -> int:
    if root:
        child_depths = list(map(max_depth, root.child))
        return max(child_depths, default=0) + 1

    else:
        return 0  # no children\

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


def change_char(string, position, char):
    string = string[:position] + str(char) + string[position + 1:]
    return string


def find_max_height_DGHs(DGHs):
    maximum_generalization_count = 1
    for dgh in DGHs.values():
        dgh.max_depth = max_depth(dgh.child)
        maximum_generalization_count *= dgh.max_depth

    return maximum_generalization_count


def create_lattice(DGHs, gen_list, current_index):
    char_index = 0
    for data in DGHs.values():
        temp = gen_list[current_index ]
        change = int(temp[char_index]) + 1
        if change == data.max_depth:
            char_index += 1
            continue
        temp = change_char(temp, char_index, change)
        if not (temp in gen_list):
            gen_list.append(temp)
        char_index += 1
    return gen_list

def is_satisfy_k_anonymity(DGHs, anonymized_dataset, k):
    k_counter = 0
    for raw_data in anonymized_dataset:
        if raw_data["check"]:
            continue
        raw_data["check"] = True
        for raw_anon_data in anonymized_dataset:
            is_equal = True
            for domain in DGHs.values():
                if raw_anon_data[domain.name] != raw_data[domain.name]:
                    is_equal = False
            if is_equal:
                k_counter += 1
        if k_counter >= (k-1):
            k_counter = 0
            continue
        else:
            return False

    return True

def try_lattice(DGHs,k, raw_dataset, lattice):
    for height in lattice:
        anonymized_dataset = copy.deepcopy(raw_dataset)
        column_index = 0
        for domain in DGHs.values():
            for raw_data in anonymized_dataset:
                column = raw_data[domain.name]
                current_node = dfs_in_dgh(domain.child,column)
                current_node = current_node.traverse(int(height[column_index]), domain.max_depth)
                raw_data[domain.name] = current_node.name
            column_index += 1
        if is_satisfy_k_anonymity(DGHs,  anonymized_dataset, k):
            return anonymized_dataset

    return []