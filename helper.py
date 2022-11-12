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