import sys

class Node:
    def __init__(self, name, parent=None, children=None):
        self.name = name
        self.parent = parent
        self.children = children if children is not None else []

def shortest_path_length(length_by_edge, start_node, end_node):
    """
    Find the shortest path length between two nodes in a graph.

    Args:
        length_by_edge (dict): A dictionary of edge lengths, where the keys are tuples of nodes and the values are the lengths.
        start_node (Node): The starting node.
        end_node (Node): The ending node.

    Returns:
        int: The shortest path length between the two nodes, or sys.maxsize if there is no path.
    """

    if start_node == end_node:
        return 0

    distances = {node: sys.maxsize for node in set([key[0] for key in length_by_edge.keys()] + [key[1] for key in length_by_edge.keys()])}
    distances[start_node] = 0

    visited = set()

    while True:
        min_distance = sys.maxsize
        current_node = None

        for node in distances:
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                current_node = node

        if current_node is None:
            break

        visited.add(current_node)

        for edge, length in length_by_edge.items():
            if edge[0] == current_node:
                neighbor = edge[1]
                new_distance = distances[current_node] + length
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance

    return distances[end_node]

def main():
    node1 = Node("1")
    node5 = Node("5")
    node4 = Node("4", None, [node5])
    node3 = Node("3", None, [node4])
    node2 = Node("2", None, [node1, node3, node4])
    node0 = Node("0", None, [node2, node5])

    length_by_edge = {
        (node0, node2): 3,
        (node0, node5): 10,
        (node2, node1): 1,
        (node2, node3): 2,
        (node2, node4): 4,
        (node3, node4): 1,
        (node4, node5): 1
    }

    # Case 1: One path
    # Output: 4
    result =  shortest_path_length(length_by_edge, node0, node1)
    print(result)

    # Case 2: Multiple path
    # Output: 7
    result = shortest_path_length(length_by_edge, node0, node5)
    print(result)

    # Case 3: Start point is same as end point
    # Output: 0
    result = shortest_path_length(length_by_edge, node2, node2)
    print(result)

    # Case 4: Unreachable path
    # Output: INT_MAX
    result = shortest_path_length(length_by_edge, node1, node5)
    print(result)

if __name__ == "__main__":
    main()