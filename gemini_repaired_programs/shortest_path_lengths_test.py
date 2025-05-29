from collections import defaultdict

def shortest_path_lengths(n, graph):
    """
    Compute shortest path lengths between all pairs of nodes in a graph.

    Args:
        n (int): The number of nodes in the graph.
        graph (dict): A dictionary representing the graph where keys are tuples
                      of (node1, node2) representing an edge and values are the
                      edge weights.

    Returns:
        dict: A dictionary where keys are tuples of (node1, node2) and values
              are the shortest path lengths between those nodes.
    """
    dist = defaultdict(lambda: float('inf'))

    # Initialize distances
    for i in range(n):
        dist[(i, i)] = 0
    for edge, weight in graph.items():
        dist[edge] = weight

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[(i, j)] = min(dist[(i, j)], dist[(i, k)] + dist[(k, j)])

    return dict(dist)


"""
Test shortest path lengths
""" 
def main():
    # Case 1: Basic graph input.
    # Output:
    graph = {
        (0, 2): 3,
        (0, 5): 5,
        (2, 1): -2,
        (2, 3): 7,
        (2, 4): 4,
        (3, 4): -5,
        (4, 5): -1
    }
    result =  shortest_path_lengths(6, graph)
    for node_pairs in result:
        if node_pairs in result:
            print(node_pairs, result[node_pairs], end=" ")
    print()

    # Case 2: Linear graph input.
    # Output:
    graph2 = {
        (0, 1): 3,
        (1, 2): 5,
        (2, 3): -2,
        (3, 4): 7
    }
    result =  shortest_path_lengths(5, graph2)
    for node_pairs in result:
        if node_pairs in result:
            print(node_pairs, result[node_pairs], end=" ")
    print()

    # Case 3: Disconnected graphs input.
    # Output:
    graph3 = {
        (0, 1): 3,
        (2, 3): 5
    }
    result =  shortest_path_lengths(4, graph3)
    for node_pairs in result:
        if node_pairs in result:
            print(node_pairs, result[node_pairs], end=" ")
    print()

    # Case 4: Strongly connected graph input.
    graph4 = {
        (0, 1): 3,
        (1, 2): 5,
        (2, 0): -1
    }
    result =  shortest_path_lengths(3, graph4)
    for node_pairs in result:
        if node_pairs in result:
            print(node_pairs, result[node_pairs], end=" ")
    print()


if __name__ == "__main__":
    main()