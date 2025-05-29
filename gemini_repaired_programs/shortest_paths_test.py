from collections import defaultdict
import heapq


def shortest_paths(start, graph):
    """
    Finds the shortest paths from a starting node to all other nodes in a graph.

    Args:
        start: The starting node.
        graph: A dictionary representing the graph where keys are tuples of (node1, node2)
               representing an edge, and values are the edge weights.

    Returns:
        A dictionary where keys are nodes and values are the shortest distances from the
        starting node.
    """

    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    pq = [(0, start)]  # Priority queue of (distance, node)

    while pq:
        dist, u = heapq.heappop(pq)

        if dist > distances[u]:
            continue  # Already processed a shorter path to u

        for edge, weight in graph.items():
            if edge[0] == u:
                v = edge[1]
                new_dist = dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

    return distances


"""
Test shortest paths
"""
def main():
    # Case 1: Graph with multiple paths
    # Output: {'A': 0, 'C': 3, 'B': 1, 'E': 5, 'D': 10, 'F': 4}
    graph = {
        ('A', 'B'): 3,
        ('A', 'C'): 3,
        ('A', 'F'): 5,
        ('C', 'B'): -2,
        ('C', 'D'): 7,
        ('C', 'E'): 4,
        ('D', 'E'): -5,
        ('E', 'F'): -1
    }
    result =  shortest_paths('A', graph)
    for path in result:
        if path in result:
            print(path, result[path], end=" ")
    print()

    # Case 2: Graph with one path
    # Output: {'A': 0, 'C': 3, 'B': 1, 'E': 5, 'D': 6, 'F': 9}
    graph2 = {
        ('A', 'B'): 1,
        ('B', 'C'): 2,
        ('C', 'D'): 3,
        ('D', 'E'): -1,
        ('E', 'F'): 4
    }
    result =  shortest_paths('A', graph2)
    for path in result:
        if path in result:
            print(path, result[path], end=" ")
    print()

    # Case 3: Graph with cycle
    # Output: {'A': 0, 'C': 3, 'B': 1, 'E': 1, 'D': 0, 'F': 5}
    graph3 = {
        ('A', 'B'): 1,
        ('B', 'C'): 2,
        ('C', 'D'): 3,
        ('D', 'E'): -1,
        ('E', 'D'): 1,
        ('E', 'F'): 4
    }
    result =  shortest_paths('A', graph3)
    for path in result:
        if path in result:
            print(path, result[path], end=" ")
    print()


if __name__ == "__main__":
    main()