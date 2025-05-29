def shortest_paths(source, weight_by_edge):
    weight_by_node = {v: float('inf') for u, v in weight_by_edge}
    weight_by_node[source] = 0

    for _ in range(len(weight_by_node) - 1):
        for (u, v), weight in weight_by_edge.items():
            if u in weight_by_node and v in weight_by_node:
                weight_by_node[v] = min(weight_by_node[u] + weight, weight_by_node[v])

    return weight_by_node

# Example usage:
# shortest_paths('A', {
#     ('A', 'B'): 3,
#     ('A', 'C'): 3,
#     ('A', 'F'): 5,
#     ('C', 'B'): -2,
#     ('C', 'D'): 7,
#     ('C', 'E'): 4,
#     ('D', 'E'): -5,
#     ('E', 'F'): -1
# })