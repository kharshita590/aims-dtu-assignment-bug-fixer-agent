def minimum_spanning_tree(graph):
    """
    Finds the minimum spanning tree of a graph using Kruskal's algorithm.

    Args:
        graph: A dictionary representing the graph where keys are edges (tuples of nodes)
               and values are the corresponding edge weights.

    Returns:
        A list of edges that form the minimum spanning tree.
    """
    parent = {}
    rank = {}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1
            return True
        return False

    mst = []
    edges = sorted(graph.items(), key=lambda item: item[1])

    for edge in graph:
        for node in edge:
            if node not in parent:
                parent[node] = node
                rank[node] = 0

    for edge, weight in edges:
        node1, node2 = edge
        if union(node1, node2):
            mst.append(edge)

    return mst


"""
Driver to test minimum spanning tree
"""
def main():
    # Case 1: Simple tree input.
    # Output: (1, 2) (3, 4) (1, 4)
    result = minimum_spanning_tree({
        (1, 2): 10,
        (2, 3): 15,
        (3, 4): 10,
        (1, 4): 10})
    for edge in result:
        print(edge),
    print()
 
    # Case 2: Strongly connected tree input.
    # Output: (2, 5) (1, 3) (2, 3) (4, 6) (3, 6)
    result = minimum_spanning_tree({
        (1, 2): 6,
        (1, 3): 1,
        (1, 4): 5,
        (2, 3): 5,
        (2, 5): 3,
        (3, 4): 5,
        (3, 5): 6,
        (3, 6): 4,
        (4, 6): 2,
        (5, 6): 6})
    for edge in result:
        print(edge),
    print()

    # Case 3: Minimum spanning tree input.
    # Output: (1, 2) (1, 3) (2, 4)
    result = minimum_spanning_tree({
            (1, 2): 6,
            (1, 3): 1,
            (2, 4): 2})
    for edge in result:
        print(edge),
    print()


if __name__ == "__main__":
    main()