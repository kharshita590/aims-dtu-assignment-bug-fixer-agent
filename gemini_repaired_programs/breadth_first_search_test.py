from collections import deque

class Node:
    def __init__(self, name, visited=False, successors=None):
        self.name = name
        self.visited = visited
        self.successors = successors if successors is not None else []

def breadth_first_search(start_node, goal_node):
    if start_node is goal_node:
        return True

    queue = deque([start_node])
    start_node.visited = True

    while queue:
        current_node = queue.popleft()

        for neighbor in current_node.successors:
            if neighbor is goal_node:
                return True
            if not neighbor.visited:
                queue.append(neighbor)
                neighbor.visited = True

    return False

"""
Driver to test breadth first search
"""
def main():
    # Case 1: Strongly connected graph
    # Output: Path found!
    station1 = Node("Westminster")
    station2 = Node("Waterloo", None, [station1])
    station3 = Node("Trafalgar Square", None, [station1, station2])
    station4 = Node("Canary Wharf",  None, [station2, station3])
    station5 = Node("London Bridge",  None, [station4, station3])
    station6 = Node("Tottenham Court Road",  None, [station5, station4])

    if breadth_first_search(station6, station1):
        print("Path found!", end=" ")
    else:
        print("Path not found!", end=" ")
    print()

    # Case 2: Branching graph
    nodef =  Node("F")
    nodee =  Node("E")
    noded =  Node("D")
    nodec =  Node("C", None, [nodef])
    nodeb =  Node("B", None, [nodee])
    nodea =  Node("A", None, [nodeb, nodec, noded])

    if breadth_first_search(nodea, nodee):
        print("Path found!", end=" ")
    else:
        print("Path not found!", end=" ")
    print()

    # Case 3: Two unconnected nodes in graph
    if breadth_first_search(nodef, nodee):
        print("Path found!", end=" ")
    else:
        print("Path not found!", end=" ")
    print()

    # Case 4: One node graph
    if breadth_first_search(nodef, nodef):
        print("Path found!", end=" ")
    else:
        print("Path not found!", end=" ")
    print()

    # Case 5: Graph with cycles
    node1 = Node("1")
    node2 = Node("2")
    node3 = Node("3")
    node4 = Node("4", None, [node1])
    node5 = Node("5", None, [node2])
    node6 = Node("6", None, [node5, node4, node3])

    node2.successors = [node6]

    if breadth_first_search(node6, node1):
        print("Path found!", end=" ")
    else:
        print("Path not found!", end=" ")
    print()

if __name__ == "__main__":
    main()