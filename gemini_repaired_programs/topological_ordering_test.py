from collections import deque

class Node:
    def __init__(self, value):
        self.value = value
        self.incoming_nodes = []
        self.outgoing_nodes = []

def topological_ordering(nodes):
    """
    Performs a topological sort on a directed acyclic graph (DAG).

    Args:
        nodes: A list of Node objects representing the graph.

    Returns:
        A list of Node objects in topological order, or None if the graph
        contains a cycle.
    """

    # Calculate in-degrees for each node
    in_degree = {}
    for node in nodes:
        in_degree[node] = len(node.incoming_nodes)

    # Initialize queue with nodes having in-degree of 0
    queue = deque([node for node in nodes if in_degree[node] == 0])

    # Initialize result list
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in node.outgoing_nodes:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(result) != len(nodes):
        return None  # Graph has a cycle

    return result


"""
Driver to test topological ordering
"""
def main():
    # Case 1: Wikipedia graph
    # Output: 5 7 3 11 8 10 2 9
    
    five = Node(5)
    seven = Node(7)
    three = Node(3)
    eleven = Node(11)
    eight = Node(8)
    two = Node(2)
    nine = Node(9)
    ten = Node(10)
 
    five.outgoing_nodes = [eleven]
    seven.outgoing_nodes = [eleven, eight]
    three.outgoing_nodes = [eight, ten]
    eleven.incoming_nodes = [five, seven]
    eleven.outgoing_nodes = [two, nine, ten]
    eight.incoming_nodes = [seven, three]
    eight.outgoing_nodes = [nine]
    two.incoming_nodes = [eleven]
    nine.incoming_nodes = [eleven, eight]
    ten.incoming_nodes = [eleven, three]

    try:
        if topological_ordering([five, seven, three, eleven, eight, two, nine, ten]) is not None:
            [print(x.value, end=" ") for x in topological_ordering([five, seven, three, eleven, eight, two, nine, ten])]
        else:
            print("Graph has a cycle")
    except Exception as e:
        print(e)
    print()


    # Case 2: GeekforGeeks example
    # Output: 4 5 0 2 3 1

    five = Node(5)
    zero = Node(0)
    four = Node(4)
    one = Node(1)
    two = Node(2)
    three = Node(3)

    five.outgoing_nodes = [two, zero]
    four.outgoing_nodes = [zero, one]
    two.incoming_nodes = [five]
    two.outgoing_nodes = [three]
    zero.incoming_nodes = [five, four]
    one.incoming_nodes = [four, three]
    three.incoming_nodes = [two]
    three.outgoing_nodes = [one]

    try:
        if topological_ordering([zero, one, two, three, four, five]) is not None:
            [print(x.value, end=" ") for x in topological_ordering([zero, one, two, three, four, five])]
        else:
            print("Graph has a cycle")
    except Exception as e:
        print(e)
    print()
    

    # Case 3: Cooking with InteractivePython
    # Output:

    milk = Node("3/4 cup milk")
    egg = Node("1 egg")
    oil = Node("1 Tbl oil")
    mix = Node("1 cup mix")
    syrup = Node("heat syrup")
    griddle = Node("heat griddle")
    pour = Node("pour 1/4 cup")
    turn = Node("turn when bubbly")
    eat = Node("eat")

    milk.outgoing_nodes = [mix]
    egg.outgoing_nodes = [mix]
    oil.outgoing_nodes = [mix]
    mix.incoming_nodes = [milk, egg, oil]
    mix.outgoing_nodes = [syrup, pour]
    griddle.outgoing_nodes = [pour]
    pour.incoming_nodes = [mix, griddle]
    pour.outgoing_nodes = [turn]
    turn.incoming_nodes = [pour]
    turn.outgoing_nodes = [eat]
    syrup.incoming_nodes = [mix]
    syrup.outgoing_nodes = [eat]
    eat.incoming_nodes = [syrup, turn]

    try:
        if topological_ordering([milk, egg, oil, mix, syrup, griddle, pour, turn, eat]) is not None:
            [print(x.value, end=" ") for x in topological_ordering([milk, egg, oil, mix, syrup, griddle, pour, turn, eat])]
        else:
            print("Graph has a cycle")
    except Exception as e:
        print(e)
    print()


if __name__ == "__main__":
    main()