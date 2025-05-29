def topological_ordering(nodes):
    """
    Performs a topological sort on a directed acyclic graph.

    Args:
        nodes: A list of nodes in the graph. Each node is expected to have
               an 'incoming_nodes' attribute (a list of nodes pointing to it)
               and an 'outgoing_nodes' attribute (a list of nodes it points to).

    Returns:
        A list of nodes in topological order, or an empty list if the input
        is invalid or the graph contains cycles.
    """
    ordered_nodes = [node for node in nodes if not node.incoming_nodes]
    visited = set(ordered_nodes)
    
    for node in ordered_nodes:
        for nextnode in node.outgoing_nodes:
            if nextnode not in visited:
                
                can_add = True
                for incoming in nextnode.incoming_nodes:
                    if incoming not in visited:
                        can_add = False
                        break
                if can_add:
                    ordered_nodes.append(nextnode)
                    visited.add(nextnode)

    if len(ordered_nodes) != len(nodes):
        return []  # Cycle detected or invalid input

    return ordered_nodes

"""
Topological Sort

Input:
    nodes: A list of directed graph nodes
 
Precondition:
    The input graph is acyclic

Output:
    An OrderedSet containing the elements of nodes in an order that puts each node before all the nodes it has edges to
"""