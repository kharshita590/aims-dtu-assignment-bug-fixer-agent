from collections import deque

def breadth_first_search(startnode, goalnode):
    queue = deque()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while queue:
        node = queue.popleft()

        if node == goalnode:
            return True
        else:
            if hasattr(node, 'successors'):
                for successor in node.successors:
                    if successor not in nodesseen:
                        queue.append(successor)
                        nodesseen.add(successor)
            else:
                return False

    return False