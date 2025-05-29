def detect_cycle(node):
    if not node:
        return False

    hare = node
    tortoise = node
 
    while True:
        if not hare or not hare.successor:
            return False

        tortoise = tortoise.successor
        hare = hare.successor.successor

        if hare is tortoise:
            return True