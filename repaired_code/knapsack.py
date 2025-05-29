from collections import defaultdict

def knapsack(capacity, items):
    memo = defaultdict(int)
    n = len(items)

    for i in range(n):
        weight, value = items[i]
        for j in range(capacity, weight - 1, -1):
            memo[i + 1, j] = max(memo[i, j], memo[i, j - weight] + value)

    return memo[n, capacity]

# Example usage:
# print(knapsack(100, [(60, 10), (50, 8), (20, 4), (20, 4), (8, 3), (3, 2)]))