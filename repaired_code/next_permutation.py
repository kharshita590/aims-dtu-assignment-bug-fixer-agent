def next_permutation(perm):
    perm = perm[:]  # Create a copy of the list to avoid modifying the original
    i = len(perm) - 2
    while i >= 0 and perm[i] >= perm[i + 1]:
        i -= 1
    if i >= 0:
        j = len(perm) - 1
        while perm[j] <= perm[i]:
            j -= 1
        perm[i], perm[j] = perm[j], perm[i]  # Swap the found elements
    perm[i + 1:] = reversed(perm[i + 1:])  # Reverse the sequence after the swapped element
    return perm

# Example usage:
# print(next_permutation([3, 2, 4, 1]))