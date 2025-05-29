def next_permutation(perm):
    """
    Next Permutation

    Input:
        perm: A list of unique ints

    Precondition:
        perm is not sorted in reverse order

    Output:
        The lexicographically next permutation of the elements of perm

    Example:
        >>> next_permutation([3, 2, 4, 1])
        [3, 4, 1, 2]
    """
    n = len(perm)
    for i in range(n - 2, -1, -1):
        if perm[i] < perm[i + 1]:
            for j in range(n - 1, i, -1):
                if perm[j] > perm[i]:
                    next_perm = list(perm)
                    next_perm[i], next_perm[j] = next_perm[j], next_perm[i]
                    next_perm[i + 1:] = reversed(next_perm[i + 1:])
                    return next_perm
    return list(reversed(perm))