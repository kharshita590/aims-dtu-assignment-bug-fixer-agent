def subsequences(a, b, k):
    if k == 0:
        return [[]]
    if a > b:
        return []
    ret = []
    for i in range(a, b + 1):
        for rest in subsequences(i + 1, b, k - 1):
            ret.append([i] + rest)
    return ret

# Example usage:
# print(subsequences(a=1, b=5, k=3))