def powerset(arr):
    if not arr:
        return [[]]
    first, *rest = arr
    rest_subsets = powerset(rest)
    return [[first] + subset for subset in rest_subsets]

# Example usage:
# print(powerset(['a', 'b', 'c']))