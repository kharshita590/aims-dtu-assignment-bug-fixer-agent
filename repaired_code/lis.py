def lis(arr):
    ends = {}
    longest = 0

    for i, val in enumerate(arr):
        prefix_lengths = [j for j in range(1, longest + 1) if arr[ends[j]] < val]

        length = max(prefix_lengths) if prefix_lengths else 0

        if length == longest or val < arr[ends[length + 1]]:
            ends[length + 1] = i
            longest = length + 1

    return longest

# Longest Increasing Subsequence
def longest_increasing_subsequence(arr):
    ends = {}
    longest = 0

    for i, val in enumerate(arr):
        prefix_lengths = [j for j in range(1, longest + 1) if arr[ends[j]] < val]

        length = max(prefix_lengths) if prefix_lengths else 0

        if length == longest or val < arr[ends[length + 1]]:
            ends[length + 1] = i
            longest = length + 1

    return longest

# Input:
arr = [4, 1, 5, 3, 7, 6, 2]

# Precondition:
# The ints in arr are unique

# Output:
# The length of the longest monotonically increasing subsequence of arr

# Example:
print(longest_increasing_subsequence(arr))  # Output: 3