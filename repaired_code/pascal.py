def pascal(n):
    rows = [[1]]
    for r in range(1, n):
        row = []
        for c in range(r):
            upleft = rows[r - 1][c - 1] if c > 0 else 0
            upright = rows[r - 1][c] if c < r else 0
            row.append(upleft + upright)
        rows.append(row)
    return rows

# Input:
# n: The number of rows to return

# Precondition:
# n >= 1

# Output:
# The first n rows of Pascal's triangle as a list of n lists

# Example:
# pascal(5)