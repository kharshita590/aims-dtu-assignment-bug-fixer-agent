def next_palindrome(digit_list):
    high_mid = len(digit_list) // 2
    low_mid = (len(digit_list) - 1) // 2
    while high_mid >= 0 and low_mid < len(digit_list):
        if digit_list[high_mid] < 9:
            digit_list[high_mid] += 1
            digit_list[low_mid] = digit_list[high_mid]
            return digit_list
        digit_list[high_mid] = 0
        high_mid -= 1
        low_mid += 1
    return [1] + [0] * (len(digit_list) - 1) + [1]

# Example usage:
# print(next_palindrome([1,4,9,4,1]))