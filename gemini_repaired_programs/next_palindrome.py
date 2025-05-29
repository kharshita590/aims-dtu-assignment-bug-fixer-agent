def next_palindrome(digit_list):
    n = len(digit_list)
    high_mid = n // 2
    low_mid = (n - 1) // 2

    # Copy the left side to the right side
    temp_list = digit_list[:]
    i = high_mid
    j = low_mid
    while i < n and j >= 0:
        temp_list[i] = temp_list[j]
        i += 1
        j -= 1

    # Check if the new number is greater than the original number
    if temp_list > digit_list:
        return temp_list

    # If not, increment the middle digit(s)
    i = low_mid
    carry = 1
    while i >= 0:
        digit_list[i] += carry
        carry = digit_list[i] // 10
        digit_list[i] %= 10
        i -= 1

    # If there is a carry, add a 1 to the beginning of the list
    if carry:
        return [1] + [0] * n

    # Copy the left side to the right side again
    temp_list = digit_list[:]
    i = high_mid
    j = low_mid
    while i < n and j >= 0:
        temp_list[i] = temp_list[j]
        i += 1
        j -= 1

    return temp_list