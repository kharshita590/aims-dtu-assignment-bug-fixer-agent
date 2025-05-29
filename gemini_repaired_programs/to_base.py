import string

def to_base(num, b):
    if not isinstance(num, int) or not isinstance(b, int):
        raise TypeError("Both num and b must be integers.")
    if num <= 0:
        return "0"  # Handle the case when num is 0
    if not 2 <= b <= 36:
        raise ValueError("Base b must be between 2 and 36.")

    result = ''
    alphabet = string.digits + string.ascii_uppercase
    while num > 0:
        i = num % b
        num = num // b
        result = alphabet[i] + result  # Prepend for correct order
    return result