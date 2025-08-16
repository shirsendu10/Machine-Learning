def find_combinations(target):
    def find_combinations_recursive(target, start, path, result):
        if target == 0:
            result.append(path)
            return
        for i in range(start, int(target**0.5) + 1):
            square = i * i
            if square <= target:
                find_combinations_recursive(target - square, i, path + [square], result)

    result = []
    find_combinations_recursive(target, 1, [], result)
    return min(result, key=len)

# Example usage:
target_number = 7
combination = find_combinations(target_number)
print(f"The combination with the least count to form {target_number} is: {combination}")
target_number = int(input("Enter the target number: "))
combination = find_combinations(target_number)
print(f"The number of numbers used to form {target_number} is: {len(combination)}")
print(f"The combination is: {combination}")