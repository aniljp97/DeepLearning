"""
Part 2:
Concatenate 2 dictionaries and sort them
"""

dict1 = {
    0: 'A',
    1: 'C',
    2: 'E',
    10: 'G'
}

dict2 = {
    6: 'B',
    7: 'D',
    8: 'F',
    9: 'H'
}

# Concatenate dict1 and dict2 into dict1
dict1.update(dict2)

print("Concatenated:", dict1)

# Sort dict1 and update it
dict1 = dict(sorted(dict1.items(), key=lambda x: x[1]))

print("Sorted:", dict1)
