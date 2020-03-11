"""
Part 1:
Recursive function to find subsets of a list of items, puts the result into the array 'res'
"""

res = []


def subsets(inp, sub, size):
    curr_sub = sub.copy()

    for i in range(len(inp)):
        curr_sub.append(inp[i])

        if curr_sub not in res:
            res.append(curr_sub)

        if len(curr_sub) < size:
            subsets(inp[i+1:], curr_sub, size)

        curr_sub = sub.copy()


LIST = [1,2,2]

subsets(LIST, [], len(LIST))

print(LIST, "subsets:")
print(res)
