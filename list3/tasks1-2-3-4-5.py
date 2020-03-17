from itertools import combinations, chain
from typing import List


def transpose(matrix: List[str]):
    return [[''.join(row.split()[i]) for row in matrix] for i in range(len(matrix))]


def flatten(collection):
    for element in collection:
        if isinstance(element, list):
            for x in flatten(element):
                yield x
        else:
            yield element


def last_column(filename: str):
    with open(filename, 'r') as file:
        return f'Total amount of bytes: {sum(int(line.split(" ")[-1]) for line in file)}'


def quick_sort_list_comprehension(collection: List):
    if len(collection) == 0:
        return []
    head, tail = collection[0], collection[1:]
    return quick_sort_list_comprehension([x for x in tail if x <= head]) + \
        [head] + \
        quick_sort_list_comprehension([x for x in tail if x > head])


def quick_sort_filter(collection: List):
    if len(collection) == 0:
        return []
    head, tail = collection[0], collection[1:]
    return quick_sort_filter(list(filter(lambda x: x <= head, tail))) + \
        [head] + \
        quick_sort_filter(list(filter(lambda x: x > head, tail)))


def subsets_short(collection: List):
    return chain.from_iterable(combinations(collection, i) for i in range(len(collection) + 1))


def subsets(collection: List):
    if len(collection) == 0:
        return [[]]
    else:
        head, tail = collection[0], collection[1:]
        return subsets(tail) + [[head, *x] for x in subsets(tail)]
        # alternative using map
        # return subsets(tail) + list(map(lambda x: [head, *x], subsets(tail)))


if __name__ == '__main__':
    # testing
    # task 1:
    M = ["1.1 2.2 3.3", "4.4 5.5 6.6", "7.7 8.8 9.9"]
    print(transpose(M))
    # task 2:
    list_ = [[1, 2, ["a", 4, "b", 5, 5, 5]], [4, 5, 6], 7, [[9, [123, [[123]]]], 10]]
    print(list(flatten(list_)))
    print(list(flatten(list_)) == [1, 2, 'a', 4, 'b', 5, 5, 5, 4, 5, 6, 7, 9, 123, 123, 10])
    # task 3:
    print(last_column('test.txt'))
    # task 4:
    print(list(quick_sort_list_comprehension([5, 7, 1, 1, 0, 9, 1, 0])))
    print(list(quick_sort_filter([5, 7, 1, 1, 0, 9, 1, 0])))
    # task 5:
    print(list(subsets_short([1, 2, 3])))
    print(subsets([1, 2, 3]))
