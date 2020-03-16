from itertools import combinations, chain
from typing import List


def transpose(matrix: List[str]):
    return [[''.join(row.split()[i]) for row in matrix] for i in range(len(matrix))]


def flatten(collection: List):
    pass


def quick_sort_list_comprehension(collection: List):
    if len(collection) == 0:
        return []
    first = collection[0]
    return quick_sort_list_comprehension([x for x in collection[1:] if x <= first]) + \
        [first] + \
        quick_sort_list_comprehension([x for x in collection[1:] if x > first])


def quick_sort_filter(collection: List):
    if len(collection) == 0:
        return []
    first = collection[0]
    return quick_sort_filter(list(filter(lambda x: x <= first, collection[1:]))) + \
        [first] + \
        quick_sort_filter(list(filter(lambda x: x > first, collection[1:])))


def subsets_list_comprehension(collection: List):
    return chain.from_iterable(combinations(collection, i) for i in range(len(collection) + 1))


def subsets(collection: List):
    pass


if __name__ == '__main__':
    # testing
    # task 1:
    M = ["1.1 2.2 3.3", "4.4 5.5 6.6", "7.7 8.8 9.9"]
    print(transpose(M))
    # task 2:
    # list_ = [[1, 2, ["a", 4, "b", 5, 5, 5]], [4, 5, 6], 7, [[9, [123, [[123]]]], 10]]
    # print(flatten(list_))
    # print(list_ == [1, 2, 'a', 4, 'b', 5, 5, 5, 4, 5, 6, 7, 9, 123, 123, 10])
    # task 4:
    print(list(quick_sort_list_comprehension([5, 7, 1, 1, 0, 9, 1, 0])))
    print(list(quick_sort_filter([5, 7, 1, 1, 0, 9, 1, 0])))
    # task 5:
    print(list(subsets_list_comprehension([1, 2, 3])))
