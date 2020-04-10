import random


def generate_random_tree(height: int):
    if height == 0:
        return None

    value= random.randint(-height**2, height**2)
    subtree_random_height = generate_random_tree(random.randint(0, height - 1))
    subtree_high = generate_random_tree(height - 1)

    if random.choice([True, False]):
        return [value, subtree_random_height, subtree_high]
    return [value, subtree_high, subtree_random_height]


def dfs(tree):
    if tree is None:
        return
    yield tree[0]
    yield from dfs(tree[1])
    yield from dfs(tree[2])


def bfs(tree):
    if tree is None:
        return
    stack = [tree]
    while len(stack) > 0:
        new_stack = []
        for st in stack:
            if st is None:
                continue
            yield st[0]
            new_stack.append(st[1])
            new_stack.append(st[2])
        stack = new_stack.copy()


if __name__ == '__main__':
    tree = generate_random_tree(3)
    print(tree)
    print(list(dfs(tree)))
    print(list(bfs(tree)))
