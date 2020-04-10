import random


class TreeNode:
    def __init__(self, node_value, subtrees):
        self.node_value = node_value
        self.subtrees = subtrees

    def __str__(self):
        return f'{self.node_value} [{", ".join(str(s) for s in self.subtrees)}]'


def generate_random_tree(height: int):
    if height == 0:
        return None
    elif height == 1:
        return TreeNode(random.randint(-(height ** 2), height ** 2), [])

    subtrees_count = random.randint(1, 6)
    highest_subtree_position = random.randrange(0, subtrees_count)
    value = random.randint(-height ** 2, height ** 2)
    subtrees = [
        generate_random_tree(random.randint(1, height - 1))
        if i != highest_subtree_position else
        generate_random_tree(height - 1)
        for i in range(subtrees_count)]
    return TreeNode(value, subtrees)


def dfs(tree: TreeNode):
    if tree is None:
        return
    yield tree.node_value
    for st in tree.subtrees:
        yield from dfs(st)


def bfs(tree):
    if tree is None:
        return
    stack = [tree]
    while len(stack) > 0:
        new_stack = []
        for st in stack:
            yield st.node_value
            for sst in st.subtrees:
                new_stack.append(sst)
        stack = new_stack.copy()


if __name__ == '__main__':
    tree = generate_random_tree(3)
    print(tree)
    print(list(dfs(tree)))
    print(list(bfs(tree)))
