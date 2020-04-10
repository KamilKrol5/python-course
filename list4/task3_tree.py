import random
from collections import deque


class TreeNode:
    def __init__(self, node_value, subtrees, height):
        self.height = height
        self.node_value = node_value
        self.subtrees = subtrees

    def __str__(self):
        return f'{self.node_value} h={self.height} [{", ".join(str(s) for s in self.subtrees)}]'

    def dfs(self):
        if self is None:
            return
        yield self.node_value
        for st in self.subtrees:
            yield from st.dfs()

    def bfs(self):
        if self is None:
            return
        queue = deque([self])
        while queue:
            st = queue.pop()
            yield st.node_value
            queue.extend(st.subtrees)


def generate_random_tree(height: int):
    if height == 0:
        return None
    elif height == 1:
        return TreeNode(random.randint(-(height ** 2), height ** 2), [], 1)

    subtrees_count = random.randint(1, 2)
    highest_subtree_position = random.randrange(0, subtrees_count)
    value = random.randint(-height ** 2, height ** 2)
    subtrees = []
    for i in range(subtrees_count):
        if i != highest_subtree_position:
            _tree = generate_random_tree(random.randint(1, height - 1))
        else:
            _tree = generate_random_tree(height - 1)
        subtrees.append(_tree)
    return TreeNode(value, subtrees, height)


if __name__ == '__main__':
    tree = generate_random_tree(3)
    print(tree)
    print(list(tree.dfs()))
    print(list(tree.bfs()))
