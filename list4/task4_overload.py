import math
from inspect import getfullargspec


class OverloadMemory:
    instances = {}

    def __init__(self, name):
        self.name = name
        self.functions = {}

    @classmethod
    def get_instance(cls, f_name):
        instance = OverloadMemory.instances.get(f_name)
        if instance is None:
            instance = OverloadMemory(f_name)
            OverloadMemory.instances[f_name] = instance
        return instance

    def __call__(self, *args, **kwargs):
        return self.functions[len(args)](*args, **kwargs)


def overload(f):

    clazz = OverloadMemory.get_instance(f.__name__)
    clazz.functions[len(getfullargspec(f).args)] = f

    return clazz


@overload
def norm(x):
    return abs(x)


@overload
def norm(x, y):
    return math.sqrt(x * x + y * y)


@overload
def norm(x, y, z):
    return abs(x) + abs(y) + abs(z)


if __name__ == '__main__':
    print(f"norm(-2) = {norm(2)}")
    print(f"norm(2,4) = {norm(2, 4)}")
    print(f"norm(2,3,4) = {norm(2, 3, 4)}")
