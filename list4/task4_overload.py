from collections import defaultdict

import math
from inspect import getfullargspec


class OverloadMemory:
    instances = {}

    def __init__(self, name):
        self.name = name
        self.functions = defaultdict(list)

    @classmethod
    def get_instance(cls, f):
        f_name = f.__name__
        instance = cls.instances.get(f_name)
        if instance is None:
            instance = cls(f_name)
            cls.instances[f_name] = instance
        return instance

    def add_function(self, f):
        self.functions[len(getfullargspec(f).args)].append(f)

    @staticmethod
    def __get_ordered_argument_types(function):
        full_arg_spec = getfullargspec(function)
        return [full_arg_spec.annotations.get(argument, None)
                for argument in full_arg_spec.args]

    def __call__(self, *args, **kwargs):
        functions = [(fun, self.__get_ordered_argument_types(fun))
                     for fun in self.functions[len(args)]]
        functions.sort(key=lambda elem: elem[1].count(None))

        for fun, types in functions:
            for arg_type, arg in zip(types, args):
                if arg_type is not None:
                    if type(arg) != arg_type:
                        break
            else:
                return fun(*args, **kwargs)
        raise TypeError(f'No matching function {self.name} with arguments: {", ".join(type(a).__name__ for a in args)}')


def overload(f):
    memory = OverloadMemory.get_instance(f)
    memory.add_function(f)

    return memory


@overload
def norm(x: int):
    print('int')
    return abs(x)


@overload
def norm(x: float):
    print('float')
    return abs(x)


@overload
def norm(x, y):
    print('not specified')
    return 0


@overload
def norm(x: int, y: int):
    print('int int')
    return math.sqrt(x * x + y * y)


@overload
def norm(x, y: float):
    print('any float')
    return y


@overload
def norm(x, y, z):
    return abs(x) + abs(y) + abs(z)


if __name__ == '__main__':
    print(f"norm(-2) = {norm(-2)}")
    print(f"norm(-2.4) = {norm(-2.4)}")

    print(f"norm(2,4) = {norm(2, 4)}")
    print(f"norm(2,4.0) = {norm(2, 4.0)}")
    print(f"norm('2',4.0) = {norm('2', 4.0)}")
    print(f"norm('2',4) = {norm('2', 4)}")
    print(f"norm(2,'4.0') = {norm(2, '4.0')}")
    print(f"norm('2','4') = {norm('2', '4')}")

    print(f"norm(2,3,4) = {norm(2, 3, 4)}")
