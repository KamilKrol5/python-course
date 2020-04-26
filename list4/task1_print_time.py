import random
from time import time


def execution_time_printer(f):
    def modified_f(*args, **kwargs):
        start = time()
        f(*args, **kwargs)
        print(f'Execution time: {time() - start}')
    return modified_f


@execution_time_printer
def generate_random_numbers_and_do_nothing_more(count: int):
    for _ in range(count):
        random.randint(10, 100000000)


if __name__ == '__main__':
    generate_random_numbers_and_do_nothing_more(1000000)
