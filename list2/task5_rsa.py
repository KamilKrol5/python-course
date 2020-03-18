import random
import sys
from typing import List

import math


def miller_rabin(n, rounds):
    if n % 2 == 0 or n == 1:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for _ in range(rounds):
        result = True
        w = random.randrange(2, n)
        if pow(w, d, n) == 1:
            continue
        for i in range(r):
            if pow(w, 2 ** i * d, n) == n - 1:
                result = False
                break
        if result:
            return False
    return True


def test_miller_rabin():
    k = 8
    print("All primes smaller than 100: ")
    for nn in range(1, 100):
        if miller_rabin(nn, k):
            print(nn, end=" ")


def modular_multiplicative_inverse(number, modulus):
    # works only when number is prime
    return pow(number, modulus - 2, modulus)


def modulo_inverse(number, modulus):
    if modulus == 1:
        return 0
    modulus_copy = modulus
    y, x = 0, 1
    while number > 1:
        q = number // modulus
        modulus, number = number % modulus, modulus
        y, x = x - q * y, y
    if x < 0:
        x = x + modulus_copy

    return x


def compute_keys(prime_1, prime_2):
    n = prime_1 * prime_2
    phi = (prime_1 - 1) * (prime_2 - 1)
    # find key1 which is relatively prime with phi
    gcd = 0
    key1 = 0
    while gcd != 1:
        key1 = random.randrange(2, phi - 1)
        gcd = math.gcd(key1, phi)

    key2 = modulo_inverse(key1, phi)
    return (key1, n), (key2, n)


def generate_keys(key_length: int):
    lower_bound = 2 ** (key_length - 1)
    higher_bound = 2 ** key_length
    rounds = 6

    prime1 = 0
    while not miller_rabin(prime1, rounds):
        prime1 = random.randrange(lower_bound, higher_bound)

    prime2 = 0
    while not miller_rabin(prime2, rounds):
        prime2 = random.randrange(lower_bound, higher_bound)

    private, public = compute_keys(prime1, prime2)

    with open('key.pub', 'w') as pub:
        pub.write(hex(public[0]))
        pub.write('\n')
        pub.write(hex(public[1]))
        pub.write('\n')
        pub.write(hex(key_length//8))

    with open('key.prv', 'w') as prv:
        prv.write(hex(private[0]))
        prv.write('\n')
        prv.write(hex(private[1]))
        prv.write('\n')
        prv.write(hex(key_length//8))


def encrypt_int(byte: int, key) -> int:
    e, n, _ = key
    return pow(byte, e, n)


def decrypt_int(byte: int, key) -> int:
    d, n, _ = key
    return pow(byte, d, n)


def encrypt(data: bytes, key) -> List[int]:
    buffer_size = key[2]
    data = data + (b' ' * (buffer_size - (len(data) % buffer_size)))
    return [
        encrypt_int(
            int.from_bytes(data[i:i+buffer_size], byteorder='big'),
            key
        )
        for i in range(0, len(data), buffer_size)
    ]


def decrypt(data: List[int], key) -> List[bytes]:
    buffer_size = key[2]
    return [decrypt_int(d, key).to_bytes(buffer_size, byteorder='big') for d in data]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Not enough arguments provided."
              "Usage:\n  --gen-keys <key length>\n  --encrypt/--decrypt <sequence to be encrypted>")
        exit(1)

    mode = sys.argv[1]
    if mode == '--gen-keys':
        print("Generating keys started. It may take a while.")
        generate_keys(int(sys.argv[2]))
        print("Keys generated successfully.")
    elif mode == '--encrypt':
        with open('key.pub', 'r') as f:
            public_key = [int(n, 16) for n in f.read().split()]
        message: str = sys.argv[2]
        print(encrypt(message.encode(), public_key))
    elif mode == '--decrypt':
        with open('key.prv', 'r') as f:
            private_key = [int(n, 16) for n in f.read().split()]
        messages: List[int] = [int(x.replace(',', '')) for x in sys.argv[2:]]

        print(''.join([x.decode().rstrip() for x in decrypt(messages, private_key)]))
    else:
        print("Unknown command. Usage:\n  --gen-keys <key length>\n  --encrypt/--decrypt <sequence to be encrypted>")
        exit(1)
