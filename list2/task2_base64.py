import sys


def decode_base64(label):
    pass


def encode_base64(arg):
    pass


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Invalid number of arguments. Valid arguments are: --encode or --decode <src/dest file> <src/dest file>")
        exit(1)
