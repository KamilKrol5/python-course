import sys


def decode_base64(label):
    pass


def encode_base64(data: bytes):
    # data: bytes = data + (6 - (len(data) % 6)) * b'='
    table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    output = []
    for i in range(0, len(data), 3):
        slice_ = data[i:i + 3]
        output.extend([
            slice_[0] >> 2,
            ((slice_[0] & 0b0000_0011) << 4) + (slice_[1] >> 4),
            ((slice_[1] & 0b0000_1111) << 2) + (slice_[2] >> 6),
            slice_[2] & 0b0011_1111,
        ])
    output = ''.join(table[s] for s in output)
    return output


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Invalid number of arguments. Valid arguments are: --encode or --decode <src/dest file> <src/dest file>")
        exit(1)

    print(encode_base64("Python3".encode()))
