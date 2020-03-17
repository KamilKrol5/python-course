import sys

table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
char_for_index = dict((c, i) for (i, c) in enumerate(table))


def decode_base64(data2: str):
    padding = data2.count('=')
    data = [char_for_index[letter] & 0b0011_1111 for letter in data2]
    output = []
    for i in range(0, len(data), 4):
        slice_ = data[i:i + 4]
        output.extend([
            (slice_[0] << 2) + (slice_[1] >> 4),
            ((slice_[1] & 0b00_11_11) << 4) + (slice_[2] >> 2),
            ((slice_[2] & 0b00_00_11) << 6) + slice_[3],
        ])
    for i in range(padding):
        output.pop()
    return bytearray(output)


def encode_base64(data: bytes):
    for_padding = 0
    output = []
    for i in range(0, len(data), 3):
        slice_ = data[i:i + 3]
        for_padding = (3 - len(slice_))
        if len(slice_) < 4:
            slice_ = slice_ + bytearray(for_padding)

        output.extend([
            slice_[0] >> 2,
            ((slice_[0] & 0b0000_0011) << 4) + (slice_[1] >> 4),
            ((slice_[1] & 0b0000_1111) << 2) + (slice_[2] >> 6),
            slice_[2] & 0b0011_1111,
        ])
    for i in range(for_padding):
        output[-i - 1] = -1

    output_str = ''.join(table[s] for s in output)
    return output_str


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Invalid number of arguments. Valid arguments are: --encode or --decode <src/dest file> <src/dest file>")
        exit(1)

    with open(sys.argv[2], 'rb') as file_src, open(sys.argv[3], 'wb') as file_dst:
        for fb in iter(lambda: file_src.read(240000), b''):
            if sys.argv[1] == '--encode':
                file_dst.write(encode_base64(fb).encode())
            elif sys.argv[1] == '--decode':
                file_dst.write(decode_base64(fb.decode()))
            else:
                print("Unknown mode. Valid modes are: --encode or --decode.")
                exit(1)

    # tests
    # print(encode_base64("Python".encode()))
    # print(encode_base64("pleasure.".encode()))
    # print(encode_base64("leasure.".encode()))
    # print(encode_base64("easure.".encode()))
    # print('---')
    # print(decode_base64(encode_base64("Python".encode())))
    # print(decode_base64(encode_base64("pleasure.".encode())))
    # print(decode_base64(encode_base64("leasure.".encode())))
    # print(decode_base64(encode_base64("easure.".encode())))
