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
    output_str = bytes(output).decode('utf-8')
    return output_str


def encode_base64(data: str):
    data = data.encode()
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

    actions = {'--encode': encode_base64, '--decode': decode_base64}
    if sys.argv[1] not in actions:
        print("Unknown mode. Valid modes are: --encode or --decode.")
        exit(1)

    with open(sys.argv[2], 'r') as file_src, open(sys.argv[3], 'w') as file_dst:
        for line in file_src:
            file_dst.write(actions[sys.argv[1]](line))

    # tests
    # print(encode_base64("Python"))
    # print(encode_base64("pleasure."))
    # print(encode_base64("leasure."))
    # print(encode_base64("easure."))
    # print('---')
    # print(decode_base64(encode_base64("Python")))
    # print(decode_base64(encode_base64("pleasure.")))
    # print(decode_base64(encode_base64("leasure.")))
    # print(decode_base64(encode_base64("easure.")))
