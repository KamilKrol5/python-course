import sys


class FileStats:
    def __init__(self, number_of_bytes, number_of_lines, number_of_words, max_line_length, ):
        self.max_line_length = max_line_length
        self.number_of_words = number_of_words
        self.number_of_lines = number_of_lines
        self.number_of_bytes = number_of_bytes

    def __str__(self):
        return f'Number of lines: {self.number_of_lines}\n' \
            f'Number of words: {self.number_of_words}\n' \
            f'Bytes: {self.number_of_bytes}\n' \
            f'Max line length: {self.max_line_length}\n'


def file_info(filename: str):
    number_of_lines = 0
    number_of_bytes = 0
    number_of_words = 0
    max_line_length = 0
    with open(filename, 'rb') as file:
        for line in file:
            number_of_bytes += len(line)
            number_of_lines += 1
            number_of_words += len(line.split())
            max_line_length = max(max_line_length, len(line))
    return FileStats(number_of_bytes=number_of_bytes, number_of_words=number_of_words,
                     number_of_lines=number_of_lines, max_line_length=max_line_length)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid number of arguments.")
        exit(0)

    print(file_info(sys.argv[1]))
