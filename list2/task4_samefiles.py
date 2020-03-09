import hashlib
import sys
from collections import defaultdict
from pathlib import Path

BYTES_TO_READ = 2 ** 12
PREFIX = ' ' * 5


def find_file_duplicates(dir_name: str):
    files = defaultdict(list)
    root = Path(dir_name)
    for path in filter(Path.is_file, root.rglob("*")):
        hash_ = hashlib.md5()
        with path.open('rb') as file:
            for fb in iter(lambda: file.read(BYTES_TO_READ), b''):
                hash_.update(fb)

        files[hash_.hexdigest()].append(path)
    return files


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid number of arguments. Valid argument is: <directory name>")
        exit(1)

    data = find_file_duplicates(sys.argv[1])
    for key, entry in ((k, v) for k, v in data.items() if len(v) > 1):
        print(f'Hash: {key} The same files:')
        for f in entry:
            print(PREFIX, f)
