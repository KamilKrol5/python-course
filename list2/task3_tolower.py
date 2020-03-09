import os
import sys


def to_lower(dir_path: str):
    with os.scandir(dir_path) as x:
        for entry in x:
            entry: os.DirEntry
            if entry.is_file():
                print(os.path.join(os.path.dirname(entry.path), entry.name.lower()))
                os.rename(entry.path, os.path.join(os.path.dirname(entry.path), entry.name.lower()))
            elif entry.is_dir():
                to_lower(entry.path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid number of arguments. Valid argument is: <directory name>")
        exit(1)

    to_lower(sys.argv[1])
