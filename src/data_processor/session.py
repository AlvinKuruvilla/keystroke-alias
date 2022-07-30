import os
from tqdm import tqdm


def find_escape_sequence(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            key = line.split(",")[1]
            print(key)


if __name__ == "__main__":
    p = os.path.join(os.getcwd(), "gen", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    for file in tqdm(onlyfiles):
        find_escape_sequence(os.path.join(p, file))
