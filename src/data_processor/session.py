import os
from tqdm import tqdm
from typing import List


def find_first_index_of(key: str, data: List[str]):
    found_index = 0
    for line in data:
        if line.split(",")[1] == key:
            return found_index
        found_index += 1


def find_first_index_of_from_position(key: str, data: List[str], start_index: int):
    found_index = 0
    remaining_data = data[start_index:]
    for line in remaining_data:
        if line.split(",")[1] == key:
            return found_index
        found_index += 1


def find_escape_sequence(path: str):
    session_data = []
    with open(path, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        while len(lines) > 0:
            ctrl_pos = find_first_index_of("Key.ctrl", lines)
            c_pos = find_first_index_of_from_position("'c'", lines, ctrl_pos)
            session = lines[: ctrl_pos + c_pos + 1]
            print(session)
            input()
            session_data.append(session)
            lines = lines[ctrl_pos + c_pos + 1 :]
    return session_data[0]


if __name__ == "__main__":
    p = os.path.join(os.getcwd(), "gen", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    for file in tqdm(onlyfiles):
        find_escape_sequence(os.path.join(p, file))
        input("File FINISHED\n")
