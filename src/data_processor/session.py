import os
import csv
from re import L
from tqdm import tqdm
from typing import List


def flatten(xss):
    """Given a list with 1 or more arrays of arrays, flatten them to a single array"""
    return [x for xs in xss for x in xs]


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


# TODO: We must ignore a specific edge case where sometime multiple CTRL C sequences may get registered back to back
# Likely we should check the generated session data and if it is empty or just contains the CTRL C sequence ignore it entirely
def find_escape_sequence(path: str):
    session_count = 0
    session_data = []
    with open(path, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        while len(lines) > 0:
            current_ctrl_pos = 0
            ctrl_pos = find_first_index_of("Key.ctrl", lines)
            c_pos = find_first_index_of_from_position("'c'", lines, ctrl_pos)
            session = [lines[: ctrl_pos + c_pos + 1]]
            # print(session)
            # input()
            session_data.append([session])
            lines = lines[ctrl_pos + c_pos + 1 :]
            if abs(current_ctrl_pos - ctrl_pos) > 10:
                session_count += 1
    print(session_count)
    return session_data


def generate_all_session():
    p = os.path.join(os.getcwd(), "gen", "km")
    onlyfiles = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    i = 0
    for file in tqdm(onlyfiles):
        data = find_escape_sequence(os.path.join(p, file))
        sess_count = len(find_escape_sequence(os.path.join(p, file)))
        # for session in sess_count:
        #     data[session]
        dir_name = os.path.splitext(file)[0]
        dir_path = os.path.join(os.getcwd(), "test", dir_name)
        if sess_count == 6:
            print(file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            while i < sess_count:
                with open(
                    os.path.join(dir_path, file + str(i + 1) + ".csv"), "w+"
                ) as f:
                    writer = csv.writer(f)
                    header = ["EID," "key," "direction," "time"]
                    writer.writerow(header)
                    writer.writerows(flatten(data[i]))
                i += 1


if __name__ == "__main__":
    p = os.path.join(os.getcwd(), "gen", "km")
    i = 0
    file = os.path.join(p, "t_31_fpd1.csv")
    data = find_escape_sequence(os.path.join(p, file))
    sess_count = len(find_escape_sequence(os.path.join(p, file)))
    print("Sess count: %d" % sess_count)
    dir_name = os.path.splitext(file)[0]
    dir_path = os.path.join(os.getcwd(), "test", dir_name)
    if sess_count == 6:
        print(file)
        while i < sess_count:
            with open(
                os.path.join(
                    dir_path,
                    file.split("_")[0]
                    + "_"
                    + file.split("_")[1]
                    + "_"
                    + str(i + 1)
                    + ".csv",
                ),
                "w+",
            ) as f:
                writer = csv.writer(f)
                header = ["EID," "key," "direction," "time"]
                writer.writerow(header)
                writer.writerows(flatten(data[i]))
            i += 1
    else:
        print("Could not make sessions for file " + file)
