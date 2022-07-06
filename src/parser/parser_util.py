import ntpath


def path_leaf(path: str):
    """An OS agnostic function to extract the filename from a path"""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
