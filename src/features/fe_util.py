import numpy as np

# Remove outlier points in the distribution using the 1.5IQR rule
def remove_outliers(x):
    a = np.asarray(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    result = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            result.append(y)
    return result


def event_to_int(event: str) -> int:
    if event == "P":
        return 1
    elif event == "R":
        return 0


# Make sure that the direction of the event is an integer rather than "P" or "R"
def conform_to_int(data):
    result = []
    for row_idx in data:
        result.append([event_to_int(row_idx[0]), row_idx[1], row_idx[2]])
    return result
