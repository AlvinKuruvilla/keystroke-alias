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
