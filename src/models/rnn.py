from sklearn import preprocessing
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from imblearn.over_sampling import SMOTE

from features.feature_lists import get_combined_features


# RNN model for classification tasks
class RNN_Net(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_classes):
        super(RNN_Net, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=3,
        )

        self.fc = nn.Linear(self.hidden_size, num_classes)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()

    def forward(self, feats):
        h_0 = Variable(torch.zeros(3, self.batch_size, self.hidden_size))
        h_0 = h_0
        # h_0 = h_0.cuda()

        out, final_h = self.rnn(feats, h_0)
        out = self.fc(final_h[-1])
        return out


# Create the appropriate train-test splits for free text classification tasks to align with the ML models
def get_train_test_splits(label_name):
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if label_name == "Gender" or label_name == "Ethnicity":
        for i in range(116):
            if Y_values[i] == "M" or Y_values[i] == "Asian":
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    Y_vector = Y_vector[:-1]
    Y_values = Y_values[:-1]
    Y_vector = Y_vector.astype("int")

    # uncomment one of the below four lines for the required feature set
    X_matrix = get_combined_features()
    # X_matrix = get_desktop_features()
    # X_matrix = get_phone_features()
    # X_matrix = get_tablet_features()

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)
    X_matrix_new = X_matrix

    print(X_matrix_new.shape)
    X_matrix_new, Y_vector = SMOTE(kind="svm").fit_sample(X_matrix_new, Y_vector)
    return X_matrix_new, Y_vector


# x = torch.randn((10, 3, 547)).cuda()
x = torch.randn((10, 3, 547))
# FIXME: AssertionError: Torch not compiled with CUDA enabled
# RNN_Net(10, 547, 10, 2).cuda().forward(x)
RNN_Net(10, 547, 10, 2).forward(x)
