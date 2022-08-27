# TODO: Combine this code with the duplicated code in custom and put into a commons folder
from sklearn import preprocessing
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from imblearn.over_sampling import SMOTE

from core.features.feature_processor import get_combined_features, get_desktop_features


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


# x = torch.randn((10, 3, 547)).cuda()
x = torch.randn((10, 3, 547))
# FIXME: AssertionError: Torch not compiled with CUDA enabled
# RNN_Net(10, 547, 10, 2).cuda().forward(x)
RNN_Net(10, 547, 10, 2).forward(x)
