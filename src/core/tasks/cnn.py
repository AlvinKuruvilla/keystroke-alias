import torch.nn as nn

# CNN model for classification tasks
class CNN_Net(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=2, bias=True
        )
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=True)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, bias=True
        )
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=True
        )

        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

    def forward(self, feats):
        out = self.bn1(self.relu(self.conv1(feats)))
        out = self.bn2(self.relu(self.conv2(out)))
        out = self.bn3(self.relu(self.conv3(out)))
        out = self.bn4(self.relu(self.conv4(out)))
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.dropout_2(self.relu(self.fc1(out)))
        out = self.dropout_2(self.relu(self.fc2(out)))
        out = self.fc3(out)
        return out
