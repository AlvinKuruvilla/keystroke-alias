import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from core.tasks.cnn import CNN_Net

from core.tasks.rnn_utils import get_train_test_splits

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.bias.data.zero_()
        nn.init.kaiming_uniform_(layer.weight.data)


def train_model_cnn(label_type):
    X_matrix_new, Y_vector = get_train_test_splits(label_type)
    X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(
        X_matrix_new, Y_vector
    )
    X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_matrix_new, Y_vector, test_size=0.3, random_state=0
    )
    kf = StratifiedKFold(n_splits=3)
    kf.get_n_splits(X_train)

    split_num = 0

    for train_index, val_index in kf.split(X_train, Y_train):
        print("Split Num: " + str(split_num))
        split_num += 1
        X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
        Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

        fcn = CNN_Net(X_train.shape[1], 2)
        fcn.apply(weights_init)
        optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
        loss_func = nn.CrossEntropyLoss()

        fcn
        loss_func

        train_tensor_x = torch.Tensor(X_train)
        train_tensor_y = torch.Tensor(Y_train)
        val_tensor_x = torch.Tensor(X_val)
        val_tensor_y = torch.Tensor(Y_val)
        test_tensor_x = torch.Tensor(X_test)
        test_tensor_y = torch.Tensor(Y_test)

        train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=10, shuffle=True, drop_last=False
        )

        val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=10, shuffle=False, drop_last=False
        )

        test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, shuffle=False, drop_last=False
        )

        for epoch in range(10):
            for itr, (x, y) in enumerate(train_dataloader):
                fcn.train()
                x = x
                y = y

                outputs = fcn(x)
                loss = loss_func(outputs, y.long())

                params = list(fcn.parameters())
                l1_regularization, l2_regularization = torch.norm(
                    params[0], 1
                ), torch.norm(params[0], 2)

                for param in params:
                    l1_regularization += torch.norm(param, 1)
                    # l2_regularization += torch.norm(param, 2)

                reg_1 = Variable(l1_regularization)
                # reg_2 = Variable(l2_regularization)

                loss += reg_1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    fcn.eval()
                    train_correct = 0
                    train_total = 0
                    val_correct = 0
                    val_total = 0
                    test_correct = 0
                    test_total = 0

                    for (x, y) in train_dataloader:
                        x = x
                        y = y

                        outputs = fcn(x)
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += y.size(0)
                        train_correct += (y == predicted).sum().item()

                    for (x, y) in val_dataloader:
                        x = x
                        y = y

                        outputs = fcn(x)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += y.size(0)
                        val_correct += (y == predicted).sum().item()

                    for (x, y) in test_dataloader:
                        x = x
                        y = y

                        outputs = fcn(x)
                        _, predicted = torch.max(outputs.data, 1)
                        test_total += y.size(0)
                        test_correct += (y == predicted).sum().item()

                if 100 * val_correct / val_total > 60:
                    print(
                        "Epoch: "
                        + str(epoch)
                        + ", Itr: "
                        + str(itr)
                        + ", Loss: "
                        + str(loss.item())
                        + ", Train Acc: "
                        + str(100 * train_correct / train_total)
                        + ", Val Acc: "
                        + str(100 * val_correct / val_total)
                        + ", Test Acc: "
                        + str(100 * test_correct / test_total)
                    )
