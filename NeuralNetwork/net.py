import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution layer
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        # Drop out
        self.drop_out1 = nn.Dropout(0.25)
        self.drop_out2 = nn.Dropout(0.5)
        # Fully connection
        self.fc1 = nn.Linear(3136, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop_out1(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2, stride=2)
        x = self.drop_out1(x)

        x = x.view(-1, self.Flatten(x))
        x = F.relu(self.fc1(x))
        x = self.drop_out2(x)
        x = F.log_softmax(self.fc2(x))
        return x

    @staticmethod
    def Flatten(x):
        """ Get flattened x's dim """
        dims = x.shape[1:]
        result = 1
        for dim in dims:
            result *= dim
        return result

def Accuracy(outputs, labels):
    outputs = torch.argmax(outputs, dim=1)      # 返回最大值的索引，而不是返回最大值
    return float(torch.sum(outputs == labels)) / float(labels.size()[0])

metrics = {
    "accuracy": Accuracy
}

