import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle
import torch
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("yes")
else:
    device = "cpu"

device = "cuda"

print(os.getcwd())
normal_images = []
potholes_images = []
path_normal = f'{os.getcwd()}/normal/'
path_potholes = f'{os.getcwd()}/potholes/'

for dirname, _, filenames in os.walk(path_normal):
    for filename in tqdm(filenames):
        try:
            img = cv2.imread(os.path.join(
                path_normal, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))
            normal_images.append(np.array(img))
        except Exception as e:
            pass

for dirname, _, filenames in os.walk(path_potholes):
    for filename in tqdm(filenames):
        try:
            print(filename)
            img = cv2.imread(os.path.join(
                path_potholes, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))
            potholes_images.append(np.array(img))
        except Exception as e:
            pass
print(len(normal_images))
print(len(potholes_images))

processed_data = []
for img in normal_images:
    t = torch.LongTensor(1)
    t[0] = 0
    img = torch.FloatTensor(img)
    img = torch.flatten(img)
    img = torch.FloatTensor(img)
    processed_data.append([img / 255, t])
for img in potholes_images:
    t = torch.LongTensor(1)
    t[0] = 1
    img = torch.FloatTensor(img)
    img = torch.flatten(img)
    img = torch.FloatTensor(img)
    processed_data.append([img / 255, t])

print(len(processed_data))
shuffle(processed_data)

train_data = processed_data[70:]
test_data = processed_data[0:70]

print(f"size of training data {len(train_data)}")
print(f"size of testing data {len(test_data)}")
print(train_data[0][0].shape)
print(train_data[0][1].shape)
print(train_data[0][1])


class Net(nn.Module, ):
    def __init__(self, featureSize=50*50):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 32, 2)
        # self.conv2 = nn.Conv2d(32, 64, 2)
        # self.conv3 = nn.Conv2d(64, 128, 2)
        # nn.Conv2d(1,32,5,)
        # nn.Conv1d(1,32,1,1)
        # nn.Conv1d
        self.conv1 = nn.Conv1d(1, 32, 4, 2)
        self.conv2 = nn.Conv1d(32, 64, 4, 2)
        self.conv3 = nn.Conv1d(64, 128, 4, 2)
        self.lin1 = nn.LazyLinear(32)
        self.lin2 = nn.LazyLinear(64)
        self.lin3 = nn.LazyLinear(32)
        x = torch.rand(1, featureSize).view(-1, 1, featureSize)
        self.linear_in = None
        self.convs(x)

        self.fc1 = nn.Linear(self.linear_in, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        #x = F.relu(self.conv1(x))
        # x = F.relu(self.lin1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self.linear_in == None:
            self.linear_in = x[0].shape[0] * x[0].shape[1]  # * x[0].shape[2]
        else:
            return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.linear_in)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


net = Net()
net.to(device)


def train_model(net, train_data, device, batchSize=10, trainSize=610, featureSize=50*50):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    for epoch in tqdm(range(30)):
        for i in (range(0, trainSize, batchSize)):
            batch = train_data[i:i + batchSize]
            batch_x = torch.cuda.FloatTensor(batchSize, 1, featureSize)
            batch_y = torch.cuda.LongTensor(batchSize, 1)

            for i in range(batchSize):
                batch_x[i] = batch[i][0]
                batch_y[i] = batch[i][1]
            batch_x.to(device)
            batch_y.to(device)
            net.zero_grad()
            outputs = net(batch_x.view(-1, 1, featureSize))
            batch_y = batch_y.view(batchSize)
            loss = F.nll_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"epoch : {epoch}  loss : {loss}")


def test_model(Net, test_data, featureSize=50*50):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_data):
            x = torch.FloatTensor(data[0])
            y = torch.LongTensor(data[1])

            x = x.view(-1, 1, featureSize)
            x = x.to(device)
            output = net(x)
            output = output.view(2)
            if (max(output[0], output[1]) == output[0]):
                index = 0
            else:
                index = 1
            if index == y[0]:
                correct += 1
            total += 1
        return round(correct/total, 5)


# print(train_data[2].shape)
train_model(net, train_data, device=device, batchSize=10, trainSize=610)

acc = test_model(net, test_data)
print(acc)
