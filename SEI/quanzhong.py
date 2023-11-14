import csv
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import time
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torchstat import stat
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from prettytable.prettytable import PrettyTable
# make fake data
import pandas as pd
import numpy as np
import pymysql
# GPU limited
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sql_connection = pymysql.connect(host='10.6.3.52', user='root', password='111111',
db='data', port=3306, autocommit=False, charset='utf8')

data = pd.read_csv('./feature_all_.csv', quoting=csv.QUOTE_NONE)
#data = pd.read_csv('./feature_math.csv', quoting=csv.QUOTE_NONE)

sql2 = "select * from data.label"
label = pd.read_sql(sql2, con=sql_connection)

def Data_Split(x, y, T):
    choose_index = np.load('choose_index_PA.npy')
    train_index = choose_index[0:int(y.shape[0] * (1 - T))]
    test_index = choose_index[int(y.shape[0] * (1 - T)):]
    return x[train_index, :], y[train_index], x[test_index], y[test_index]

data = np.array(data)
label = np.array(label)
X_train, Y_train, X_test, Y_test = Data_Split(data, label, 0.2)  # 8:2

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
X_train.float()
Y_train.float()


X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(Y_test)
X_test.float()
Y_test.float()

batch_size = 32
dataset_train = TensorDataset(X_train, Y_train)
train_iter = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
dataset_test = TensorDataset(X_test, Y_test)
test_iter = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # ³õÊ¼»¯»ìÏý¾ØÕó£¬ÔªËØ¶¼Îª0
        self.num_classes = num_classes  # Àà±ðÊýÁ¿£¬Êý¾Ý¼¯Àà±ðÎª6
        self.labels = labels  # Àà±ð±êÇ©

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # predÎªÔ¤²â½á¹û£¬labelsÎªÕæÊµ±êÇ©
            self.matrix[p, t] += 1  # ¸ù¾ÝÔ¤²â½á¹ûºÍÕæÊµ±êÇ©µÄÖµÍ³¼ÆÊýÁ¿£¬ÔÚ»ìÏý¾ØÕóÏàÓ¦Î»ÖÃ+1

    def summary(self):  # ¼ÆËãÖ¸±êº¯Êý
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # »ìÏý¾ØÕó¶Ô½ÇÏßµÄÔªËØÖ®ºÍ£¬Ò²¾ÍÊÇ·ÖÀàÕýÈ·µÄÊýÁ¿
        acc = sum_TP / n  # ×ÜÌå×¼È·ÂÊ
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # ´´½¨Ò»¸ö±í¸ñ
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1 Score"]
        for i in range(self.num_classes):  # ¾«È·¶È¡¢ÕÙ»ØÂÊ¡¢ÌØÒì¶ÈµÄ¼ÆËã
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # Ã¿Ò»Àà×¼È·¶È
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1_Score = round(2 * Precision * Recall / (Precision + Recall), 3) if TN + FP != 0 else 0.
            #Ð¡ÊýµãÈ¡ºóÈýÎ»
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1_Score])
        print(table)
        return str(acc)

    def plot(self):  # »æÖÆ»ìÏý¾ØÕó
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # ÉèÖÃxÖá×ø±êlabel
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # ÉèÖÃyÖá×ø±êlabel
        plt.yticks(range(self.num_classes), self.labels)
        # ÏÔÊ¾colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # ÔÚÍ¼ÖÐ±ê×¢ÊýÁ¿/¸ÅÂÊÐÅÏ¢
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # ×¢ÒâÕâÀïµÄmatrix[y, x]²»ÊÇmatrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

class se_block(nn.Module):
    def __init__(self, in_channel, ratio=8):
        super(se_block, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    # Ç°Ïò´«²¥
    def forward(self, inputs):

        b = inputs.shape[0]
        c = inputs.shape[1]
        h = inputs.shape[2]
        w = inputs.shape[3]

        x = self.avg_pool(inputs)
        x = x.view([b, c])
        #x = x.float()

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        y = x.view([b, c, 1, 1])

        outputs = y * inputs
        return x, outputs

# read class_indict
json_label_path = './class_indices_pa.json'
assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
json_file = open(json_label_path, 'r')
class_indict = json.load(json_file)

labels = [label for _, label in class_indict.items()]
confusion = ConfusionMatrix(num_classes=6, labels=labels)

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(56, 56, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(56),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(56, 56, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(56),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(56, 56, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(56),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(56, 6),
            #torch.nn.Softmax(dim=1)
        )
        self.attention = se_block(56)


    def forward(self, x):
        quanzhong, x1 = self.attention(x)
        x0 = self.conv2d(x1)
        x1 = x0.view(x0.shape[0], -1)
        output = self.output(x1)

        self.x0 = x0
        return output, quanzhong
        #return output

    def get_fea(self):
        return self.x0


net = net()
net = net.to(device)
print(next(net.parameters()).device)
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()
epochs = 200 #200
train_loss, test_loss = [], []

dataset_train = TensorDataset(X_train, Y_train)
train_iter = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
dataset_test = TensorDataset(X_test, Y_test)
test_iter = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

time_start = time.time()
for t in range(epochs):
    running_loss = 0

    for datas, labels in train_iter:
        optimizer.zero_grad()
        datas = torch.unsqueeze(datas, dim=-1)
        datas = torch.unsqueeze(datas, dim=-1)
        datas = datas.to(device)
        labels = labels.to(device)
        net = net.to(device)
        y_hat, quanzhong = net(datas.float())
        #y_hat = net(datas.float())
        labels = labels.squeeze()
        loss = loss_func(y_hat, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    net.train()
    train_loss.append(running_loss / len(train_iter))
    print("Training Error: {:.3f}.. ".format(running_loss / len(train_iter)))
time_end = time.time() # 计算结束，终止计时
trainTime = time_end - time_start
print('Training time is ', trainTime, 's')

time_start = time.time()
test_runningloss = 0
test_acc = 0
with torch.no_grad():
    net.eval()
    for datas, labels in test_iter:
        datas = torch.unsqueeze(datas, dim=-1)
        datas = torch.unsqueeze(datas, dim=-1)
      #  datas = datas.permute(0, 2, 3, 1)
        datas = datas.to(device)
        labels = labels.to(device)
        y_hat, quanzhong = net(datas.float())
        #y_hat = net(datas.float())
        labels = labels.squeeze()
        test_runningloss += loss_func(y_hat, labels.long())
        ps = torch.exp(y_hat)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)

        ret, predictions = torch.max(y_hat.data, 1)
        confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())

        test_acc += torch.mean(equals.type(torch.FloatTensor))

test_loss.append(test_runningloss/len(test_iter))
print("Testing Error: {:.3f}.. ".format(test_runningloss / len(test_iter)),
      "Classification Accuracy: {:.3f}".format(100 * test_acc / len(test_iter)))
time_end = time.time()
testTime = time_end - time_start
print('Testing time is ', testTime, 's')
#confusion.plot()
confusion.summary()

def plot_tsne(quanzhong):
    #area = (50 * np.random.rand(56)) ** 2
    #sc = plt.scatter(quanzhong.cpu()[0], quanzhong.cpu()[1], s=area, c=colors, marker='o',cmap='Blues_r')
    #colors = np.random.rand(56)
    #sc = plt.scatter(quanzhong.cpu()[0], quanzhong.cpu()[1], s=30, c=colors, alpha=0.5)
    #area = (50 * np.random.rand(56)) ** 2
    x = range(1,57)
    plt.xlim(0,60)
    plt.ylim(0,1)
    plt.xlabel('Number of features')
    plt.ylabel('Feature weight')
    plt.grid()
    sc = plt.scatter(x, quanzhong.cpu()[0,:], s=50, alpha=0.5, marker='o')
    #plt.colorbar(sc)
    #plt.show()


if __name__ == '__main__':
    param = count_param(net)
    param = param *32
    print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    #plot_tsne(quanzhong)


