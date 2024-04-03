import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.tree import DecisionTreeClassifier
import time
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(GRU, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # GRU Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        # Dropout layer
        # self.dropout = nn.Dropout(drop_prob)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, h0)

        # print("Shape after GRU layer:", end = ' ')
        # print(out.shape)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        # out = self.dropout(out)
        out = self.fc(out)

        # print("Shape after Linear layer:", end = ' ')
        # print(out.shape)

        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


def get_train_test(df):
    X = df.drop(['outcome'], axis=1)
    Y = df[['outcome']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


# Test validation
def test_validation(test_loader, accuracy_list, DEBUG=False):
    # Calculate Accuracy
    correct = 0
    total = 0

    # Iterate through test dataset
    for data, labels in test_loader:
        data = data.reshape(-1, seq_length, input_dim)

        # Forward propagation
        outputs = model(data)

        # flatten labels
        labels = labels.reshape(-1)

        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]  # NOTE: May need to change this to torch.max(outputs.data, 1)[0]

        # Total number of labels
        total += labels.size(0)

        correct += (predicted == labels).sum()

    accuracy = 100 * correct / float(total)
    # print("Total:", total)
    # print("Correct:", correct)

    # store loss and iteration
    accuracy_list.append(accuracy)
    if DEBUG:
        # Print Loss
        print('Loss: {}  Accuracy: {} %'.format(loss.item(), accuracy))

    return accuracy


df = pd.read_csv("kddcup.data_10_percent_corrected")
# set columns
df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

cleanup_nums = {"protocol_type":     {"tcp": 1, "icmp": 2, "udp": 3},
                "service": {"vmnet": 1, "smtp": 2, "ntp_u":3, "shell":4, "kshell":5, "aol":6, "imap4":7, "urh_i":8, "netbios_ssn":9,
                           "tftp_u":10, "mtp":11, "uucp":12, "nnsp":13, "echo":14, "tim_i":15, "ssh":16, "iso_tsap":17, "time":18,
                           "netbios_ns":19,"systat":20, "hostnames":21, "login":22, "efs":23, "supdup":24, "http_8001":25, "courier":26,
                           "ctf":27,"finger":28,"nntp":29,"ftp_data":30,"red_i":31,"ldap":32,"http":33,"ftp":34,"pm_dump":35,"exec":36,
                           "klogin":37,"auth":38,"netbios_dgm":39,"other":40,"link":41,"X11":42,"discard":43,"private":44,"remote_job":45,
                           "IRC":46,"daytime":47,"pop_3":48,"pop_2":49,"gopher":50,"sunrpc":51,"name":52,"rje":53,"domain":54,"uucp_path":55,
                           "http_2784":56,"Z39_50":57,"domain_u":58,"csnet_ns":59,"whois":60,"eco_i":61,"bgp":62,"sql_net":63,"printer":64,
                           "telnet":65,"ecr_i":66,"urp_i":67,"netstat":68,"http_443":69,"harvest":70},
                "outcome":{"back.":1, "land.":1, "neptune.":1, "pod.":1, "smurf.":1, "teardrop.":1, "satan.":2, "ipsweep.":2, "nmap.":2, "portsweep.":2, "normal.":0, "guess_passwd.":3, "ftp_write.":3, "imap.":3, "phf.":3, "multihop.":3, "warezmaster.":3, "warezclient.":3, "spy.":3, "buffer_overflow.":4, "loadmodule.":4, "perl.":4, "rootkit.":4},
                "flag":{"RSTR":1,"S3":2,"SF":3,"RSTO":4,"SH":5,"OTH":6,"S2":7,"RSTOS0":8,"S1":9,"S0":10,"REJ":11}}


df.replace(cleanup_nums, inplace=True)

# Split train and test
X_train, X_test, Y_train, Y_test = get_train_test(df)

batch_size = 10
seq_length = 1     # The number of previous data points we want to use to predict
input_dim = len(X_train.columns)    # The number of classes we have
print(batch_size, seq_length, input_dim)

# convert to torch
X_train = torch.from_numpy(X_train.values).float()
X_test = torch.from_numpy(X_test.values).float()
Y_train = torch.from_numpy(Y_train.values).float()
Y_test = torch.from_numpy(Y_test.values).float()

# Pytorch train and test sets
train = TensorDataset(X_train, Y_train)
test = TensorDataset(X_test, Y_test)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)

output_size = 23 # 23 classes
hidden_layers = 10
num_layers = 10
drop_prob = 0.3

# model = RNN(input_dim, output_size, 10, 5)
# model = LSTM(input_dim, output_size, 10, 5, drop_prob)
model = GRU(input_dim, output_size, 10, 5)


learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
# criterion = SoftTreeSupLoss(dataset='CIFAR10', criterion=criterion)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# model = SoftNBDT(dataset='CIFAR10', model=model)  # `model` is your original model

DEBUG = True
n_epochs = 25
print_every = 10
count = 0

loss_list = []
# iteration_list = []
accuracy_list = []
outputs = []
true = []

for epoch in range(1, n_epochs + 1):
    start_time = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.reshape(-1, seq_length, input_dim)
        # data = data.squeeze(1)
        # print(data.size())
        # targets = Variable(targets)

        scores = model(data)

        if DEBUG:
            print(data.shape)
            print(scores)
            print(targets)
            DEBUG = False

        # out = scores.numpy().flatten()

        # outputs.append(scores.numpy())
        # true.append(targets.tolist())
        # print(outputs)
        # print(true)

        loss = criterion(scores, targets.view(-1).long())  # Tried targets.softmax(dim = 1)

        # Change gradients
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        count += 1

        # Store loss
        if count % 500 == 0:
            loss_list.append(loss.data)

    test_accuracy = test_validation(test_loader, accuracy_list)
    time_taken = time.time() - start_time

    print('Epoch: {}/{} .............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(loss.item()), end=' - ')
    print("Accuracy: {:.4f}".format(test_accuracy), end=' - ')
    print("Time: {:0.0f}".format(time_taken))