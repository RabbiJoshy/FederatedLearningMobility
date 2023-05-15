import seaborn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def ODIN(n = 100000):
    df = pd.read_csv("Odin/fused_dataset_plain.csv").sample(n).fillna(0)
    leavout = ['trip_id', 'sted_o', 'sted_d', 'av_car', 'av_carp', 'av_transit', 'av_cycle', 'av_walk', 'c_carp',
               'c_cycle', 'c_walk', 'c_car', 'pc_car_tue', 'pc_car_npr_mean', 'pc_car_npr_max', 'aankpc', 'vertpc',
               'pc_car_tue_nan', 'pc_car_npr_mean_nan', 'pc_car_npr_max_nan',
               'creation_datetime']
    df = df.drop(leavout, axis=1)
    dep_var = 'choice'
    choicedict = {1: 'Auto bestuurder',
                  2: 'Auto passagier', 3: 'OV bus, tram, metro of trein',
                  4: 'Fiets', 5: 'Lopen'}
    df = df.replace({"choice": choicedict})

    cat_names = ['ovstkaart', 'weekday', 'd_hhchildren', 'd_high_educ', 'gender', 'age', 'pur_home', 'pur_work',
                 'pur_busn', 'pur_other', 'driving_license', 'car_ownership',
                 'main_car_user', 'hh_highinc10', 'hh_lowinc10', 'hh_highinc20', 'av_car', 'av_carp', 'av_transit',
                 'av_cycle', 'av_walk']
    cat_names = [x for x in cat_names if x not in leavout]
    cont_names = [x for x in df.columns if x not in cat_names]
    cont_names.remove(dep_var)

    df[cont_names] = MinMaxScaler().fit_transform(df[cont_names])
    df = pd.get_dummies(df, columns=cat_names)

    le = LabelEncoder()
    df['choice'] = le.fit_transform(df['choice'])

    return df, le
dep_var = 'choice'
df, le = ODIN()

# Balance classes
g = df.groupby('choice')
df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
df = df.reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop = True)



dftrain, dftest = train_test_split(df, test_size=0.33, random_state=42)
# dftrain = dftrain.sort_values(by=dep_var)

# Define input data
num_clients = 2
num_features = len(df.columns) - 1
num_rounds = 80
batch_size = 100
learning_rate = 0.1
num_classes = 5

criterion = nn.CrossEntropyLoss()
clients = np.array_split(dftrain, num_clients)


def generate_data(df):
    X = df.drop([dep_var], axis=1).to_numpy()
    y = df[dep_var].to_numpy()
    return X, y


client_data = []
for client_id in clients:
    X, y = generate_data(client_id)
    X = torch.tensor(X).float()
    # y = torch.tensor(y).float()
    y = torch.tensor(y).long()
    client_data.append((X, y))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 20)
        self.fc2 = nn.Linear(20, 8)
        self.fc3 = nn.Linear(8, 3)
        self.fc4 = nn.Linear(3, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        h3 = x  # store the activations of the third hidden layer
        x = self.fc4(x)
        x = nn.functional.softmax(x, dim=1)
        return x, h3


def evaluateACC(model, X_test, y_test):
    y_pred = model(torch.tensor(X_test).float())[0].argmax(dim=1)
    accuracy = (y_pred == torch.tensor(y_test)).float().mean()
    return accuracy

# Split the data into training and testing sets
X_test, y_test = generate_data(dftest)
X_train = np.concatenate([client_data[i][0].numpy() for i in range(num_clients)])
y_train = np.concatenate([client_data[i][1].numpy() for i in range(num_clients)])

def update_model(model, dataset, learning_rate):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(batch_size):
        idx = np.random.randint(len(dataset[0]), size=batch_size)
        X_batch, y_batch = dataset[0][idx], dataset[1][idx]
        y_pred, h = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, h


# Train the non-federated model
unfed_model = Net()
for round_num in range(num_rounds):
    dataset = client_data[0]
    unfed_model, h = update_model(unfed_model, dataset, learning_rate)
    rmse = evaluateACC(unfed_model, X_test, y_test)
    print(f"Round {round_num}: Acc = {rmse}")

view = pd.DataFrame({'pred': unfed_model(torch.tensor(X_test).float())[0].argmax(dim=1), 'true': y_test})
view['pred'] = le.inverse_transform(view.pred)
view['true'] = le.inverse_transform(view.true)



idx = np.random.randint(len(dataset[0]), size=1000)
X_batch, y_batch = dataset[0][idx], dataset[1][idx]
y_pred, h = unfed_model(X_batch)
y_predi = y_pred.float().argmax(dim=1)


embeddings = pd.DataFrame({'h0':h[:, 0].detach().numpy(), 'h1':h[:, 1].detach().numpy(), 'h2':h[:, 2].detach().numpy(), 'choice': y_batch.detach().numpy(), 'pred' : y_pred.float().argmax(dim=1)})

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data = embeddings), x = 'h1', y = 'h0', hue= 'choice')
sns.scatterplot(data = embeddings, x = 'h1', y = 'h0', hue= 'pred')
plt.show()

# torch.save(unfed_model.state_dict(), 'embed')