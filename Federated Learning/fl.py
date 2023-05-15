import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def ODIN():
    df = pd.read_csv("Odin/fused_dataset_plain.csv").sample(50000).fillna(0)
    leavout = ['trip_id', 'sted_o','sted_d', 'av_car', 'av_carp', 'av_transit', 'av_cycle', 'av_walk', 'c_carp', 'c_cycle', 'c_walk', 'c_car', 'pc_car_tue', 'pc_car_npr_mean', 'pc_car_npr_max', 'aankpc', 'vertpc',
           'pc_car_tue_nan', 'pc_car_npr_mean_nan', 'pc_car_npr_max_nan',
           'creation_datetime']
    df=df.drop(leavout, axis = 1)
    dep_var = 'choice'
    choicedict = {1: 'Auto bestuurder',
    2: 'Auto passagier', 3: 'OV bus, tram, metro of trein',
    4: 'Fiets', 5: 'Lopen'}
    df=df.replace({"choice": choicedict})

    cat_names = ['ovstkaart','weekday','d_hhchildren', 'd_high_educ', 'gender', 'age', 'pur_home', 'pur_work', 'pur_busn', 'pur_other','driving_license', 'car_ownership',
           'main_car_user', 'hh_highinc10', 'hh_lowinc10', 'hh_highinc20', 'av_car', 'av_carp', 'av_transit', 'av_cycle', 'av_walk']
    cat_names = [x for x in cat_names if x not in leavout]
    cont_names = [x for x in df.columns if x not in cat_names]
    cont_names.remove(dep_var)


    df[cont_names] = MinMaxScaler().fit_transform(df[cont_names])
    df = pd.get_dummies(df, columns=cat_names)
    le = LabelEncoder()
    df['choice'] = le.fit_transform(df['choice'])

    #Balance classes
    g = df.groupby('choice')
    df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

df = pd.read_pickle('Change Data/movers')

dftrain, dftest = train_test_split(df, test_size=0.33, random_state=42)
dftrain = dftrain.sort_values(by = dep_var)


# Define input data
num_clients = 15
num_features = len(df.columns)-1
num_rounds = 25
batch_size = 100
learning_rate = 0.25
num_classes = 5

# Define the loss function
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

clients = np.array_split(dftrain, num_clients)

def generate_data(df):
    X = df.drop([dep_var], axis = 1).to_numpy()
    y = df[dep_var].to_numpy()
    return X, y

client_data = []
for client_id in clients:
    X,y = generate_data(client_id)
    X = torch.tensor(X).float()
    # y = torch.tensor(y).float()
    y = torch.tensor(y).long()
    client_data.append((X, y))

# Define the model architecture
class NetReg(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NetSmall(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 10)
        self.fc4 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = nn.functional.softmax(x, dim=1)
        return x

# Define the client update function to train the model on client data
def client_update_fn(model, dataset, learning_rate):
    # Create a copy of the model to use on the client
    client_model = copy.deepcopy(model)
    # Define the optimizer
    optimizer = optim.SGD(client_model.parameters(), lr=learning_rate)
    # Train the model on the client's data
    for i in range(batch_size):
        idx = np.random.randint(len(dataset[0]), size=batch_size)
        X_batch, y_batch = dataset[0][idx], dataset[1][idx]
        y_pred = client_model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Return the updated model weights
    return client_model.state_dict()

# Define the server update function to aggregate the client model weights and update the server model
def server_update_fn(model, client_weights):
    # Average the client model weights
    new_weights = {}
    for key in model.state_dict().keys():
        weight_sum = torch.zeros_like(client_weights[0][key])
        for j in range(len(client_weights)):
            weight_sum += client_weights[j][key]
        new_weights[key] = weight_sum / len(client_weights)
    # Update the server model with the new weights
    model.load_state_dict(new_weights)
    # Return the updated server model
    return model

# # Define the evaluation function to evaluate the model on test data
def evaluateRMSE(model, X_test, y_test):
    y_pred = model(torch.tensor(X_test).float()).detach().numpy().flatten()
    rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
    return rmse

def evaluateACC(model, X_test, y_test):
    y_pred = model(torch.tensor(X_test).float()).argmax(dim=1)
    accuracy = (y_pred == torch.tensor(y_test)).float().mean()
    return accuracy

# Split the data into training and testing sets
X_test, y_test = generate_data(dftest)
X_train = np.concatenate([client_data[i][0].numpy() for i in range(num_clients)])
y_train = np.concatenate([client_data[i][1].numpy() for i in range(num_clients)])

# Train the federated model
model = Net()
for round_num in range(num_rounds):
    client_weights = []
    # for client_id in range(num_clients):
    for client_id in range(len(clients)):
        # Get the client's data
        dataset = client_data[client_id]
        # Update the client model with the client's data
        client_weights.append(client_update_fn(model, dataset, learning_rate))
    # Update the server model with the client model weights
    model = server_update_fn(model, client_weights)
    # Evaluate the model on the test data
    rmse = evaluateACC(model, X_test, y_test)
    print(f"Round {round_num}: Accuracy = {rmse}")

print("Training complete!")

def update_model(model, dataset, learning_rate):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(batch_size):
        idx = np.random.randint(len(dataset[0]), size=batch_size)
        X_batch, y_batch = dataset[0][idx], dataset[1][idx]
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

# Train the non-federated model
unfed_model = Net()
for round_num in range(num_rounds):
    dataset = client_data[0]
    model2 = update_model(unfed_model, dataset, learning_rate)
    rmse = evaluateACC(unfed_model, X_test, y_test)
    print(f"Round {round_num}: Acc = {rmse}")

view = pd.DataFrame({'pred': model(torch.tensor(X_test).float()).argmax(dim=1), 'true': y_test})
view['pred'] = le.inverse_transform(view.pred)
view['true'] = le.inverse_transform(view.true)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(view.true, view.pred, labels = view.pred.unique())
view.pred.unique()