import streamlit as st

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent

st.title("Crickentron 🏏")
st.write("Predict the Winner! Enter the names of all 11 players for each team and discover the victorious team!")

with st.sidebar:
    st.title('Crickentron')
    st.subheader("by Krishna Advait Sripada, Katarina Poffley, Snehal Sobti, Lianne Choong")
    st.caption("© 2023 by Crickentron")

# team1_player1 = 'SR Tendulkar'
# team1_player2 = 'ST Jayasuriya'
# team1_player3 = 'R Dravid'
# team1_player4 = 'RT Ponting'
# team1_player5 = 'KC Sangakkara'
# team1_player6 = 'MS Dhoni'
# team1_player7 = 'JH Kallis'
# team1_player8 = 'M Muralidaran'
# team1_player9 = 'B Lee'
# team1_player10 = 'A Kumble'
# team1_player11 = 'Wasim Akram'

# team2_player1 = 'PP Shaw'
# team2_player2 = 'RD Gaikwad'
# team2_player3 = 'V Shankar'
# team2_player4 = 'KM Jadhav'
# team2_player5 = 'DJ Hooda'
# team2_player6 = 'NV Ojha'
# team2_player7 = 'MA Agarwal'
# team2_player8 = 'Ravi Bishnoi'
# team2_player9 = 'DL Chahar'
# team2_player10 = 'RD Chahar'
# team2_player11 = 'JD Unadkat'

st.subheader("Enter Team 1 players:")

team1_player1 = st.text_input("Player 1")
team1_player2 = st.text_input("Player 2")
team1_player3 = st.text_input("Player 3")
team1_player4 = st.text_input("Player 4")
team1_player5 = st.text_input("Player 5")
team1_player6 = st.text_input("Player 6")
team1_player7 = st.text_input("Player 7")
team1_player8 = st.text_input("Player 8")
team1_player9 = st.text_input("Player 9")
team1_player10 = st.text_input("Player 10")
team1_player11 = st.text_input("Player 11")

st.subheader("Enter Team 2 players:")

team2_player1 = st.text_input("Opponent Player 1")
team2_player2 = st.text_input("Opponent Player 2")
team2_player3 = st.text_input("Opponent Player 3")
team2_player4 = st.text_input("Opponent Player 4")
team2_player5 = st.text_input("Opponent Player 5")
team2_player6 = st.text_input("Opponent Player 6")
team2_player7 = st.text_input("Opponent Player 7")
team2_player8 = st.text_input("Opponent Player 8")
team2_player9 = st.text_input("Opponent Player 9")
team2_player10 = st.text_input("Opponent Player 10")
team2_player11 = st.text_input("Opponent Player 11")

torch.manual_seed(42) # set the random seed

team1 = np.array([team1_player1, team2_player2, team1_player3, team1_player4, team1_player5, team1_player6, team1_player7, team1_player8, team1_player9, team1_player10, team1_player11])

team2 = np.array([team2_player1, team2_player2, team2_player3, team2_player4, team2_player5, team2_player6, team2_player7, team2_player8, team2_player9, team2_player10, team2_player11])

def get_features_dataset(team1, team2):
    career_features_df = pd.read_excel('career_bat_bowl_combined.xlsx')

    numFeatures = 9	

    # Create a dictionary for player names to their combined bat_bowl features
    name_to_features_dict = {}

    for i in range(len(career_features_df)):
        record = career_features_df.iloc[i,:]

        name = record['Player']
        name = name.split('(')[0].strip()
        name_to_features_dict[name] = np.array(record[1:numFeatures + 1])

    data = []
    data_row = []

    # Get the 9 features for each player
    for player in team1:
        data_row = np.append(data_row, name_to_features_dict[player.strip()])

    for player in team2:
        data_row = np.append(data_row, name_to_features_dict[player.strip()])

    data.append(data_row)

    return torch.Tensor(data)

def get_training_data():
    data_df = pd.read_excel('primary_model_data.xlsx')
    labels_df = pd.read_excel('primary_model_labels.xlsx')

    # Normalize each of the desired features of data_df to keep values between 0 and 1
    data_df = (data_df - np.min(data_df, axis = 0)) / (np.max(data_df, axis = 0) - np.min(data_df, axis = 0))

    # Convert data and labels to PyTorch tensors
    data_tensor = torch.Tensor(data_df.values)
    labels_tensor = torch.Tensor(labels_df.values.squeeze(1))

    print('Shape of data_tensor: ', data_tensor.shape)
    print('Shape of labels_tensor: ', labels_tensor.shape)

    # Create a TensorDataset from data and labels
    dataset = TensorDataset(data_tensor, labels_tensor)
    dataset_size = len(dataset)

    # Split data into train, val and test data in ratio 70:15:15
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    return train_set, val_set


class Primary_Model(nn.Module):
    def __init__(self):
        super(Primary_Model, self).__init__()
        self.name = 'Primary'
        self.layer1 = nn.Linear(22 * 9, 60)
        self.layer2 = nn.Linear(60, 20)
        self.layer3 = nn.Linear(20, 3) # Output's shape is 3 because we have 3 classes (0, 1, 2)
    def forward(self, input):
        flattened = input.view(-1, 22 * 9)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output



def get_accuracy(model, data):
    correct = 0
    total = 0
    for inputs, labels in torch.utils.data.DataLoader(data, batch_size=64):
        output = model(inputs)
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]
    return correct / total

def train(model, data, val_set, batch_size=64, num_epochs=1 , learning_rate = 0.01, print_stat = 1):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.7)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for inputs, labels in iter(train_loader):

            out = model(inputs)             # forward pass

            loss = criterion(out, labels.long()) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, data)) # compute training accuracy
            val_acc.append(get_accuracy(model, val_set))  # compute validation accuracy
            n += 1

        # print('Epoch {0}: Training Accuracy: {1} | Validation Accuracy: {2}'.format(epoch, train_acc[n - 1], val_acc[n - 1]))

    if print_stat:
      # plotting
      plt.title("Training Loss")
      plt.plot(iters, losses, label="Train")
      plt.xlabel("Iterations")
      plt.ylabel("Loss")
      plt.show()

      plt.title("Training and Validation Accuracy")
      plt.plot(iters, train_acc, label="Train")
      plt.plot(iters, val_acc, label="Validation")
      plt.xlabel("Iterations")
      plt.ylabel("Accuracy")
      plt.legend(loc='best')
      plt.show()

      print("Final Training Accuracy: {}".format(train_acc[-1]))
      print("Final Validation Accuracy: {}".format(val_acc[-1]))

def winIndex(team1, team2):
    test_data = get_features_dataset(team1, team2)
    train_set, val_set = get_training_data()

    model = Primary_Model()
    train(model, train_set, val_set, learning_rate=0.01, batch_size=64, num_epochs=150, print_stat=0)

    output = F.softmax(model(test_data))

    #select index with maximum prediction score
    prediction = output.max(1, keepdim=True)[1]

result = winIndex(team1, team2)

if (result == 0):
    st.write('Team 1 wins')
elif (result == 1):
    st.write('Team 2 wins')
else:
    st.write('No Result')
