import streamlit as st

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

st.title("Crickentron 🏏")
st.write("Predict the Winner! Enter the names of all 11 players for each team and discover the victorious team!")

with st.sidebar:
    st.title('Crickentron')
    st.subheader("by Krishna Advait Sripada, Katarina Poffley, Snehal Sobti, Lianne Choong")
    st.caption("© 2023 by Crickentron")

# team0_player1 = 'SR Tendulkar'
# team0_player2 = 'ST Jayasuriya'
# team0_player3 = 'R Dravid'
# team0_player4 = 'RT Ponting'
# team0_player5 = 'KC Sangakkara'
# team0_player6 = 'MS Dhoni'
# team0_player7 = 'JH Kallis'
# team0_player8 = 'M Muralidaran'
# team0_player9 = 'B Lee'
# team0_player10 = 'A Kumble'
# team0_player11 = 'Wasim Akram'

# team1_player1 = 'PP Shaw'
# team1_player2 = 'RD Gaikwad'
# team1_player3 = 'V Shankar'
# team1_player4 = 'KM Jadhav'
# team1_player5 = 'DJ Hooda'
# team1_player6 = 'NV Ojha'
# team1_player7 = 'MA Agarwal'
# team1_player8 = 'Ravi Bishnoi'
# team1_player9 = 'DL Chahar'
# team1_player10 = 'RD Chahar'
# team1_player11 = 'JD Unadkat'

st.subheader("Enter Team 0 players:")

team0_player1 = st.text_input("Player 1")
team0_player2 = st.text_input("Player 2")
team0_player3 = st.text_input("Player 3")
team0_player4 = st.text_input("Player 4")
team0_player5 = st.text_input("Player 5")
team0_player6 = st.text_input("Player 6")
team0_player7 = st.text_input("Player 7")
team0_player8 = st.text_input("Player 8")
team0_player9 = st.text_input("Player 9")
team0_player10 = st.text_input("Player 10")
team0_player11 = st.text_input("Player 11")

st.subheader("Enter Team 1 players:")

team1_player1 = st.text_input("Opponent Player 1")
team1_player2 = st.text_input("Opponent Player 2")
team1_player3 = st.text_input("Opponent Player 3")
team1_player4 = st.text_input("Opponent Player 4")
team1_player5 = st.text_input("Opponent Player 5")
team1_player6 = st.text_input("Opponent Player 6")
team1_player7 = st.text_input("Opponent Player 7")
team1_player8 = st.text_input("Opponent Player 8")
team1_player9 = st.text_input("Opponent Player 9")
team1_player10 = st.text_input("Opponent Player 10")
team1_player11 = st.text_input("Opponent Player 11")

torch.manual_seed(42) # set the random seed

team0 = np.array([team0_player1, team1_player2, team0_player3, team0_player4, team0_player5, team0_player6, team0_player7, team0_player8, team0_player9, team0_player10, team0_player11])

team1 = np.array([team1_player1, team1_player2, team1_player3, team1_player4, team1_player5, team1_player6, team1_player7, team1_player8, team1_player9, team1_player10, team1_player11])

def get_features_dataset(team0, team1):
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
    for player in team0:
        if (player in name_to_features_dict):
            data_row = np.append(data_row, name_to_features_dict[player.strip()])
        else:
            data_row = np.append(data_row, np.zeros(numFeatures))

    for player in team1:
        if (player in name_to_features_dict):
            data_row = np.append(data_row, name_to_features_dict[player.strip()])
        else:
            data_row = np.append(data_row, np.zeros(numFeatures))

    data.append(data_row)

    return torch.Tensor(data)

def getWinner(team0, team1):
    model = torch.load('ourModel.pt')
    model.eval()

    test_data = get_features_dataset(team0, team1)

    output = F.softmax(model(test_data), dim = 1)

    #select index with maximum prediction score
    prediction = output.max(1, keepdim=True)[1]

    return prediction.item()

result = getWinner(team0, team1)

if (result == 0):
    st.header(':orange[**Team 0 wins**]')
elif (result == 1):
    st.header(':orange[**Team 1 wins**]')
else:
    st.header(':orange[**No Result**]')
