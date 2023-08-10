import streamlit as st

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

st.title("Crickentron 🏏")
st.write("Predict the Winner! Enter the match year, team names and names of all 11 players for each team and discover the victorious team!")

with st.sidebar:
    st.title('Crickentron')
    st.subheader("by Krishna Advait Sripada, Katarina Poffley, Snehal Sobti, Lianne Choong")
    st.caption("© 2023 by Crickentron")

st.subheader("Enter Match Year:")

year_number = st.text_input("Match year")
if year_number.strip() == '':
    year_number = 2023
else:
    year_number = int(year_number)

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

st.subheader("Enter Team 0 name:")

team0_name = st.text_input("Team 0")

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

st.subheader("Enter Team 1 name:")

team1_name = st.text_input("Team 1")

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

team0 = [team0_player1, team1_player2, team0_player3, team0_player4, team0_player5, team0_player6, team0_player7, team0_player8, team0_player9, team0_player10, team0_player11]
team1 = [team1_player1, team1_player2, team1_player3, team1_player4, team1_player5, team1_player6, team1_player7, team1_player8, team1_player9, team1_player10, team1_player11]

def get_features_dataset(team0_players, team1_players, year_number):
    career_features_df = pd.read_excel('career_bat_bowl_combined.xlsx')
    year_wise_df = pd.read_excel('year-wise_bat_bowl_combined.xlsx')

    numFeatures = 9

    # Create a dictionary for player names to their combined bat_bowl career features
    name_to_features_dict = {}

    for i in range(len(career_features_df)):
        record = career_features_df.iloc[i,:]

        name = record['Player']
        name = name.split('(')[0].strip()
        name_to_features_dict[name] = np.array(record[1:numFeatures + 1])

    # Create a dictionary of dictionary where
    # key is Year and
    # value is another dictionary with player names as keys and arrays of 9 recent performance features of players as the values
    year_to_nameFeatures_dict = {}
    name_to_recentFeatures_dict = {}

    startYear = 1971
    endYear = 2023
    prevYear = 1971

    i = 0
    record = year_wise_df.iloc[0,:]
    currYear = record['Year']

    while i < len(year_wise_df):

        while currYear == prevYear:

            # Player names are like SR Tendulkar (IND)
            # We do not want stuff inside parantheses
            # Because there are cases where there is different data inside parantheses for same player on
            # different webpages
            name = record['Player']
            name = name.split('(')[0].strip()

            # column 0 is Player Name and column 1 is Year
            name_to_recentFeatures_dict[name] = np.array(record[2:numFeatures + 2])

            i = i + 1
            if i == len(year_wise_df):
                break
            record = year_wise_df.iloc[i,:]
            currYear = record['Year']

        year_to_nameFeatures_dict[prevYear] = name_to_recentFeatures_dict
        name_to_recentFeatures_dict = {}
        prevYear = currYear

    data = []
    data_row = []

    # Get the 18 features for each player for team0
    for player in team0_players:
        if type(player) != str:
            data_row = np.append(data_row, np.zeros(numFeatures * 2))
        elif year_number <= startYear or year_number >= endYear + 2:
            PName = player.split('(')[0].strip()
            if PName in name_to_features_dict:
                data_row = np.append(data_row, name_to_features_dict[PName])
                data_row = np.append(data_row, np.zeros(numFeatures))
            else:
                data_row = np.append(data_row, np.zeros(numFeatures * 2))
        else:
            PName = player.split('(')[0].strip()
            recent_form_row = np.zeros(numFeatures)

            # Get career features
            if PName in name_to_features_dict:
                data_row = np.append(data_row, name_to_features_dict[PName])
            else:
                data_row = np.append(data_row, np.zeros(numFeatures))

            # Get one year before data
            if PName in year_to_nameFeatures_dict[year_number - 1]:
                recent_form_row = recent_form_row + (year_to_nameFeatures_dict[year_number - 1])[PName] # Element-wise addition

            # Get two years before data
            if year_number > startYear + 1 and PName in year_to_nameFeatures_dict[year_number - 2]:
                recent_form_row = recent_form_row + (year_to_nameFeatures_dict[year_number - 2])[PName] # Element-wise addition

            data_row = np.append(data_row, recent_form_row)


    # Get the 18 features for each player for team0
    for player in team1_players:
        if type(player) != str:
            data_row = np.append(data_row, np.zeros(numFeatures * 2))
        elif year_number <= startYear or year_number >= endYear + 2:
            PName = player.split('(')[0].strip()
            if PName in name_to_features_dict:
                data_row = np.append(data_row, name_to_features_dict[PName])
                data_row = np.append(data_row, np.zeros(numFeatures))
            else:
                data_row = np.append(data_row, np.zeros(numFeatures * 2))
        else:
            PName = player.split('(')[0].strip()
            recent_form_row = np.zeros(numFeatures)

            # Get career features
            if PName in name_to_features_dict:
                data_row = np.append(data_row, name_to_features_dict[PName])
            else:
                data_row = np.append(data_row, np.zeros(numFeatures))

            # Get one year before data
            if PName in year_to_nameFeatures_dict[year_number - 1]:
                recent_form_row = recent_form_row + (year_to_nameFeatures_dict[year_number - 1])[PName] # Element-wise addition

            # Get two years before data
            if year_number > startYear + 1 and PName in year_to_nameFeatures_dict[year_number - 2]:
                recent_form_row = recent_form_row + (year_to_nameFeatures_dict[year_number - 2])[PName] # Element-wise addition

            data_row = np.append(data_row, recent_form_row)

    data.append(data_row)

    return torch.Tensor(data)

class Final_Model(nn.Module):
    def __init__(self):
        super(Final_Model, self).__init__()
        self.name = 'Primary'
        self.layer1 = nn.Linear(22 * 18, 180)
        self.layer2 = nn.Linear(180, 90)
        self.layer3 = nn.Linear(90, 40)
        self.layer4 = nn.Linear(40, 3) # Output's shape is 3 because we have 3 classes (0, 1, 2)
    def forward(self, input):
        flattened = input.view(-1, 22 * 18)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        output = self.layer4(activation3)
        return output

def getWinner(team0, team1, match_year):
    model = torch.load('ourModel.pt')
    model.eval()

    test_data = get_features_dataset(team0, team1, match_year)

    output = F.softmax(model(test_data), dim = 1)

    #select index with maximum prediction score
    prediction = output.max(1, keepdim=True)[1]

    return prediction.item()

# Initializing the variable result
result = -1

predict_button = st.button("Predict")
if predict_button:
    with st.spinner("Analyzing..."):
        result = getWinner(team0, team1, year_number)

if (result == 0):
    header_text = f":orange[**{team0_name} wins**]"
elif (result == 1):
    header_text = f":orange[**{team1_name} wins**]"
else:
    header_text = f":orange[**No Result**]"

st.header(header_text)
