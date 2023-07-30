import streamlit as st

st.title("Crickentron 🏏")
st.write("Predict the Winner! Enter the names of all 11 players for each team and discover the victorious team!")

with st.sidebar:
    st.title('Crickentron')
    st.subheader("by Krishna Advait Sripada, Katarina Poffley, Snehal Sobti, Lianne Choong")
    st.caption("© 2023 by Crickentron")

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

