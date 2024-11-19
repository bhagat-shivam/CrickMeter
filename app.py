import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', message='Thread "MainThread": missing  ScriptRunContext')

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Team and city lists
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
         'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
          'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
          'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
          'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
          'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 
          'Cardiff', 'Christchurch', 'Trinidad']

# Player data simulation
players = ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6']
bowlers = ['Bowler1', 'Bowler2', 'Bowler3', 'Bowler4']

# App title
st.title('CrickMeter')

# Prediction Interface
st.subheader("Cricket Score Prediction")
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done (works for over>5)')
with col5:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': city,
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })
    result = pipe.predict(input_df)
    st.header("Predicted Score: " + str(int(result[0])))

# Scenario-Based Match Simulation
st.subheader("Scenario-Based Match Simulation")
st.write("Simulate potential outcomes by adjusting match variables.")

batting_order = st.multiselect("Reorder Batting Lineup (Drag and Drop)", players, default=players)
current_bowler = st.selectbox("Select Current Bowler", bowlers)
fielding_setup = st.radio(
    "Fielding Setup",
    options=["Aggressive", "Defensive", "Balanced"],
    index=2  # Default to Balanced
)

if st.button("Simulate Match"):
    st.write(f"Simulated Batting Order: {', '.join(batting_order)}")
    st.write(f"Current Bowler: {current_bowler}")
    st.write(f"Fielding Strategy: {fielding_setup}")
    st.info("Simulation complete. Integration with the prediction engine for dynamic updates is in progress.")

# Player Form Analysis
st.subheader("Player Form Analysis")
st.write("Analyze player performance trends.")

# Simulated Player Form Data
player_runs = np.random.randint(20, 70, size=5)
player_wickets = np.random.randint(0, 4, size=5)
match_dates = pd.date_range(end=pd.Timestamp.today(), periods=5).strftime("%d-%b-%Y")

selected_player = st.selectbox("Select Player", players)
if selected_player:
    st.write(f"Performance of {selected_player} in Last 5 Matches:")
    fig, ax1 = plt.subplots()
    ax1.bar(match_dates, player_runs, color="blue", alpha=0.6, label="Runs Scored")
    ax1.set_ylabel("Runs", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(match_dates, player_wickets, color="red", marker="o", label="Wickets Taken")
    ax2.set_ylabel("Wickets", color="red")
    fig.tight_layout()
    st.pyplot(fig)

# ChatGPT-like Assistant
st.subheader("Chat with CrickMeter Assistant")

# Chatbot response function
def chatbot_response(user_input):
    user_input = user_input.lower()
    if 'score' in user_input:
        return "To predict scores, provide inputs like batting team, bowling team, and other match details."
    elif 'weather' in user_input:
        return "Currently, weather prediction is not integrated. Consider using WeatherAPI for such data."
    elif 'match' in user_input:
        return "You can simulate match outcomes by modifying inputs like batting order, runs, and overs."
    elif 'team' in user_input:
        return "The teams supported are: " + ', '.join(teams)
    elif 'player' in user_input:
        return "You can view a player's performance trends in the Player Form Analysis section."
    else:
        return "I'm here to assist with cricket score predictions and match insights. Please try asking something related!"

user_query = st.text_input("Ask me anything about cricket or the app:")
if user_query:
    response = chatbot_response(user_query)
    st.write(f"Assistant: {response}")
