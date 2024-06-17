import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title = 'History Page',
    page_icon = '🏠',
    layout='wide'
)


# Function to save predictions to a CSV file
def save_to_csv(make_predictions):
    # Check if the CSV file exists, if not create it with headers
    try:
        history_df = pd.read_csv('history.csv') #read history.csv if it exists
    except FileNotFoundError:
        history_df = pd.DataFrame(columns=['PredictedOutcome'])
 
    # Create a new DataFrame with the new prediction
    prediction_df = pd.DataFrame({'PredictedOutcome': [make_predictions]})
 
    # Concatenate the existing DataFrame with the new DataFrame
    history_df = pd.concat([history_df, prediction_df], axis = 0, ignore_index=True )
 
    # Save the updated DataFrame to CSV
    history_df.to_csv('history.csv', index=False)
 
    st.success("Prediction saved successfully.")
 
def show_history():
    st.title('Prediction History')
   
    # Check if history dataframe exists in session state
    if 'history_df' in st.session_state:
        history_df = st.session_state.history_df.copy()  # Make a copy to prevent modifying the original dataframe
        history_df['PredictedOutcome'] = history_df['PredictedOutcome'].map({0: 'Not Churned', 1: 'Churned'})
        st.dataframe(history_df)
    else:
        st.write("No predictions have been made yet.")
 
 
if __name__ == "__main__":
    make_predictions = 1  
    save_to_csv(make_predictions)
    show_history()
 