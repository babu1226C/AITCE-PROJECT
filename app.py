# Import all the necessary libraries
import pandas as pd
import joblib
import streamlit as st

# Load the model and model column structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Streamlit User Interface
st.title("💧 Water Pollutants Predictor")
st.write("Predict the water pollutants based on **Year** and **Station ID**")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# Prediction button
if st.button('Predict'):
    if not station_id.strip():
        st.warning('⚠️ Please enter a valid Station ID.')
    else:
        try:
            # Prepare the input DataFrame
            input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})

            # One-hot encode the Station ID
            input_encoded = pd.get_dummies(input_df, columns=['id'])

            # Add any missing columns with zero (align with training columns)
            for col in model_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            # Remove any extra columns not in model_cols
            input_encoded = input_encoded[[col for col in model_cols if col in input_encoded.columns]]

            # Add missing columns again (in case some were missing after reindex)
            for col in model_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            # Ensure column order matches model_cols
            input_encoded = input_encoded[model_cols]

            # Make prediction
            predicted_pollutants = model.predict(input_encoded)[0]

            # Display predicted pollutant levels
            pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
            st.subheader(f"🔍 Predicted pollutant levels for Station ID '{station_id}' in {year_input}:")
            for p, val in zip(pollutants, predicted_pollutants):
                st.write(f"**{p}**: {val:.2f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
