import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page title and layout
st.set_page_config(page_title="Raisin Variety Classifier", layout="centered")

st.title("🍇 Raisin Variety Classifier App")
st.write("Adjust the features below to predict whether the raisin variety is **Kecimen** or **Besni**.")

# 1. Load the saved model securely
@st.cache_resource
def load_model():
    with open("Randomforest_sample.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'Randomforest_sample.pkl' not found. Please run train.py first to generate the model file.")
    st.stop()

# 2. List out the exact features your dataset expects
# These match the columns of your X_train dataset
feature_names = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter']

st.subheader("Input Feature Values")

# 3. Create dynamic UI sliders or number inputs for each feature
input_data = {}

# We create two columns in the UI for a clean look
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    # Split the inputs evenly between column 1 and column 2
    current_col = col1 if i % 2 == 0 else col2
    
    # We use number input boxes here so users can type or click step buttons
    # You can change st.number_input to st.slider if you prefer dragging sliders
    input_data[feature] = current_col.number_input(
        label=f"Enter {feature}",
        value=0.0,
        step=1.0
    )

# 4. Predict Button Logic
st.write("---")
if st.button("🔮 Predict Raisin Variety", type="primary"):
    # Convert input dictionary directly into a 1-row DataFrame matching the model's expectations
    input_df = pd.DataFrame([input_data])
    
    # Run the prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # 5. Display the final results cleanly to the user
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("🎉 The predicted variety is **Kecimen**!")
        st.info(f"Confidence Level: {prediction_proba[1] * 100:.1f}%")
    else:
        st.warning("🍂 The predicted variety is **Besni**!")
        st.info(f"Confidence Level: {prediction_proba[0] * 100:.1f}%")
