import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

# Load model
try:
    with open("iris.pkl", "rb") as f:
        model = pickle.load(f)
except:
    st.error("Model file iris.pkl not found")
    st.stop()

# Load dataset info
iris = load_iris()
target_names = iris.target_names

# Title
st.title("🌸 Iris Species Prediction App")
st.markdown("Predict the **Iris flower species** using sepal and petal measurements.")

st.divider()

# Input section
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=20.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=20.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=20.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=20.0,
        value=0.2,
        step=0.1
    )

st.divider()

# Prediction button
if st.button("Predict Species 🌼"):

    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    predicted_class = int(prediction[0])
    species = target_names[predicted_class]

    st.success(f"Predicted Species: **{species.upper()}**")

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame(
        prediction_proba,
        columns=target_names
    )

    st.bar_chart(prob_df.T)

st.divider()

st.markdown(
"""
### About
This app uses a **Decision Tree model** trained on the Iris dataset to classify flowers into:

- Setosa
- Versicolor
- Virginica
"""
)
