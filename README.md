# Iris-Classifier-App
## Project Overview

This project implements a machine learning pipeline to classify Iris flower species using a Support Vector Machine (SVM) classifier. The dataset used is the classic Iris dataset, which contains measurements for 3 different species of Iris flowers.

In addition to training the model, the project includes an interactive web application built with **Streamlit** that allows users to input flower measurements and predicts the species in real-time.
## Dataset

- The data is loaded from a CSV file named `Iris.csv`.
- It contains the following columns:
  - `SepalLengthCm`
  - `SepalWidthCm`
  - `PetalLengthCm`
  - `PetalWidthCm`
  - `Species`
    ## Code Explanation

### 1. Data Loading and Label Encoding

The Iris dataset is loaded using pandas. The species names are encoded into numeric labels using `LabelEncoder` to prepare them for the classifier.


`import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Iris.csv')
encoder = LabelEncoder()
df['species_encode'] = encoder.fit_transform(df['Species'])`
### 2. Feature Selection and Scaling

The features selected for training are the 4 measurement columns (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`). 

These features are scaled using `StandardScaler` to normalize the data and improve SVM performance.
from sklearn.preprocessing import StandardScaler

`X = df.iloc[:, 1:5].values # select columns 1 to 4 for features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)`

### 3. Train-Test Split

Split the prepared dataset into training and testing sets for model evaluation.


`from sklearn.model_selection import train_test_split
y = df['species_encode'].values # target labels
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42
)`

### 4. Model Training

A Support Vector Machine with a linear kernel is used as the classifier. The model is trained on the training set.


`from sklearn.svm import SVC
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)`

### 5. Model Evaluation

Evaluate model accuracy on the test set.
`from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")`

### 6. Prediction Function

You can make predictions using the model and scaler by providing flower measurements as input.


`def predict_species(sepal_length, sepal_width, petal_length, petal_width):
features = [[sepal_length, sepal_width, petal_length, petal_width]]
features_scaled = scaler.transform(features)
pred_encoded = model.predict(features_scaled)
species = encoder.inverse_transform([pred_encoded])
return species`

---

## Streamlit Web App

The project includes a Streamlit app (`app.py`) that provides an easy interface to input measurements and get classification results.

- Users adjust sliders for flower dimensions.
- Press a “Predict” button to see the predicted Iris species instantly.

Example slider input and prediction snippet in Streamlit:
import streamlit as st

`sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
if st.button("Predict"):
species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
st.success(f"Predicted Iris species: {species}")`
## Running the Project

### Prerequisites

Make sure you have Python 3.x installed. Install required packages using:
pip install pandas scikit-learn streamlit
### Running the Streamlit App

Run the app with this command:
`streamlit run app.py`

The app should open in your default web browser.

---

## Summary

- The project demonstrates classic machine learning techniques on the Iris dataset.
- Uses SVM for classification with scalable and readable code.
- Interactive Streamlit UI allows dynamic user interaction without programming.

## Acknowledgements

- [scikit-learn](https://scikit-learn.org) for the implementation of SVM and data preprocessing tools.
- [Streamlit](https://streamlit.io) for providing an easy, fast web app framework.
- The UCI Machine Learning Repository for the Iris dataset.










