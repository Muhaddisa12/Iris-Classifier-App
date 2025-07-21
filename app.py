import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('Iris.csv')
    encoder = LabelEncoder()
    df['species_encode'] = encoder.fit_transform(df['species'])
    return df, encoder

df, encoder = load_data()

x = df.drop(['species', 'species_encode'], axis=1)
y = df['species_encode']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
accuracy = clf.score(x_test, y_test) * 100

st.title("Iris Species Prediction App")
st.write("Model Accuracy on Test Data: ", accuracy, "%")

st.write("Iris Dataset Pairplot")
fig = sns.pairplot(df, hue='species', diag_kind='kde', palette='husl')
st.pyplot(fig)

st.write("Enter the values of the flower features:")

sepal_length = st.slider(
    'Sepal Length (cm)',
    float(x['sepal_length'].min()),
    float(x['sepal_length'].max()),
    float(x['sepal_length'].mean())
)

sepal_width = st.slider(
    'Sepal Width (cm)',
    float(x['sepal_width'].min()),
    float(x['sepal_width'].max()),
    float(x['sepal_width'].mean())
)

petal_length = st.slider(
    'Petal Length (cm)',
    float(x['petal_length'].min()),
    float(x['petal_length'].max()),
    float(x['petal_length'].mean())
)

petal_width = st.slider(
    'Petal Width (cm)',
    float(x['petal_width'].min()),
    float(x['petal_width'].max()),
    float(x['petal_width'].mean())
)

if st.button('Predict'):
    input_features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=x.columns
    )
    prediction = clf.predict(input_features)[0]
    predict_species = encoder.inverse_transform([prediction])[0]
    st.success(f"The Predicted Iris species is: {predict_species}")
