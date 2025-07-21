import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class IrisSVMClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def load_and_prepare_data(self, file_path='Iris.csv'):
        df = pd.read_csv(file_path)
        df['species_encode'] = self.encoder.fit_transform(df['Species'])
        X = df.iloc[:, 1:5].values  # Feature columns (sepal and petal measurements)
        y = df['species_encode'].values
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {acc:.2f}")

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        features_scaled = self.scaler.transform(features)
        pred_encoded = self.model.predict(features_scaled)[0]
        pred_species = self.encoder.inverse_transform([pred_encoded])[0]
        return pred_species

if __name__ == "__main__":
    clf = IrisSVMClassifier()
    clf.train()
    # Example prediction
    prediction = clf.predict(5.1, 3.5, 1.4, 0.2)
    print(f"Predicted species: {prediction}")
