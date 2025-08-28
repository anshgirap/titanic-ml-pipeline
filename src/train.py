import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    # Dataset path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "titanic_clean.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_csv(data_path)

    # Candidate features
    candidate_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    # Keep only existing ones
    available_features = [f for f in candidate_features if f in df.columns]

    if "Survived" not in df.columns:
        raise KeyError("'Survived' column not found in dataset")

    print("Using features:", available_features)

    # Encode categorical + prepare
    df = pd.get_dummies(df[available_features + ["Survived"]], drop_first=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    return model, X.columns.tolist()