import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/titanic_clean.csv")
    return df

def preprocess_data(df):
    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test