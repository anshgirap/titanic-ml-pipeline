import os
import json
import joblib
import pandas as pd

def make_dataframe(data, features):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "titanic_model.pkl")
    json_path = os.path.join(base_dir, "data", "sample.json")

    # Load model
    model, features = joblib.load(model_path)

    # Load input
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Please create it.")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Predict
    X = make_dataframe(data, features)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[0][1]

    print(f"Prediction: {y_pred[0]} | Probability of survival: {y_proba:.4f}")

if __name__ == "__main__":
    main()