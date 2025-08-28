import os
import joblib
from train import train_model

if __name__ == "__main__":
    model, features = train_model()

    # Save model + features
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump((model, features), os.path.join(model_dir, "titanic_model.pkl"))
    print("Model saved as titanic_model.pkl")