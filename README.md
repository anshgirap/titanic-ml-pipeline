# Titanic ML Pipeline

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)](https://scikit-learn.org/stable/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-yellowgreen)](https://pandas.pydata.org/)

A clean and modular machine learning pipeline for the **Titanic dataset**, built using **scikit-learn** and **pandas**. This project demonstrates a standard ML workflow: data preprocessing, training, saving models, and making predictions.

---

## 🚀 Features

- End-to-end ML pipeline (train → save → predict)
- Logistic Regression baseline model
- Modular code structure (train, predict, main)
- Reusable for other datasets with minor tweaks

---

## 📂 Project Structure

```
titanic-ml-pipeline/
│── data/
│   ├── titanic.csv
│   ├── titanic_clean.csv
│── models/
│   ├── titanic_model.pkl
│── src/
│   ├── main.py
│   ├── train.py
│   ├── predict.py
│── .gitignore
│── README.md
```

---

## ⚡ Installation

```bash
git clone https://github.com/anshgirap/titanic-ml-pipeline.git
cd titanic-ml-pipeline
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Run:

```bash
python src/main.py
```

This trains the model on `titanic_clean.csv`, evaluates accuracy, and saves it to `models/titanic_model.pkl`.

---

## 🔮 Making Predictions

Prepare an input JSON (`data/sample.json`):

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "Fare": 7.25
}
```

Then run:

```bash
python src/predict.py
```

Output will display prediction (`0 = Did not survive`, `1 = Survived`).

---

## 🛠️ Tech Stack

- Python 3.9+
- pandas
- scikit-learn
- joblib

---

## 📜 License

MIT License. Free to use and modify.
