# Road Accident Prediction Model

> A supervised machine learning model that classifies road accident risk levels using environmental conditions and traffic features, built with Support Vector Machine (SVM) and scikit-learn.

---

## Overview

This project develops a predictive classification model to assess the probability of road accidents under different environmental and traffic conditions. Using publicly available accident datasets, the model identifies high-risk patterns that can support road safety analysis and early warning systems.

---

## Problem Statement

Road accidents are a leading cause of injury and fatality globally. Predicting risk based on observable conditions (weather, road type, traffic density, time of day) enables proactive safety interventions. This model classifies whether given conditions pose **Low**, **Medium**, or **High** accident risk.

---

## Dataset

- **Source:** Publicly available road accident datasets (e.g., UK Road Safety Data / Kaggle)
- **Features:** Weather conditions, road surface, light conditions, road type, speed limit, junction type, time of day, day of week, vehicle type
- **Target:** Accident severity / risk category (Low / Medium / High)

*(Update this section with the exact dataset you used and its source URL)*

---

## Approach

```
Raw Dataset
    │
    ▼
Exploratory Data Analysis (EDA)
    │
    ▼
Preprocessing
(Encoding, Scaling, Missing Value Handling)
    │
    ▼
Feature Selection
    │
    ▼
SVM Model Training
    │
    ▼
Evaluation (Accuracy, Precision, Recall, F1)
    │
    ▼
High-Risk Pattern Analysis
```

---

## Key Code Snippets

### Preprocessing

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('accident_data.csv')
df.dropna(inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in ['weather', 'road_type', 'light_conditions']:
    df[col] = le.fit_transform(df[col])

X = df.drop('severity', axis=1)
y = df['severity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### Model Training & Evaluation

```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
```

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | *(add your result)* |
| Precision | *(add your result)* |
| Recall | *(add your result)* |
| F1 Score | *(add your result)* |

*(Fill in actual model performance metrics from your run)*

---

## Key Findings

- Weather condition and road surface type were the strongest predictors of accident severity
- Night-time conditions with wet roads produced the highest proportion of high-risk classifications
- SVM with RBF kernel outperformed linear kernel for this multi-class classification task

*(Update with your actual findings)*

---

## Setup

### Prerequisites
- Python 3.8+

### Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
```

### Run

```bash
jupyter notebook road_accident_prediction.ipynb
```

---

## Project Structure

```
road-accident-prediction/
├── road_accident_prediction.ipynb   # Main notebook
├── data/
│   └── accident_data.csv            # Dataset (or download instructions)
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Language:** Python
- **ML:** Scikit-learn (SVM, preprocessing, evaluation)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## Author

**Ajayaman Kantumuchu**
MS in Computer Science, CSUSB | ajayamankantumuchu@gmail.com
[LinkedIn](https://linkedin.com/in/YOUR_USERNAME) | [GitHub](https://github.com/Ajayaman2627)
