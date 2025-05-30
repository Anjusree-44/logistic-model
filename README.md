## binary classifier using logistic regression

## 📊 Dataset
[Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

##CODE :
#logistic model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv(r"C:\Users\katta\OneDrive\Python-ai\myenv\BreastCancer.csv")

# Display basic info
print("Dataset Head:\n", df.head())
print("\nDataset Info:\n", df.info())

# Drop any non-numeric or irrelevant columns if needed (adjust for actual columns)
df = df.drop(['id'], axis=1, errors='ignore')

# Check unique values in Class column
print("Unique values in original Class column:", df['Class'].unique())

# Encode target variable if it's in string form
#df['Class'] = df['Class'].map({'benign': 0, 'malignant': 1})

# Drop rows where mapping produced NaN (i.e., invalid or missing class)
df = df.dropna(subset=['Class']) 

# Split into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Drop rows where any feature is NaN
X = X.dropna()
y = y.loc[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Any NaN in X?", X.isnull().any().any())
print("Any NaN in y?", y.isnull().any())
print("X indices:", X.index)
print("y indices:", y.index)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Show sigmoid behavior for a few inputs
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sample_scores = np.linspace(-10, 10, 100)
sigmoid_vals = sigmoid(sample_scores)
plt.figure()
plt.plot(sample_scores, sigmoid_vals)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid()
plt.show()

---
## Screenshots 
![4 1](https://github.com/user-attachments/assets/80398c89-55c2-49c4-b0e0-3cc41e98a0bd)
![4 2](https://github.com/user-attachments/assets/0557be35-618c-48df-a309-3dc33c8ddecf)

