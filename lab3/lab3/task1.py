import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('student-por.csv')

X_reg = df[['G1', 'G2', 'studytime', 'failures', 'absences']]
y_reg = df['G3']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = dt_reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")


df['Passed'] = (df['G3'] >= 10).astype(int)

X_clf = df[['G1', 'G2', 'studytime', 'failures', 'absences']]
y_clf = df['Passed']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = dt_clf.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print("\nОтчет классификации:")
print(classification_report(y_test_clf, y_pred_clf, target_names=['Не сдал', 'Сдал']))

print("Матрица ошибок:")
cm = confusion_matrix(y_test_clf, y_pred_clf)
print(cm)
print(f"TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

y_proba = dt_clf.predict_proba(X_test_clf)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_clf, y_proba)
auc = roc_auc_score(y_test_clf, y_proba)

print(f"AUC: {auc:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'Decision Tree (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Случайный (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривая')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
