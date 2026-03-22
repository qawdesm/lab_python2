import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

df = pd.read_csv('student-por.csv')

X_reg = df[['G1', 'G2', 'studytime', 'failures', 'absences']]
y_reg = df['G3']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.4, random_state=42)
print(f"Регрессия - обучающая: {len(X_train_reg)}, тестовая: {len(X_test_reg)}")
print("Линейная регрессия:")
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)
y_pred_reg = linear_model.predict(X_test_reg)

MSE = mean_squared_error(y_test_reg, y_pred_reg)
RMSE = root_mean_squared_error(y_test_reg, y_pred_reg)
MAE = mean_absolute_error(y_test_reg, y_pred_reg)

print(f"MSE: {MSE:.3f}")
print(f"RMSE: {RMSE:.3f}")
print(f"MAE: {MAE:.3f}")

n=2
poly_features = PolynomialFeatures(n)
X_train_poly = poly_features.fit_transform(X_train_reg)
X_test_poly = poly_features.transform(X_test_reg)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_reg)
y_pred_poly = poly_model.predict(X_test_poly)

MSE_poly = mean_squared_error(y_test_reg, y_pred_poly)
print(f"MSE линейной регрессии: {MSE:.3f}")
print(f"MSE полиномиальной регрессии: {MSE_poly:.3f}")

if MSE_poly < MSE:
    print(f"Улучшение достигнуто! MSE уменьшилась на {MSE - MSE_poly:.3f}")
else:
    print("Улучшение не достигнуто")



df['Passed'] = (df['G3'] >= 10).astype(int)
X_clf = df[['G1', 'G2', 'studytime', 'failures', 'absences']]
y_clf = df['Passed']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.4, random_state=42)

print("\nЛогистическая регрессия (без регуляризации):")
print(f"Классификация - обучающая: {len(X_train_clf)}, тестовая: {len(X_test_clf)}")
logreg_model = LogisticRegression(C=np.inf)
logreg_model.fit(X_train_clf, y_train_clf)
y_pred_test_clf = logreg_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_test_clf)
f1_orig = f1_score(y_test_clf, y_pred_test_clf)
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1_orig:.3f}")
print(f"Отчет классификации:")
print(classification_report(y_test_clf, y_pred_test_clf, target_names=['Не сдал', 'Сдал']))

print("Матрица ошибок:")
cm = confusion_matrix(y_test_clf, y_pred_test_clf)
print(cm)
print(f"TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

print("\nУлучшение модели (L2 регуляризация):")
logreg_l2 = LogisticRegression()
logreg_l2.fit(X_train_clf, y_train_clf)
y_pred_l2 = logreg_l2.predict(X_test_clf)

f1_l2 = f1_score(y_test_clf, y_pred_l2)
print(f"Без регуляризации - F1: {f1_orig:.3f}")
print(f"С L2 регуляризацией - F1: {f1_l2:.3f}")

if f1_l2 > f1_orig:
    print(f"Улучшение достигнуто! F1 вырос на {f1_l2 - f1_orig:.3f}")
else:
    print("Улучшение не достигнуто")