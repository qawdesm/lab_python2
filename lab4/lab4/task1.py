import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('student-por.csv')

df['Passed'] = (df['G3'] >= 10).astype(int)

X = df[['G1', 'G2', 'studytime', 'failures', 'absences']]
y = df['Passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Случайный лес:")
print(f"OOB Accuracy: {rf.oob_score_:.3f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(classification_report(y_test, y_pred_rf, target_names=['Не сдал', 'Сдал']))

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

print("AdaBoost:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada):.3f}")
print(classification_report(y_test, y_pred_ada, target_names=['Не сдал', 'Сдал']))

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("Градиентный бустинг:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")
print(classification_report(y_test, y_pred_gb, target_names=['Не сдал', 'Сдал']))

plt.figure(figsize=(8, 6))

for name, scores, color in [
    ('Random Forest', rf.predict_proba(X_test)[:, 1], 'blue'),
    ('AdaBoost', ada.decision_function(X_test), 'green'),
    ('Gradient Boosting', gb.decision_function(X_test), 'red')
]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    plt.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Случайный')
plt.xlabel('FPR'), plt.ylabel('TPR')
plt.title('ROC-кривые')
plt.legend(), plt.grid(alpha=0.3)
plt.show()