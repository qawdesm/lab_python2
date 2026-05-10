import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
data = pd.DataFrame({
    "text": [
        "Win a free lottery prize now",
        "Limited offer buy today",
        "Congratulations you won money",
        "You have been selected for a free gift card",
        "Claim your prize immediately act now",
        "Earn cash fast work from home",
        "Free iPhone click to claim your reward",
        "Exclusive deal only for you limited time",
        #разделение
        "Meeting scheduled at 10 am",
        "Project report attached",
        "Let's discuss the budget",
        "Can we reschedule the call to Friday",
        "Please find the invoice attached",
        "Team lunch is at noon today",
        "Reminder to submit your timesheet",
        "Great work on the presentation yesterday",
    ],
    "spam": [1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0]
})
 
X = data["text"]
y = data["spam"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42,
    stratify=y
)
 
model = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("clf", LogisticRegression(max_iter=1000))
])
 
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
 
# Кросс-валидация на всём датасете (5 фолдов)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print(f"F1 по кросс-валидации (5 фолдов): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
 
msg = ["You have been selected for a free gift card"]
 
prediction = model.predict(msg)[0]
prob = model.predict_proba(msg)[0][1]
 
print("Спам" if prediction == 1 else "Не спам")
print("Вероятность спама:", round(prob, 2))