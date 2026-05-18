import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score,
    classification_report, confusion_matrix)

reviews = {
    "text": [
        # Положительные отзывы (label = 1)
        "Отличный товар, очень доволен покупкой",
        "Быстрая доставка, упаковка целая, всё супер",
        "Качество превзошло ожидания, рекомендую",
        "Пользуюсь каждый день, вещь очень удобная",
        "Заказал уже второй раз, не разочаровал",
        "Прекрасное качество за свои деньги",
        "Товар соответствует описанию, доволен",
        "Очень нравится, буду заказывать ещё",
        "Хорошая вещь, служит уже полгода",
        "Отличное соотношение цена-качество",
        # Отрицательные отзывы (label = 0)
        "Пришёл брак, возврат оформили с трудом",
        "Качество отвратительное, развалилось за неделю",
        "Не соответствует описанию, очень разочарован",
        "Долгая доставка, товар пришёл повреждённым",
        "Деньги выброшены на ветер, не покупайте",
        "Сломалось через два дня, ужасное качество",
        "Продавец не отвечает, жду возврат",
        "Полный обман, товар не соответствует фото",
        "Разочарован, материал очень низкого сорта",
        "Не рекомендую, много дефектов",
    ],
    "sentiment": [1]*10 + [0]*10
}
df = pd.DataFrame(reviews)

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    lowercase=True,      # приводим к нижнему регистру
    max_features=500,    # берём 500 наиболее значимых слов
    ngram_range=(1, 2)   # учитываем пары слов (биграммы)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
models = {
    "Логистическая регрессия": LogisticRegression(
        C=1.0,          # параметр регуляризации
        max_iter=1000,  # максимальное число итераций
        random_state=42
    ),
    "Случайный лес": RandomForestClassifier(
        n_estimators=100,  # число деревьев
        max_depth=10,      # максимальная глубина
        random_state=42
    ),
    "Нейронная сеть (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32),  # два скрытых слоя
        activation="relu",            # функция активации
        solver="adam",                # оптимизатор
        alpha=0.01,                   # L2-регуляризация
        max_iter=500,                 # эпохи обучения
        learning_rate_init=0.001,     # шаг обучения
        random_state=42
    )
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    predictions[name] = y_pred
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }
    print(f"\n=== {name} ===")
    print(f"Точность: {results[name]["accuracy"]:.2f}")
    print(results[name]["report"])

# График 1: Сравнение точности моделей
accuracies = {name: res["accuracy"]
              for name, res in results.items()}

plt.figure(figsize=(9, 5))
bars = plt.bar(accuracies.keys(), accuracies.values(),
               color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylim(0, 1.15)
plt.title("Сравнение моделей по точности (Accuracy)",
          fontsize=14, pad=15)
plt.ylabel("Accuracy")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             h + 0.03, f"{h:.2f}",
             ha="center", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150)
plt.show()


# График 2: Матрицы ошибок
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Матрицы ошибок", fontsize=15)
for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                ax=axes[idx], cbar=False,
                xticklabels=["Негат.", "Позит."],
                yticklabels=["Негат.", "Позит."])
    axes[idx].set_title(name, fontsize=11)
    axes[idx].set_xlabel("Предсказано")
    axes[idx].set_ylabel("Истина")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
new_reviews = [
    "Покупкой очень доволен, отличный товар",
    "Полный брак, не рекомендую никому",
    "Нормально, ожидал большего"
]

new_tfidf = vectorizer.transform(new_reviews)

mlp = models["Нейронная сеть (MLP)"]
preds = mlp.predict(new_tfidf)
probs = mlp.predict_proba(new_tfidf)

for text, pred, prob in zip(new_reviews, preds, probs):
    label = "Положительный" if pred == 1 else "Отрицательный"
    confidence = max(prob)
    print(f"Отзыв: {text[:40]}...")
    print(f"Тональность: {label} (уверенность: {confidence:.0%})")
    print()
