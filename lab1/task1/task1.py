import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

df = pd.read_csv("titanic.csv")
print("Первые 15 строк нашего датасета:")
print(df.head(15))
print("Информация о нашем датасете:")
print(df.info())
print("Название колонок нашего датасета:")
print(df.columns.tolist())
print("Количество пропущенных значений для кадого столбца в нашем датасете ДО:")
print(df.isnull().sum())


pclass_mode = df['pclass'].mode()[0]
df['pclass'] = df['pclass'].fillna(pclass_mode)

survived_mode = df['survived'].mode()[0]
df['survived'] = df['survived'].fillna(survived_mode)

name_mode = df['name'].mode()[0]
df['name'] = df['name'].fillna(name_mode)

sex_mode = df['sex'].mode()[0]
df['sex'] = df['sex'].fillna(sex_mode)

age_median = df['age'].median()
df['age'] = df['age'].fillna(age_median)

sibsp_median = df['sibsp'].median()
df['sibsp'] = df['sibsp'].fillna(sibsp_median)

parch_median = df['parch'].median()
df['parch'] = df['parch'].fillna(parch_median)

ticket_mode = df['ticket'].mode()[0]
df['ticket'] = df['ticket'].fillna(ticket_mode)

fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)

cabin_mode = df['cabin'].mode()[0]
df['cabin'] = df['cabin'].fillna(cabin_mode)

embarked_mode = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(embarked_mode)

boat_mode = df['boat'].mode()[0]
df['boat'] = df['boat'].fillna(boat_mode)

body_median = df['body'].median()
df['body'] = df['body'].fillna(body_median)

home_mode = df['home.dest'].mode()[0]
df['home.dest'] = df['home.dest'].fillna(home_mode)

print("Количество пропущенных значений для каждого столбца в нашем датасете после:")
print(df.isnull().sum())

columns_to_normalize_mm = ['age', 'fare', 'sibsp']
print("Нормализация MinMaxScaler():")
for col in columns_to_normalize_mm:
    print(f"до: мин={df[col].min():.2f}, макс={df[col].max():.2f}")
scaler = MinMaxScaler()
for col in columns_to_normalize_mm:
    print(f"Нормализуем столбец: {col}")
    df[col] = scaler.fit_transform(df[[col]])
    print(f"после: мин={df[col].min():.2f}, макс={df[col].max():.2f}")

columns_to_normalize_st = ['parch', 'body']
print("Нормализация StandardScaler():")
for col in columns_to_normalize_st:
    print(f"до: мин={df[col].min():.2f}, макс={df[col].max():.2f}")
scaler = StandardScaler()
for col in columns_to_normalize_st:
    print(f"Нормализуем столбец: {col}")
    df[col] = scaler.fit_transform(df[[col]])
    print(f"после: мин={df[col].min():.2f}, макс={df[col].max():.2f}")


categorical_cols = ['sex', 'embarked', 'cabin', 'boat']

print(f"Категориальные столбцы: {categorical_cols}")
print("\nУникальные значения до преобразования:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()[:5]} (всего {df[col].nunique()})")

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nПосле One-Hot Encoding:")
print(f"Всего столбцов: {len(df.columns)}")

new_cols = [col for col in df.columns if any(cat in col for cat in ['sex_', 'embarked_', 'cabin_', 'boat_'])]
print(f"\nСоздано {len(new_cols)} новых столбцов")
print("Первые 5 строк после преобразования:")
print(df.head())
df.to_csv('titanic_final.csv', index=False)