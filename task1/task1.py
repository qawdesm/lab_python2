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

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(f"Размер обучающей выборки: {len(train_df)}")
print(f"Размер тестовой выборки: {len(test_df)}")

columns_to_normalize_mm = ['age', 'fare', 'sibsp']
print("Нормализация MinMaxScaler():")
scaler_mm = MinMaxScaler()
train_df[columns_to_normalize_mm] = scaler_mm.fit_transform(train_df[columns_to_normalize_mm])
print("Train данные после нормализации:")
for col in columns_to_normalize_mm:
    print(f"{col}: мин={train_df[col].min():.2f}, макс={train_df[col].max():.2f}")
test_df[columns_to_normalize_mm] = scaler_mm.transform(test_df[columns_to_normalize_mm])
print("Test данные после нормализации:")
for col in columns_to_normalize_mm:
    print(f"{col}: мин={test_df[col].min():.2f}, макс={test_df[col].max():.2f}")


columns_to_normalize_st = ['parch', 'body']
print("Нормализация StandardScaler():")
scaler_st = StandardScaler()
train_df[columns_to_normalize_st] = scaler_st.fit_transform(train_df[columns_to_normalize_st])
print("Train данные после нормализации:")
for col in columns_to_normalize_st:
    print(f"{col}: среднее={train_df[col].mean():.2f}, стд={train_df[col].std():.2f}")
test_df[columns_to_normalize_st] = scaler_st.transform(test_df[columns_to_normalize_st])
print("Test данные после нормализации:")
for col in columns_to_normalize_st:
    print(f"{col}: среднее={test_df[col].mean():.2f}, стд={test_df[col].std():.2f}")

categorical_cols = ['sex', 'embarked', 'cabin', 'boat']

print(f"Категориальные столбцы: {categorical_cols}")
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

print("\nПосле One-Hot Encoding:")
print(f"Train: {len(train_df.columns)} столбцов")
print(f"Test: {len(test_df.columns)} столбцов")
train_df.to_csv('titanic_train.csv', index=False)
test_df.to_csv('titanic_test.csv', index=False)