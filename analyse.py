import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Путь к директории с файлами MovieLens 100K
data_path = "/Users/pzof/Downloads/ml-100k"  # Замените на путь к вашей папке

# Названия столбцов для файлов
columns = ["user_id", "item_id", "rating", "timestamp"]

# Создаем пустые DataFrame для train и test
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)

# Читаем все пары файлов u1.base, u1.test, ..., u5.base, u5.test
for i in range(1, 6):  # Номера от 1 до 5
    # Формирование путей к файлам
    base_file = os.path.join(data_path, f"u{i}.base")
    test_file = os.path.join(data_path, f"u{i}.test")
    
    # Чтение данных
    base_df = pd.read_csv(base_file, sep="\t", names=columns)
    test_df_part = pd.read_csv(test_file, sep="\t", names=columns)
    
    # Добавляем в общие DataFrame
    train_df = pd.concat([train_df, base_df], ignore_index=True)
    test_df = pd.concat([test_df, test_df_part], ignore_index=True)

# Проверка итоговых размеров данных
print(f"Размер общего train DataFrame: {train_df.shape}")
print(f"Размер общего test DataFrame: {test_df.shape}")

# Анализ данных
def analyze_data(df):
    # 1. Количество пользователей и предметов
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    print(f"Количество пользователей: {num_users}")
    print(f"Количество предметов: {num_items}")

    # 2. Распределение по пользователям
    user_ratings = df.groupby('user_id').size()
    print(f"Среднее количество оценок на пользователя: {user_ratings.mean():.2f}")
    plt.figure(figsize=(8, 6))
    plt.hist(user_ratings, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    plt.title("Распределение количества оценок на пользователя")
    plt.xlabel("Количество оценок")
    plt.ylabel("Частота")
    plt.show()

    # 3. Распределение по предметам
    item_ratings = df.groupby('item_id').size()
    print(f"Среднее количество оценок на предмет: {item_ratings.mean():.2f}")
    plt.figure(figsize=(8, 6))
    plt.hist(item_ratings, bins=30, alpha=0.7, color="lightgreen", edgecolor="black")
    plt.title("Распределение количества оценок на предмет")
    plt.xlabel("Количество оценок")
    plt.ylabel("Частота")
    plt.show()

    # 4. Распределение оценок
    plt.figure(figsize=(8, 6))
    plt.hist(df['rating'], bins=np.arange(0.5, 6, 1), alpha=0.7, color="lightcoral", edgecolor="black")
    plt.title("Распределение оценок")
    plt.xlabel("Оценка")
    plt.ylabel("Частота")
    plt.show()

    # 5. Поиск выбросов
    users_with_one_rating = user_ratings[user_ratings == 1].index
    items_with_one_rating = item_ratings[item_ratings == 1].index
    print(f"Пользователи с одной оценкой: {len(users_with_one_rating)}")
    print(f"Предметы с одной оценкой: {len(items_with_one_rating)}")
    return users_with_one_rating, items_with_one_rating

# Удаление выбросов
def clean_data(df, users_with_one_rating, items_with_one_rating):
    df_cleaned = df[~df['user_id'].isin(users_with_one_rating)]
    df_cleaned = df_cleaned[~df_cleaned['item_id'].isin(items_with_one_rating)]
    print(f"Данные после очистки: {df_cleaned.shape[0]} оценок")
    return df_cleaned

# Разбиение данных (user-item train-test split)
def train_test_user_item_split(df, test_size=0.2):
    train_data = []
    test_data = []

    for user, group in df.groupby('user_id'):
        if len(group) > 1:
            train, test = train_test_split(group, test_size=test_size, random_state=42)
            train_data.append(train)
            test_data.append(test)
        else:
            train_data.append(group)

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    print(f"Размер обучающей выборки: {train_df.shape[0]}")
    print(f"Размер тестовой выборки: {test_df.shape[0]}")
    return train_df, test_df

# Анализ данных
print(analyze_data(train_df)) 
print(analyze_data(test_df)) 