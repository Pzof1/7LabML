import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1) Загрузка MovieLens 100k в train_df, test_df (Ваш код)
# ------------------------------

data_path = "/Users/pzof/Downloads/ml-100k"

columns = ["user_id", "item_id", "rating", "timestamp"]
train_df = pd.DataFrame(columns=columns)
test_df = pd.DataFrame(columns=columns)

for i in range(1, 6):
    base_file = os.path.join(data_path, f"u{i}.base")
    test_file = os.path.join(data_path, f"u{i}.test")
    
    base_df = pd.read_csv(base_file, sep="\t", names=columns)
    test_df_part = pd.read_csv(test_file, sep="\t", names=columns)
    
    train_df = pd.concat([train_df, base_df], ignore_index=True)
    test_df = pd.concat([test_df, test_df_part], ignore_index=True)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Опционально убедимся, что id пользователей/фильмов начинаются с 1 (MovieLens так устроен)
# но надо привести к 0-based индексам для удобства:
train_df['user_id'] -= 1
train_df['item_id'] -= 1
test_df['user_id']  -= 1
test_df['item_id']  -= 1

num_users = train_df['user_id'].max() + 1
num_items = train_df['item_id'].max() + 1
print("num_users =", num_users, "num_items =", num_items)

# ------------------------------
# 2) Подготовим numpy-структуры
# ------------------------------

# Храним train ratings как список (u, i, r)
train_data = train_df[['user_id','item_id','rating']].values
test_data  = test_df[['user_id','item_id','rating']].values

# ------------------------------
# 3) Реализация ALS с нуля
# ------------------------------

def als_train(
    train_data, 
    num_users, 
    num_items, 
    rank=20,       # размер латентного пространства
    reg=0.1,       # регуляризация
    num_iters=10,  # кол-во итераций ALS
    seed=42
):
    """
    Реализуем ALS (Alternating Least Squares) для явных рейтингов.
    
    :param train_data: np.array формы (N, 3) [user, item, rating]
    :param num_users:  число пользователей
    :param num_items:  число фильмов
    :param rank: размерность латентного фактора
    :param reg:  коэффициент регуляризации (lambda)
    :param num_iters: кол-во итераций ALS
    :param seed: для воспроизводимости рандомной инициализации
    :return: (P, Q), где:
             P shape=(num_users, rank),
             Q shape=(num_items, rank).
    """
    np.random.seed(seed)
    
    # Инициализация латентных векторов
    P = 0.1 * np.random.randn(num_users, rank)
    Q = 0.1 * np.random.randn(num_items, rank)
    
    # Для удобства создадим "список" item->(user, rating), user->(item, rating)
    # чтобы быстро находить, какие (i,r) принадлежат пользователю u и наоборот.
    
    user_ratings = [[] for _ in range(num_users)]
    item_ratings = [[] for _ in range(num_items)]
    for (u, i, r) in train_data:
        user_ratings[u].append((i, r))
        item_ratings[i].append((u, r))
    
    # Основной цикл ALS:
    for iteration in range(num_iters):
        # ----- Обновляем P (пользователи), фиксируя Q -----
        for u in range(num_users):
            # Собираем все (i, r_ui) для данного пользователя
            user_data = user_ratings[u]
            if len(user_data) == 0:
                continue  # если вдруг нет рейтингов
            
            # Формируем матрицу A и вектор b для решения A * p_u = b
            # A = Q_i^T * Q_i + lambdaI, b = Q_i^T * r_ui
            # где Q_i -- вектора фильмов, оцененных пользователем u
            # Real eq: p_u = (sum_i(q_i q_i^T) + regI)^{-1} * sum_i(r_ui * q_i)
            
            # rank x rank
            A = reg * np.eye(rank)
            # rank x 1
            b = np.zeros(rank)
            
            for (i, r_ui) in user_data:
                q_i = Q[i]
                A += np.outer(q_i, q_i)  # q_i q_i^T
                b += r_ui * q_i
            
            # Решаем систему A p_u = b
            p_u = np.linalg.solve(A, b)
            P[u] = p_u
        
        # ----- Обновляем Q (фильмы), фиксируя P -----
        for i in range(num_items):
            item_data = item_ratings[i]
            if len(item_data) == 0:
                continue
            
            A = reg * np.eye(rank)
            b = np.zeros(rank)
            
            for (u, r_ui) in item_data:
                p_u = P[u]
                A += np.outer(p_u, p_u)
                b += r_ui * p_u
            
            q_i = np.linalg.solve(A, b)
            Q[i] = q_i
        
        # Можно вывести промежуточный loss/iter (не обязательно)
        # loss = compute_mse(train_data, P, Q)
        # print(f"Iteration {iteration+1}/{num_iters}, MSE on train = {loss:.4f}")
    
    return P, Q

def compute_rmse(data, P, Q):
    """
    Вычислим RMSE на заданном наборе data (user, item, rating).
    """
    mse_sum = 0.0
    for (u, i, r_ui) in data:
        r_pred = np.dot(P[u], Q[i])  # скалярное произведение
        err = r_ui - r_pred
        mse_sum += err*err
    rmse = np.sqrt(mse_sum / len(data))
    return rmse

# ------------------------------
# 4) Запуск обучения ALS
# ------------------------------
rank = 20
num_iters = 10
reg = 0.1

P, Q = als_train(
    train_data=train_data,
    num_users=num_users,
    num_items=num_items,
    rank=rank,
    reg=reg,
    num_iters=num_iters,
    seed=42
)

# ------------------------------
# 5) Оценка на тесте
# ------------------------------
train_rmse = compute_rmse(train_data, P, Q)
test_rmse = compute_rmse(test_data, P, Q)

print(f"Final Train RMSE = {train_rmse:.4f}")
print(f"Final Test  RMSE = {test_rmse:.4f}")
