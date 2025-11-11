# ============================================================
# Лабораторная работа №4
# "Обработка набора данных и регрессионный анализ"
# Выполнил: ...
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ------------------------------------------------------------
# 4.1. Статистические характеристики сигналов
# ------------------------------------------------------------
print("\n=== 4.1. Статистические характеристики сигналов ===")

# --- Хаотическая система (логистическое отображение)
def logistic_map(r, x0, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

n = 1000
chaotic_series = logistic_map(r=3.9, x0=0.5, n=n)

# --- Равномерное распределение
uniform_series = np.random.uniform(0, 1, n)

# --- Нормальное распределение
normal_series = np.random.normal(loc=0, scale=1, size=n)

datasets = {
    "Хаотический ряд": chaotic_series,
    "Равномерное распределение": uniform_series,
    "Нормальное распределение": normal_series
}

for name, data in datasets.items():
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    print(f"{name}: среднее = {mean:.4f}, дисперсия = {var:.4f}")

# --- Гистограммы
plt.figure(figsize=(12, 6))
for i, (name, data) in enumerate(datasets.items(), 1):
    plt.subplot(1, 3, i)
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(name)
    plt.xlabel("Значения")
    plt.ylabel("Частота")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4.2. Парная линейная регрессия
# ------------------------------------------------------------
print("\n=== 4.2. Парная линейная регрессия ===")

np.random.seed(42)
n = 50
x = np.linspace(0, 10, n)
y = 3 * x + 7 + np.random.normal(0, 3, n)  # линейная зависимость с шумом

x_reshaped = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x_reshaped, y)

a = model.coef_[0]
b = model.intercept_
r2 = model.score(x_reshaped, y)

print(f"Уравнение регрессии: y = {a:.3f} * x + {b:.3f}")
print(f"Коэффициент детерминации R² = {r2:.4f}")

plt.scatter(x, y, color='blue', label='Наблюдения')
plt.plot(x, model.predict(x_reshaped), color='red', label='Линия регрессии')
plt.title("Парная линейная регрессия")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 4.3 (вариант 1). Нелинейная (квадратичная) регрессия
# ------------------------------------------------------------
print("\n=== 4.3 (вариант 1). Нелинейная регрессия ===")

np.random.seed(0)
x = np.linspace(-3, 3, 100)
y = 2 * x**2 + 3 * x + 5 + np.random.normal(0, 2, size=x.shape)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x.reshape(-1, 1))

model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred = model_poly.predict(X_poly)

print(f"Уравнение регрессии: y = {model_poly.coef_[2]:.3f}x² + "
      f"{model_poly.coef_[1]:.3f}x + {model_poly.intercept_:.3f}")
print(f"Коэффициент детерминации R² = {model_poly.score(X_poly, y):.4f}")

plt.scatter(x, y, color='blue', label='Наблюдения')
plt.plot(x, y_pred, color='red', label='Квадратичная модель')
plt.title("Нелинейная (квадратичная) регрессия")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 4.3 (вариант 2). Фильтрация выборочных данных
# ------------------------------------------------------------
print("\n=== 4.3 (вариант 2). Фильтрация выборочных данных ===")

np.random.seed(1)
n = 100
x = np.linspace(0, 10, n)
y_true = np.sin(x)
noise = np.random.normal(0, 0.3, n)
y_noisy = y_true + noise

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def median_filter(data, window_size=5):
    filtered = []
    for i in range(len(data) - window_size + 1):
        filtered.append(np.median(data[i:i+window_size]))
    return np.array(filtered)

y_ma = moving_average(y_noisy, window_size=5)
y_med = median_filter(y_noisy, window_size=5)
x_filtered = x[2:-2]  # для совмещения длины

plt.figure(figsize=(10, 6))
plt.plot(x, y_true, 'k--', label="Истинный сигнал")
plt.plot(x, y_noisy, 'gray', alpha=0.6, label="С шумом")
plt.plot(x_filtered, y_ma, 'r', label="Скользящее среднее")
plt.plot(x_filtered, y_med, 'b', label="Медианный фильтр")
plt.title("Фильтрация выборочных данных")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print("\nРабота завершена успешно ✅")
