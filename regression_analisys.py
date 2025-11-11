import numpy as np
import matplotlib.pyplot as plt
np.random.seed(35)

#-----
# 4.1 Статистические характеристики сигналов
# ----------------------------
# 1. Хаотическая система — логистическое отображение
# ----------------------------
def logistic_map(r, x0, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

n = 1000  # количество точек

# хаотический ряд
chaotic_series = logistic_map(r=3.9, x0=0.5, n=n)

# ----------------------------
# 2. Ряд с равномерным распределением
# ----------------------------
uniform_series = np.random.uniform(0, 1, n)

# ----------------------------
# 3. Ряд с нормальным распределением
# ----------------------------
normal_series = np.random.normal(loc=0, scale=1, size=n)

# ----------------------------
# Вычисление статистик
# ----------------------------
datasets = {
    "Хаотический ряд": chaotic_series,
    "Равномерное распределение": uniform_series,
    "Нормальное распределение": normal_series
}

for name, data in datasets.items():
    mean = np.mean(data)
    var = np.var(data, ddof=1)  # выборочная дисперсия
    print(f"{name}:")
    print(f"  Среднее = {mean:.4f}")
    print(f"  Дисперсия = {var:.4f}\n")

# ----------------------------
# Построение гистограмм
# ----------------------------
plt.figure(figsize=(12, 6))

for i, (name, data) in enumerate(datasets.items(), 1):
    plt.subplot(1, 3, i)
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(name)
    plt.xlabel("Значения")
    plt.ylabel("Частота")

plt.tight_layout()
plt.show()
