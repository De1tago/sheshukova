import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import medfilt
# 4.1 Статистические характеристики
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
#a) Временной ряд с хаотической динамикой
r = 3.8 # параметр роста
x_a = np.zeros(1000)
x_a[0] = 0.5
for i in range(1, len(x_a)):
    x_a[i] = r * x_a[i - 1] * (1 - x_a[i - 1])
#b) Случайные числа
x_b = np.random.rand(1000)  # 0/1
#c) Нормальное распределение с шумом
x_c = np.random.normal(loc=0, scale=1, size=1000)
datasets = {'Хаотическая динамика (a)': x_a,
            'Случайные числа (b)': x_b,
            'Нормальное распределение (c)': x_c} # характеризуется наибольшим разбросом значений (высокая дисперсия)
print("\n4.1 Статистические характеристики")
for name, data in datasets.items():
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    print(f"\n{name}:")
    print(f"Среднее (мат.ожидание): {mean:.3f}")
    print(f"Дисперсия: {var:.3f}")
    print(f"СКО: {std:.3f}")
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue')
    plt.title(f"Гистограмма: {name}")
    plt.xlabel("Значение")
    plt.ylabel("Плотность вероятности")
    plt.grid(True)

# 4.2 Простая линейная регрессия
# Генерация линейной зависимости с шумом
x = np.linspace(0, 10, 50)
y = 2.5 * x + 5 + np.random.normal(0, 5, len(x))
# МНК
coef = np.polyfit(x, y, 1) # (массив, массив, степень полинома линейный)
y_pred = np.polyval(coef, x)
# y=a1 * x + a0<, a1 - наклон, a0 - свободный член --> параметры функции регресии
print("\n4.2 Линейная регрессия:")
print(f"y = {coef[0]:.3f} * x + {coef[1]:.3f}")
#print(f"Наклон (a1) = {coef[0]:.3f}")
#print(f"Свободный член (a0) = {coef[1]:.3f}")
# Визуализация
plt.figure(figsize=(6,4))
plt.scatter(x, y, label="Исходные данные", color="blue")
plt.plot(x, y_pred, color="red", label="Линейная регрессия")
plt.legend()
plt.title("Линейная регрессия")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
# Шум увеличен: рассеяние точек вокруг линии регрессии увеличится, точность оценки  а1 а0 снизится.
# Шум уменьшен: точки ближе к прямой.

# 4.3 Параметрическая (нелинейная) регрессия
def model(x, a, b):
    return a * np.exp(b * x)
# Синтетические данные
x_nl = np.linspace(-3, 3, 70)
y_nl = 2 * np.exp(1.2 * x_nl) + np.random.normal(0, 1.5, len(x_nl))
# Оценка параметров
params, cov = curve_fit(model, x_nl, y_nl, p0=[0, 5])
a_fit, b_fit = params
print("\n4.3 Нелинейная регрессия:")
print(f"Модель: y = {a_fit:.3f} * exp({b_fit:.3f} * x)")
# Визуализация
plt.figure(figsize=(6,4))
plt.scatter(x_nl, y_nl, label="Данные", color="green")
plt.plot(x_nl, model(x_nl, *params), 'r', label="Аппроксимация exp")
plt.legend()
plt.title("Нелинейная регрессия")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
# Увеличение шума: точки сильнее разлетятся от экспоненты
# Шум уменьшен: данные ближе к кривой, параметры аппроксимации точнее.

# 4.4 Фильтрация выборочных данных
signal = np.cos(x_nl) + np.random.normal(0, 0.3, len(x_nl))
window = 5
smooth_mean = np.convolve(signal, np.ones(window)/window, mode='valid') # Уменьшает случайные колебания, если увеличить размер окна сильнее сглаживание, но сигнал теряет детали.
smooth_median = medfilt(signal, kernel_size=5) # Убирает резкие выбросы


plt.figure(figsize=(8,4))
plt.plot(x_nl, signal, 'gray', alpha=0.5, label="Исходный сигнал")
plt.plot(x_nl[window-1:], smooth_mean, 'r', label="Скользящее среднее")
plt.legend()
plt.title("Фильтрация данных — Скользящее среднее")
plt.xlabel("x")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(x_nl, signal, 'gray', alpha=0.5, label="Исходный сигнал")
plt.plot(x_nl, smooth_median, 'g', label="Медианная фильтрация")
plt.legend()
plt.title("Фильтрация данных — Медианный фильтр")
plt.xlabel("x")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.show()
