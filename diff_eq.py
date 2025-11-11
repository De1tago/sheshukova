"""
--------------------------------------------
Уравнение: du/dt = α * d²u/dx²
Область: x ∈ [0, L], t ∈ [0, T]
Граничные условия: u(0, t) = u(L, t) = 0
Начальное условие: u(x, 0) = sin(πx)
Аналитическое решение: u(x, t) = e^(-απ²t) * sin(πx)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# -------------------------------
# ПАРАМЕТРЫ
# -------------------------------
L = 1.0          # длина стержня
T = 0.2          # конечное время
alpha = 0.01     # коэффициент теплопроводности

Nx = 50           # число узлов по пространству
Nt = 2000         # число шагов по времени
dx = L / (Nx - 1)
dt = T / Nt
r = alpha * dt / dx**2  # число Фурье

print(f"--- Параметры ---")
print(f"L = {L}, T = {T}, alpha = {alpha}")
print(f"Nx = {Nx}, Nt = {Nt}")
print(f"dx = {dx:.5f}, dt = {dt:.5e}, r = {r:.4f}")
if r > 0.5:
    print("⚠️ Внимание: схема может быть неустойчива (r > 0.5)\n")

# -------------------------------
# ИНИЦИАЛИЗАЦИЯ
# -------------------------------
x = np.linspace(0, L, Nx)
u_explicit = np.sin(np.pi * x)  # начальное условие
u_implicit = np.copy(u_explicit)

# -------------------------------
# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ
# -------------------------------
def u_analytic(x, t, alpha):
    return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)

# -------------------------------
# ЯВНАЯ СХЕМА
# -------------------------------
u_new = np.zeros_like(u_explicit)
for n in range(Nt):
    for i in range(1, Nx - 1):
        u_new[i] = u_explicit[i] + r * (u_explicit[i+1] - 2*u_explicit[i] + u_explicit[i-1])
    u_explicit[:] = u_new

# -------------------------------
# НЕЯВНАЯ СХЕМА (Кранк–Николсон)
# -------------------------------
A = np.zeros((Nx-2, Nx-2))
B = np.zeros((Nx-2, Nx-2))
for i in range(Nx-2):
    A[i, i] = 1 + r
    B[i, i] = 1 - r
    if i > 0:
        A[i, i-1] = -r/2
        B[i, i-1] = r/2
    if i < Nx-3:
        A[i, i+1] = -r/2
        B[i, i+1] = r/2

for n in range(Nt):
    rhs = B @ u_implicit[1:-1]
    u_implicit[1:-1] = solve(A, rhs)

# -------------------------------
# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ В МОМЕНТ ВРЕМЕНИ T
# -------------------------------
u_exact = u_analytic(x, T, alpha)

# -------------------------------
# ВИЗУАЛИЗАЦИЯ
# -------------------------------
plt.figure(figsize=(9,6))
plt.plot(x, u_exact, 'k--', lw=2, label='Аналитическое решение')
plt.plot(x, u_explicit, 'r-', label='Явная схема (r ≤ 0.5)')
plt.plot(x, u_implicit, 'b-', label='Кранк–Николсон')
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title(f'Решение уравнения теплопроводности при t={T}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# АНАЛИЗ ОШИБОК
# -------------------------------
err_explicit = np.sqrt(np.mean((u_explicit - u_exact)**2))
err_implicit = np.sqrt(np.mean((u_implicit - u_exact)**2))
print(f"\nСреднеквадратичная ошибка:")
print(f"  Явная схема       = {err_explicit:.3e}")
print(f"  Кранк–Николсон    = {err_implicit:.3e}")

# -------------------------------
# АНАЛИЗ ЗАВИСИМОСТИ ОШИБКИ ОТ ШАГА
# -------------------------------
def compute_error(Nx, Nt):
    dx = L / (Nx - 1)
    dt = T / Nt
    r = alpha * dt / dx**2
    x = np.linspace(0, L, Nx)
    u = np.sin(np.pi * x)
    u_new = np.zeros_like(u)
    for n in range(Nt):
        for i in range(1, Nx - 1):
            u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        u[:] = u_new
    u_exact = u_analytic(x, T, alpha)
    return np.sqrt(np.mean((u - u_exact)**2)), r

steps = [20, 40, 80, 160, 320]
errors = []
r_values = []

for n in steps:
    e, rv = compute_error(n, Nt)
    errors.append(e)
    r_values.append(rv)

plt.figure(figsize=(8,5))
plt.loglog(steps, errors, 'o-', label='Ошибка явной схемы')
plt.xlabel('Число узлов Nx')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('Зависимость ошибки от пространственного шага')
plt.grid(True, which='both')
plt.legend()
plt.show()
