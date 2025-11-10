"""
Программа для численного решения обыкновенных дифференциальных уравнений (ОДУ)
методами Рунге-Кутта 4-го порядка и модифицированным методом Эйлера.

Исследуемые системы:
1. Линейный гармонический осциллятор (с известным аналитическим решением)
2. Система Лоренца (нелинейная система с классическим странным аттрактором)

Проводится сравнение методов по точности, времени вычисления и влиянию на фазовые портреты.
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib может отсутствовать в окружении
    plt = None

# Тип для функции правой части системы ОДУ: f(t, y) -> dy/dt
# t - время, y - вектор состояния, возвращает производную состояния
VectorFunc = Callable[[float, np.ndarray], np.ndarray]

# Глобальные параметры задач
LINEAR_PARAMS = dict(omega=1.5, x0=1.0, v0=0.0, t0=0.0, t_end=20.0)
LINEAR_STEPS = (0.2, 0.1, 0.05, 0.02)
LINEAR_INITIAL_STATE = np.array([LINEAR_PARAMS["x0"], LINEAR_PARAMS["v0"]], dtype=float)

# Базовые параметры системы Лоренца
LORENZ_BASE_PARAMS = dict(sigma=10.0, beta=8.0 / 3.0, rho=28.0)
LORENZ_T0 = 0.0
LORENZ_T_END = 60.0
LORENZ_STEPS = (0.02, 0.01, 0.005)
LORENZ_TRANSIENT = 10.0
LORENZ_INITIAL_STATE = np.array([1.0, 1.0, 1.0], dtype=float)

# Набор параметров, демонстрирующих смену поведения: стационар → периодика → хаос
# Источник: классические примеры для системы Лоренца
LORENZ_SCENARIOS = (
    {
        "name": "Стационарный режим",
        "description": "rho < 1 — траектория быстро притягивается к одной фиксированной точке",
        "params": {**LORENZ_BASE_PARAMS, "rho": 10.0},
    },
    {
        "name": "Квазипериодический режим",
        "description": "rho чуть выше критического ~24.74 — переход через бифуркацию, появляются колебания",
        "params": {**LORENZ_BASE_PARAMS, "rho": 24.5},
    },
    {
        "name": "Хаотический режим",
        "description": "rho = 28 — классический странный аттрактор Лоренца",
        "params": {**LORENZ_BASE_PARAMS, "rho": 28.0},
    },
    {
        "name": "Сложный хаос",
        "description": "rho = 100 — расширение аттрактора, усиливается чувствительность",
        "params": {**LORENZ_BASE_PARAMS, "rho": 100.0},
    },
)
LORENZ_SCENARIO_STEP = 0.01
LORENZ_SCENARIO_T_END = 80.0


def rk4_step(func: VectorFunc, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """
    Выполняет один шаг метода Рунге-Кутта 4-го порядка (RK4).
    
    Метод имеет порядок точности O(h^4) и использует 4 вычисления функции правой части.
    
    Args:
        func: Функция правой части системы ОДУ f(t, y)
        t: Текущее время
        y: Текущее состояние системы (вектор)
        h: Шаг интегрирования
    
    Returns:
        Новое состояние системы после шага интегрирования
    """
    # Вычисляем 4 коэффициента (k1, k2, k3, k4) для метода Рунге-Кутта
    k1 = func(t, y)  # Производная в начальной точке
    k2 = func(t + 0.5 * h, y + 0.5 * h * k1)  # Производная в средней точке (используя k1)
    k3 = func(t + 0.5 * h, y + 0.5 * h * k2)  # Производная в средней точке (используя k2)
    k4 = func(t + h, y + h * k3)  # Производная в конечной точке (используя k3)
    # Взвешенная комбинация коэффициентов для получения следующего значения
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def modified_euler_step(func: VectorFunc, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """
    Выполняет один шаг модифицированного метода Эйлера (метод Хойна).
    
    Метод имеет порядок точности O(h^2) и использует 2 вычисления функции правой части.
    Более точный, чем явный метод Эйлера, но менее точный, чем RK4.
    
    Args:
        func: Функция правой части системы ОДУ f(t, y)
        t: Текущее время
        y: Текущее состояние системы (вектор)
        h: Шаг интегрирования
    
    Returns:
        Новое состояние системы после шага интегрирования
    """
    # Первый шаг: предсказание (явный метод Эйлера)
    k1 = func(t, y)
    y_predict = y + h * k1
    # Второй шаг: коррекция (используем производную в предсказанной точке)
    k2 = func(t + h, y_predict)
    # Усредняем два наклона для повышения точности
    return y + 0.5 * h * (k1 + k2)


def integrate(
    func: VectorFunc,
    t0: float,
    y0: np.ndarray,
    h: float,
    steps: int,
    stepper: Callable[[VectorFunc, float, np.ndarray, float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Интегрирует систему ОДУ от начального момента времени t0 на заданное число шагов.
    
    Args:
        func: Функция правой части системы ОДУ f(t, y)
        t0: Начальное время
        y0: Начальное состояние системы (вектор)
        h: Шаг интегрирования
        steps: Количество шагов интегрирования
        stepper: Функция, выполняющая один шаг интегрирования (rk4_step или modified_euler_step)
    
    Returns:
        Кортеж (ts, ys), где:
        - ts: массив временных точек
        - ys: массив состояний системы в каждой временной точке
    """
    # Инициализация массивов для хранения результатов
    ts = np.empty(steps + 1)  # Временные точки (включая начальную)
    ys = np.empty((steps + 1, y0.size))  # Состояния системы (включая начальное)
    ts[0] = t0
    ys[0] = y0
    t = t0
    y = y0.copy()  # Копируем начальное состояние, чтобы не изменять исходное
    # Последовательное интегрирование на каждом шаге
    for i in range(1, steps + 1):
        y = stepper(func, t, y, h)  # Выполняем один шаг интегрирования
        t = t0 + i * h  # Обновляем время
        ts[i] = t
        ys[i] = y
    return ts, ys


def harmonic_oscillator(omega: float) -> VectorFunc:
    """
    Создает функцию правой части для линейного гармонического осциллятора.
    
    Уравнение: x'' + omega^2 * x = 0
    Преобразуется в систему первого порядка:
        dx/dt = v
        dv/dt = -omega^2 * x
    
    Args:
        omega: Частота колебаний осциллятора
    
    Returns:
        Функция правой части системы ОДУ
    """
    def system(t: float, state: np.ndarray) -> np.ndarray:
        x, v = state  # x - координата, v - скорость
        dxdt = v  # Производная координаты равна скорости
        dvdt = -omega**2 * x  # Ускорение пропорционально смещению (закон Гука)
        return np.array([dxdt, dvdt])

    return system


def harmonic_exact_solution(t: float, omega: float, x0: float, v0: float) -> Tuple[float, float]:
    """
    Вычисляет точное аналитическое решение линейного гармонического осциллятора.
    
    Решение имеет вид: x(t) = A*cos(omega*t) + B*sin(omega*t)
    где A и B определяются из начальных условий.
    
    Args:
        t: Время
        omega: Частота колебаний
        x0: Начальная координата
        v0: Начальная скорость
    
    Returns:
        Кортеж (x, v) - координата и скорость в момент времени t
    """
    # Коэффициенты разложения по синусам и косинусам
    a = x0  # Коэффициент при cos
    b = v0 / omega if omega != 0 else 0.0  # Коэффициент при sin
    # Точное решение для координаты
    x = a * math.cos(omega * t) + b * math.sin(omega * t)
    # Точное решение для скорости (производная от координаты)
    v = -a * omega * math.sin(omega * t) + b * omega * math.cos(omega * t)
    return x, v


@dataclass
class SolverStats:
    """
    Класс для хранения статистики работы численного метода решения ОДУ.
    
    Attributes:
        method: Название метода (например, "RK4" или "Modified Euler")
        step: Шаг интегрирования
        steps: Количество шагов интегрирования
        max_abs_error: Максимальная абсолютная погрешность по сравнению с точным решением
        rms_error: Среднеквадратичная погрешность (Root Mean Square)
        runtime_ms: Время выполнения в миллисекундах
    """
    method: str
    step: float
    steps: int
    max_abs_error: float
    rms_error: float
    runtime_ms: float


def measure_solver(
    func: VectorFunc,
    exact: Callable[[float], np.ndarray],
    t0: float,
    y0: np.ndarray,
    t_end: float,
    h: float,
    stepper: Callable[[VectorFunc, float, np.ndarray, float], np.ndarray],
    method_name: str,
) -> SolverStats:
    """
    Измеряет производительность численного метода решения ОДУ.
    
    Выполняет интегрирование, измеряет время выполнения и вычисляет погрешности
    путем сравнения с точным решением.
    
    Args:
        func: Функция правой части системы ОДУ
        exact: Функция, возвращающая точное решение в момент времени t
        t0: Начальное время
        y0: Начальное состояние
        t_end: Конечное время интегрирования
        h: Шаг интегрирования
        stepper: Функция одного шага интегрирования
        method_name: Название метода (для статистики)
    
    Returns:
        Объект SolverStats с результатами измерений
    """
    # Вычисляем количество шагов
    steps = int((t_end - t0) / h)
    # Измеряем время выполнения интегрирования
    start = time.perf_counter()
    ts, ys = integrate(func, t0, y0, h, steps, stepper)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    # Вычисляем точные значения для сравнения
    exact_vals = np.array([exact(t) for t in ts])
    # Вычисляем норму разности между численным и точным решением в каждой точке
    errors = np.linalg.norm(ys - exact_vals, axis=1)
    # Максимальная абсолютная погрешность
    max_abs = float(np.max(errors))
    # Среднеквадратичная погрешность (RMS)
    rms = float(np.sqrt(np.mean(errors**2)))
    return SolverStats(method_name, h, steps, max_abs, rms, runtime_ms)


def lorenz_system(sigma: float, beta: float, rho: float) -> VectorFunc:
    """
    Создает функцию правой части для классической системы Лоренца.

    Уравнения:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Args:
        sigma: Параметр Прандтля (отношение вязкости к теплопроводности)
        beta: Геометрический коэффициент (обычно 8/3)
        rho: Число Рэлея (задает внешний нагрев, регулирует поведение системы)

    Returns:
        Функция правой части системы ОДУ
    """

    def system(t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return np.array([dxdt, dydt, dzdt])

    return system


def run_linear_oscillator_analysis() -> List[SolverStats]:
    """
    Проводит анализ численных методов на примере линейного гармонического осциллятора.
    
    Сравнивает методы Рунге-Кутта 4-го порядка и модифицированный метод Эйлера
    при различных шагах интегрирования. Вычисляет погрешности относительно точного решения.
    
    Returns:
        Список статистик для каждого метода и шага интегрирования
    """
    # Параметры задачи (используем глобальные константы для единообразия)
    omega = LINEAR_PARAMS["omega"]
    x0 = LINEAR_PARAMS["x0"]
    v0 = LINEAR_PARAMS["v0"]
    t0 = LINEAR_PARAMS["t0"]
    t_end = LINEAR_PARAMS["t_end"]
    initial_state = LINEAR_INITIAL_STATE.copy()
    system = harmonic_oscillator(omega)

    # Функция точного решения (для вычисления погрешностей)
    def exact_vec(t: float) -> np.ndarray:
        x, v = harmonic_exact_solution(t, omega, x0, v0)
        return np.array([x, v])

    # Тестируем оба метода на разных шагах интегрирования
    stats = []
    for h in LINEAR_STEPS:
        # Модифицированный метод Эйлера
        stats.append(
            measure_solver(system, exact_vec, t0, initial_state, t_end, h, modified_euler_step, "Modified Euler")
        )
        # Метод Рунге-Кутта 4-го порядка
        stats.append(measure_solver(system, exact_vec, t0, initial_state, t_end, h, rk4_step, "RK4"))
    return stats


def run_lorenz_analysis() -> Tuple[dict, dict]:
    """
    Проводит анализ численных методов на примере системы Лоренца.

    Поскольку точное решение неизвестно, сравниваются траектории, полученные разными методами.
    Исследуется влияние шага интегрирования и метода на фазовый портрет (проекция x-z).
    Первые LORENZ_TRANSIENT единиц времени отбрасываются для исключения переходного процесса.

    Returns:
        Кортеж (results, phase_diffs), где:
        - results: словарь с результатами интегрирования для каждого метода и шага
        - phase_diffs: словарь с разницами между траекториями методов при одинаковом шаге
    """
    system = lorenz_system(**LORENZ_BASE_PARAMS)
    t0 = LORENZ_T0
    t_end = LORENZ_T_END
    transient = LORENZ_TRANSIENT
    initial_state = LORENZ_INITIAL_STATE.copy()

    results = {}
    for h in LORENZ_STEPS:
        steps = int((t_end - t0) / h)
        for method_name, stepper in (("Modified Euler", modified_euler_step), ("RK4", rk4_step)):
            start = time.perf_counter()
            ts, ys = integrate(system, t0, initial_state, h, steps, stepper)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            mask = ts >= transient
            ts_trim = ts[mask]
            ys_trim = ys[mask]

            results[(method_name, h)] = {
                "times": ts_trim,
                "states": ys_trim,
                "runtime_ms": runtime_ms,
            }

    phase_diffs = {}
    for h in LORENZ_STEPS:
        me_key = ("Modified Euler", h)
        rk_key = ("RK4", h)
        me_states = results[me_key]["states"]
        rk_states = results[rk_key]["states"]
        min_len = min(len(me_states), len(rk_states))
        diff = np.linalg.norm(me_states[:min_len] - rk_states[:min_len], axis=1)
        phase_diffs[h] = {
            "mean": float(np.mean(diff)),
            "max": float(np.max(diff)),
        }
    return results, phase_diffs


def run_lorenz_scenarios() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Запускает численное интегрирование системы Лоренца для различных наборов параметров,
    демонстрирующих смену характера поведения.

    Returns:
        Словарь вида {name: {"times": ts_trim, "states": ys_trim, "params": params, "description": str}}
    """
    scenario_results: Dict[str, Dict[str, np.ndarray]] = {}
    h = LORENZ_SCENARIO_STEP
    steps = int((LORENZ_SCENARIO_T_END - LORENZ_T0) / h)

    for scenario in LORENZ_SCENARIOS:
        params = scenario["params"]
        system = lorenz_system(**params)
        ts, ys = integrate(system, LORENZ_T0, LORENZ_INITIAL_STATE.copy(), h, steps, rk4_step)
        mask = ts >= LORENZ_TRANSIENT
        scenario_results[scenario["name"]] = {
            "times": ts[mask],
            "states": ys[mask],
            "params": params,
            "description": scenario["description"],
        }
    return scenario_results


def plot_linear_error_summary(stats: List[SolverStats]) -> None:
    """
    Строит логарифмический график зависимости погрешности от шага интегрирования
    для линейного осциллятора.
    """
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    grouped: dict[str, List[SolverStats]] = {}
    for stat in stats:
        grouped.setdefault(stat.method, []).append(stat)

    for method, method_stats in grouped.items():
        ordered = sorted(method_stats, key=lambda s: s.step, reverse=True)
        steps = [s.step for s in ordered]
        max_errors = [s.max_abs_error for s in ordered]
        rms_errors = [s.rms_error for s in ordered]
        ax.loglog(steps, max_errors, marker="o", label=f"{method} max|error|")
        ax.loglog(steps, rms_errors, marker="s", linestyle="--", label=f"{method} rms_error")

    ax.set_xlabel("Шаг интегрирования h")
    ax.set_ylabel("Погрешность")
    ax.set_title("Зависимость погрешности от шага (линейный осциллятор)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.invert_xaxis()


def plot_linear_time_series(selected_steps: Tuple[float, ...] = (0.2, 0.05)) -> None:
    """
    Строит графики x(t) для линейного осциллятора и сравнивает численное решение с точным.
    """
    if plt is None:
        return

    omega = LINEAR_PARAMS["omega"]
    t0 = LINEAR_PARAMS["t0"]
    t_end = LINEAR_PARAMS["t_end"]
    x0 = LINEAR_PARAMS["x0"]
    v0 = LINEAR_PARAMS["v0"]
    system = harmonic_oscillator(omega)
    methods = (
        ("Modified Euler", modified_euler_step),
        ("RK4", rk4_step),
    )

    fig, axes = plt.subplots(len(selected_steps), 1, sharex=True, figsize=(9, 3 * len(selected_steps)))
    if isinstance(axes, np.ndarray):
        ax_array = axes
    else:
        ax_array = np.array([axes])

    for ax, h in zip(ax_array, selected_steps):
        method_results = []
        steps = int((t_end - t0) / h)
        for method_name, stepper in methods:
            ts, ys = integrate(system, t0, LINEAR_INITIAL_STATE.copy(), h, steps, stepper)
            method_results.append((method_name, ts, ys[:, 0]))

        reference_ts = method_results[0][1]
        exact_values = np.array([harmonic_exact_solution(t, omega, x0, v0)[0] for t in reference_ts])
        ax.plot(reference_ts, exact_values, label="Exact", color="black", linewidth=1.2)

        for method_name, ts, x_vals in method_results:
            ax.plot(ts, x_vals, label=method_name)

        ax.set_title(f"Линейный осциллятор, h = {h}")
        ax.set_ylabel("x(t)")
        ax.grid(True, alpha=0.3)

    ax_array[-1].set_xlabel("t")
    handles, labels = ax_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.98))


def plot_lorenz_phase_portraits(lorenz_results: dict) -> None:
    """
    Строит фазовые портреты (проекция x-z) для системы Лоренца
    при разных методах интегрирования и шагах.
    """
    if plt is None:
        return

    methods = ("Modified Euler", "RK4")
    rows = len(LORENZ_STEPS)
    cols = len(methods)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), sharex=False, sharey=False)

    if rows == 1:
        axes = np.array([axes])

    for row, h in enumerate(LORENZ_STEPS):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            states = lorenz_results[(method, h)]["states"]
            if len(states) > 20000:
                states = states[::5]
            ax.plot(states[:, 0], states[:, 2], linewidth=0.7)
            ax.set_title(f"{method}, h = {h}")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("z")
            if row == rows - 1:
                ax.set_xlabel("x")

    fig.suptitle("Фазовые портреты системы Лоренца (проекция x-z)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))


def plot_lorenz_scenario_comparison(scenario_results: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Отображает временные ряды и фазовые портреты (проекция x-z) для разных параметров системы Лоренца.
    """
    if plt is None:
        return

    scenario_names = list(scenario_results.keys())
    rows = len(scenario_names)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows), sharex=False)
    if rows == 1:
        axes = np.array([axes])

    for idx, name in enumerate(scenario_names):
        data = scenario_results[name]
        ts = data["times"]
        states = data["states"]
        description = data["description"]
        params = data["params"]

        ax_time = axes[idx, 0]
        ax_phase = axes[idx, 1]

        ax_time.plot(ts, states[:, 0], linewidth=0.9)
        ax_time.set_ylabel("x(t)")
        ax_time.set_title(f"{name}\n{description}")
        ax_time.grid(True, alpha=0.3)
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        ax_time.text(
            0.02,
            0.95,
            param_str,
            transform=ax_time.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

        sample_states = states if len(states) <= 20000 else states[::5]
        ax_phase.plot(sample_states[:, 0], sample_states[:, 2], linewidth=0.7)
        ax_phase.set_xlabel("x")
        ax_phase.set_ylabel("z")
        ax_phase.grid(True, alpha=0.3)
        ax_phase.set_title("Фазовый портрет (x-z)")

    axes[-1, 0].set_xlabel("t")
    fig.suptitle(
        "Изменение поведения системы Лоренца при вариации параметра rho",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))


def main() -> None:
    """
    Главная функция программы.
    
    Выполняет анализ численных методов на двух системах:
    1. Линейный осциллятор - сравнение с точным решением
    2. Система Лоренца - сравнение методов между собой
    
    Выводит результаты в виде таблиц.
    """
    # Анализ линейного осциллятора
    oscillator_stats = run_linear_oscillator_analysis()
    print("Линейный осциллятор (x'' + omega^2 x = 0, omega = 1.5):")
    # Заголовок таблицы
    print("{:<15} {:>6} {:>8} {:>14} {:>14} {:>12}".format("Метод", "h", "шагов", "max|error|", "rms_error", "time, ms"))
    # Вывод результатов для каждого метода и шага
    for s in oscillator_stats:
        print(
            "{:<15} {:>6.3f} {:>8} {:>14.6e} {:>14.6e} {:>12.3f}".format(
                s.method, s.step, s.steps, s.max_abs_error, s.rms_error, s.runtime_ms
            )
        )

    # Анализ системы Лоренца
    lorenz_results, phase_diffs = run_lorenz_analysis()
    print("\nНелинейная система (уравнения Лоренца):")
    print("{:<15} {:>6} {:>12}".format("Метод", "h", "time, ms"))
    # Вывод времени выполнения для каждого метода и шага
    for (method, h), data in lorenz_results.items():
        print("{:<15} {:>6.3f} {:>12.3f}".format(method, h, data["runtime_ms"]))

    # Сравнение траекторий, полученных разными методами
    print("\nСредняя и максимальная разница траекторий между методами при одинаковом шаге:")
    print("{:>6} {:>16} {:>16}".format("h", "mean_diff", "max_diff"))
    for h, diff_stats in phase_diffs.items():
        print("{:>6.3f} {:>16.6e} {:>16.6e}".format(h, diff_stats["mean"], diff_stats["max"]))

    scenario_results = run_lorenz_scenarios()
    print("\nСистема Лоренца: смена режима поведения при варьировании rho:")
    for name, data in scenario_results.items():
        params = data["params"]
        description = data["description"]
        param_str = ", ".join(f"{key}={value}" for key, value in params.items())
        print(f"- {name}: {description} ({param_str})")

    if plt is None:
        print("\nВизуализация недоступна: библиотека matplotlib не установлена.")
        return

    plot_linear_error_summary(oscillator_stats)
    plot_linear_time_series()
    plot_lorenz_phase_portraits(lorenz_results)
    plot_lorenz_scenario_comparison(scenario_results)
    plt.show()


if __name__ == "__main__":
    main()

