import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Определення функції
def function(x):
    return 2 * np.sin(x)

# Визначення області пошуку
x_values = np.linspace(-2*np.pi, 2*np.pi, 1000)

# Знаходження локальних та глобального мінімумів і максимумів
local_minimum = minimize_scalar(function, bounds=(-0.25*np.pi, 0.25 *np.pi), method='bounded')
global_minimum = minimize_scalar(function, bounds=(-10, 10), method='bounded')
local_maximum = minimize_scalar(lambda x: -function(x), bounds=(-0.25*np.pi, 0.25 *np.pi), method='bounded')
global_maximum = minimize_scalar(lambda x: -function(x), bounds=(-10, 10), method='bounded')

# Зміщення значення глобального мінімуму та максимуму
global_minimum_value = global_minimum.fun
global_maximum_value = -global_maximum.fun

# Побудова графіку
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_values, function(x_values), label='Function', linewidth=2)
ax.scatter(local_minimum.x, local_minimum.fun, color='red', marker='o', s=100, label=f'Local Min\n({local_minimum.x:.2f}, {local_minimum.fun:.2f})')
ax.scatter(global_minimum.x, global_minimum_value, color='green', marker='o', s=100, label=f'Global Min\n({global_minimum.x:.2f}, {global_minimum_value:.2f})')
ax.scatter(local_maximum.x, -local_maximum.fun, color='blue', marker='o', s=100, label=f'Local Max\n({local_maximum.x:.2f}, {-local_maximum.fun:.2f})')
ax.scatter(global_maximum.x, global_maximum_value, color='purple', marker='o', s=100, label=f'Global Max\n({global_maximum.x:.2f}, {global_maximum_value:.2f})')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.tick_params(axis='both', which='both', length=5, width=1.5)
plt.title('Graph of the f(x)=2sin(x) and the Extreme points')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
