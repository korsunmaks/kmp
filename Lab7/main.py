import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def function(x):
    return np.abs(x - 2)

search_domain_1 = (-2, 5)

# Finding a localized minimum
result1 = minimize(function, 0, bounds=[search_domain_1])

# print("The minimum of the function is at x =", result1.x[0])
# print("The value of the function at this point =", result1.fun)

def function_v2(x):
    return np.abs(x[0] - x[1]-2)

search_domain_2 = [(-3, 3), (-5, 5)]

# Finding a localized minimum
result2 = minimize(function_v2, [0, 0], bounds=search_domain_2)

# print("The minimum of the function is at x =", result2.x[0], "and y =", result2.x[1])
# print("The value of the function at this point =", result2.fun)



# Graph 1
# x_values = np.linspace(search_domain_1[0], search_domain_1[1], 100)
# y_values = function(x_values)

# plt.plot(x_values, y_values, label='f(x) = |x - 2|')
# plt.scatter(result.x, result.fun, color='red', label='Min')
# plt.axhline(0, color='black',linewidth=1)
# plt.axvline(0, color='black',linewidth=1)
# plt.title('Graph of the function and the minimum')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()
# plt.grid(True)
# plt.show()


# Graph 2

def function(x, y):
    return np.abs(x - y - 2)

x_domain = np.linspace(-3, 3, 100)
y_domain = np.linspace(-5, 5, 100)
x_values, y_values = np.meshgrid(x_domain, y_domain)
z_values = function(x_values, y_values)

# Construction of a three-dimensional graph
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_values, y_values, z_values, cmap='viridis', alpha=0.8, edgecolor='k')

# # Marking the minimum point
# ax.scatter(result2.x[0], result2.x[1], result2.fun, color='red', s=100, label='Мінімум')

# ax.set_title('Graph of the function and the minimum')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('f(X, Y)')
# ax.legend()
# plt.show()



fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# XOZ
ax1 = axes[0]
contour1 = ax1.contour(x_values, y_values, z_values, cmap='viridis', levels=20)
ax1.scatter(result2.x[0], result2.x[1], color='red', label='Min')
ax1.set_title('XOZ')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

# XOY
ax2 = axes[1]
contour2 = ax2.contour(x_values, y_values, z_values, cmap='viridis', levels=20)
ax2.scatter(result2.x[0], result2.x[1], color='red', label='Min')
ax2.set_title('XOY')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend()

# YOZ
ax3 = axes[2]
contour3 = ax3.contour(x_values, y_values, z_values, cmap='viridis', levels=20)
ax3.scatter(result2.x[0], result2.x[1], color='red', label='Min')
ax3.set_title('YOZ')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.legend()


plt.tight_layout()
plt.show()
