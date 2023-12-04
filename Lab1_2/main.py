import numpy as np
import matplotlib.pyplot as plt

# Параметри
m = 1 # Маса тіла (кг)
g = 9.8  # Прискорення вільного падіння (м/с^2)
h = 20  # Висота (м)
k = 0.1  # Коефіцієнт опору середовища

# Задаємо діапазон швидкостей від 10м/с до 20м/с 
v_values = np.arange(10, 21, 1)

# Обчислення енергії за формулою
energy_values = m * g * h - k * v_values**2

# Побудова графіка, гістограми та таблички
plt.figure(figsize=(10, 6))

# Графік залежності
plt.subplot(2, 2, 1)
plt.plot(v_values, energy_values, linestyle='-', label='Енергія')
plt.title('Залежність енергії від швидкості')
plt.xlabel('Швидкість (м/с)')
plt.ylabel('Енергія (Дж)')
plt.legend()

# Табличка
plt.subplot(2,2,2)
table_data = np.column_stack((v_values, energy_values))
plt.axis('off')
plt.table(cellText=table_data, colLabels=['Швидкість (м/с)', 'Енергія (Дж)'], cellLoc='center', loc='center')

# Гістограма
plt.subplot(2, 2, 3)
plt.bar(v_values,energy_values,width=1,edgecolor='black')
plt.title('Гістограма')
plt.xlabel('Швидкість (м/с)')
plt.ylabel('Енергія (Дж)')
plt.xticks(np.arange(10, 21, 1))


plt.tight_layout()
plt.show()
