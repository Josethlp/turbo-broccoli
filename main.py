'''
Entrenamiento de Goku en la cámara del tiempo.

Este script utiliza la regresión lineal con objeto de estimar el tiempo
necesario para aumentar el nivel de ki de Goku a 120.000,
basándose en datos de entrenamiento.

Requiere:
- Archivo 'entrenamiento.csv' con datos de ki y tiempo.
- Bibliotecas: matplotlib, pandas, sklearn.

'''

# Importa las bibliotecas necesarias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Lee el archivo 'entrenamiento.csv'
entrenamiento = pd.read_csv('entrenamiento.csv')

# Extrae los valores de las columnas 'ki' y 'tiempo'
x = entrenamiento[['ki']].values  # Variable independiente: ki
y = entrenamiento[['tiempo']].values  # Variable dependiente: tiempo

# Gráfica de la relación entre las variables ki y tiempo
entrenamiento.plot(kind='scatter', grid=True, x='ki', y='tiempo')
# plt.show()  # Comentar o descomentar según se desee visualizar la gráfica

# Crea un modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(x, y)

# Precdice el tiempo necesario para alcanzar un ki de 120.000
ki_nuevo = [[120000]]
tiempo_predicho = modelo.predict(ki_nuevo)
print(tiempo_predicho)
