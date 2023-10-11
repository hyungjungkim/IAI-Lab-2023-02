import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('gas-production.csv')

X = df[['Por', 'Brittle']].values.reshape(-1,2)
Y = df['Prod']

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(6, 24, 30)
y_pred = np.linspace(0, 100, 30)
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

reg_model = linear_model.LinearRegression().fit(X, Y)
predicted = reg_model.predict(model_viz)

plt.style.use('default')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=10, edgecolor='blue')
ax.set_xlabel('Porosity (%)', fontsize=12)
ax.set_ylabel('Brittleness', fontsize=12)
ax.set_zlabel('Gas Prod. (Mcf/day)', fontsize=12)
ax.locator_params(nbins=4, axis='x')
ax.locator_params(nbins=5, axis='x')

fig.tight_layout()
plt.show()