import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add your code below!

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('2 component PCA example')

targets = ['4.9', '4.8', '4.7']
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indicesToKeep = (final_data['User Rating'] == float(target))

    ax.scatter(final_data.loc[indicesToKeep, 'principal component 1'], 
               final_data.loc[indicesToKeep, 'principal component 2'],
               c = color, s = 50)

ax.grid()
plt.show()