import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adalinegd import AdalineGD

# get the iris data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',
	header = None)

# Plot 100 samples of the data
y = df.iloc[0:100, 3].values
y = np.where(y == 2, -1, 1)
X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = '<5 de años')
plt.scatter(X[60:100, 0], X[60:100, 1], color = 'blue', marker = 'x', label = '>5 años')
plt.xlabel('N° nodos axilares positivos')
plt.ylabel('Edad del paciente')
plt.legend(loc = 'upper left')
plt.show()

# Standardize the data
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Create the AdalineGD model
model1 = AdalineGD(n_iter = 50, eta = 0.01)

# Train the model
model1.fit(X_std, y)

# Plot the training error
plt.plot(range(1, len(model1.cost_) + 1), model1.cost_, marker = 'o', color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

