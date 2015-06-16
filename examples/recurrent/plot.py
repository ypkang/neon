import numpy as np
predicted = np.loadtxt('train-pred.txt')
actual = np.loadtxt('train-true.txt')

from matplotlib import pyplot as plt
plt.plot(predicted.T[:, 0])
plt.plot(actual.T[:, 0])
plt.show()

predicted = np.loadtxt('test-pred.txt')
actual = np.loadtxt('test-true.txt')

from matplotlib import pyplot as plt
plt.plot(predicted.T[:, 0])
plt.plot(actual.T[:, 0])
plt.show()