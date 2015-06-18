import numpy as np
predicted = np.loadtxt('train-pred.txt')
actual = np.loadtxt('train-true.txt')

from matplotlib import pyplot as plt
plt.plot(predicted.T[:, 0][:-1], label='RNN Next Day Prediction')
plt.plot(actual.T[:, 0][:-1], label='MSFT Open')
plt.title('RNN Train Set')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

predicted = np.loadtxt('test-pred.txt')
actual = np.loadtxt('test-true.txt')

from matplotlib import pyplot as plt
plt.plot(predicted.T[:, 0], label='RNN Next Day Prediction')
plt.plot(actual.T[:, 0], label='MSFT Open')
plt.title('RNN Test Set')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()