import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
y1 = [0.49, 0.215, 0.215, 0.14, 0.125, 0.085, 0.08, 0.07, 0.07]
y2 = [0.4020618556701031, 0.27835051546391754, 0.32989690721649484, 0.17525773195876287,
             0.25773195876288657, 0.25773195876288657, 0.24742268041237114, 0.25773195876288657,
             0.25773195876288657]

plt.plot(x, y1, label="Train Error")
plt.plot(x, y2, label="Test Error")
plt.legend()
plt.ylabel('Error rate')
plt.xlabel('Depth of the tree')
plt.savefig('heart_error_plot.png')

