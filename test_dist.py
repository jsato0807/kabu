import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

x = np.linspace(-10, 10, 1000)
pdf = cauchy.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='Cauchy Distribution')
plt.title('Cauchy Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import laplace

x = np.linspace(-10, 10, 1000)
pdf = laplace.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='Laplace Distribution')
plt.title('Laplace Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import norm

x = np.linspace(-10, 10, 1000)
pdf = norm.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='Normal Distribution')
plt.title('Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import pareto

x = np.linspace(0.1, 10, 1000)
pdf = pareto.pdf(x, 3)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='Pareto Distribution')
plt.title('Pareto Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()



