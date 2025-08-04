import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 100, 50)
y = np.linspace(0, 1e6, 50) + np.random.normal(0, 2e5, 50)

plt.scatter(x, y)
plt.plot(np.linspace(0, 1e6, 100))
plt.xlabel("Percent of attention I pay to Claude Code")
plt.ylabel("Number of bugs I find")
plt.show()
