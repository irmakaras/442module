import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module import FitFunction1D, plot_data, plot_fit, draw_fit_info, style

plt.style.use(style)

# Generate a random dataset from a normal distribution
data = np.random.normal(365, 45, 1000)
bins = 25

# Fit a gaussian function to the data
f = FitFunction1D("gaussian")
f.fit(data, initial_values=[5,350,30], histogram=[bins]) # Here, the order of parameters are [A, μ, σ]

# Create a plot
fig, ax = plt.subplots(figsize=(9,8))

histogram_style = dict(fill=True,
                       label="Data",
                       color="green")

errorbar_style = dict(marker= "",
                    capsize=    3,
                    color=      "red",
                    linestyle=  "",
                    label=      "Error",
                    elinewidth= 2)
plot_data(data, bins=bins, errorbar_style=errorbar_style, histogram_style=histogram_style)

fit_style = dict(label=       "Fit",
                 linestyle=   "-",
                 linewidth=   3,
                 color=       "orange")
plot_fit(f, fit_style=fit_style)

draw_fit_info(f, loc="upper right", offset=(0.01, 0.01))
ax.legend(loc="upper left")

ax.set_title("I CAN DO HISTOGRAMS TOO!", pad=15)
ax.set_xlabel("x (Observable)")
ax.set_ylabel("y (Counts)")

plt.savefig("plots/gaussian.pdf", bbox_inches="tight")
plt.show()