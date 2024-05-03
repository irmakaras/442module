import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module import FitFunction1D, plot_data, plot_fit, draw_fit_info, style

plt.style.use(style)

# Import the data
data = pd.read_csv("data/qm-data.txt", sep="\t", header=None)
x = data[0].to_numpy()
y = data[1].to_numpy()
x_err = data[2].to_numpy()
y_err = data[3].to_numpy()

# Example of defining a custom function
def my_linear_function(x, α, β):
    return α*x+β

f = FitFunction1D(my_linear_function)
f.set_expression("α*x+β")
f.fit(x, y, y_err, initial_values=[1, 1])

# Create a plot
fig, axes = plt.subplots(figsize=(7,7))

errorbar_style = dict(linestyle="",
                      elinewidth=1.5,
                      capsize=4,
                      capthick=1.5,
                      marker="o",
                      markersize=8,
                      markeredgecolor="red",
                      markerfacecolor="green",
                      color="magenta",
                      label="Data")
plot_data(x, y, x_err=x_err, y_err=y_err, errorbar_style=errorbar_style)

fit_style = dict(linewidth=3,
                 color="blue",
                 label="Fit")
plot_fit(f, fit_style=fit_style)

draw_fit_info(f, fontsize=12, loc="upper left", offset=(0.03, -0.03), edges=True)
axes.legend(loc="lower right")

axes.set_title("An example of a colorful line fit", pad=10)
axes.set_xlabel("x (Observable 1)")
axes.set_ylabel("y (Observable 2)")

plt.savefig("plots/linefit.pdf", bbox_inches="tight")
plt.show()