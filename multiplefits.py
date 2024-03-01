import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module import FitFunction1D, plot_fit, plot_data, draw_fit_info, style

plt.style.use(style)

# ----- Import the data
data = pd.read_csv("data/cav-data.txt", sep="\t", header=None)
data_x = data[0].to_numpy()
data_y = data[1].to_numpy()
data_y_err = 0.0001

# ----- Fit some functions to the data in different ranges
f1 = FitFunction1D("a*exp(b*x)*sin(c*x+d)+e")
f1.fit(data_x, data_y, data_y_err, initial_params=[0.31, -0.002, 0.029, 0.8, 0.001], range=[2600, 4700])

f2 = FitFunction1D("a*sin(c*x+d)+e")
f2.fit(data_x, data_y, data_y_err, initial_params=[0.31, 0.029, 0.8, 0.001], range=[1995,2550])

# ----- Create a plot
fig, ax = plt.subplots(figsize=(12,8))


errorbar_style = dict(label="Data",
                  marker=".",
                  markersize=2,
                  linestyle="",
                  color="black",)

plot_data(data_x, y=data_y, errorbar_style=errorbar_style)


fit1_style = dict(label=       "Fit 1",
                 linestyle=   "-",
                 linewidth=   2,
                 color=       "red")

fit2_style = dict(label=       "Fit 2",
                 linestyle=   "-",
                 linewidth=   2,
                 color=       "green")

plot_fit([f1, f2], fit_style=[fit1_style, fit2_style])


draw_fit_info([f1, f2], edges=True, fontsize=14, loc=(1.01,1))
ax.legend(loc="lower right")

ax.set_title("Cavendish Torsion Balance")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Potential (V)")

plt.savefig("plots/multiplefits", bbox_inches="tight")
plt.show()