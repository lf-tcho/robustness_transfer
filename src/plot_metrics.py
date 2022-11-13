from numpy import genfromtxt
import matplotlib.pyplot as plt
from pathlib import Path

path_lp = Path(
    r"C:\Users\mauri\Documents\Stanford\dmtml\project\dmtml\experiments\lp_bs_128_eps_10_lr_0.001\logs\lp_bs_128_eps_10_lr_0.001_logs.csv"
)
path_ff = Path(
    r"C:\Users\mauri\Documents\Stanford\dmtml\project\dmtml\experiments\lp_bs_128_eps_10_lr_0.001_lp_0\logs\lp_bs_128_eps_10_lr_0.001_lp_0_logs.csv"
)
ff = genfromtxt(path_ff, delimiter=",", names=True)
lp = genfromtxt(path_lp, delimiter=",", names=True)

fig, ax = plt.subplots()
ax.plot(ff["Step"], ff["Value"], marker="o", label="Full fine-tuning")
ax.set_xlabel("Iteration")
ax.set_ylabel("Test accuracy")
ax.plot(lp["Step"], lp["Value"], marker="o", label="Linear probing")
ax.legend(loc="center right")
plt.show()
plt.savefig('test_accuracy.png')
