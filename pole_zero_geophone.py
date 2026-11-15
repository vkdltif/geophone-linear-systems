import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

# JF-20DX parameters from Table 1 of Ma et al. (Sensors 2023)
G = 27.3                      # V/(m/s)  (273e-3 V/(cm/s))
f0 = 10.0                     # Hz
gamma0 = 0.3                  # open-circuit damping ratio
w0 = 2*np.pi*f0               # rad/s

num = [-G, 0.0, 0.0]                          # -G s^2
den = [1.0, 2*gamma0*w0, w0**2]               # s^2 + 2*gamma0*w0*s + w0^2
zeros = np.roots(num[:1] + [0, 0])            # double zero at s = 0
poles = np.roots(den)

alpha = gamma0*w0
wd = w0*np.sqrt(1-gamma0**2)
print("w0 =", w0, " alpha =", alpha, " wd =", wd)
print("poles =", poles, " |p| =", abs(poles[0]))
print("den coeffs =", den)

# h(t) = -G*delta(t) + G*exp(-alpha t)[A cos(wd t) + B sin(wd t)]
A = 2*alpha
B = (w0**2 - 2*alpha**2)/wd
print("A =", A, " B =", B, " sqrt(A^2+B^2) =", np.hypot(A, B))
reg = lambda t: G*np.exp(-alpha*t)*(A*np.cos(wd*t)+B*np.sin(wd*t))
I_reg = quad(lambda t: abs(reg(t)), 0, 2, limit=500)[0]
print("int |regular part| dt =", I_reg, " total with delta weight =", I_reg+G)
print("upper bound =", G + G*np.hypot(A,B)/alpha)

plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":9,"mathtext.fontset":"stix"})
fig, ax = plt.subplots(figsize=(3.5, 3.3))
ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
ax.fill_between([-95, 0], -85, 85, color="0.92", zorder=0)
ax.plot(poles.real, poles.imag, "x", color="k", ms=8, mew=1.6, label="Poles")
ax.plot([0], [0], "o", mfc="none", mec="k", ms=8, mew=1.3, label="Double zero")
ax.annotate("(2)", (0, 0), xytext=(6, 7), textcoords="offset points", fontsize=8)
ax.annotate(r"$-18.85 + j59.94$", (poles[0].real, abs(poles[0].imag)), xytext=(-8, -3), textcoords="offset points", fontsize=8, ha="right")
ax.annotate(r"$-18.85 - j59.94$", (poles[1].real, -abs(poles[1].imag)), xytext=(-8, -3), textcoords="offset points", fontsize=8, ha="right")
ax.set_xlim(-95, 40); ax.set_ylim(-85, 85)
ax.set_xlabel(r"$\sigma$ (rad/s)"); ax.set_ylabel(r"$j\omega$ (rad/s)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8, numpoints=1)
ax.text(-90, 76, "stable region (Re $s<0$)", fontsize=7.5, color="0.35")
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig_pole_zero.png", dpi=300)
