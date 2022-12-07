import numpy as np
import matplotlib.pyplot as plt

# Runge Kutta 4th order method
def rk4(f, x, t, h, N, beta, gamma, alpha, k, miu, ro, a, b):
    k1 = h * f(x, t, N, beta, gamma, alpha, k, miu, ro, a, b)
    k2 = h * f(x + 0.5 * k1, t + 0.5 * h, N, beta, gamma, alpha, k, miu, ro, a, b)
    k3 = h * f(x + 0.5 * k2, t + 0.5 * h, N, beta, gamma, alpha, k, miu, ro, a, b)
    k4 = h * f(x + k3, t + h, N, beta, gamma, alpha, k, miu, ro, a, b)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# SIR model differential equations.
def deriv(y, t, N, beta, gamma, alpha, k, miu, ro, a, b):
    S, I, R = y
    dSdt = alpha - beta * S - (1 + a * I) * k * I * S / (1 + b * I ** 2) + gamma * R
    dIdt = (1 + a * I) * ro * k * I * S / (1 + b * I ** 2) - (beta + miu) * I
    dRdt = miu * I - (beta + gamma) * R + (1 + a * I) * (1 - ro) * k * I * S / (1 + b * I ** 2)
    return np.array([dSdt, dIdt, dRdt])

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 100, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
alpha = 0.9
beta = 0.82
k = 2.11
miu = 0.11
ro = 0.4
a = 0.2
b = 0.4
gamma = 0.001

ro_list = np.linspace(0.1, 0.9, 9)
for p in ro_list:
    R_0 = p * alpha * k / (beta * (beta + miu))
    print(f'ro = {round(p, 1)}, R_0 = {R_0}')

# A grid of time points (in days)
t = np.linspace(0, 100, 2000)

# Initial conditions vector
y0 = np.array([S0, I0, R0])
# Integrate the SIR equations over the time grid, t.
ret = np.array([y0])
for i in range(len(t) - 1):
    ret = np.vstack((ret, rk4(deriv, ret[-1], t[i], t[i + 1] - t[i], N, beta, gamma, alpha, k, miu, ro, a, b)))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number of People')
ax.set_ylim(0,200)
# Limit the x axis range
ax.set_xlim(-0.5, 20)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
