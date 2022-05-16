from math import pi, exp
import numpy as np
import matplotlib.pyplot as plt

count = 2

Q = 2
h = 20 * count
v = 4 * count
tvert = 2
U = 15
h_neriv = 2
N = 1000
sigma_y = 0.005
sigma_z = 0.005
A = Q / (pi * sigma_y * sigma_z * U ** 3)
H = h + tvert * v
K = 10 ** 6


def fi(x, y, H):
    return A * exp(-(y ** 2) / (2 * sigma_y * U * x) - (H ** 2) / (2 * sigma_z * U * x)) / (x ** 2)


x = np.linspace(10, 15000, N)
y0 = np.zeros(N)
y1 = np.linspace(-150, 150, N)
gauss_x = [fi(x[i], y0[i], H) for i in range(N)]
max_g = max(gauss_x)
max_g_ind = np.argmax(gauss_x)
x_max = x[max_g_ind]

gauss_y = [fi(x_max, y1[i], H) for i in range(N)]

fig1, ax1 = plt.subplots()
ax1.set_xlabel('х')
ax1.set_ylabel('C(x,0)')
ax1.plot(x, gauss_x, color="blue", label='fi(x,0)')
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.set_xlabel('y')
ax2.set_ylabel('C(xmax,y)')
ax2.plot(y1, gauss_y, color='red', label='fi(xmax,y)')
ax2.legend()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
fig3.suptitle('Gauss Dispersion at xmax')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('fi')
ax3.plot(x, y0, gauss_x, color='blue', label='концентрація токсинів від джерела по х')
ax3.plot([x_max for i in range(N)], y1, gauss_y, color='red', label='розсіювання токсинів по y при Cmax')
ax3.scatter(x_max, y0[0], max_g, color='green', label='xmax,0,Cmax')
print(x_max, y0[0], max_g)
ax3.legend()
plt.show()

# count = 1
#2605.865865865866 0.0 1.4951480467718294e-07
# count = 2
#10453.483483483484 0.0 9.34475188177006e-09