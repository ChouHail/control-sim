from common import *
from control_math import *
from numpy import pi
from scipy.signal import ss2tf, tf2ss

"""Plant"""
m = 1
k = 0
b = 0

A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([1, 0])
D = np.array([0])
rigid_plant_num, rigid_plant_den = ss2tf(A, B, C, D)
#
res_freq = 200
anti_res_freq = 150
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

beta1 = 0.1
beta2 = 0.1
res_num = (
    res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2])
)
res_den = np.array([1, 2 * beta2 * res_omega, res_omega ** 2, 0, 0])

num = np.convolve(res_num, rigid_plant_num.flatten())
den = np.convolve(res_den, rigid_plant_den.flatten())

# num = res_num
# den = res_den
A, B, C, D = tf2ss(res_num, res_den)
ss = StateSpaceModel(A, B, C, D, DT)

"""Chirp参数"""
start_freq = 50
end_freq = 5000
start_freq_ = 0.8 * start_freq
end_freq_ = 1.1 * end_freq
# 扫频时间
T = 1
T = (
    int(((end_freq_ - start_freq_) / (end_freq - start_freq) * T) * SERVO_FREQ)
    / SERVO_FREQ
)

t = np.linspace(0, T, int(SERVO_FREQ * T) + 1)
u = np.sin(2 * np.pi * ((end_freq_ - start_freq_) / T * t ** 2 / 2 + start_freq_ * t))

# u = np.linspace(0, 1, len(t4dyn))
# u = np.sin(2 * pi * 500 * t4dyn)

y = np.zeros_like(u)
for i in range(len(u)):
    input_sig = u[i]
    y_output, x_state = ss.response(input_sig, method="zoh")
    y[i] = y_output
p = y
y = np.diff(y, 2) / DT / DT
y = np.pad(y, (2, 0), "constant", constant_values=(0, 0))


u_detrend = u - np.mean(u)
y_detrend = y - np.mean(y)

f_u, fw_u = fft(hamm(u_detrend), DT)
f_y, fw_y = fft(hamm(y_detrend), DT)

resolution = 1 / DT / len(f_u)
start_point = int(start_freq / resolution)
end_point = int(end_freq / resolution)

f_u = f_u[start_point:end_point]
fw = fw_y[start_point:end_point] / fw_u[start_point:end_point]

dd_decay = -(
    (
        np.e ** (-4 * 1j * f_u * pi * DT)
        * (-1 + np.e ** (2 * 1j * f_u * pi * DT)) ** 2
        * (
            1
            + np.e ** (4 * 1j * f_u * pi * DT)
            - np.e ** (2 * 1j * f_u * pi * DT) * np.e ** (1j * 2 * pi * f_u * DT)
        )
    )
    / (4 * f_u ** 2 * pi ** 2 * DT ** 2)
)
zoh_decay = (1 - np.e ** (-1j * 2 * pi * f_u * DT)) / (1j * 2 * pi * f_u * DT)
linear_decay = (
    np.e ** (1j * 2 * pi * f_u * DT)
    * (1 - np.e ** (-1j * 2 * pi * f_u * DT)) ** 2
    / (1j * 2 * pi * f_u * DT) ** 2
)
fw = fw / zoh_decay / zoh_decay  # / zoh_decay
plt.figure()
plt.plot(t, u, label="u")
plt.legend()
plt.figure()
plt.plot(t, y, label="y")
plt.legend()

plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.xscale("log")
plt.plot(f_u, 20 * np.log10(np.abs(fw)))

plt.subplot(122)
plt.xscale("log")
plt.plot(f_u, np.angle(fw, deg=True))
# dd_decay
# plt.figure(figsize=(14, 4))
# plt.subplot(121)
# plt.xscale("log")
# plt.plot(f_u, 20 * np.log10(np.abs(dd_decay)))
#
# plt.subplot(122)
# plt.xscale("log")
# plt.plot(f_u, np.angle(dd_decay, deg=True))
#
# # dd_decay
# plt.figure(figsize=(14, 4))
# plt.subplot(121)
# plt.xscale("log")
# plt.plot(f_u, 20 * np.log10(np.abs(zoh_decay)))
#
# plt.subplot(122)
# plt.xscale("log")
# plt.plot(f_u, np.angle(zoh_decay, deg=True))
plt.show()
