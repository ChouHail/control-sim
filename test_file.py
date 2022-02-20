import numpy as np
import scipy.signal

from common import *
from control_math import *
from numpy import pi
from scipy.signal import ss2tf, tf2ss, impulse, impulse2, step, butter, lsim, lsim2
from sympy import solve
from sympy.abc import a, b, c
from scipy.linalg import toeplitz

"""Plant"""
m = 0.03
k = 0
bb = 0

A = np.array([[0, 1], [-k / m, -bb / m]])
B = np.array([[0], [1 / m]])
C = np.array([1, 0])
D = np.array([0])
# rigid_plant_num, rigid_plant_den = ss2tf(A, B, C, D)
#
res_freq = 200
anti_res_freq = 150
res_omega = res_freq * 2 * np.pi
anti_res_omega = anti_res_freq * 2 * np.pi

mass = 0.4
beta1 = 0.1
beta2 = 0.1
res_num = (
    1
    / mass
    * res_omega ** 2
    / anti_res_omega ** 2
    * np.array([1, 2 * beta1 * anti_res_omega, anti_res_omega ** 2])
)
res_den = np.array([1, 2 * beta2 * res_omega, res_omega ** 2, 0, 0])

# num = np.convolve(res_num, rigid_plant_num.flatten())
# den = np.convolve(res_den, rigid_plant_den.flatten())

num = res_num
den = res_den
# A, B, C, D = tf2ss(num, den)
sys_ss = StateSpaceModel(A, B, C, D, DT)
sys_tf = TransferFunc(num, den, DT)

"""Chirp信号初始化"""
start_freq = 5
end_freq = 5000
start_freq_ = 0.8 * start_freq
end_freq_ = 1.1 * end_freq
# 扫频时间
T4chirp = 1
T4chirp = (
    int(((end_freq_ - start_freq_) / (end_freq - start_freq) * T4chirp) * SERVO_FREQ)
    / SERVO_FREQ
)

t4dyn = np.linspace(0, T4chirp, int(SERVO_FREQ * T4chirp) + 1)
u = np.sin(
    2
    * np.pi
    * ((end_freq_ - start_freq_) / T4chirp * t4dyn ** 2 / 2 + start_freq_ * t4dyn)
)

# u = np.linspace(0, 1, len(t4dyn))
# u = np.sin(2 * pi * 500 * t4dyn)

"""计算Chirp响应"""
y = np.zeros_like(u)
for i in range(len(u)):
    input_sig = u[i]
    y_output, x_state = sys_ss.response(input_sig, method="zoh2")
    y[i] = y_output
y = np.diff(y, 2) / DT / DT
y = np.pad(y, (2, 0), "constant", constant_values=(0, 0))


u_detrend = u - np.mean(u)
y_detrend = y - np.mean(y)

f_bode, fw_u = fft(hamm(u_detrend), DT)
f_y, fw_y = fft(hamm(y_detrend), DT)

resolution = 1 / DT / len(f_bode)
start_point = int(start_freq / resolution)
end_point = int(end_freq / resolution)

f_bode = f_bode[start_point:end_point]
fw = fw_y[start_point:end_point] / fw_u[start_point:end_point]

"""修正差分和离散化方法造成的伯德图误差"""
dd_decay = -(
    (
        np.e ** (-4 * 1j * f_bode * pi * DT)
        * (-1 + np.e ** (2 * 1j * f_bode * pi * DT)) ** 2
        * (
            1
            + np.e ** (4 * 1j * f_bode * pi * DT)
            - np.e ** (2 * 1j * f_bode * pi * DT) * np.e ** (1j * 2 * pi * f_bode * DT)
        )
    )
    / (4 * f_bode ** 2 * pi ** 2 * DT ** 2)
)
zoh_decay = (1 - np.e ** (-1j * 2 * pi * f_bode * DT)) / (1j * 2 * pi * f_bode * DT)
linear_decay = (
    np.e ** (1j * 2 * pi * f_bode * DT)
    * (1 - np.e ** (-1j * 2 * pi * f_bode * DT)) ** 2
    / (1j * 2 * pi * f_bode * DT) ** 2
)
fw = fw / dd_decay / zoh_decay

# 输入和加速度时域响应
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].set_xlabel("t4dyn/[s]")
axes[0].set_ylabel("u/[N]")
axes[0].plot(t4dyn, u, label="u")

axes[1].set_xlabel("t4dyn/[s]")
axes[1].set_ylabel("acc/[m/s**2]")
axes[1].plot(t4dyn, y, label="acc")
plt.suptitle("Input and Acc")
plt.legend()

# 修正后和拟合的伯德图
# f_fit, fw_fit = sys_fit.bode(np.array(range(start_freq, end_freq)))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].set_xlabel("f/[Hz]")
axes[0].set_ylabel("Gain/[dB]")
axes[0].semilogx(f_bode, 20 * np.log10(np.abs(fw)), label="Gain")
# axes[0].semilogx(f_fit, 20 * np.log10(np.abs(fw_fit)), label="Fit Gain")

axes[1].set_xlabel("f/[Hz]")
axes[1].set_ylabel("Phase/[deg]")
axes[1].semilogx(f_bode, np.angle(fw, deg=True), label="Phase")
# axes[1].semilogx(f_fit, np.angle(fw_fit, deg=True), label="Fit Phase")
plt.suptitle("sys id bode plot")
plt.legend()

plt.show()
