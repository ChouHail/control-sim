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

# A = np.array([[0, 1], [-k / m, -bb / m]])
# B = np.array([[0], [1 / m]])
# C = np.array([1, 0])
# D = np.array([0])
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
A, B, C, D = tf2ss(num, den)
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
p = y
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

# 拟合传递函数
Bn, Am = fit(f_bode, fw, 2, 2)
sys_fit = TransferFunc(Bn, Am, DT)
print("num:", num, "den:", den[:-2])
print("Bn:", Bn, "Am:", Am)

"""设计PID"""
f_fit4pid, fw_fit4pid = sys_fit.bode(np.array(range(1, 500)))
plant_gain = np.mean(20 * np.log10(np.abs(fw_fit4pid)))
kp, ki, kd = pole_placement(plant_gain, 100, 0, SERVO_FREQ)
# kp = 1000
# ki = 0.1
# kd = 1000
print(kp, ki, kd)
pid_controller = PID(kp, ki, kd, SERVO_FREQ)
pid_tf = TransferFunc([kd, kp, ki], [1, 0], DT)


# SPG
T4dyn = 0.01
T4settling = T4dyn
T4move = T4dyn + T4settling
y1 = 0.005
sol = solve(
    [
        a * T4dyn ** 5 + b * T4dyn ** 4 + c * T4dyn ** 3 - y1,
        5 * a * T4dyn ** 4 + 4 * b * T4dyn ** 3 + 3 * c * T4dyn ** 2,
        20 * a * T4dyn ** 3 + 12 * b * T4dyn ** 2 + 6 * c * T4dyn,
    ],
    [a, b, c],
)

t4dyn = np.arange(0, T4dyn, DT)
set_p = sol[a] * t4dyn ** 5 + sol[b] * t4dyn ** 4 + sol[c] * t4dyn ** 3
set_v = 5 * sol[a] * t4dyn ** 4 + 4 * sol[b] * t4dyn ** 3 + sol[c] * 3 * t4dyn ** 2
set_a = 20 * sol[a] * t4dyn ** 3 + 12 * sol[b] * t4dyn ** 2 + sol[c] * 6 * t4dyn ** 1

t4move = np.arange(0, T4move, DT)
T4move = t4move[-1] + DT

set_p = np.append(set_p, y1 * np.ones(len(t4move) - len(t4dyn)))
set_v = np.append(set_v, np.zeros(len(t4move) - len(t4dyn)))
set_a = np.append(set_a, np.zeros(len(t4move) - len(t4dyn)))

"""计算PS响应"""
identity_tf = TransferFunc([1], [1], DT)
closed_loop_tf = pid_tf * sys_tf / (identity_tf + pid_tf * sys_tf)
ps_tf = sys_tf / (identity_tf + pid_tf * sys_tf)
# ps_inv_tf = pid_tf + identity_tf / sys_tf
# ps_inv_tf_scipy = scipy.signal.TransferFunction(ps_inv_tf.num, ps_inv_tf.den)

T4ir = T4move
t4ir = np.arange(DT, T4ir + DT, DT)
# t4ir = np.arange(0, T4ir, DT)
t4ir, process_sensitivity_ir = impulse(
    (ps_tf.num, ps_tf.den),
    T=t4ir,
)
process_sensitivity_ir = process_sensitivity_ir
ps_response_mat = np.mat(
    toeplitz(process_sensitivity_ir, np.zeros_like(process_sensitivity_ir)), dtype=float
)
ps_pinv_mat = np.linalg.inv(ps_response_mat)

# plt.figure()
# plt.plot(t4ir, process_sensitivity_ir)

# Filter
Q_num, Q_den = butter(3, 1000, "low", analog=True)
# Q_num, Q_den = ([1], [1])
butter_filter = scipy.signal.TransferFunction(Q_num, Q_den)

# iteration
f_kp1 = np.zeros_like(set_p)
datalog = dict()
datalog["p_fbk"] = np.zeros_like(set_p, dtype=float)
datalog["err"] = np.zeros_like(set_p, dtype=float)
datalog["pid_output"] = np.zeros_like(set_p, dtype=float)
datalog["f_k"] = np.zeros_like(set_p, dtype=float)
datalog["plant_input"] = np.zeros_like(set_p, dtype=float)
print("Total Samples:", len(set_p))
for k in range(15):
    p_fbk = 0
    sys_ss.reset()
    pid_controller.reset()
    f_k = f_kp1
    for i in range(len(set_p)):
        if i % 1000 == 0:
            print(k + 1, i)
        p_cmd = set_p[i]
        a_cmd = set_a[i]
        err = p_cmd - p_fbk
        pid_output = pid_controller.response(err)

        plant_input = pid_output + f_k[i]
        p_fbk, x_state = sys_ss.response(
            plant_input, method="zoh2"
        )  # +random.normal()/4/5
        p_fbk = p_fbk[0, 0]

        datalog["p_fbk"][i] = p_fbk
        datalog["err"][i] = err
        datalog["pid_output"][i] = pid_output
        datalog["plant_input"][i] = plant_input
    datalog["f_k"] = f_k
    plant_input_list = datalog["plant_input"]
    # current plant in as next ffc
    # Le = np.zeros_like(set_p)
    # Le_after_Q = np.zeros_like(set_p)
    # current_ILC_after_Q = np.zeros_like(set_p)
    # next_ILC = current_ILC + pid_output

    # inverse plant
    # Le = np.array(plant_pinv_mat * err.reshape(-1, 1)).reshape(-1)
    # Le_after_Q = np.zeros_like(set_p)
    # current_ILC_after_Q = np.zeros_like(set_p)
    # next_ILC = current_ILC + pid_output + Le

    # inverse ps (paper method)
    # 矩阵计算方法
    Le = np.array(ps_pinv_mat * datalog["err"].reshape(-1, 1)).reshape(-1)
    # scipy库计算，行不通，因为非因果系统。Le = L * e = ps_inv * e
    # t4move, Le, x_out = lsim2(ps_inv_tf_scipy, datalog["err"], t4move)

    # without Q
    plant_input_list_after_Q = plant_input_list
    # with Q
    # t4move, f_k_after_Q, x_out = lsim(butter_filter, plant_input_list, t4move)

    f_kp1 = plant_input_list_after_Q + Le

    fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    fig.suptitle("Iteration " + str(k + 1))
    axs[0, 0].plot(t4move, set_p, label="cmd")
    axs[0, 0].plot(t4move, datalog["p_fbk"], label="p_fbk")
    axs[0, 0].legend(loc="upper left")
    axs[0, 1].plot(t4move, datalog["err"], label="err")
    axs[0, 1].legend(loc="upper left")
    axs[1, 0].plot(t4move, datalog["pid_output"], label="pid out")
    axs[1, 0].legend(loc="upper left")
    axs[1, 1].plot(t4move, datalog["f_k"], label="f_k")
    axs[1, 1].legend(loc="upper left")
    axs[2, 0].plot(t4move, Le, label="Le")
    axs[2, 0].legend(loc="upper left")
    axs[2, 1].plot(t4move, plant_input_list_after_Q, label="plant_input_list aft Q")
    axs[2, 1].legend(loc="upper left")
    axs[3, 0].plot(t4move, plant_input_list, label="plant_input_list")
    axs[3, 0].legend(loc="upper left")
    axs[3, 1].plot(t4move, f_kp1, label="f_k+1 (aft Q)")
    axs[3, 1].legend(loc="upper left")
    # plt.show()


# 输入和加速度时域响应
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# axes[0].set_xlabel("t4dyn/[s]")
# axes[0].set_ylabel("u/[N]")
# axes[0].plot(t4dyn, u, label="u")
#
# axes[1].set_xlabel("t4dyn/[s]")
# axes[1].set_ylabel("acc/[m/s**2]")
# axes[1].plot(t4dyn, y, label="acc")
# plt.suptitle("Input and Acc")
# plt.legend()

# 修正后和拟合的伯德图
# f_fit, fw_fit = sys_fit.bode(np.array(range(start_freq, end_freq)))
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# axes[0].set_xlabel("f/[Hz]")
# axes[0].set_ylabel("Gain/[dB]")
# axes[0].semilogx(f_bode, 20 * np.log10(np.abs(fw)), label="Gain")
# axes[0].semilogx(f_fit, 20 * np.log10(np.abs(fw_fit)), label="Fit Gain")
#
# axes[1].set_xlabel("f/[Hz]")
# axes[1].set_ylabel("Phase/[deg]")
# axes[1].semilogx(f_bode, np.angle(fw, deg=True), label="Phase")
# axes[1].semilogx(f_fit, np.angle(fw_fit, deg=True), label="Fit Phase")
# plt.suptitle("sys id bode plot")
# plt.legend()

# #
# f_pid, fw_fit4pid = closed_loop_tf.bode(np.array(range(1, 2000)))
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# axes[0].set_xlabel("f/[Hz]")
# axes[0].set_ylabel("Gain/[dB]")
# axes[0].semilogx(f_pid, 20 * np.log10(np.abs(fw_fit4pid)), label="Gain")
#
# axes[1].set_xlabel("f/[Hz]")
# axes[1].set_ylabel("Phase/[deg]")
# axes[1].semilogx(f_pid, np.angle(fw_fit4pid, deg=True), label="Phase")
# plt.suptitle("PID bode plot")
# plt.legend()

plt.show()
