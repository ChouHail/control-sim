import numpy as np
from sympy.core import symbols
from sympy import simplify, cancel, Poly, fraction, solve
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import integrate
from itertools import chain, zip_longest


class TransferFunc(object):
    def __init__(self, num, den, dt):
        self.num = np.array(num)
        self.den = np.array(den)
        self.dt = dt
        z = symbols("z")
        _num_d = _den_d = 0
        # tustin
        s = 2 / dt * (z - 1) / (z + 1)
        # 前向差分
        # s = (z - 1) / dt
        # 后向差分
        # s = (1 - z ** -1) / dt

        for _i in range(len(num)):
            _num_d += num[-_i - 1] * s ** _i
        for _i in range(len(den)):
            _den_d += den[-_i - 1] * s ** _i

        num_d, den_d = cancel(simplify(_num_d / _den_d)).as_numer_denom()
        self.num_d = num_d.as_poly(z).all_coeffs()
        self.den_d = den_d.as_poly(z).all_coeffs()
        self.input_array = np.zeros_like(self.num_d, dtype=float)
        self.output_array = np.zeros_like(self.den_d, dtype=float)
        self.output = 0

    def __get_operation_coeffs(self, other):
        s = symbols("s")
        self_num = Poly(self.num, s).as_expr(s)
        self_den = Poly(self.den, s).as_expr(s)
        other_num = Poly(other.num, s).as_expr(s)
        other_den = Poly(other.den, s).as_expr(s)
        return s, self_num, self_den, other_num, other_den

    def __add__(self, other):
        s, self_num, self_den, other_num, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_num / self_den + other_num / other_den))
        _res_num, _res_den = res_sys_s.as_numer_denom()
        res_num = np.array(_res_num.as_poly(s).all_coeffs(), dtype=float)
        res_den = np.array(_res_den.as_poly(s).all_coeffs(), dtype=float)
        res_sys = TransferFunc(res_num, res_den, self.dt)
        return res_sys

    def __sub__(self, other):
        s, self_num, self_den, other_num, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_num / self_den - other_num / other_den))
        _res_num, _res_den = res_sys_s.as_numer_denom()
        res_num = np.array(_res_num.as_poly(s).all_coeffs(), dtype=float)
        res_den = np.array(_res_den.as_poly(s).all_coeffs(), dtype=float)
        res_sys = TransferFunc(res_num, res_den, self.dt)
        return res_sys

    def __mul__(self, other):
        s, self_num, self_den, other_num, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_num / self_den * other_num / other_den))
        _res_num, _res_den = res_sys_s.as_numer_denom()
        res_num = np.array(_res_num.as_poly(s).all_coeffs(), dtype=float)
        res_den = np.array(_res_den.as_poly(s).all_coeffs(), dtype=float)
        res_sys = TransferFunc(res_num, res_den, self.dt)
        return res_sys

    def __truediv__(self, other):
        s, self_num, self_den, other_num, other_den = self.__get_operation_coeffs(other)
        res_sys_s = cancel(simplify(self_num / self_den / (other_num / other_den)))
        _res_num, _res_den = res_sys_s.as_numer_denom()
        res_num = np.array(_res_num.as_poly(s).all_coeffs(), dtype=float)
        res_den = np.array(_res_den.as_poly(s).all_coeffs(), dtype=float)
        res_sys = TransferFunc(res_num, res_den, self.dt)
        return res_sys

    def zpk(self):
        s = symbols("s")
        self_num = Poly(self.num, s).as_expr(s)
        self_den = Poly(self.den, s).as_expr(s)
        z = np.array(solve(self_num, s))
        p = np.array(solve(self_den, s))
        k = (self_num / self_den).subs(s, 0)
        return z, p, k

    def response(self, input_sig):
        self.input_array = np.delete(np.insert(self.input_array, 0, input_sig), -1)
        self.output_array = np.delete(np.insert(self.output_array, 0, 0), -1)
        self.output = (
            np.dot(self.input_array, self.num_d)
            - np.dot(self.output_array[1::], self.den_d[1::])
        ) / self.den_d[0]
        self.output_array[0] = self.output
        return self.output

    def reset(self):
        self.__init__(self.num, self.den, self.dt)

    def bode(self, f, plot=False):
        # f = np.arange(low, up, 1)
        omega = 2 * np.pi * f
        num = den = 0
        for i in range(len(self.num)):
            num = num + self.num[-i - 1] * (1j * omega) ** i
        for i in range(len(self.den)):
            den = den + self.den[-i - 1] * (1j * omega) ** i
        num = num.astype("complex")
        den = den.astype("complex")

        fw = num / den
        gain = 20 * np.log10(np.abs(fw))
        phase = np.angle(fw, deg=True)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            axes[0].set_xlabel("f/[Hz]")
            axes[0].set_ylabel("Gain/[dB]")
            axes[0].semilogx(f, gain)

            axes[1].set_xlabel("f/[Hz]")
            axes[1].set_ylabel("Phase/[deg]")
            axes[1].semilogx(f, phase)

            plt.suptitle("Bode plot")
            plt.show()

        return f, fw


class StateSpaceModel(object):
    def __init__(self, A, B, C, D, dt):
        n_states = int(np.asarray(A).size ** 0.5)
        self.n_states = n_states
        self.A = np.asarray(A).reshape((n_states, n_states))
        self.B = np.asarray(B).reshape((n_states, -1))
        self.C = np.asarray(C).reshape((-1, n_states))
        self.D = np.asarray(D).reshape((self.C.shape[0], self.B.shape[1]))
        self.dt = dt
        self.x_state = np.zeros((n_states, 1))
        self.y_output = np.zeros((self.C.shape[0], 1))
        self.last_u = np.zeros((self.B.shape[1], 1))

        identity = np.identity(n_states)
        self.expM_zoh_x = expm(A * dt)
        self.expM_zoh_u = np.linalg.pinv(A).dot(self.expM_zoh_x - identity).dot(B)
        n_inputs = self.B.shape[1]
        M_linear = np.block(
            [
                [A * dt, B * dt, np.zeros((n_states, n_inputs))],
                [
                    np.zeros((n_inputs, n_states + n_inputs)),
                    np.identity(n_inputs),
                ],
                [np.zeros((n_inputs, n_states + 2 * n_inputs))],
            ]
        )
        expM_linear = expm(M_linear)
        self.Ad_linear = expM_linear[:n_states, :n_states]
        self.Bd1_linear = expM_linear[:n_states, n_states + n_inputs :]
        self.Bd0_linear = (
            expM_linear[:n_states, n_states : n_states + n_inputs] - self.Bd1_linear
        )
        M_step = np.block(
            [[A * dt, B * dt], [np.zeros((n_inputs, n_states + n_inputs))]]
        )
        expM_step = expm(M_step)
        self.Ad_step = expM_step[:n_states, :n_states]
        self.Bd1_step = expM_step[:n_states, n_states:]

        def model(t, vX, Force):
            vU = Force
            vX = vX.reshape(len(self.B), 1)
            dx = self.A.dot(vX) + self.B.dot(vU)
            return dx

        self.ssmodel = model

    # def __init__(self, mass, damping_ratio, stiffness, R, L, kf, fs_pos, fs_current):
    #     self.mass = mass
    #     self.damping_ratio = damping_ratio
    #     self.stiffness = stiffness
    #     self.R = R
    #     self.L = L
    #     self.kf = kf
    #     self.A = np.mat(
    #         [[0, 1], [-self.stiffness / self.mass, -self.damping_ratio / self.mass]]
    #     )
    #     self.B = np.mat([[0], [1 / self.mass]])
    #     self.C = np.mat([1, 0])
    #     self.D = np.mat([0])
    #
    #     def model(t, vX, Force):
    #         vU = Force
    #         vX = vX.reshape(len(self.B), 1)
    #         dx = self.A.dot(vX) + self.B.dot(vU)
    #         return dx
    #
    #     self.ssmodel = model
    #     self.Xstates = np.zeros((self.A.shape[1], 1))
    #     self.T = 1 / fs_current
    #
    #
    # def run(self, t1, force):
    #     r = integrate.ode(self.ssmodel).set_integrator("dopri5")
    #     r.set_initial_value(self.Xstates, t1).set_f_params(force)
    #     r.integrate(r.t + self.T)
    #     self.Xstates = r.y
    #     self.pos = self.C.dot(self.Xstates)
    #     return self.pos

    def response(self, u, method="zoh"):
        last_u = self.last_u
        u = np.asarray(u).reshape((-1, 1))
        x_state = self.x_state
        if method == "zoh":  # 教科书的解法，输入为上一拍的step，延时半拍，可用于模拟带有零阶保持器的系统
            x_state = self.expM_zoh_x.dot(x_state) + self.expM_zoh_u.dot(last_u)
            self.y_output = self.C.dot(x_state) + self.D.dot(last_u)
        if method == "zoh2":  # lsim解法，输入为上一拍的step，延时半拍，可用于模拟带有零阶保持器的系统
            from scipy.signal import lsim, lsim2, step, impulse

            # from control import forced_response

            x_state = np.dot(self.Ad_step, x_state) + np.dot(self.Bd1_step, last_u)
            self.y_output = self.C.dot(x_state) + self.D.dot(last_u)
        if method == "interp":  # 输入为当前拍和上一拍的平均值的step，没有延时
            u_mean = (last_u + u) / 2
            x_state = self.expM_zoh_x.dot(x_state) + self.expM_zoh_u.dot(u_mean)
            self.y_output = self.C.dot(x_state) + self.D.dot(u_mean)
        elif method == "linear":  # 三角形保持器，forced_response解法，没有延时
            x_state = (
                np.dot(self.Ad_linear, x_state)
                + np.dot(self.Bd0_linear, last_u)
                + np.dot(self.Bd1_linear, u)
            )
            self.y_output = self.C.dot(x_state) + self.D.dot(u)
        elif method == "ode":
            r = integrate.ode(self.ssmodel).set_integrator("dopri5")
            r.set_initial_value(self.x_state, 0).set_f_params(self.last_u)
            r.integrate(r.t + self.dt)
            x_state = r.y
            self.y_output = self.C.dot(x_state)
            # return self.y_output, self.x_state

        self.last_u = u
        self.x_state = x_state
        return self.y_output, x_state

    def impulse(self, t):
        y_output = np.zeros_like(t)
        A = self.A
        B = self.B
        C = self.C
        for i in range(len(t)):
            y_output[i] = (C.dot(expm(A) * t[i]).dot(B))[0, 0]
        return y_output

    def reset(self):
        self.__init__(self.A, self.B, self.C, self.D, self.dt)


class PID(object):
    def __init__(self, kp, ki, kd, servo_freq):
        self.kp = kp
        self.ki = ki / servo_freq
        self.kd = kd * servo_freq
        self.servo_freq = servo_freq
        self.last_u = np.zeros_like(kp)
        self.ki_output = self.last_u
        self.kd_output = self.last_u

    def response(self, u):
        self.ki_output = self.ki_output + self.ki * u
        self.kd_output = self.kd * (u - self.last_u)
        self.last_u = u
        return self.kp * u + self.ki_output + self.kd_output

    def reset(self):
        self.__init__(self.kp, self.ki, self.kd, 1)


def pole_placement(dB, bandwidth, alpha, servo_freq):
    damping = 0.707
    ratio = 0.5
    k = 10 ** (dB / 20)
    omega = 2 * np.pi * bandwidth
    kp = (1 + 2 * damping * ratio) * omega ** 2 / k
    ki = ratio * omega ** 3 / k
    kd = (2 * damping * omega + ratio * omega - alpha) / k
    return kp, ki, kd


def dft(freq, data, dt):
    data = np.array(data)
    t = np.array(range(len(data))) * dt
    cos_wave = np.cos(2 * np.pi * freq * t)
    sine_wave = np.sin(2 * np.pi * freq * t)
    real = np.sum(np.multiply(data, cos_wave))
    imagine = np.sum(np.multiply(data, sine_wave))
    return real - 1j * imagine


def dft_slow(src_data):
    num = len(src_data)
    fw = np.zeros(num, dtype=complex)
    for k in range(num):
        for _n in range(num):
            fw[k] += src_data[_n] * np.e ** (-1j * 2 * np.pi / num * _n * k)
    return fw


def dft_vectorized(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    num = x.shape[0]
    n = np.arange(num)
    k = n.reshape((num, 1))
    m = np.exp(-2j * np.pi * k * n / num)
    return np.dot(m, x)


def fft_slow(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    num = x.shape[0]

    if num % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif num <= 32:  # this cutoff should be optimized
        return dft_vectorized(x)
    else:
        x_even = fft_slow(x[::2])
        x_odd = fft_slow(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(num) / num)
        return np.concatenate(
            [
                x_even + factor[: int(num / 2)] * x_odd,
                x_even + factor[int(num / 2) :] * x_odd,
            ]
        )


def _fft(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    num = x.shape[0]

    if np.log2(num) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # _N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    num_min = min(num, 32)

    # Perform an O[N^2] DFT on all length-_N_min sub-problems at once
    n = np.arange(num_min)
    k = n[:, None]
    m = np.exp(-2j * np.pi * n * k / num_min)
    mat_x = np.dot(m, x.reshape((num_min, -1)))

    # build-up each level of the recursive calculation all at once
    while mat_x.shape[0] < num:
        x_even = mat_x[:, : int(mat_x.shape[1] / 2)]
        x_odd = mat_x[:, int(mat_x.shape[1] / 2) :]
        factor = np.exp(-1j * np.pi * np.arange(mat_x.shape[0]) / mat_x.shape[0])[
            :, None
        ]
        mat_x = np.vstack([x_even + factor * x_odd, x_even - factor * x_odd])

    return mat_x.ravel()


def hamm(x):
    x = np.array(x)
    num = len(x)
    hamm_window = 0.53836 - 0.46164 * np.cos(2 * np.pi * np.arange(num) / num)
    x = x * hamm_window

    return x


def anti_hamm(x):
    x = np.array(x)
    num = len(x)
    hamm_window = 0.5 * (1 + np.cos(2 * np.pi * np.arange(num) / (num - 1)))
    x = x * hamm_window

    return x


def half_hamm(x):
    x = np.array(x)
    num = len(x)
    hamm_window = 0.53836 + 0.46164 * np.cos(np.pi * np.arange(num) / (num - 1))

    x = x * hamm_window
    return x


def pad(x):
    num = len(x)
    num_pad = int(2 ** np.ceil(np.log2(num)))
    x_pad = np.pad(x, (0, num_pad - num))
    return x_pad


def fft(x, dt):
    x_pad = pad(x)
    num_pad = len(x_pad)
    fs = 1 / (num_pad * dt)
    f_pad = fs * np.arange(num_pad)

    fw_pad = _fft(x_pad)

    return f_pad, fw_pad


# 分母最高次项系数为1
def fit2(f, fw, bn, am):
    s = 1j * 2 * np.pi * f
    num = len(f)
    A = np.zeros([num, am], dtype="complex")
    for i in range(am):
        A[:, i] = s ** i * fw
    Re_A = A.real
    Im_A = A.imag
    for i in range(bn + 1):
        if i % 2 == 0:
            Re_A = np.concatenate((Re_A, -(s ** i).real.reshape(num, 1)), axis=1)
        else:
            Im_A = np.concatenate((Im_A, -(s ** i).imag.reshape(num, 1)), axis=1)

    # 加权
    # Re_weight = np.mat(np.diag(np.linspace(1, 0.0, num)))
    # Re_A = Re_weight * Re_A
    # Im_A = Re_weight * Im_A

    Re_A = np.mat(Re_A)
    Im_A = np.mat(Im_A)
    B = np.mat(-fw * s ** am)
    Re_B = np.mat(B.real).T
    Im_B = np.mat(B.imag).T
    Re_X = ((Re_A.T * Re_A).I * Re_A.T * Re_B).real.tolist()  # 最小二乘法
    Im_X = ((Im_A.T * Im_A).I * Im_A.T * Im_B).real.tolist()  # 最小二乘法
    #
    # Re_X = (np.linalg.pinv(Re_A) * Re_B).real.tolist()  # numpy库求伪逆
    # Im_X = (np.linalg.pinv(Im_A) * Re_B).real.tolist()  # numpy库求伪逆

    Re_X = [i[0] for i in Re_X]
    Im_X = [i[0] for i in Im_X]
    Bn1 = Re_X[am:]  # [b0, b2, b4, ...]
    Bn2 = Im_X[am:]  # [b1, b3, b5, ...]

    Bn = [x for x in chain.from_iterable(zip_longest(Bn1, Bn2)) if x is not None]
    Am = Re_X[0:am] + [1]
    Bn.reverse()
    Am.reverse()
    return Bn, Am


# 分母0次项系数为1
def fit(f, fw, bn, am):
    s = 1j * 2 * np.pi * f
    num = len(f)
    A = np.zeros([num, am], dtype="complex")
    for i in range(am):
        A[:, i] = s ** (i + 1) * fw
    Re_A = A.real
    Im_A = A.imag
    for i in range(bn + 1):
        if ((s[0] ** i).real) != 0:
            Re_A = np.concatenate((Re_A, -(s ** i).real.reshape(num, 1)), axis=1)
        else:
            Im_A = np.concatenate((Im_A, -(s ** i).imag.reshape(num, 1)), axis=1)
    # 加权
    Re_weight = np.mat(np.diag(np.linspace(1, 0.0, num)))
    Re_A = Re_weight * Re_A
    Im_A = Re_weight * Im_A

    Re_A = np.mat(Re_A)
    Im_A = np.mat(Im_A)
    B = np.mat(-fw)
    Re_B = np.mat(B.real).T
    Im_B = np.mat(B.imag).T
    Re_X = ((Re_A.T * Re_A).I * Re_A.T * Re_B).real.tolist()  # 最小二乘法
    Im_X = ((Im_A.T * Im_A).I * Im_A.T * Im_B).real.tolist()  # 最小二乘法

    # Re_X = (np.linalg.pinv(Re_A) * Re_B).real.tolist()  # numpy库求伪逆
    # Im_X = (np.linalg.pinv(Im_A) * Im_B).real.tolist()  # numpy库求伪逆

    Re_X = [i[0] for i in Re_X]
    Im_X = [i[0] for i in Im_X]
    Bn1 = Re_X[am:]  # [b0, b2, b4, ...]
    Bn2 = Im_X[am:]  # [b1, b3, b5, ...]

    Bn = [x for x in chain.from_iterable(zip_longest(Bn1, Bn2)) if x is not None]
    Am = ([1] + Re_X)[0 : am + 1]
    Bn.reverse()
    Am.reverse()
    return Bn, Am


def chirp_iden(sys, start_freq, end_freq, t, plot=False):
    dt = sys.dt
    start_freq_ = 0.8 * start_freq
    end_freq_ = 1.2 * end_freq
    t_ = t * (end_freq_ - start_freq_) / (end_freq - start_freq)
    t_list = np.arange(0, t_, dt)
    pad_len = int(0.1 / dt)

    u = np.sin(
        2
        * np.pi
        * ((end_freq_ - start_freq_) / t_ * t_list ** 2 / 2 + start_freq_ * t_list)
    )
    u = np.pad(u, (0, pad_len))
    a = np.array([])
    for i in range(len(u)):
        input_sig = u[i]
        a_current = sys.response(input_sig)  # +random.normal()/4/5
        a = np.append(a, a_current)

    csv = np.asarray([u, a]).T
    # np.savetxt("datalog.csv", csv, delimiter=",")

    a = np.array(a, dtype=float)
    # a = np.pad(np.diff(pos, 2), (0, 2))
    y = a

    u_detrend = u - np.mean(u)
    y_detrend = y - np.mean(y)

    # f_bode, fw_u = fft(half_hamm(u_detrend), dt)
    # f_y, fw_y = fft(half_hamm(y_detrend), dt)
    f_u, fw_u = fft(u_detrend, dt)
    f_y, fw_y = fft(y_detrend, dt)

    resolution = 1 / dt / len(f_u)
    start_point = int(start_freq / resolution)
    end_point = int(end_freq / resolution)

    fw = fw_y / fw_u
    if plot:
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            20 * np.log10(np.abs(fw)[start_point:end_point]),
        )

        plt.subplot(122)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            np.angle(fw[start_point:end_point], deg=True),
        )
        plt.show()

    return f_u[1:end_point], fw[1:end_point]


def chirp_iden_cross(sys, start_freq, end_freq, t, plot=False):
    dt = sys.dt
    start_freq_ = 0.8 * start_freq
    end_freq_ = 1.2 * end_freq
    t_ = t * (end_freq_ - start_freq_) / (end_freq - start_freq)
    t_list = np.arange(0, t_, dt)
    pad_len = int(0.1 / dt)

    u = np.sin(
        2
        * np.pi
        * ((end_freq_ - start_freq_) / t_ * t_list ** 2 / 2 + start_freq_ * t_list)
    )
    u = np.pad(u, (0, pad_len))
    a = np.array([])
    for i in range(len(u)):
        input_sig = u[i]
        a_current = sys.response(input_sig)  # +random.normal()/4/5
        a = np.append(a, a_current)

    csv = np.asarray([u, a]).T
    # np.savetxt("datalog.csv", csv, delimiter=",")

    a = np.array(a, dtype=float)
    y = a

    u_detrend = u - np.mean(u)
    y_detrend = y - np.mean(y)

    cross_num = int(len(u))
    Ruy = np.zeros(cross_num)
    Ruu = np.zeros(cross_num)
    for i in range(cross_num):
        print(i)
        Ruy[i] = sum(u_detrend[0 : -i - 1] * y_detrend[i:-1])
        Ruu[i] = sum(u_detrend[0 : -i - 1] * u_detrend[i:-1])
    # Ruy = half_hamm(Ruy)
    # Ruu = half_hamm(Ruu)
    f_u, fw_Ruy = fft(Ruy, dt)
    f_y, fw_Ruu = fft(Ruu, dt)
    fw = fw_Ruy / fw_Ruu

    resolution = 1 / dt / len(f_u)
    start_point = int(start_freq / resolution)
    end_point = int(end_freq / resolution)

    if plot:
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            20 * np.log10(np.abs(fw)[start_point:end_point]),
        )

        plt.subplot(122)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            np.angle(fw[start_point:end_point], deg=True),
        )
        plt.show()

    return f_u[1:end_point], fw[1:end_point]


def chirp_iden_pos(sys, start_freq, end_freq, t, plot=False):
    dt = sys.dt
    start_freq_ = 0.8 * start_freq
    end_freq_ = 1.2 * end_freq
    t_ = t * (end_freq_ - start_freq_) / (end_freq - start_freq)
    t_list = np.arange(0, t_, dt)
    pad_len = int(0.1 / dt)

    u = np.sin(
        2
        * np.pi
        * ((end_freq_ - start_freq_) / t * t_list ** 2 / 2 + start_freq_ * t_list)
    )
    u = np.pad(u, (0, pad_len))
    a = np.array([])
    v = np.array([0])
    p = np.array([0])
    for i in range(len(u)):
        input_sig = u[i]
        a_current = sys.response(input_sig)  # +random.normal()/4/5
        v_current = v[-1] + a_current * dt
        p_current = p[-1] + v_current * dt
        a = np.append(a, a_current)
        v = np.append(v, v_current)
        p = np.append(p, p_current)

    a = np.array(a, dtype=float)
    v = np.array(v, dtype=float)
    p = np.array(p[1:], dtype=float)
    # a = np.pad(np.diff(pos, 2), (0, 2))
    y = np.diff(p, 2) / dt / dt
    y = np.pad(y, (1, 1), "constant", constant_values=(0, 0))

    u_detrend = u - np.mean(u)
    y_detrend = y - np.mean(y)

    # f_bode, fw_u = fft(half_hamm(u_detrend), dt)
    # f_y, fw_y = fft(half_hamm(y_detrend), dt)
    f_u, fw_u = fft(u_detrend, dt)
    f_y, fw_y = fft(y_detrend, dt)

    resolution = 1 / dt / len(f_u)
    start_point = int(start_freq / resolution)
    end_point = int(end_freq / resolution)

    fw = fw_y / fw_u
    if plot:
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            20 * np.log10(np.abs(fw)[start_point:end_point]),
        )

        plt.subplot(122)
        plt.xscale("log")
        plt.plot(
            f_u[start_point:end_point],
            np.angle(fw[start_point:end_point], deg=True),
        )
        plt.show()

    return f_u[1:end_point], fw[1:end_point]
