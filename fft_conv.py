import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.fftpack import fft, ifft, fftshift, fftfreq
import time
import numpy

scipy.signal.tf2sos()
a = np.random.random(3)
b = np.random.random(3)
T1 = time.time()
c = np.correlate(a, b, "full")
T2 = time.time()
print("互相关算法", T2 - T1, "秒")

T1 = time.time()
num = len(a)
left_pad = round((num - 1) / 2 + 0.1)
right_pad = num - 1 - left_pad
a1 = np.pad(a, (left_pad, right_pad), "constant", constant_values=0)
b1 = np.pad(np.flip(b), (left_pad, right_pad), "constant", constant_values=0)
ahat = fft(a1)
bhat = fft(b1)
chat = ahat * bhat
c1 = np.real(ifft(chat))
c1 = fftshift(c1)
if num % 2 == 1:
    c1 = np.roll(c1, 1)
T2 = time.time()
print("fft算法", T2 - T1, "秒")
# print(c)
# print(c1)
print(np.allclose(c, c1))
#
# n = 64
# L = 30
# dx = L / n
# # x = np.arange(-L / 2, L / 2, dx)
# x = np.arange(0, L, dx)
# f = np.sin(x)
# # fhat = np.fft.fft(f)
# fhat = fft(f)
# fft_freq = fftfreq(n, d=dx) * 2 * np.pi
# kappa = (2 * np.pi / L) * np.arange(-n / 2, n / 2)
# # kappa = (2 * np.pi / L) * np.arange(0, n)
# # kappa = np.fft.fftshift(kappa)
# kappa = fftshift(kappa)
# dfhat = kappa * fhat * (1j)
# # dfFFt = np.real(np.fft.ifft(dfhat))
# dfFFt = np.real(ifft(dfhat))
#
# plt.figure()
# plt.plot(x, dfFFt)
# plt.show()
