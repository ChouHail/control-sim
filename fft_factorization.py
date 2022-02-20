import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq
import time

a = np.array([1, 3, 2, 4])
b = np.array([5, 3, 2, 4])
b = a
a0 = np.array([5, 5, 5, 5])
b0 = np.array([1, 1, 1, 1])
c = np.correlate(a, b, "full")
c0 = np.correlate(a0, b0, "full")

cc = np.correlate(a, a, "full")
aa = ifft(np.sqrt((fft(fftshift(cc)))))
np.correlate(aa, np.flip(aa), "full")
np.correlate(aa, aa, "full")
num = len(a)
left_pad = round((num - 1) / 2 + 0.1)
right_pad = num - 1 - left_pad
a1 = np.pad(a, (left_pad, right_pad), "constant", constant_values=0)
b1 = np.pad(np.flip(b), (left_pad, right_pad), "constant", constant_values=0)
ahat = fft(a1)
bhat = fft(b1)
chat = ahat * ahat

a01 = np.pad(a0, (left_pad, right_pad), "constant", constant_values=0)
b01 = np.pad(np.flip(b0), (left_pad, right_pad), "constant", constant_values=0)
a0hat = fft(a01)
b0hat = fft(b01)
c0hat = a0hat * a0hat

c1 = np.real(ifft(chat))
c1 = fftshift(c1)
if num % 2 == 1:
    c1 = np.roll(c1, 1)
print(c)
print(c1)
print(np.allclose(c, c1))
print(c0)
