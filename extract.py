import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import periodogram, spectrogram

rate, t = wavfile.read('Data Sets/animals023.wav')
#plt.plot(t)

# Average energy
avg_eng = reduce(lambda p, x: p + (x ** 2), t) / len(t)

# Zero-crossing rate
zcr = np.sum(np.abs(np.diff(t))) / len(t)

# Silence ratio
sr = len(list(filter(lambda x: x == 0, t))) / len(t)

# Get frequency domain
fourier = fft(t)
half = len(fourier)/2
f = abs(fourier[:int(half - 1)])
plt.plot(f)
plt.show()

# Bandwidth
c = 0
bottom, top = 0, len(f)
bw = top

while f[c] == 0:
    bottom = c
    c += 1

c = len(f) - 1
while f[c] == 0:
    top = c
    c -= 1

bw = top - bottom

# Power spectral density
freq, pfreq = periodogram(t)
plt.semilogy(freq, pfreq)
plt.show()

# Dominant frequency
dom = np.argmax(f)

# Spectrogram
freq, time, spec = spectrogram(t)
plt.specgram(t, Fs=1000)
plt.show()
