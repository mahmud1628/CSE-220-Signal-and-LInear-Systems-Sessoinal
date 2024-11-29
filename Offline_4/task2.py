import numpy as np
import matplotlib.pyplot as plt
import time

def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0

def pad_zeors(signal):
        n = len(signal)
        new_n = 2**int(np.ceil(np.log2(n)))
        return np.pad(signal, (0, new_n-n), 'constant')

def DFT(signal):
    n = len(signal)
    real = np.zeros(n)
    imaginary = np.zeros(n)
    for k in range(n):
        for i in range(n):
            angle = 2 * np.pi * k * i / n
            real[k] += signal[i] * np.cos(angle)
            imaginary[k] -= signal[i] * np.sin(angle)
    return real, imaginary

def IDFT(signal_tuple):
    real, imaginary = signal_tuple
    n = len(real)
    signal = np.zeros(n)
    for i in range(n):
        for k in range(n):
            angle = 2 * np.pi * k * i / n
            signal[i] += real[k] * np.cos(angle) - imaginary[k] * np.sin(angle)
    return signal/n

def FFT(signal):
    real, imaginary = signal
    n = len(real)

    if not is_power_of_two(n):
        real = pad_zeors(real)
        imaginary = pad_zeors(imaginary)
        n = len(real)

    if(n == 1):
        return real,imaginary
    
    real_even = real[::2]
    real_odd = real[1::2]
    imaginary_even = imaginary[::2]
    imaginary_odd = imaginary[1::2]
    
    result_real_even, result_imaginary_even = FFT((real_even, imaginary_even))
    result_real_odd, result_imaginary_odd = FFT((real_odd, imaginary_odd))

    result_real, result_imaginary = np.zeros(n), np.zeros(n)

    for i in range(0,n//2):
        twiddle_factor_real = np.cos(2 * np.pi * i / n)
        twiddle_factor_imag = -np.sin(2 * np.pi * i/ n)

        result_real[i] = result_real_even[i] + twiddle_factor_real * result_real_odd[i] - twiddle_factor_imag * result_imaginary_odd[i]
        result_imaginary[i] = result_imaginary_even[i] + twiddle_factor_real * result_imaginary_odd[i] + twiddle_factor_imag * result_real_odd[i]
        result_real[i + n//2] = result_real_even[i] - twiddle_factor_real * result_real_odd[i] + twiddle_factor_imag * result_imaginary_odd[i]
        result_imaginary[i + n//2] = result_imaginary_even[i] - twiddle_factor_real * result_imaginary_odd[i] - twiddle_factor_imag * result_real_odd[i]
    
    return result_real,result_imaginary

def IFFT(signal_tuple):
    n = len(signal_tuple[0])
    signal = recursiveIFFT(signal_tuple)
    return signal[0] / n
        
def recursiveIFFT(signal_tuple):
    real, imaginary = signal_tuple
    n = len(real)

    if not is_power_of_two(n):
        real = pad_zeors(real)
        imaginary = pad_zeors(imaginary)
        n = len(real)

    if(n == 1):
        return real,imaginary
    
    real_even = real[::2]
    real_odd = real[1::2]
    imaginary_even = imaginary[::2]
    imaginary_odd = imaginary[1::2]
    
    result_real_even, result_imaginary_even = recursiveIFFT((real_even, imaginary_even))
    result_real_odd, result_imaginary_odd = recursiveIFFT((real_odd, imaginary_odd))

    result_real, result_imaginary = np.zeros(n), np.zeros(n)

    for i in range(0,n//2):
        twiddle_factor_real = np.cos(2 * np.pi * i / n)
        twiddle_factor_imag = np.sin(2 * np.pi * i / n)

        result_real[i] = result_real_even[i] + twiddle_factor_real * result_real_odd[i]-twiddle_factor_imag * result_imaginary_odd[i]
        result_imaginary[i] = result_imaginary_even[i] + twiddle_factor_real * result_imaginary_odd[i] + twiddle_factor_imag * result_real_odd[i]
        result_real[i + n//2] = result_real_even[i] - twiddle_factor_real * result_real_odd[i] + twiddle_factor_imag * result_imaginary_odd[i]
        result_imaginary[i + n//2] = result_imaginary_even[i] - twiddle_factor_real * result_imaginary_odd[i] - twiddle_factor_imag * result_real_odd[i]
    
    return result_real,result_imaginary


average_time_dft, average_time_fft, average_time_idft, average_time_ifft = [], [], [], []

N = 10

signal_sizes = [2 ** i for i in range(1, 10)] 

for n in signal_sizes:
    signals = [np.random.rand(n) for s in range(N)]

    time_dft, time_fft, time_idft, time_ifft = 0, 0, 0, 0

    for signal in signals:

        start_time = time.perf_counter()
        signal_dft = DFT(signal)
        time_dft += time.perf_counter() - start_time

        start_time = time.perf_counter()
        signal_fft = FFT((signal, np.zeros_like(signal)))
        time_fft += time.perf_counter() - start_time

        start_time = time.perf_counter()
        signal_idft = IDFT(signal_dft)
        time_idft += time.perf_counter() - start_time

        start_time = time.perf_counter()
        signal_ifft = IFFT(signal_fft)
        time_ifft += time.perf_counter() - start_time
    
    average_time_dft.append(time_dft / N)
    average_time_fft.append(time_fft / N)
    average_time_idft.append(time_idft / N)
    average_time_ifft.append(time_ifft / N)


plt.figure(figsize=(10, 6))
plt.plot(signal_sizes, average_time_dft, label='DFT', color='red')
plt.plot(signal_sizes, average_time_fft, label='FFT', color='green')
plt.title('Comparison between DFT and FFT')
plt.grid(True)
plt.xlabel('size')
plt.ylabel('time')
plt.xscale('log', base = 2)
plt.yscale('log')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(signal_sizes, average_time_idft, label='IDFT', color='red')
plt.plot(signal_sizes, average_time_ifft, label='IFFT', color='green')
plt.title('Comparison between IDFT and IFFT')
plt.grid(True)
plt.xlabel('size')
plt.ylabel('time')
plt.xscale('log', base = 2)
plt.yscale('log')
plt.legend()
plt.show()
