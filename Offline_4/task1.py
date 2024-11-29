import numpy as np
import matplotlib.pyplot as plt
n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000



#use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):

    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    
     # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B

#implement other functions and logic
def discrete_fourier_transform(signal):
    real = np.zeros(n)
    imaginary = np.zeros(n)
    for k in range(n):
        for i in range(n):
            angle = 2 * np.pi * k * i / n
            real[k] += signal[i] * np.cos(angle)
            imaginary[k] -= signal[i] * np.sin(angle)
    return real, imaginary

def inverse_discrete_fourier_transform(signal_tuple):
    real, imaginary = signal_tuple
    signal = np.zeros(n)
    for i in range(n):
        for k in range(n):
            angle = 2 * np.pi * k * i / n
            signal[i] += real[k] * np.cos(angle) - imaginary[k] * np.sin(angle)
    return signal/n

def cross_correlation(signal_A, signal_B):
    real_A, imaginary_A = discrete_fourier_transform(signal_A)
    real_B, imaginary_B = discrete_fourier_transform(signal_B)
    multiplied_real = real_A * real_B - imaginary_A * imaginary_B
    multiplied_imaginary = real_A * imaginary_B + imaginary_A * real_B
    multiplied_signal = (multiplied_real, multiplied_imaginary)
    return inverse_discrete_fourier_transform(multiplied_signal)

def sample_lag(signal_A, signal_B):
    cross_correlation_values = cross_correlation(signal_A, signal_B)
    index = 0
    for i in range(0,len(cross_correlation_values)):
        if(cross_correlation_values[i]>cross_correlation_values[index]):
            index = i

    if(index>n//2):
        index-=n
    return index

def calculate_distance(lag):
    return abs(lag) * wave_velocity / sampling_rate

def plot_signal_A(signal_A):
    plt.figure(figsize=(6, 6))
    plt.stem(samples,signal_A, basefmt='b-')
    plt.title("Signal A (Station A)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

def plot_signal_B(signal_B):
    plt.figure(figsize=(6, 6))
    plt.stem(samples,signal_B , linefmt='r-')
    plt.title("Signal B (Station B)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

def plot_frequency_spectrum_A(dft_signal_A):
    plt.figure(figsize=(6, 6))
    plt.stem(samples,np.sqrt(dft_signal_A[0] ** 2 + dft_signal_A[1] ** 2),basefmt='b-')
    plt.title("Frequency Spectrum of Signal A")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

def plot_frequency_spectrum_B(dft_signal_B):
    plt.figure(figsize=(6, 6))
    plt.stem(samples,np.sqrt(dft_signal_B[0] ** 2+dft_signal_B[1] ** 2),linefmt='r-')
    plt.title("Frequency Spectrum of Signal B")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

def plot_cross_correlation(cross_correlation_values):
    plt.figure(figsize=(6, 6))
    plt.stem(samples,cross_correlation_values,linefmt='g-',basefmt='g-')
    plt.title("DFT-based Cross-Correlation")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation")
    plt.show()

def main():
    signal_A, signal_B = generate_signals()
    dft_signal_A = discrete_fourier_transform(signal_A)
    dft_signal_B = discrete_fourier_transform(signal_B)

    plot_signal_A(signal_A)
    plot_frequency_spectrum_A(dft_signal_A)

    plot_signal_B(signal_B)
    plot_frequency_spectrum_B(dft_signal_B)

    cross_correlation_values = cross_correlation(signal_A, signal_B)
    plot_cross_correlation(cross_correlation_values)

    lag = sample_lag(signal_A, signal_B)
    distance = calculate_distance(lag)
    print(f"Lag: {lag}")
    print(f"Distance between the stations: {distance} meters")

main()