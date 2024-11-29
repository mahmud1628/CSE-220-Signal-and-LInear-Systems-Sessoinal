import numpy as np
import matplotlib.pyplot as plt

def generate_function(x,name):
    if(name == "parabolic"):
        y = np.where((-2 <= x) & (x <= 2), x**2, 0)
    elif(name == "triangular"):
        y = np.where((-2 <= x) & (x <= 2), 1 - abs(x) / 2, 0)
    elif(name == "sawtooth"):
        y = np.where((-2 <= x) & (x <= 2), x + 2, 0)
    elif(name == "rectangular"):
        y = np.where((-2 <= x) & (x <= 2), 1, 0)
    else:
        raise ValueError("Invalid function name")
    return y

# Define the interval and function and generate appropriate x values and y values
x_values = np.linspace(-10, 10, 1000)
y_values = generate_function(x_values, "parabolic")

# Plot the original function
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2")
plt.title("Original Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# Define the sampled times and frequencies
sampled_times = x_values
L = 1
frequencies = np.linspace(-L, L, 1000)

# Fourier Transform 
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
    # use trapezoidal integration to calculate the real and imaginary parts of the FT

    for i, freq in enumerate(frequencies):
        # Calculate the real and imaginary parts of the Fourier Transform
        ft_result_real[i] = np.trapz(signal * np.cos(2 * np.pi * freq * sampled_times), sampled_times)
        ft_result_imag[i] = -1*np.trapz(signal * np.sin(2 * np.pi * freq * sampled_times), sampled_times)

    return ft_result_real, ft_result_imag

# Apply FT to the sampled data
ft_data = fourier_transform(y_values, frequencies, sampled_times)
#  plot the FT data
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
plt.title("Frequency Spectrum of y = x^2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()


# Inverse Fourier Transform 
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n = len(sampled_times)
    reconstructed_signal = np.zeros(n)
    # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
    # use trapezoidal integration to calculate the real part
    # You have to return only the real part of the reconstructed signal
    for i, time in enumerate(sampled_times):
        reconstructed_signal[i] = np.trapz(ft_signal[0] * np.cos(2 * np.pi * frequencies * time) - ft_signal[1] * np.sin(2 * np.pi * frequencies * time), frequencies)
    
    return reconstructed_signal

# Reconstruct the signal from the FT data
reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
# Plot the original and reconstructed functions for comparison
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
plt.title("Original vs Reconstructed Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
