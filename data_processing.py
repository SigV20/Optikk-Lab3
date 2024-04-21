import numpy as np
import matplotlib.pyplot as plt
import statistics

def load_data():
    data = np.loadtxt('H2_data', delimiter=' ')
    N = len(data) 
    total_time = 30  
    time = np.linspace(0, total_time, N)  

    # Calculate indices for data-slicing
    index_start = int((0 / total_time) * N)
    index_end = int((30 / total_time) * N)

    return time[index_start:index_end], data[index_start:index_end]

def plot_time_domain_individual(time, data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # One row, one column
    
    ax.plot(time, data[:, 1], color='green', label='Grønn kanal')  
    ax.set_xlabel('Tid (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(7, 16)  
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_time_domain(time, data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  
    
    global channel_colors, channel_labels
    channel_colors = ['red', 'green', 'blue']  
    channel_labels = ['Rød kanal', 'Grønn kanal', 'Blå kanal']
    
    for i in range(3):  
        axs[i].plot(time, data[:, i], color=channel_colors[i], label=channel_labels[i])
        axs[i].set_xlabel('Tid (s)')
        axs[i].set_ylabel('Amplitude')
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_xlim(7, 16)  
    #fig.suptitle('Reflektans fra håndledd for RGB fargekanaler', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_fft_and_calculate_snr_individual(time, data):
    zero_padding = 10  
    N = len(time)
    timestep = time[1] - time[0]
    freq = np.fft.fftfreq(N * zero_padding, d=timestep) * 60  # Frequency in BPM

    window = np.hanning(N)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  
    snrs = []  # To store the SNR of the green channel

    windowed_data = data[:, 1] * window 
    fft_result = np.fft.fft(windowed_data, n=N * zero_padding)
    magnitude = np.abs(fft_result)[:N * zero_padding // 2]
    freq_positive = freq[:N * zero_padding // 2]

    noise_freq_range = (freq_positive > 0.7 * 60) & (freq_positive < 3.5 * 60)
    exclude_signal_peak = (freq_positive < 1.26 * 60) | (freq_positive > 1.4 * 60)
    noise_indices = noise_freq_range & exclude_signal_peak
    average_noise = np.mean(magnitude[noise_indices])

    signal_peak_indices = (freq_positive > 1.26 * 60) & (freq_positive < 1.4 * 60)
    if np.any(signal_peak_indices):
        signal_peak_magnitude = np.max(magnitude[signal_peak_indices])
        snr_db = 20 * np.log10(signal_peak_magnitude / average_noise)
    else:
        snr_db = 0

    snrs.append(snr_db)

    channel_colors = ['red', 'green', 'blue']  
    channel_labels = ['Rød kanal', 'Grønn kanal', 'Blå kanal']
    ax.plot(freq_positive, magnitude, label=channel_labels[1], color = channel_colors[1])
    ax.set_xlabel('Frekvens (BPM)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.set_xlim(0.75 * 60, 2 * 80)
    ax.set_ylim(0, 100)  
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return snrs  


def plot_fft_and_calculate_snr(time, data):
    N = len(time)
    timestep = time[1] - time[0]
    freq = np.fft.fftfreq(N, d=timestep) * 60  # Frequency in Beats Per Minute (BPM)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    snrs = [] 

    channel_colors = ['red', 'green', 'blue']  
    channel_labels = ['Rød kanal', 'Grønn kanal', 'Blå kanal']

    for i in range(0, 3): 
        fft_result = np.fft.fft(data[:, i])
        magnitude = np.abs(fft_result)[:N // 2]  
        freq_positive = freq[:N // 2]

        noise_freq_range = (freq_positive >= 0.5*60) & (freq_positive <= 3.5*60)
        exclude_signal_peak = (freq_positive < 1.29*60) | (freq_positive > 1.35*60)
        noise_indices = noise_freq_range & exclude_signal_peak
        average_noise = np.mean(magnitude[noise_indices])

        signal_peak_indices = (freq_positive >= 1.29*60) & (freq_positive <= 1.35*60)
        if np.any(signal_peak_indices):
            signal_peak_magnitude = np.max(magnitude[signal_peak_indices])
            snr_db = 20 * np.log10(signal_peak_magnitude / average_noise)
        else:
            snr_db = 0
        
        snrs.append(snr_db)

        axs[i].plot(freq_positive, magnitude, label= channel_labels[i], color = channel_colors[i])
        axs[i].set_xlabel('Frekvens (BPM)')
        axs[i].set_ylabel('Amplitude')
        axs[i].grid(True)
        axs[i].legend()
        axs[i].set_xlim(0.75*60, 2.75*60)
        axs[i].set_ylim(0, 400) 

    fig.suptitle('FFT and SNR for Green and Blue Channels', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return snrs



time, data = load_data()
#plot_fft_and_calculate_snr(time, data)
plot_time_domain_individual(time, data)
plot_fft_and_calculate_snr_individual(time,data)
snr = plot_fft_and_calculate_snr_individual(time, data)
print(f"SNR: {snr}")


mean = 0
std = 0

print(f"Average pulse: {mean}. Standard deviation: {std}")












