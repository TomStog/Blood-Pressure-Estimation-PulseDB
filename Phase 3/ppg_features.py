import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def plot_epochs(epochs_dict):

    fig, ax = plt.subplots(figsize=(12, 6))
    all_signals = []
    x_values = None

    # Plot each epoch's derivative
    for epoch_label, df in epochs_dict.items():
        # Get x values (first column/index) and signal values
        x = df.index.values
        signal = df['Signal'].values

        # Store for averaging
        all_signals.append(signal)
        if x_values is None:
            x_values = x

        # Plot with dashed line
        ax.plot(x, signal, '--', alpha=0.6, linewidth=1,
                label=f'Epoch {epoch_label}')

    # Calculate and plot average signal
    #avg_derivative = np.nanmedian(all_derivatives, axis=0)
    avg_signal = np.nanmedian(all_signals, axis=0)
    avg_derivative = gaussian_filter1d(np.gradient(avg_signal, x), sigma = 2, radius = 2)
    ax.plot(x_values, avg_signal, 'r-', linewidth=2.5,
            alpha=0.5, label='Average', zorder=100)

    # Find zero crossings of the average derivative
    zero_crossings_x = []
    zero_crossings_y = []
    for i in range(len(avg_derivative) - 1):
        if avg_derivative[i] * avg_derivative[i + 1] < 0:  # Sign change
            # Linear interpolation to find exact crossing point
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = avg_derivative[i], avg_derivative[i + 1]
            x_cross = x1 - y1 * (x2 - x1) / (y2 - y1)

            # Interpolate y-coordinate from avg_signal
            sig_y1, sig_y2 = avg_signal[i], avg_signal[i + 1]
            y_cross = sig_y1 + (x_cross - x1) * (sig_y2 - sig_y1) / (x2 - x1)

            zero_crossings_x.append(x_cross)
            zero_crossings_y.append(y_cross)

    # Plot zero crossings with green circles
    if zero_crossings_x:
        ax.plot(zero_crossings_x, zero_crossings_y, 'go',
                markersize=10, label='Zero Crossings', zorder=101)

    # Formatting
    ax.set_xlabel('Time/Position', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Average Signal Extrema', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return zero_crossings_x, zero_crossings_y

def average_signal(epochs_dict):
    all_signals = []
    x_values = None

    # Plot each epoch's derivative
    for epoch_label, df in epochs_dict.items():
        # Get x values (first column/index) and signal values
        x = df.index.values
        signal = df['Signal'].values

        # Store for averaging
        all_signals.append(signal)
        if x_values is None:
            x_values = x

    # Calculate and plot average derivative
    avg_signal = np.nanmedian(all_signals, axis=0)

    return avg_signal, x_values

def find_ascending_thresholds(avg_signal, x_values, thresholds=[0.0, 0.10, 0.25, 0.33, 0.50, 0.66, 0.75, 1.0]):
    """
    Find indices where signal crosses amplitude thresholds after global minimum.

    Parameters:
    -----------
    avg_signal : array
        The signal values
    x_values : array
        The corresponding x-axis values (time points)
    thresholds : list
        List of threshold percentages (e.g., 0.10 for 10%)

    Returns:
    --------
    dict : Dictionary with threshold percentages as keys and (index, x_value, signal_value) as values
    """

    # Find global minimum
    min_idx = np.argmin(avg_signal)
    max_idx = np.argmax(avg_signal)
    min_value = avg_signal[min_idx]

    # Find global maximum after the minimum
    max_value = np.max(avg_signal[min_idx:])

    # Calculate amplitude (peak-to-peak after minimum)
    amplitude = max_value - min_value

    # Initialize results dictionary
    results = {}
    print(f"\n{'='*60}")
    print("ASCENDING PHASE (After Global Minimum)")
    print(f"{'='*60}")
    print(f"Global minimum: {min_value:.4f} at index {min_idx} (x = {x_values[min_idx]:.4f})")
    print(f"Global maximum (after min): {max_value:.4f} at index {max_idx} (x = {x_values[max_idx]:.4f})")
    print(f"Amplitude: {amplitude:.4f}\n")

    # For each threshold, find first crossing after minimum
    for threshold in thresholds:
        target_value = min_value + (threshold * amplitude)

        # Search only after the minimum
        for i in range(min_idx + 1, len(avg_signal)):
            if avg_signal[i] >= target_value:
                results[threshold] = {
                    'index': i,
                    'x_value': x_values[i],
                    'signal_value': avg_signal[i],
                    'target_value': target_value
                }
                print(f"{int(threshold*100)}% threshold:")
                print(f"  Target value: {target_value:.4f}")
                print(f"  Crossed at index: {i}")
                print(f"  X value: {x_values[i]:.4f}")
                print(f"  Signal value: {avg_signal[i]:.4f}\n")
                break

    return results

def find_descending_thresholds(avg_signal, x_values, thresholds=[0.10, 0.25, 0.33, 0.50, 0.66, 0.75, 1.0]):
    """
    Find indices where signal crosses amplitude thresholds again after global maximum (descending).

    Parameters:
    -----------
    avg_signal : array
        The signal values
    x_values : array
        The corresponding x-axis values (time points)
    thresholds : list
        List of threshold percentages (e.g., 0.10 for 10%)

    Returns:
    --------
    dict : Dictionary with threshold percentages as keys and (index, x_value, signal_value) as values
    """

    # Find global minimum
    min_idx = np.argmin(avg_signal)
    min_value = avg_signal[min_idx]

    # Find global maximum after the minimum
    #max_idx = min_idx + np.argmax(avg_signal[min_idx:])
    max_idx = np.argmax(avg_signal)
    max_value = avg_signal[max_idx]

    # Calculate amplitude
    amplitude = max_value - min_value

    # Initialize results dictionary
    results = {}

    print(f"\n{'='*60}")
    print("DESCENDING PHASE (After Global Maximum)")
    print(f"{'='*60}")
    print(f"Global minimum: {min_value:.4f} at index {min_idx} (x = {x_values[min_idx]:.4f})")
    print(f"Global maximum: {max_value:.4f} at index {max_idx} (x = {x_values[max_idx]:.4f})")
    print(f"Amplitude: {amplitude:.4f}\n")

    # For each threshold, find crossing after maximum (descending)
    for threshold in thresholds:
        target_value = min_value + (threshold * amplitude)

        # Search only after the maximum
        for i in range(max_idx + 1, len(avg_signal)):
            # Check if signal crosses below the threshold (descending)
            if avg_signal[i] <= target_value:
                results[threshold] = {
                    'index': i,
                    'x_value': x_values[i],
                    'signal_value': avg_signal[i],
                    'target_value': target_value
                }
                print(f"{int(threshold*100)}% threshold (descending):")
                print(f"  Target value: {target_value:.4f}")
                print(f"  Crossed at index: {i}")
                print(f"  X value: {x_values[i]:.4f}")
                print(f"  Signal value: {avg_signal[i]:.4f}\n")
                break

    return results

SAMPLING_RATE = 125
HEART_RATE = 75
DURATION = 10.4

ppg = nk.ppg_simulate(heart_rate=HEART_RATE, duration=DURATION, sampling_rate=SAMPLING_RATE)
ppg_clean = nk.ppg_clean(ppg, method='nabian2018', sampling_rate=SAMPLING_RATE)
ppg_norm = NormalizeData(ppg_clean)

peaks = nk.ppg_findpeaks(ppg_norm, method="bishop", sampling_rate=SAMPLING_RATE, show=False)

# differences between consecutive peaks (in samples)
diffs = np.diff(np.array(peaks["PPG_Peaks"]))

# median interval in seconds
interval_sec = np.nanmedian(np.abs(diffs)) / SAMPLING_RATE
#print(interval_sec, "seconds")

ppg_epochs = nk.ppg_segment(ppg_norm, sampling_rate=SAMPLING_RATE, show=False)

zero_crossings_x, zero_crossings_y = plot_epochs(ppg_epochs)
print("Median Systolic Peak Amplitude: ", zero_crossings_y[1])
print("Systolic Peak x-position: ", zero_crossings_x[1],"\n")

print("Median Dicrotic Notch Amplitude: ", zero_crossings_y[2])
print("Dicrotic Notch x-position: ", zero_crossings_x[2],"\n")

print("Median Diastolic Peak Amplitude: ", zero_crossings_y[3])
print("Diastolic Peak x-position: ", zero_crossings_x[3])

avg_signal, x_values = average_signal(ppg_epochs)

# Find thresholds
ascending_results = find_ascending_thresholds(avg_signal, x_values)

# Find thresholds
descending_results = find_descending_thresholds(avg_signal, x_values)

#print(ascending_results)
#print(descending_results)

print("A - N:", ascending_results[1.0]['x_value'] - ascending_results[0.0]['x_value'])

print("B - M:", ascending_results[1.0]['x_value'] - ascending_results[0.1]['x_value'])

print("C - L:", ascending_results[1.0]['x_value'] - ascending_results[0.25]['x_value'])

print("D - K:", ascending_results[1.0]['x_value'] - ascending_results[0.33]['x_value'])

print("E - J:", ascending_results[1.0]['x_value'] - ascending_results[0.5]['x_value'])

print("F - I:", ascending_results[1.0]['x_value'] - ascending_results[0.66]['x_value'])

print("G - H:", ascending_results[1.0]['x_value'] - ascending_results[0.75]['x_value'])

print("O - N:", interval_sec - (ascending_results[1.0]['x_value'] - ascending_results[0.0]['x_value']))

print("P - M:", descending_results[0.1]['x_value'] - ascending_results[1.0]['x_value'])

print("Q - L:", descending_results[0.25]['x_value'] - ascending_results[1.0]['x_value'])

print("R - K:", descending_results[0.33]['x_value'] - ascending_results[1.0]['x_value'])

print("S - J:", descending_results[0.5]['x_value'] - ascending_results[1.0]['x_value'])

print("T - I:", descending_results[0.66]['x_value'] - ascending_results[1.0]['x_value'])

print("U - H:", descending_results[0.75]['x_value'] - ascending_results[1.0]['x_value'])

print("V - W:", interval_sec)