import neurokit2 as nk
import numpy as np
from scipy.ndimage import gaussian_filter1d

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def find_zero_crossings(epochs_dict):
    # Calculate and plot average signal
    avg_signal, x_values = average_signal(epochs_dict)
    avg_derivative = gaussian_filter1d(np.gradient(avg_signal, x_values), sigma = 1, radius = 1)

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

    return zero_crossings_x, zero_crossings_y

def average_signal(epochs_dict):
    all_signals = []
    x_values = None

    # Plot each epoch's derivative
    for _, df in epochs_dict.items():
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

def find_thresholds(avg_signal, x_values, thresholds=[0.25, 0.33, 0.50, 0.66, 0.75]):

    # Find global minimum
    min_idx = np.argmin(avg_signal)
    max_idx = np.argmax(avg_signal)
    min_value = avg_signal[min_idx]

    # Find global maximum after the minimum
    max_value = np.max(avg_signal[min_idx:])

    # Calculate amplitude (peak-to-peak after minimum)
    amplitude = max_value - min_value

    # Initialize results dictionary
    asc_results = {}
    desc_results = {}

    # For each threshold, find first crossing after minimum
    for threshold in thresholds:
        target_value = min_value + (threshold * amplitude)

        # Search only after the minimum
        for i in range(min_idx + 1, len(avg_signal)):
            if avg_signal[i] >= target_value:
                asc_results[threshold] = {
                    'x_value': x_values[i]
                }
                break

        # Search only after the maximum
        for i in range(max_idx + 1, len(avg_signal)):
            # Check if signal crosses below the threshold (descending)
            if avg_signal[i] <= target_value:
                desc_results[threshold] = {
                    'x_value': x_values[i]
                }
                break

    return asc_results, desc_results

def ppg_func_v1(ppg_norm, sampling_rate, print_opt=False):
    # Find peaks in PPG
    peaks = nk.ppg_findpeaks(ppg_norm, method="bishop", sampling_rate=sampling_rate, show=False)
    # differences between consecutive peaks (in samples)
    diffs = np.diff(np.array(peaks["PPG_Peaks"]))
    # median interval in seconds
    interval_sec = np.nanmedian(np.abs(diffs)) / sampling_rate

    ppg_epochs = nk.ppg_segment(ppg_norm, sampling_rate=sampling_rate, show=False)

    zero_crossings_x, zero_crossings_y = find_zero_crossings(ppg_epochs)

    avg_signal, x_values = average_signal(ppg_epochs)

    # Find thresholds
    ascending_results, descending_results = find_thresholds(avg_signal, x_values)

    avg_sys_amp = zero_crossings_y[1]
    avg_dic_notch_amp = zero_crossings_y[2]
    avg_dia_amp = zero_crossings_y[3]

    sys_dic_notch_time = zero_crossings_x[2] - zero_crossings_x[1]
    sys_dia_time = zero_crossings_x[3] - zero_crossings_x[1]

    a_n = zero_crossings_x[1] - zero_crossings_x[0]
    c_l = zero_crossings_x[1] - ascending_results[0.25]['x_value']
    d_k = zero_crossings_x[1] - ascending_results[0.33]['x_value']
    e_j = zero_crossings_x[1] - ascending_results[0.5]['x_value']
    f_i = zero_crossings_x[1] - ascending_results[0.66]['x_value']
    g_h = zero_crossings_x[1] - ascending_results[0.75]['x_value']
    o_n = interval_sec - (zero_crossings_x[1] - zero_crossings_x[0])
    q_l = descending_results[0.25]['x_value'] - zero_crossings_x[1]
    r_k = descending_results[0.33]['x_value'] - zero_crossings_x[1]
    s_j = descending_results[0.5]['x_value'] - zero_crossings_x[1]
    t_i = descending_results[0.66]['x_value'] - zero_crossings_x[1]
    u_h = descending_results[0.75]['x_value'] - zero_crossings_x[1]

    if print_opt:
        print("Median Systolic Peak Amplitude: ", zero_crossings_y[1])
        print("Systolic Peak x-position: ", zero_crossings_x[1],"\n")

        print("Median Dicrotic Notch Amplitude: ", zero_crossings_y[2])
        print("Dicrotic Notch x-position: ", zero_crossings_x[2],"\n")

        print("Median Diastolic Peak Amplitude: ", zero_crossings_y[3])
        print("Diastolic Peak x-position: ", zero_crossings_x[3])        

        print("A - N:", zero_crossings_x[1] - zero_crossings_x[0])
        print("C - L:", zero_crossings_x[1] - ascending_results[0.25]['x_value'])
        print("D - K:", zero_crossings_x[1] - ascending_results[0.33]['x_value'])
        print("E - J:", zero_crossings_x[1] - ascending_results[0.5]['x_value'])
        print("F - I:", zero_crossings_x[1] - ascending_results[0.66]['x_value'])
        print("G - H:", zero_crossings_x[1] - ascending_results[0.75]['x_value'])
        print("O - N:", interval_sec - (zero_crossings_x[1] - zero_crossings_x[0]))
        print("Q - L:", descending_results[0.25]['x_value'] - zero_crossings_x[1])
        print("R - K:", descending_results[0.33]['x_value'] - zero_crossings_x[1])
        print("S - J:", descending_results[0.5]['x_value'] - zero_crossings_x[1])
        print("T - I:", descending_results[0.66]['x_value'] - zero_crossings_x[1])
        print("U - H:", descending_results[0.75]['x_value'] - zero_crossings_x[1])

    return np.array([avg_sys_amp, avg_dic_notch_amp, avg_dia_amp, sys_dic_notch_time, sys_dia_time, a_n, c_l, d_k, e_j, f_i, g_h, o_n, q_l, r_k, s_j, t_i, u_h])

def ppg_func_v2(ppg_norm, sampling_rate, print_opt=False):
    try:
        # Find peaks in PPG
        peaks = nk.ppg_findpeaks(ppg_norm, method="bishop", sampling_rate=sampling_rate, show=False)
        
        # Check if peaks were found
        if peaks is None or "PPG_Peaks" not in peaks or len(peaks["PPG_Peaks"]) < 2:
            return None
        
        # differences between consecutive peaks (in samples)
        diffs = np.diff(np.array(peaks["PPG_Peaks"]))
        
        # median interval in seconds
        interval_sec = np.nanmedian(np.abs(diffs)) / sampling_rate
        
        # Check if interval is valid
        if np.isnan(interval_sec) or interval_sec == 0:
            return None

        ppg_epochs = nk.ppg_segment(ppg_norm, sampling_rate=sampling_rate, show=False)
        
        # Check if segmentation was successful
        if ppg_epochs is None or len(ppg_epochs) == 0:
            return None

        zero_crossings_x, zero_crossings_y = find_zero_crossings(ppg_epochs)
        
        # Check if zero crossings were found (need at least 4 points)
        if zero_crossings_x is None or zero_crossings_y is None or len(zero_crossings_x) < 4 or len(zero_crossings_y) < 4:
            return None

        avg_signal, x_values = average_signal(ppg_epochs)
        
        # Check if average signal is valid
        if avg_signal is None or x_values is None:
            return None

        # Find thresholds
        ascending_results, descending_results = find_thresholds(avg_signal, x_values)
        
        # Check if thresholds were found
        if ascending_results is None or descending_results is None:
            return None
        
        # Check if required threshold values exist
        required_ascending = [0.25, 0.33, 0.5, 0.66, 0.75]
        required_descending = [0.25, 0.33, 0.5, 0.66, 0.75]
        
        for threshold in required_ascending:
            if threshold not in ascending_results or 'x_value' not in ascending_results[threshold]:
                return None
        
        for threshold in required_descending:
            if threshold not in descending_results or 'x_value' not in descending_results[threshold]:
                return None

        avg_sys_amp = zero_crossings_y[1]
        avg_dic_notch_amp = zero_crossings_y[2]
        avg_dia_amp = zero_crossings_y[3]

        sys_dic_notch_time = zero_crossings_x[2] - zero_crossings_x[1]
        sys_dia_time = zero_crossings_x[3] - zero_crossings_x[1]

        a_n = zero_crossings_x[1] - zero_crossings_x[0]
        c_l = zero_crossings_x[1] - ascending_results[0.25]['x_value']
        d_k = zero_crossings_x[1] - ascending_results[0.33]['x_value']
        e_j = zero_crossings_x[1] - ascending_results[0.5]['x_value']
        f_i = zero_crossings_x[1] - ascending_results[0.66]['x_value']
        g_h = zero_crossings_x[1] - ascending_results[0.75]['x_value']
        o_n = interval_sec - (zero_crossings_x[1] - zero_crossings_x[0])
        q_l = descending_results[0.25]['x_value'] - zero_crossings_x[1]
        r_k = descending_results[0.33]['x_value'] - zero_crossings_x[1]
        s_j = descending_results[0.5]['x_value'] - zero_crossings_x[1]
        t_i = descending_results[0.66]['x_value'] - zero_crossings_x[1]
        u_h = descending_results[0.75]['x_value'] - zero_crossings_x[1]
        
        # Check for any None or NaN values in the calculated features
        features = [avg_sys_amp, avg_dic_notch_amp, avg_dia_amp, sys_dic_notch_time, sys_dia_time, 
                   a_n, c_l, d_k, e_j, f_i, g_h, o_n, q_l, r_k, s_j, t_i, u_h]
        
        if any(f is None or (isinstance(f, (int, float)) and np.isnan(f)) for f in features):
            return None

        if print_opt:
            print("Median Systolic Peak Amplitude: ", zero_crossings_y[1])
            print("Systolic Peak x-position: ", zero_crossings_x[1],"\n")

            print("Median Dicrotic Notch Amplitude: ", zero_crossings_y[2])
            print("Dicrotic Notch x-position: ", zero_crossings_x[2],"\n")

            print("Median Diastolic Peak Amplitude: ", zero_crossings_y[3])
            print("Diastolic Peak x-position: ", zero_crossings_x[3])        

            print("A - N:", a_n)
            print("C - L:", c_l)
            print("D - K:", d_k)
            print("E - J:", e_j)
            print("F - I:", f_i)
            print("G - H:", g_h)
            print("O - N:", o_n)
            print("Q - L:", q_l)
            print("R - K:", r_k)
            print("S - J:", s_j)
            print("T - I:", t_i)
            print("U - H:", u_h)

        return np.array(features)
    
    except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as e:
        # Return None if any error occurs during calculation
        return None

SAMPLING_RATE = 125
HEART_RATE = 75
DURATION = 10.4

ppg = nk.ppg_simulate(heart_rate=HEART_RATE, duration=DURATION, sampling_rate=SAMPLING_RATE)
ppg_clean = nk.ppg_clean(ppg, method='nabian2018', sampling_rate=SAMPLING_RATE)
ppg_norm = NormalizeData(ppg_clean)

print(ppg_func_v1(ppg_norm, SAMPLING_RATE))
print(ppg_func_v2(ppg_norm, SAMPLING_RATE))