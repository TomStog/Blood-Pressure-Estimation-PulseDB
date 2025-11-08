import neurokit2 as nk
import numpy as np

def ecg_func_v1(ecg, sampling_rate, print_opt = False):
    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    
    # Delineate the normalized ECG signal
    _, waves_dwt = nk.ecg_delineate(ecg, 
                                    rpeaks, 
                                    sampling_rate=sampling_rate, 
                                    method="dwt", 
                                    show=False, 
                                    show_type='all')
    
    avg_heart_rate = 60/(np.nanmedian(np.diff(rpeaks['ECG_R_Peaks']))/sampling_rate)
    avg_rpeak_amp = ecg[rpeaks['ECG_R_Peaks']].mean()
    avg_ppeak_amp = ecg[waves_dwt['ECG_P_Peaks']].mean()
    avg_ponset_amp = ecg[waves_dwt['ECG_P_Onsets']].mean()
    avg_poffset_amp = ecg[waves_dwt['ECG_P_Offsets']].mean()
    avg_qpeak_amp = ecg[waves_dwt['ECG_Q_Peaks']].mean()
    avg_ronset_amp = ecg[waves_dwt['ECG_R_Onsets']].mean()
    avg_roffset_amp = ecg[waves_dwt['ECG_R_Offsets']].mean()
    avg_speak_amp = ecg[waves_dwt['ECG_S_Peaks']].mean()
    avg_tpeak_amp = ecg[waves_dwt['ECG_T_Peaks']].mean()
    avg_tonset_amp = ecg[waves_dwt['ECG_T_Onsets']].mean()
    avg_toffset_amp = ecg[waves_dwt['ECG_T_Offsets']].mean()

    rr_interval = []
    for idx in range(len(rpeaks['ECG_R_Peaks']) - 1):
        rr_interval.append(rpeaks['ECG_R_Peaks'][idx + 1] - rpeaks['ECG_R_Peaks'][idx])
    avg_rr_interval = np.nanmedian(rr_interval)/sampling_rate

    pp_interval= []
    for idx in range(len(waves_dwt['ECG_P_Peaks']) - 1):
        pp_interval.append(waves_dwt['ECG_P_Peaks'][idx + 1] - waves_dwt['ECG_P_Peaks'][idx])
    avg_pp_interval = np.nanmedian(pp_interval)/sampling_rate

    p_duration = []
    for idx in range(len(waves_dwt['ECG_P_Onsets'])):
        p_duration.append(waves_dwt['ECG_P_Offsets'][idx] - waves_dwt['ECG_P_Onsets'][idx])
    avg_p_duration = np.nanmedian(p_duration)/sampling_rate

    pr_segment = []
    for idx in range(len(waves_dwt['ECG_P_Offsets'])):
        pr_segment.append(waves_dwt['ECG_R_Onsets'][idx]-waves_dwt['ECG_P_Offsets'][idx])
    avg_pr_seg = np.nanmedian(pr_segment)/sampling_rate

    avg_pr_interval = (np.nanmedian(p_duration) + np.nanmedian(pr_segment))/sampling_rate

    qrs_duration = []
    for idx in range(len(waves_dwt['ECG_R_Offsets'])):
        qrs_duration.append(waves_dwt['ECG_R_Offsets'][idx] - waves_dwt['ECG_R_Onsets'][idx])
    avg_qrs_duration = np.nanmedian(qrs_duration)/sampling_rate

    st_segment = []
    for idx in range(len(waves_dwt['ECG_T_Onsets'])):
        st_segment.append(waves_dwt['ECG_T_Onsets'][idx]-waves_dwt['ECG_R_Offsets'][idx])
    avg_st_seg = np.nanmedian(st_segment)/sampling_rate

    st_t_segment = []
    for idx in range(len(waves_dwt['ECG_T_Offsets'])):
        st_t_segment.append(waves_dwt['ECG_T_Offsets'][idx]-waves_dwt['ECG_R_Offsets'][idx])
    avg_st_t_seg = np.nanmedian(st_t_segment)/sampling_rate

    tp_interval = []
    for idx in range(len(waves_dwt['ECG_T_Offsets']) - 1):
        tp_interval.append(waves_dwt['ECG_P_Onsets'][idx + 1] - waves_dwt['ECG_T_Offsets'][idx])
    avg_tp_interval = np.nanmedian(tp_interval)/sampling_rate

    qt_duration = []
    for idx in range(len(waves_dwt['ECG_T_Offsets'])):
        qt_duration.append(waves_dwt['ECG_T_Offsets'][idx] - waves_dwt['ECG_R_Onsets'][idx])
    avg_qt_duration = np.nanmedian(qt_duration)/sampling_rate

    if print_opt:
        print("0. Average Heart Rate: ", avg_heart_rate)
        print("1. Average R-peak amplitude:", avg_rpeak_amp)
        print("2. Average P-peak amplitude:", avg_ppeak_amp)
        print("3. Average P-onset amplitude:", avg_ponset_amp)
        print("4. Average P-offset amplitude:", avg_poffset_amp)
        print("5. Average Q-peak amplitude:", avg_qpeak_amp)
        print("6. Average R-onset amplitude:", avg_ronset_amp)
        print("7. Average R-offset amplitude:", avg_roffset_amp)
        print("8. Average S-peak amplitude:", avg_speak_amp)
        print("9. Average T-peak amplitude:", avg_tpeak_amp)
        print("10. Average T-onset amplitude:", avg_tonset_amp)
        print("11. Average T-offset amplitude:", avg_toffset_amp)
        print("12. Average RR-interval:", avg_rr_interval, 'seconds')
        print("13. Average PP-interval:", avg_pp_interval, 'seconds')
        print("14. Average P-duration:", avg_p_duration, 'seconds')
        print("15. Average PR-segment:", avg_pr_seg, 'seconds')
        print("16. Average PR-interval:", avg_pr_interval, 'seconds')
        print("17. Average QRS-duration:", avg_qrs_duration, 'seconds')
        print("18. Average ST-segment:", avg_st_seg, 'seconds')
        print("19. Average ST_T-segment:", avg_st_t_seg, 'seconds')
        print("20. Average TP-interval:", avg_tp_interval, 'seconds')
        print("21. Average QT-duration:", avg_qt_duration, 'seconds')

    return np.array([avg_heart_rate, avg_rpeak_amp, avg_ppeak_amp, avg_ponset_amp, avg_poffset_amp, avg_qpeak_amp, avg_ronset_amp, avg_roffset_amp, avg_speak_amp, avg_tpeak_amp, avg_tonset_amp, avg_toffset_amp, avg_rr_interval, avg_pp_interval, avg_p_duration, avg_pr_seg, avg_pr_interval, avg_qrs_duration, avg_st_seg, avg_st_t_seg, avg_tp_interval, avg_qt_duration])

def ecg_func_v2(ecg, sampling_rate, print_opt=False):
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    _, waves_dwt = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, 
                                    method="dwt", show=False, show_type='all')
    
    # Direct computations without intermediate lists
    avg_heart_rate = 60 / (np.nanmedian(np.diff(rpeaks['ECG_R_Peaks'])) / sampling_rate)
    
    # Amplitudes
    avg_rpeak_amp = ecg[rpeaks['ECG_R_Peaks']].mean()
    avg_ppeak_amp = ecg[waves_dwt['ECG_P_Peaks']].mean()
    avg_ponset_amp = ecg[waves_dwt['ECG_P_Onsets']].mean()
    avg_poffset_amp = ecg[waves_dwt['ECG_P_Offsets']].mean()
    avg_qpeak_amp = ecg[waves_dwt['ECG_Q_Peaks']].mean()
    avg_ronset_amp = ecg[waves_dwt['ECG_R_Onsets']].mean()
    avg_roffset_amp = ecg[waves_dwt['ECG_R_Offsets']].mean()
    avg_speak_amp = ecg[waves_dwt['ECG_S_Peaks']].mean()
    avg_tpeak_amp = ecg[waves_dwt['ECG_T_Peaks']].mean()
    avg_tonset_amp = ecg[waves_dwt['ECG_T_Onsets']].mean()
    avg_toffset_amp = ecg[waves_dwt['ECG_T_Offsets']].mean()
    
    # Intervals using numpy operations (convert lists to arrays)
    avg_rr_interval = np.nanmedian(np.diff(rpeaks['ECG_R_Peaks'])) / sampling_rate
    avg_pp_interval = np.nanmedian(np.diff(waves_dwt['ECG_P_Peaks'])) / sampling_rate
    
    # Convert to arrays for arithmetic operations
    p_offsets = np.array(waves_dwt['ECG_P_Offsets'])
    p_onsets = np.array(waves_dwt['ECG_P_Onsets'])
    r_onsets = np.array(waves_dwt['ECG_R_Onsets'])
    r_offsets = np.array(waves_dwt['ECG_R_Offsets'])
    t_onsets = np.array(waves_dwt['ECG_T_Onsets'])
    t_offsets = np.array(waves_dwt['ECG_T_Offsets'])
    
    avg_p_duration = np.nanmedian(p_offsets - p_onsets) / sampling_rate
    avg_pr_seg = np.nanmedian(r_onsets - p_offsets) / sampling_rate
    avg_pr_interval = np.nanmedian((p_offsets - p_onsets) + (r_onsets - p_offsets)) / sampling_rate
    
    avg_qrs_duration = np.nanmedian(r_offsets - r_onsets) / sampling_rate
    avg_st_seg = np.nanmedian(t_onsets - r_offsets) / sampling_rate
    avg_st_t_seg = np.nanmedian(t_offsets - r_offsets) / sampling_rate
    
    # TP interval needs offset indexing
    avg_tp_interval = np.nanmedian(p_onsets[1:] - t_offsets[:-1]) / sampling_rate
    
    avg_qt_duration = np.nanmedian(t_offsets - r_onsets) / sampling_rate
    
    if print_opt:
        print("0. Average Heart Rate: ", avg_heart_rate)
        print("1. Average R-peak amplitude:", avg_rpeak_amp)
        print("2. Average P-peak amplitude:", avg_ppeak_amp)
        print("3. Average P-onset amplitude:", avg_ponset_amp)
        print("4. Average P-offset amplitude:", avg_poffset_amp)
        print("5. Average Q-peak amplitude:", avg_qpeak_amp)
        print("6. Average R-onset amplitude:", avg_ronset_amp)
        print("7. Average R-offset amplitude:", avg_roffset_amp)
        print("8. Average S-peak amplitude:", avg_speak_amp)
        print("9. Average T-peak amplitude:", avg_tpeak_amp)
        print("10. Average T-onset amplitude:", avg_tonset_amp)
        print("11. Average T-offset amplitude:", avg_toffset_amp)
        print("12. Average RR-interval:", avg_rr_interval, 'seconds')
        print("13. Average PP-interval:", avg_pp_interval, 'seconds')
        print("14. Average P-duration:", avg_p_duration, 'seconds')
        print("15. Average PR-segment:", avg_pr_seg, 'seconds')
        print("16. Average PR-interval:", avg_pr_interval, 'seconds')
        print("17. Average QRS-duration:", avg_qrs_duration, 'seconds')
        print("18. Average ST-segment:", avg_st_seg, 'seconds')
        print("19. Average ST_T-segment:", avg_st_t_seg, 'seconds')
        print("20. Average TP-interval:", avg_tp_interval, 'seconds')
        print("21. Average QT-duration:", avg_qt_duration, 'seconds')
    
    return np.array([avg_heart_rate, avg_rpeak_amp, avg_ppeak_amp, avg_ponset_amp, 
                     avg_poffset_amp, avg_qpeak_amp, avg_ronset_amp, avg_roffset_amp, 
                     avg_speak_amp, avg_tpeak_amp, avg_tonset_amp, avg_toffset_amp, 
                     avg_rr_interval, avg_pp_interval, avg_p_duration, avg_pr_seg, 
                     avg_pr_interval, avg_qrs_duration, avg_st_seg, avg_st_t_seg, 
                     avg_tp_interval, avg_qt_duration])

def ecg_func_v3(ecg, sampling_rate, print_opt=False):
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
    _, waves_dwt = nk.ecg_delineate(ecg, rpeaks, sampling_rate=sampling_rate, 
                                    method="dwt", show=False, show_type='all')
    
    # Helper function to safely compute mean amplitude
    def safe_mean_amp(indices):
        if indices is None or len(indices) == 0:
            return None
        indices_arr = np.array(indices)
        valid_indices = indices_arr[~np.isnan(indices_arr)].astype(int)
        if len(valid_indices) == 0:
            return None
        return ecg[valid_indices].mean()
    
    # Helper function to safely compute median interval
    def safe_median_interval(values):
        if values is None or len(values) == 0:
            return None
        values_arr = np.array(values)
        valid_values = values_arr[~np.isnan(values_arr)]
        if len(valid_values) < 2:
            return None
        result = np.nanmedian(np.diff(valid_values)) / sampling_rate
        return None if np.isnan(result) else result
    
    # Helper function to safely compute duration
    def safe_duration(arr1, arr2):
        if arr1 is None or arr2 is None or len(arr1) == 0 or len(arr2) == 0:
            return None
        arr1_clean = np.array(arr1)[~np.isnan(arr1)]
        arr2_clean = np.array(arr2)[~np.isnan(arr2)]
        min_len = min(len(arr1_clean), len(arr2_clean))
        if min_len == 0:
            return None
        result = np.nanmedian(arr1_clean[:min_len] - arr2_clean[:min_len]) / sampling_rate
        return None if np.isnan(result) else result
    
    # Heart rate
    avg_heart_rate = None
    if 'ECG_R_Peaks' in rpeaks and len(rpeaks['ECG_R_Peaks']) > 1:
        rr_med = np.nanmedian(np.diff(rpeaks['ECG_R_Peaks'])) / sampling_rate
        if not np.isnan(rr_med) and rr_med > 0:
            avg_heart_rate = 60 / rr_med
    
    # Amplitudes
    avg_rpeak_amp = safe_mean_amp(rpeaks.get('ECG_R_Peaks'))
    avg_ppeak_amp = safe_mean_amp(waves_dwt.get('ECG_P_Peaks'))
    avg_ponset_amp = safe_mean_amp(waves_dwt.get('ECG_P_Onsets'))
    avg_poffset_amp = safe_mean_amp(waves_dwt.get('ECG_P_Offsets'))
    avg_qpeak_amp = safe_mean_amp(waves_dwt.get('ECG_Q_Peaks'))
    avg_ronset_amp = safe_mean_amp(waves_dwt.get('ECG_R_Onsets'))
    avg_roffset_amp = safe_mean_amp(waves_dwt.get('ECG_R_Offsets'))
    avg_speak_amp = safe_mean_amp(waves_dwt.get('ECG_S_Peaks'))
    avg_tpeak_amp = safe_mean_amp(waves_dwt.get('ECG_T_Peaks'))
    avg_tonset_amp = safe_mean_amp(waves_dwt.get('ECG_T_Onsets'))
    avg_toffset_amp = safe_mean_amp(waves_dwt.get('ECG_T_Offsets'))
    
    # Basic intervals
    avg_rr_interval = safe_median_interval(rpeaks.get('ECG_R_Peaks'))
    avg_pp_interval = safe_median_interval(waves_dwt.get('ECG_P_Peaks'))
    
    # Durations and segments
    avg_p_duration = safe_duration(
        waves_dwt.get('ECG_P_Offsets'), 
        waves_dwt.get('ECG_P_Onsets')
    )
    
    avg_pr_seg = safe_duration(
        waves_dwt.get('ECG_R_Onsets'), 
        waves_dwt.get('ECG_P_Offsets')
    )
    
    # PR interval (P duration + PR segment)
    avg_pr_interval = None
    if all(k in waves_dwt for k in ['ECG_P_Onsets', 'ECG_P_Offsets', 'ECG_R_Onsets']):
        p_onsets = np.array(waves_dwt['ECG_P_Onsets'])[~np.isnan(waves_dwt['ECG_P_Onsets'])]
        p_offsets = np.array(waves_dwt['ECG_P_Offsets'])[~np.isnan(waves_dwt['ECG_P_Offsets'])]
        r_onsets = np.array(waves_dwt['ECG_R_Onsets'])[~np.isnan(waves_dwt['ECG_R_Onsets'])]
        min_len = min(len(p_onsets), len(p_offsets), len(r_onsets))
        if min_len > 0:
            result = np.nanmedian((p_offsets[:min_len] - p_onsets[:min_len]) + 
                                 (r_onsets[:min_len] - p_offsets[:min_len])) / sampling_rate
            avg_pr_interval = None if np.isnan(result) else result
    
    avg_qrs_duration = safe_duration(
        waves_dwt.get('ECG_R_Offsets'), 
        waves_dwt.get('ECG_R_Onsets')
    )
    
    avg_st_seg = safe_duration(
        waves_dwt.get('ECG_T_Onsets'), 
        waves_dwt.get('ECG_R_Offsets')
    )
    
    avg_st_t_seg = safe_duration(
        waves_dwt.get('ECG_T_Offsets'), 
        waves_dwt.get('ECG_R_Offsets')
    )
    
    # TP interval (needs offset indexing)
    avg_tp_interval = None
    if all(k in waves_dwt for k in ['ECG_P_Onsets', 'ECG_T_Offsets']):
        p_onsets = np.array(waves_dwt['ECG_P_Onsets'])[~np.isnan(waves_dwt['ECG_P_Onsets'])]
        t_offsets = np.array(waves_dwt['ECG_T_Offsets'])[~np.isnan(waves_dwt['ECG_T_Offsets'])]
        if len(p_onsets) > 1 and len(t_offsets) > 0:
            min_len = min(len(p_onsets) - 1, len(t_offsets))
            if min_len > 0:
                result = np.nanmedian(p_onsets[1:min_len+1] - t_offsets[:min_len]) / sampling_rate
                avg_tp_interval = None if np.isnan(result) else result
    
    avg_qt_duration = safe_duration(
        waves_dwt.get('ECG_T_Offsets'), 
        waves_dwt.get('ECG_R_Onsets')
    )
    
    if print_opt:
        print("0. Average Heart Rate: ", avg_heart_rate)
        print("1. Average R-peak amplitude:", avg_rpeak_amp)
        print("2. Average P-peak amplitude:", avg_ppeak_amp)
        print("3. Average P-onset amplitude:", avg_ponset_amp)
        print("4. Average P-offset amplitude:", avg_poffset_amp)
        print("5. Average Q-peak amplitude:", avg_qpeak_amp)
        print("6. Average R-onset amplitude:", avg_ronset_amp)
        print("7. Average R-offset amplitude:", avg_roffset_amp)
        print("8. Average S-peak amplitude:", avg_speak_amp)
        print("9. Average T-peak amplitude:", avg_tpeak_amp)
        print("10. Average T-onset amplitude:", avg_tonset_amp)
        print("11. Average T-offset amplitude:", avg_toffset_amp)
        print("12. Average RR-interval:", avg_rr_interval, 'seconds' if avg_rr_interval else '')
        print("13. Average PP-interval:", avg_pp_interval, 'seconds' if avg_pp_interval else '')
        print("14. Average P-duration:", avg_p_duration, 'seconds' if avg_p_duration else '')
        print("15. Average PR-segment:", avg_pr_seg, 'seconds' if avg_pr_seg else '')
        print("16. Average PR-interval:", avg_pr_interval, 'seconds' if avg_pr_interval else '')
        print("17. Average QRS-duration:", avg_qrs_duration, 'seconds' if avg_qrs_duration else '')
        print("18. Average ST-segment:", avg_st_seg, 'seconds' if avg_st_seg else '')
        print("19. Average ST_T-segment:", avg_st_t_seg, 'seconds' if avg_st_t_seg else '')
        print("20. Average TP-interval:", avg_tp_interval, 'seconds' if avg_tp_interval else '')
        print("21. Average QT-duration:", avg_qt_duration, 'seconds' if avg_qt_duration else '')
    
    # Collect all features
    features = [avg_heart_rate, avg_rpeak_amp, avg_ppeak_amp, avg_ponset_amp, 
                avg_poffset_amp, avg_qpeak_amp, avg_ronset_amp, avg_roffset_amp, 
                avg_speak_amp, avg_tpeak_amp, avg_tonset_amp, avg_toffset_amp, 
                avg_rr_interval, avg_pp_interval, avg_p_duration, avg_pr_seg, 
                avg_pr_interval, avg_qrs_duration, avg_st_seg, avg_st_t_seg, 
                avg_tp_interval, avg_qt_duration]
    
    # Return None if any feature couldn't be calculated
    if any(f is None for f in features):
        return None
    
    return np.array(features)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

SAMPLING_RATE = 125
DURATION = 10
HEART_RATE = 70

# Generate {DURATION} seconds of ECG signal (recorded at {SAMPLING_RATE} samples/second)
ecg = nk.ecg_simulate(duration=DURATION, sampling_rate=SAMPLING_RATE, heart_rate=HEART_RATE)
features_1 = ecg_func_v1(NormalizeData(ecg), sampling_rate=SAMPLING_RATE)
features_2 = ecg_func_v2(NormalizeData(ecg), sampling_rate=SAMPLING_RATE)
features_3 = ecg_func_v3(NormalizeData(ecg), sampling_rate=SAMPLING_RATE)

print(features_1)
print(features_2)
print(features_3)