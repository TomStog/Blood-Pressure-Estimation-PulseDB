import numpy as np
import neurokit2 as nk

SAMPLING_RATE = 125
DURATION = 10
HEART_RATE = 70

# Generate {DURATION} seconds of ECG signal (recorded at {SAMPLING_RATE} samples/second)
ecg = nk.ecg_simulate(duration=DURATION, sampling_rate=SAMPLING_RATE, heart_rate=HEART_RATE)

# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg, sampling_rate=SAMPLING_RATE)

# Delineate the ECG signal
signal_dwt, waves_dwt = nk.ecg_delineate(ecg, 
                                         rpeaks, 
                                         sampling_rate=SAMPLING_RATE, 
                                         method="dwt", 
                                         show=True, 
                                         show_type='all')

print("1. Average R-peak amplitude:", ecg[rpeaks['ECG_R_Peaks']].mean())

print("2. Average P-peak amplitude:", ecg[waves_dwt['ECG_P_Peaks']].mean())

print("3. Average P-onset amplitude:", ecg[waves_dwt['ECG_P_Onsets']].mean())

print("4. Average P-offset amplitude:", ecg[waves_dwt['ECG_P_Offsets']].mean())

print("5. Average Q-peak amplitude:", ecg[waves_dwt['ECG_Q_Peaks']].mean())

print("6. Average R-onset amplitude:", ecg[waves_dwt['ECG_R_Onsets']].mean())

print("7. Average R-offset amplitude:", ecg[waves_dwt['ECG_R_Offsets']].mean())

print("8. Average S-peak amplitude:", ecg[waves_dwt['ECG_S_Peaks']].mean())

print("9. Average T-peak amplitude:", ecg[waves_dwt['ECG_T_Peaks']].mean())

print("10. Average T-onset amplitude:", ecg[waves_dwt['ECG_T_Onsets']].mean())

print("11. Average T-offset amplitude:", ecg[waves_dwt['ECG_T_Offsets']].mean())

# RR - interval
rr_interval = []
for idx in range(len(rpeaks['ECG_R_Peaks']) - 1):
    rr_interval.append(rpeaks['ECG_R_Peaks'][idx + 1] - rpeaks['ECG_R_Peaks'][idx])
print("12. Average RR-interval:", np.nanmedian(rr_interval)/SAMPLING_RATE, 'seconds')

# PP - interval
pp_interval= []
for idx in range(len(waves_dwt['ECG_P_Peaks']) - 1):
    pp_interval.append(waves_dwt['ECG_P_Peaks'][idx + 1] - waves_dwt['ECG_P_Peaks'][idx])
print("13. Average PP-interval:", np.nanmedian(pp_interval)/SAMPLING_RATE, 'seconds')

# P - duration
p_duration = []
for idx in range(len(waves_dwt['ECG_P_Onsets'])):
    p_duration.append(waves_dwt['ECG_P_Offsets'][idx] - waves_dwt['ECG_P_Onsets'][idx])
print("14. Average P-duration:", np.nanmedian(p_duration)/SAMPLING_RATE, 'seconds')

# PR - segment
pr_segment = []
for idx in range(len(waves_dwt['ECG_P_Offsets'])):
    pr_segment.append(waves_dwt['ECG_R_Onsets'][idx]-waves_dwt['ECG_P_Offsets'][idx])
print("15. Average PR-segment:", np.nanmedian(pr_segment)/SAMPLING_RATE, 'seconds')

# PR - interval
print("16. Average PR-interval:", (np.nanmedian(p_duration) + np.nanmedian(pr_segment))/SAMPLING_RATE, 'seconds')

# QRS - duration
qrs_duration = []
for idx in range(len(waves_dwt['ECG_R_Offsets'])):
    qrs_duration.append(waves_dwt['ECG_R_Offsets'][idx] - waves_dwt['ECG_R_Onsets'][idx])
print("17. Average QRS-duration:", np.nanmedian(qrs_duration)/SAMPLING_RATE, 'seconds')

# ST - segment
st_segment = []
for idx in range(len(waves_dwt['ECG_T_Onsets'])):
    st_segment.append(waves_dwt['ECG_T_Onsets'][idx]-waves_dwt['ECG_R_Offsets'][idx])
print("18. Average ST-segment:", np.nanmedian(st_segment)/SAMPLING_RATE, 'seconds')

# ST_T - segment
st_t_segment = []
for idx in range(len(waves_dwt['ECG_T_Offsets'])):
    st_t_segment.append(waves_dwt['ECG_T_Offsets'][idx]-waves_dwt['ECG_R_Offsets'][idx])
print("19. Average ST_T-segment:", np.nanmedian(st_t_segment)/SAMPLING_RATE, 'seconds')

# TP - interval
tp_interval = []
for idx in range(len(waves_dwt['ECG_T_Offsets']) - 1):
    tp_interval.append(waves_dwt['ECG_P_Onsets'][idx + 1] - waves_dwt['ECG_T_Offsets'][idx])
print("20. Average TP-interval:", np.nanmedian(tp_interval)/SAMPLING_RATE, 'seconds')

# QT - duration
qt_duration = []
for idx in range(len(waves_dwt['ECG_T_Offsets'])):
    qt_duration.append(waves_dwt['ECG_T_Offsets'][idx] - waves_dwt['ECG_R_Onsets'][idx])
print("21. Average QT-duration:", np.nanmedian(qt_duration)/SAMPLING_RATE, 'seconds')