import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt
import pyabf

# ============================================================
#                  GLOBAL USER SETTINGS
# ============================================================
fPSP_direction = 'min'        # 'min' or 'max'
apply_highpass_filter = True
highpass_cutoff, highpass_order = 1.0, 2
root = '/Users/gs075/Desktop/Data/LFP/HF_RSP_Plexxikon/LFP_input'
output_path = '/Users/gs075/Desktop/Data/LFP/HF_RSP_Plexxikon/LFP_output'
delay_after_stim_s = 0.0008
fPSP_min_deflection = 0.05
min_valid_amplitude, max_valid_amplitude = fPSP_min_deflection, 0.4
zoom_window, fPSP_window = 0.02, 0.005
artifact_search_window_ms = 1.0
stim_start, stim_interval, n_stimulations = 8.969, 10.0, 50
amplitudes = np.linspace(50, 500, 10)
n_repeats = int(np.ceil(n_stimulations / len(amplitudes)))
currents = np.tile(amplitudes, n_repeats)[:n_stimulations]

os.makedirs(output_path, exist_ok=True)

# ============================================================
#                  Load All ABF Files into Memory
# ============================================================
abf_files = [f for f in os.listdir(root) if f.lower().endswith('.abf')]
abf_files.sort()
recordings_data = []

def highpass_filter(sig, fs, cutoff=1.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, sig)

print("\nLoading all recordings...")

for recording_file in abf_files:
    recording = os.path.splitext(recording_file)[0]
    abf_path = os.path.join(root, recording_file)
    csv_path = os.path.join(output_path, f"LFP_results_{recording}.csv")

    abf = pyabf.ABF(abf_path)
    abf.setSweep(0, channel=0)
    fs = abf.dataRate
    abf_filtered = highpass_filter(abf.sweepY, fs, highpass_cutoff, highpass_order) if apply_highpass_filter else abf.sweepY.copy()

    # detect stimulation artifacts
    artifact_search_window = artifact_search_window_ms / 1000.0
    artifact_indices = []
    L = len(abf_filtered)
    for i in range(n_stimulations):
        t = stim_start + i * stim_interval
        s = max(int((t - artifact_search_window) * fs), 0)
        e = min(int((t + artifact_search_window) * fs), L)
        seg = abf_filtered[s:e]
        if seg.size:
            artifact_indices.append(s + int(np.argmax(np.abs(seg))))
    stim_times = np.array(artifact_indices) / fs

    # initialize baseline & peak storage
    fPSP_peaks = [None] * len(stim_times)
    baseline_times = []
    df_existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    # detect fPSP peaks & baselines
    for idx, stim_time in enumerate(stim_times):
        stim_idx = int(stim_time * fs)
        artifact_samples = int(artifact_search_window * fs)
        window = abf_filtered[stim_idx:stim_idx + artifact_samples]
        artifact_idx = stim_idx + int(np.argmax(np.abs(window))) if window.size else stim_idx
        search_start = artifact_idx + int(delay_after_stim_s * fs)
        search_end = search_start + int(fPSP_window * fs)
        seg = abf_filtered[search_start:search_end]
        if seg.size:
            local_peak = (np.argmax(seg) if fPSP_direction == 'max' else np.argmin(seg))
            peak_idx = search_start + local_peak
        else:
            peak_idx = search_start
        peak_val = abf_filtered[peak_idx]
        if not (min_valid_amplitude <= abs(peak_val) <= max_valid_amplitude):
            peak_val = np.nan
            peak_idx = search_start
        fPSP_peaks[idx] = (abf.sweepX[peak_idx], peak_val)

        if df_existing is not None and idx < len(df_existing):
            baseline_times.append([
                float(df_existing.loc[idx, 'Baseline1 Time (s)']),
                float(df_existing.loc[idx, 'Baseline1 Vm (mV)']),
                float(df_existing.loc[idx, 'Baseline2 Time (s)']) if not pd.isna(df_existing.loc[idx, 'Baseline2 Time (s)']) else None,
                float(df_existing.loc[idx, 'Baseline2 Vm (mV)']) if not pd.isna(df_existing.loc[idx, 'Baseline2 Vm (mV)']) else None
            ])
        else:
            baseline_times.append([stim_time, abf_filtered[int(stim_time * fs)], None, None])

    recordings_data.append({
        "recording": recording,
        "abf": abf,
        "abf_filtered": abf_filtered,
        "fs": fs,
        "stim_times": stim_times,
        "fPSP_peaks": fPSP_peaks,
        "baseline_times": baseline_times,
        "csv_path": csv_path
    })

print(f"Loaded {len(recordings_data)} recordings.\n")

# ============================================================
#                  Interactive Threshold GUI
# ============================================================
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].axhline(0, ls='--', color='k')
axs[0].set_ylabel('Vm (mV)')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Current (pA)')

current_recording = 0
current_index = 0
dragging_base = None

def generate_biphasic_pulse(amplitude_pA, duration_us=100, fs_wave=1e6, window_ms=1):
    total = max(int(duration_us * 1e-6 * fs_wave), 2)
    half = total // 2
    pulse = np.zeros(total)
    pulse[:half] = amplitude_pA
    pulse[half:] = -amplitude_pA
    window_samples = int(window_ms * 1e-3 * fs_wave)
    full = np.zeros(window_samples)
    c = window_samples // 2
    s = c - total // 2
    full[s:s + total] = pulse
    t_ms = (np.arange(window_samples) - c) / fs_wave * 1000
    return t_ms, full

def display_stim():
    rec = recordings_data[current_recording]
    abf, abf_filtered, fs = rec["abf"], rec["abf_filtered"], rec["fs"]
    stim_times, fPSP_peaks, baseline_times = rec["stim_times"], rec["fPSP_peaks"], rec["baseline_times"]
    idx = current_index
    L = len(abf_filtered)
    recording = rec["recording"]

    peak_time, peak_val = fPSP_peaks[idx]
    b1_t, b1_v, b2_t, b2_v = baseline_times[idx]
    stim_time = stim_times[idx]

    i0 = max(0, int((stim_time - zoom_window / 2) * fs))
    i1 = min(L, int((stim_time + zoom_window / 2) * fs))
    x_full = abf.sweepX[i0:i1]
    y_full = abf_filtered[i0:i1]

    axs[0].cla()
    axs[0].plot(x_full, y_full, lw=1, color='dimgrey')
    axs[0].axhline(0, ls='--', color='k')
    axs[0].axvline(stim_time, color='red', ls='--', lw=1)
    axs[0].axvspan(stim_time + delay_after_stim_s,
                   stim_time + delay_after_stim_s + fPSP_window,
                   color='steelblue', alpha=0.2)
    if not np.isnan(peak_val):
        axs[0].plot(peak_time, peak_val, 'go', ms=6)
    axs[0].plot(b1_t, b1_v, 'o', color='orange', ms=6)
    if b2_t is not None:
        axs[0].plot(b2_t, b2_v, 'o', color='orangered', ms=6)
    axs[0].set_ylim(-0.4, 0.2)
    axs[0].set_title(f"{recording} — Stim {idx+1}/{len(stim_times)} "
                     f"({current_recording+1}/{len(recordings_data)})")

    axs[1].cla()
    t_wave, y_wave = generate_biphasic_pulse(currents[idx])
    axs[1].plot(t_wave, y_wave, color='maroon')
    axs[1].set_xlim(-0.5, 0.5)
    axs[1].set_ylim(-600, 600)
    axs[1].set_title(f"Stimulation Waveform — {int(currents[idx])} pA")
    fig.canvas.draw_idle()

def on_key(event):
    global current_recording, current_index
    rec = recordings_data[current_recording]
    n_stim = len(rec["stim_times"])
    if event.key == 'right':
        if current_index < n_stim - 1:
            current_index += 1
        else:
            if current_recording < len(recordings_data) - 1:
                current_recording += 1
                current_index = 0
            else:
                print("Reached last recording.")
    elif event.key == 'left':
        if current_index > 0:
            current_index -= 1
        else:
            if current_recording > 0:
                current_recording -= 1
                current_index = len(recordings_data[current_recording]["stim_times"]) - 1
            else:
                print("At first recording.")
    display_stim()

fig.canvas.mpl_connect('key_press_event', on_key)

# simple dragging hooks (kept minimal for clarity)
def on_press(event):
    global dragging_base
    if event.inaxes != axs[0]:
        return
    rec = recordings_data[current_recording]
    b1_t, _, b2_t, _ = rec["baseline_times"][current_index]
    d1 = abs(event.xdata - (b1_t or 0))
    d2 = abs(event.xdata - (b2_t or 1e9))
    if d1 < 0.001:
        dragging_base = 'base1'
    elif d2 < 0.001:
        dragging_base = 'base2'

def on_motion(event):
    global dragging_base
    if dragging_base is None or event.inaxes != axs[0]:
        return
    rec = recordings_data[current_recording]
    abf_filtered = rec["abf_filtered"]
    fs = rec["fs"]
    xi = int(event.xdata * fs)
    if xi < 0 or xi >= len(abf_filtered):
        return
    yv = abf_filtered[xi]
    if dragging_base == 'base1':
        rec["baseline_times"][current_index][0] = event.xdata
        rec["baseline_times"][current_index][1] = yv
    else:
        rec["baseline_times"][current_index][2] = event.xdata
        rec["baseline_times"][current_index][3] = yv
    display_stim()

def on_release(event):
    global dragging_base
    dragging_base = None

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

def export_csv(event):
    for rec in recordings_data:
        rows = []
        abf = rec["abf"]
        fs = rec["fs"]
        abf_filtered = rec["abf_filtered"]
        for i, (b1_t, b1_v, b2_t, b2_v) in enumerate(rec["baseline_times"]):
            p_t, p_v = rec["fPSP_peaks"][i]
            try:
                i1, i2 = sorted([int(b1_t * fs), int(b2_t * fs)]) if b2_t else (int(b1_t * fs), int(b1_t * fs))
                area = np.trapz(abf_filtered[i1:i2] - b1_v, abf.sweepX[i1:i2]) * 1e3
            except Exception:
                area = np.nan
            rows.append([i+1, b1_t, b1_v, b2_t, b2_v, p_t, p_v, area, int(currents[i])])
        df = pd.DataFrame(rows, columns=[
            'Stim #', 'Baseline1 Time (s)', 'Baseline1 Vm (mV)',
            'Baseline2 Time (s)', 'Baseline2 Vm (mV)',
            'Peak Time (s)', 'Peak Vm (mV)',
            'Area (mV·ms)', 'Current (pA)'
        ])
        df.to_csv(rec["csv_path"], index=False)
        print(f"Exported: {os.path.basename(rec['csv_path'])}")

export_ax = plt.axes([0.8, 0.01, 0.15, 0.05])
export_button = Button(export_ax, 'Export CSV')
export_button.on_clicked(export_csv)

display_stim()
print("Press → or ← to navigate between stims/recordings.")
plt.show()

print("\nAll recordings processed successfully.")
