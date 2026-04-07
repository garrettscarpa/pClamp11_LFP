import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pyabf

# ================= User Settings =================
root = '/Users/gs075/Desktop/Data/LFP/HF_RSP_Tumor_1stCohort_Reanalysis/LFP_input'
output_root = root.replace("LFP_input", "LFP_output")
stim_index_to_show = 9   # 10th stim (0-based index)

apply_highpass_filter = True
highpass_cutoff = 1
highpass_order = 1

stim_start = 8.969
stim_interval = 10.0
artifact_search_window_ms = 1.0

delay_after_stim_s = 0.002
fPSP_window = 0.005
zoom_window = 0.07

fPSP_direction = 'min'   # 'min' or 'max'

# ================= Helpers =================
def highpass_filter(sig, fs, cutoff=1.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, sig)

def get_stim_times(fs, signal, n_stim=50):
    artifact_search_window = artifact_search_window_ms / 1000.0
    indices = []
    L = len(signal)

    for i in range(n_stim):
        t = stim_start + i * stim_interval
        s = max(int((t - artifact_search_window) * fs), 0)
        e = min(int((t + artifact_search_window) * fs), L)

        seg = signal[s:e]
        if seg.size:
            indices.append(s + np.argmax(np.abs(seg)))

    return np.array(indices) / fs

def compute_peak(signal, fs, stim_time):
    stim_idx = int(stim_time * fs)
    search_start = stim_idx + int(delay_after_stim_s * fs)
    search_end = search_start + int(fPSP_window * fs)

    seg = signal[search_start:search_end]
    if seg.size == 0:
        return np.nan, np.nan

    if fPSP_direction == 'min':
        local_idx = np.argmin(seg)
    else:
        local_idx = np.argmax(seg)

    peak_idx = search_start + local_idx
    return peak_idx, signal[peak_idx]

def find_output_npy(fname):
    # extract core ID from ABF filename
    # assumes something like: 2025_10_23_0009.abf
    base = os.path.splitext(fname)[0]

    for f in os.listdir(output_root):
        if f.endswith("_FV_removed.npy") and base in f:
            return os.path.join(output_root, f)

    return None

# ================= Load ABF files =================
abf_files = sorted([f for f in os.listdir(root) if f.endswith('.abf')])

records = []

for fname in abf_files:
    path = os.path.join(root, fname)
    abf = pyabf.ABF(path)
    abf.setSweep(0)

    fs = abf.dataRate
    signal = abf.sweepY.copy()
    edited_signal = None
    npy_path = find_output_npy(fname)
    
    if npy_path is not None:
        try:
            edited_signal = np.load(npy_path)
        except:
            edited_signal = None
    if apply_highpass_filter:
        signal = highpass_filter(signal, fs, highpass_cutoff, highpass_order)

    stim_times = get_stim_times(fs, signal)

    if len(stim_times) <= stim_index_to_show:
        continue

    stim_time = stim_times[stim_index_to_show]

    peak_idx, peak_val = compute_peak(signal, fs, stim_time)

    # Zoom window
    i0 = max(0, int((stim_time - zoom_window / 2) * fs))
    i1 = min(len(signal), int((stim_time + zoom_window / 2) * fs))

    x = abf.sweepX[i0:i1]
    y = signal[i0:i1]

    records.append({
        "name": fname,
        "x": x,
        "y": y,
        "edited_signal": edited_signal,
        "stim_time": stim_time,
        "peak_idx": peak_idx,
        "peak_val": peak_val
    })
    
    

# ================= Plot + Navigation =================
fig, ax = plt.subplots(figsize=(10, 6))
current_index = 0

def plot_record(idx):
    ax.clear()

    rec = records[idx]

    # Raw / filtered signal
    ax.plot(rec["x"], rec["y"], lw=1, color="dimgrey", label="Raw")

    # Edited signal (if exists)
    if rec["edited_signal"] is not None:
        i0 = max(0, int((rec["stim_time"] - zoom_window / 2) * fs))
        i1 = min(len(rec["edited_signal"]), int((rec["stim_time"] + zoom_window / 2) * fs))

        y_edit = rec["edited_signal"][i0:i1]
        x_edit = rec["x"][:len(y_edit)]  # align lengths safely

        ax.plot(x_edit, y_edit, lw=1, color='k', linestyle='--', label="Edited")

    ax.axhline(0, linestyle='--')

    # stim line
    ax.axvline(rec["stim_time"], color='red', linestyle='--')

    # peak
    if not np.isnan(rec["peak_val"]):
        ax.plot(abf.sweepX[rec["peak_idx"]], rec["peak_val"], 'go')

    ax.set_ylim(-1, 0.5)

    window = 0.030
    ax.set_xlim(rec["stim_time"] - window/2, rec["stim_time"] + window/2)

    ax.set_title(f"{rec['name']} | Stim 10")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vm (mV)")

    ax.legend()
    fig.canvas.draw_idle()

def on_key(event):
    global current_index

    if event.key == 'right':
        current_index = min(current_index + 1, len(records) - 1)
    elif event.key == 'left':
        current_index = max(current_index - 1, 0)

    plot_record(current_index)

fig.canvas.mpl_connect('key_press_event', on_key)

# Initial display
if records:
    plot_record(current_index)
else:
    print("No valid recordings found with at least 10 stimuli.")

plt.show()