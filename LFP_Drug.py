import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyabf
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline

################################### File ######################################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data/'
recording = '2025_10_21_0014'
abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

############################## User Settings ##################################
fPSP_direction = 'min'       # 'max' or 'min'
apply_highpass_filter = True
highpass_cutoff = 1.0        # Hz
highpass_order = 2
artifact_threshold = -3.0     # mV change
min_interval_s = 0.5
zoom_window = 0.05
fPSP_window_s = 0.01
delay_after_stim_s = 0.002
fPSP_min_deflection = 0.05
pre_stim = 0
post_stim = 0.01
min_valid_amplitude = fPSP_min_deflection
max_valid_amplitude = 0.4
depth_per_stim_um = 10.0
initial_depth_um = 0.0

abf.setSweep(0, channel=0)
fs = abf.dataRate
t = abf.sweepX.copy()
y = abf.sweepY.copy()

########################### Optional Truncation ###############################
truncate_start_s = None  # e.g., 0.1
truncate_end_s = None    # e.g., 2.0

if truncate_start_s is not None or truncate_end_s is not None:
    start_idx = 0 if truncate_start_s is None else int(truncate_start_s * fs)
    end_idx = len(y) if truncate_end_s is None else int(truncate_end_s * fs)
    t = t[start_idx:end_idx]
    y = y[start_idx:end_idx]
    artifact_offset = start_idx
    print(f"Truncated trace to {t[0]:.3f} s → {t[-1]:.3f} s")
else:
    artifact_offset = 0

############################## Helper Functions #################################
def highpass_filter(signal, fs, cutoff=1.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, signal)

############################## Apply High-pass Filter ##########################
if apply_highpass_filter:
    y = highpass_filter(y, fs, highpass_cutoff, highpass_order)
    print(f"Applied high-pass filter: {highpass_cutoff} Hz cutoff")
else:
    print("Skipping high-pass filter")

############################## Detect Stimulus Artifacts #######################
dy = np.diff(y)
threshold = np.std(dy) * 5
crossings = np.where(np.abs(dy) > threshold)[0]

clusters = []
if len(crossings) > 0:
    cluster = [crossings[0]]
    for c in crossings[1:]:
        if (c - cluster[-1]) < int(0.002 * fs):
            cluster.append(c)
        else:
            clusters.append(cluster)
            cluster = [c]
    clusters.append(cluster)

artifact_indices = []
for cl in clusters:
    artifact_indices.append(cl[1] if len(cl) >= 2 else cl[0])

artifact_indices = [idx - artifact_offset for idx in artifact_indices]
artifact_indices = [idx for idx in artifact_indices if 0 <= idx < len(y)]

stim_times = t[artifact_indices]
print(f"Detected {len(artifact_indices)} stim artifacts")

############################## Detect fPSP Peaks ###############################
fPSP_peaks = []
for stim_idx in artifact_indices:
    start = stim_idx + int(delay_after_stim_s * fs)
    end = min(len(y), start + int(fPSP_window_s * fs))
    seg = y[start:end]
    if len(seg) == 0:
        fPSP_peaks.append(None)
        continue

    if fPSP_direction == 'min':
        peak_local = np.argmin(seg)
        peak_value = seg[peak_local]
        deflection = seg[0] - peak_value
    else:
        peak_local = np.argmax(seg)
        peak_value = seg[peak_local]
        deflection = peak_value - seg[0]

    if deflection < fPSP_min_deflection:
        fPSP_peaks.append(None)
    else:
        fPSP_peaks.append(start + peak_local)

# Adjust fPSP peaks for truncation
for i, peak in enumerate(fPSP_peaks):
    if peak is not None:
        peak_adj = peak - artifact_offset
        if 0 <= peak_adj < len(y):
            fPSP_peaks[i] = peak_adj
        else:
            fPSP_peaks[i] = None

print(f"Detected fPSPs in {sum(p is not None for p in fPSP_peaks)} / {len(fPSP_peaks)} stimuli")

############################## Compute Amplitudes & Depth ######################
fPSP_amplitudes = []
fPSP_depths = []

for i, peak_idx in enumerate(fPSP_peaks):
    stim_idx = artifact_indices[i]
    if peak_idx is None:
        fPSP_amplitudes.append(np.nan)
    else:
        baseline = np.mean(y[max(0, stim_idx - int(0.002 * fs)):stim_idx])
        fPSP_amplitudes.append(baseline - y[peak_idx])
    fPSP_depths.append(initial_depth_um + i * depth_per_stim_um)

fPSP_amplitudes = np.array(fPSP_amplitudes)
fPSP_depths = np.array(fPSP_depths)
valid_mask = (fPSP_amplitudes >= min_valid_amplitude) & (fPSP_amplitudes <= max_valid_amplitude)
valid_indices = np.where(valid_mask)[0]

############################## Build DataFrame for Overlay Plot ##################
sorted_indices = valid_indices[np.argsort(fPSP_depths[valid_mask])]
n_valid = len(sorted_indices)
cmap = cm.get_cmap('cividis')
overlay_data = []

for i, idx in enumerate(sorted_indices):
    peak_idx = fPSP_peaks[idx]
    stim_idx = artifact_indices[idx]
    start = max(0, stim_idx - int(pre_stim * fs))
    end = min(len(y), stim_idx + int(post_stim * fs))
    x_data = t[start:end] - t[stim_idx]
    baseline = np.mean(y[max(0, stim_idx - int(0.002*fs)):stim_idx])
    seg_aligned = y[start:end] - baseline
    overlay_data.append({
        'sorted_index': i,
        'color_value': 1 - (i / max(n_valid - 1, 1)),
        'amplitude': fPSP_amplitudes[idx],
        'x_data': x_data,
        'y_data': seg_aligned
    })

df_overlay = pd.DataFrame(overlay_data)

# Prepare valid stim lists for interactive viewer
valid_stim_indices = [artifact_indices[idx] for idx in sorted_indices]
valid_fPSP_peaks = [fPSP_peaks[idx] for idx in sorted_indices]

############################## Plot 1: Interactive fPSP Viewer ##################
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title(f"{recording} — Detected fPSPs")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Vm (mV)")
ax.axhline(0, color='k', ls='--')
highlight_line, = ax.plot([], [], color='blue', lw=2)
peak_marker, = ax.plot([], [], 'go', markersize=6)
text_annot = ax.annotate('', xy=(0,0), xytext=(10,10), textcoords='offset points',
                         color='darkgreen', fontsize=8, fontweight='bold')
current_idx = 0

def display_stim(idx):
    stim_idx = valid_stim_indices[idx]
    peak_idx = valid_fPSP_peaks[idx]

    start = max(0, stim_idx - int(zoom_window/2 * fs))
    end = min(len(y), stim_idx + int(zoom_window/2 * fs))
    if end <= start:
        highlight_line.set_data([], [])
        peak_marker.set_data([], [])
        text_annot.set_text(f"Stim {idx+1}/{len(valid_stim_indices)} — empty slice")
        ax.set_xlim(0, zoom_window)
        ax.set_ylim(-1, 1)
        fig.canvas.draw_idle()
        return

    y_segment = y[start:end]
    t_segment = t[start:end]
    ymin, ymax = np.min(y_segment), np.max(y_segment)
    margin = 0.1 * (ymax - ymin) if ymax != ymin else 0.1
    ax.set_xlim(t_segment[0], t_segment[-1])
    ax.set_ylim(-0.8, 0.2)  # Fixed y-axis
    highlight_line.set_data(t_segment, y_segment)

    for coll in ax.collections:
        coll.remove()
    ax.axvline(t[stim_idx], color='red', linestyle='--', lw=1.5)

    f_start = stim_idx + int(delay_after_stim_s * fs)
    f_end = min(f_start + int(fPSP_window_s * fs), len(y))
    if f_end > f_start:
        ax.axvspan(t[f_start], t[f_end], color='orange', alpha=0.2)

    if peak_idx is not None and 0 <= peak_idx < len(y):
        peak_marker.set_data([t[peak_idx]], [y[peak_idx]])
        text_annot.set_text(f"Stim {idx+1}/{len(valid_stim_indices)}\nPeak Vm = {y[peak_idx]:.3f} mV")
    else:
        peak_marker.set_data([], [])
        text_annot.set_text(f"Stim {idx+1}/{len(valid_stim_indices)}\nNo fPSP detected")

    fig.canvas.draw_idle()

def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = (current_idx + 1) % len(valid_stim_indices)
    elif event.key == 'left':
        current_idx = (current_idx - 1) % len(valid_stim_indices)
    display_stim(current_idx)

fig.canvas.mpl_connect('key_press_event', on_key)
display_stim(current_idx)
plt.tight_layout()
plt.show()

############################## Plot 2: Amplitude vs. Depth ######################
plt.figure(figsize=(6,4))
x = stim_times[sorted_indices]
y_amp = df_overlay['amplitude']
sc = plt.scatter(x, y_amp, c=df_overlay['color_value'], cmap=cmap, s=50)
spline = UnivariateSpline(x, y_amp, s=0.5)
x_fit = np.linspace(x[0], x[-1], 200)
plt.plot(x_fit, spline(x_fit), '--', color='darkorange')

# plt.axvline(1000, color='red', linestyle='--', lw=1.5, label='CNQX (50uM)')
# plt.axvline(1405, color='blue', linestyle='--', lw=1.5, label='Washout')

plt.xlabel("Stimulation Time (s)")
plt.ylabel("fPSP Amplitude (mV)")
plt.title("fPSP Amplitude vs. Time")
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=9)

cbar = plt.colorbar(sc)
cbar.set_ticks([0,1])
cbar.set_ticklabels(['End (Blue)', 'Start (Yellow)'])
cbar.set_label('Recording Timepoint', rotation=90, labelpad=0)
cbar.ax.yaxis.set_label_position('left')

plt.tight_layout()
plt.show()

############################## Plot 3: Overlay of Baseline-Aligned fPSPs #######
fig_overlay, ax_overlay = plt.subplots(figsize=(6,4))
ax_overlay.set_title(f"{recording} — Overlay of Valid fPSPs")
ax_overlay.set_xlabel("Time after stim (s)")
ax_overlay.set_ylabel("Vm (mV)")
ax_overlay.set_xlim(pre_stim, post_stim)
ax_overlay.set_ylim(-0.8, 0.2)
ax_overlay.axhline(0, color='k', ls='--')

for _, row in df_overlay.iterrows():
    ax_overlay.plot(row['x_data'], row['y_data'], color=cmap(row['color_value']), alpha=0.1)

import matplotlib as mpl
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig_overlay.colorbar(sm, ax=ax_overlay)
cbar.set_label('Recording Timepoint', rotation=90, labelpad=0)
cbar.set_ticks([0,1])
cbar.set_ticklabels(['End (Blue)', 'Start (Yellow)'])
cbar.ax.yaxis.set_label_position('left')

plt.tight_layout()
plt.show()
