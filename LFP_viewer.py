import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pyabf
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

############################## User Settings ##################################
fPSP_direction = 'max'      # 'max' or 'min'

############################## Load the ABF file ##################################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data'
output_path = '/Users/gs075/Desktop/LFP'
recording = '2025_10_01_0015'
abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

############################## Prepare Sweep ##################################
print("Total sweeps =", abf.sweepCount)
abf.setSweep(0, channel=0)
fs = abf.dataRate  # Sampling rate (Hz)

############################## High-Pass Filter ##################################
def highpass_filter(signal, fs, cutoff=1.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

abf_filtered = highpass_filter(abf.sweepY, fs, cutoff=1.0)

############################## Define Stimulations ############################
currents = [300, 0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
n_stimulations = len(currents)
stim_start = 3.829
stim_interval = 10
stim_times = [stim_start + i * stim_interval for i in range(n_stimulations)]
zoom_window = 0.1  # seconds
fPSP_window = 0.01  # seconds after stim

########################## Detect fPSP Peaks & Baselines #######################
fPSP_peaks = []
fPSP_slopes = []
baseline_times = []

artifact_search_window_ms = 1.0  # ±1 ms around nominal stim

for stim_time in stim_times:
    stim_idx = int(stim_time * fs)
    artifact_samples = int(artifact_search_window_ms / 1000 * fs)
    artifact_window = abf_filtered[stim_idx: stim_idx + artifact_samples]

    # Detect stim artifact
    artifact_idx_local = np.argmax(np.abs(artifact_window))
    artifact_idx_global = stim_idx + artifact_idx_local

    # Detect fEPSP peak after artifact
    fPSP_window_samples = int(fPSP_window * fs)
    segment = abf_filtered[artifact_idx_global: artifact_idx_global + fPSP_window_samples]

    peak_idx_local = np.argmax(segment) if fPSP_direction == 'max' else np.argmin(segment)
    peak_idx_global = artifact_idx_global + peak_idx_local

    # Baseline: last 0 mV crossing before peak
    pre_peak_segment = abf_filtered[artifact_idx_global:peak_idx_global+1]
    zero_cross_idx = np.where(np.diff(np.sign(pre_peak_segment)))[0]
    baseline_idx = artifact_idx_global + zero_cross_idx[-1] if len(zero_cross_idx) > 0 else artifact_idx_global
    baseline_time = abf.sweepX[baseline_idx]
    baseline_val = abf_filtered[baseline_idx]

    # Full slope from baseline to peak
    x_slope = abf.sweepX[baseline_idx:peak_idx_global]
    y_slope = abf_filtered[baseline_idx:peak_idx_global]
    slope = np.polyfit(x_slope, y_slope, 1)[0]  # mV/s

    # Save
    fPSP_peaks.append((abf.sweepX[peak_idx_global], abf_filtered[peak_idx_global]))
    baseline_times.append([baseline_time, baseline_val])  # list for mutability
    fPSP_slopes.append(slope)

########################## Use Existing CSV Baselines If Available ####################
filename = "LFP_results"
csv_path = os.path.join(output_path, f"{filename}_{recording}.csv")
if os.path.exists(csv_path):
    print(f"Found existing CSV for recording {recording}. Using baseline times from file.")
    df_existing = pd.read_csv(csv_path)
    baseline_times = [[df_existing.loc[i, 'Baseline Time (s)'],
                       df_existing.loc[i, 'Baseline Vm (mV)']] 
                      for i in range(n_stimulations)]
else:
    print("No existing CSV found. Using calculated baseline times.")

############################## Plot Setup ####################################
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

fPSP_marker, = axs[0].plot([], [], 'o', markersize=4, color='darkgreen')
baseline_marker, = axs[0].plot([], [], 'o', markersize=6, color='k')
slope_line, = axs[0].plot([], [], color='blue', lw=2)
fPSP_annotation = axs[0].annotate('', xy=(0,0), xytext=(10,10),
                                  textcoords='offset points', color='darkgreen', fontsize=8, fontweight='bold')
axs[0].axhline(0, color='k', ls='--')
axs[0].set_ylabel('Vm (mV)')
axs[0].set_ylim(-0.2, 0.5)

# Bottom plot: stimulation waveform
stim_line, = axs[1].plot([], [], color='olive')
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("Current (pA)")
axs[1].set_title("Stimulation Waveform")
axs[1].set_xlim(-0.5, 0.5)
axs[1].set_ylim(-350, 350)
plt.subplots_adjust(hspace=0.4)

########################## Biphasic Pulse Generator ##########################
def generate_biphasic_pulse(amplitude_pA, duration_us=100, fs=1_000_000, window_ms=1):
    total_samples_pulse = max(int(duration_us * 1e-6 * fs), 2)
    half_samples = total_samples_pulse // 2
    pulse = np.zeros(total_samples_pulse)
    pulse[:half_samples] = amplitude_pA
    pulse[half_samples:] = -amplitude_pA
    total_samples_window = int(window_ms * 1e-3 * fs)
    pulse_full = np.zeros(total_samples_window)
    center_idx = total_samples_window // 2
    start_idx = center_idx - total_samples_pulse // 2
    pulse_full[start_idx:start_idx + total_samples_pulse] = pulse
    t = (np.arange(total_samples_window) - center_idx) / fs * 1000
    return t, pulse_full

############################## Export Button ###############################
def export_csv(event):
    data = []
    for i in range(n_stimulations):
        peak_time, peak_val = fPSP_peaks[i]
        baseline_time, baseline_val = baseline_times[i]
        slope = fPSP_slopes[i]
        data.append([i+1, currents[i], baseline_time, baseline_val, peak_time, peak_val, slope])
    df = pd.DataFrame(data, columns=['Stim #','Current (pA)','Baseline Time (s)','Baseline Vm (mV)',
                                     'Peak Time (s)','Peak Vm (mV)','Slope (mV/s)'])
    df.to_csv(csv_path, index=False)
    print(f"CSV exported as {os.path.basename(csv_path)}")

export_ax = plt.axes([0.8, 0.01, 0.15, 0.05])
export_button = Button(export_ax, 'Export CSV')
export_button.on_clicked(export_csv)

########################## Display stim function ###########################
current_index = 0
def display_stim(idx):
    peak_time, peak_val = fPSP_peaks[idx]
    baseline_time, baseline_val = baseline_times[idx]
    slope = fPSP_slopes[idx]

    # Plot full segment around stim
    stim_time = stim_times[idx]
    idx_start = max(0, int((stim_time - zoom_window/2)*fs))
    idx_end = min(len(abf_filtered), int((stim_time + zoom_window/2)*fs))
    x_full = abf.sweepX[idx_start:idx_end]
    y_full = abf_filtered[idx_start:idx_end]
    axs[0].cla()
    axs[0].plot(x_full, y_full, color='maroon')
    axs[0].axhline(0, color='k', ls='--')
    axs[0].set_ylabel('Vm (mV)')
    axs[0].set_ylim(-0.2, 0.5)
    axs[0].set_title(f"High-Pass Filtered Vm - Stim {idx+1}/{n_stimulations}")

    # Re-add markers and slope
    fPSP_marker.set_data([peak_time], [peak_val])
    baseline_marker.set_data([baseline_time], [baseline_val])
    slope_line.set_data(abf.sweepX[int(baseline_time*fs):int(peak_time*fs)],
                        abf_filtered[int(baseline_time*fs):int(peak_time*fs)])
    axs[0].add_line(fPSP_marker)
    axs[0].add_line(baseline_marker)
    axs[0].add_line(slope_line)
    axs[0].add_artist(fPSP_annotation)
    fPSP_annotation.xy = (peak_time, peak_val)
    fPSP_annotation.set_text(f"{peak_val:.3f} mV, slope={slope:.2f} mV/s")

    # Stimulation waveform
    t, y = generate_biphasic_pulse(currents[idx])
    stim_line.set_data(t, y)
    axs[1].relim()
    axs[1].autoscale_view()
    axs[1].set_ylim(-350, 350)
    axs[1].set_xlim(-0.5, 0.5)
    axs[1].set_title(f"Stimulation Waveform - {currents[idx]} pA")
    fig.canvas.draw_idle()

########################## Keyboard navigation ###########################
def on_key(event):
    global current_index
    if event.key == 'right':
        current_index = min(current_index + 1, n_stimulations-1)
    elif event.key == 'left':
        current_index = max(current_index - 1, 0)
    else:
        return
    display_stim(current_index)

fig.canvas.mpl_connect('key_press_event', on_key)

########################## Draggable baseline (x-axis only) ##################
dragging = False
def on_press(event):
    global dragging
    if event.inaxes != axs[0]:
        return
    contains, _ = baseline_marker.contains(event)
    if contains:
        dragging = True

def on_release(event):
    global dragging
    dragging = False

def on_motion(event):
    if not dragging or event.inaxes != axs[0]:
        return
    # Snap baseline to signal
    xdata = abf.sweepX
    mouse_x = event.xdata
    idx = np.searchsorted(xdata, mouse_x)
    idx = np.clip(idx, 0, len(xdata)-1)
    new_x = xdata[idx]
    new_y = abf_filtered[idx]
    baseline_times[current_index][0] = new_x
    baseline_times[current_index][1] = new_y
    baseline_marker.set_data([new_x], [new_y])

    # Update slope
    peak_time, peak_val = fPSP_peaks[current_index]
    idx_start = int(new_x*fs)
    idx_end = int(np.searchsorted(abf.sweepX, peak_time))
    slope_y = abf_filtered[idx_start:idx_end]
    slope_x = abf.sweepX[idx_start:idx_end]
    slope_line.set_data(slope_x, slope_y)
    slope = np.polyfit(slope_x, slope_y, 1)[0]
    fPSP_annotation.set_text(f"{peak_val:.3f} mV, slope={slope:.2f} mV/s")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Show first stim on startup
display_stim(current_index)
plt.show()
