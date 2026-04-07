import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt
import pyabf
import time

# -------------------- User settings --------------------
fPSP_direction = 'min'        # 'min' or 'max'
apply_highpass_filter = True
highpass_cutoff, highpass_order = 1, 1
root = '/Users/gs075/Desktop/Data/LFP/HF_RSP_Tumor_1stCohort_Reanalysis/LFP_input'
recording = '2025_10_31_0017'
output_path = '/Users/gs075/Desktop/Data/LFP/HF_RSP_Tumor_1stCohort_Reanalysis/LFP_output'
abf_path = os.path.join(root, recording + ".abf")
csv_path = os.path.join(output_path, f"LFP_results_{recording}.csv")
delay_after_stim_s = 0.001
fPSP_min_deflection = 0.05
min_valid_amplitude, max_valid_amplitude = fPSP_min_deflection, 0.4
zoom_window, fPSP_window = 0.04, 0.010
artifact_search_window_ms = 1.0
stim_start, stim_interval, n_stimulations = 8.969, 10.0, 50
y_ax_lim = -0.9, .3
amplitudes = np.linspace(50, 500, 10)
n_repeats = int(np.ceil(n_stimulations / len(amplitudes)))
currents = np.tile(amplitudes, n_repeats)[:n_stimulations]
interp_tolerance = 1e-3  # mV threshold for "off-signal"
snap_bases = True  # default: snapping disabled
undo_stack = []
# -------------------- Load ABF --------------------
abf = pyabf.ABF(abf_path)
abf.setSweep(0, channel=0)
fs = abf.dataRate

# -------------------- Filtering --------------------
def highpass_filter(sig, fs, cutoff=1.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, sig)

# Original filtered signal (ALWAYS preserved)
abf_original = highpass_filter(abf.sweepY, fs, highpass_cutoff, highpass_order) if apply_highpass_filter else abf.sweepY.copy()

# Working signal (may be FV removed)
fv_save_path = os.path.join(output_path, f"{recording}_FV_removed.npy")

if os.path.exists(fv_save_path):
    abf_filtered = np.load(fv_save_path)
    print("Loaded FV-removed trace")
else:
    abf_filtered = abf_original.copy()

# -------------------- Stim / artifact detection --------------------
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
# -------------------- Helpers & data containers --------------------
fPSP_peaks = [None] * len(stim_times)
baseline_times = []  # each entry: [base1_time, base1_val, base2_time, base2_val]
# existing CSV baseline load if present
df_existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
def update_all_peaks():
    for idx in range(len(stim_times)):
        b = baseline_times[idx]
        if b[0] is None or b[2] is None:
            fPSP_peaks[idx] = (np.nan, np.nan); continue
        i1 = max(0, min(int(b[0] * fs), L - 1))
        i2 = max(0, min(int(b[2] * fs), L - 1))
        if i2 <= i1: i2 = min(i1 + 1, L - 1)
        seg = abf_filtered[i1:i2 + 1]
        if seg.size == 0:
            fPSP_peaks[idx] = (np.nan, np.nan); continue
        local_idx = np.argmin(seg) if fPSP_direction == 'min' else np.argmax(seg)
        peak_idx = i1 + local_idx
        fPSP_peaks[idx] = (abf.sweepX[peak_idx], abf_filtered[peak_idx])
        

# -------------------- Detect fPSPs & baselines --------------------
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

    # baseline initialization (from CSV if available)
    if df_existing is not None and idx < len(df_existing):
        baseline_times.append([
            float(df_existing.loc[idx, 'Baseline1 Time (s)']),
            float(df_existing.loc[idx, 'Baseline1 Vm (mV)']),
            float(df_existing.loc[idx, 'Baseline2 Time (s)']) if not pd.isna(df_existing.loc[idx, 'Baseline2 Time (s)']) else None,
            float(df_existing.loc[idx, 'Baseline2 Vm (mV)']) if not pd.isna(df_existing.loc[idx, 'Baseline2 Vm (mV)']) else None
        ])
    else:
        pre_peak = abf_filtered[artifact_idx:peak_idx + 1] if peak_idx >= artifact_idx else abf_filtered[artifact_idx:artifact_idx + 1]
        zero_cross = np.where(np.diff(np.sign(pre_peak)))[0]
        base_idx = artifact_idx + (zero_cross[-1] if zero_cross.size else 0)
        base_idx = max(base_idx, search_start)
        base_idx = min(base_idx, L - 1)
        baseline_times.append([abf.sweepX[base_idx], abf_filtered[base_idx], None, None])

# ensure 4 entries per baseline
for b in baseline_times:
    while len(b) < 4: b.append(None)
update_all_peaks()




# -------------------- Biphasic pulse generator --------------------
def generate_biphasic_pulse(amplitude_pA, duration_us=100, fs_wave=1e6, window_ms=1):
    total = max(int(duration_us * 1e-6 * fs_wave), 2)
    half = total // 2
    pulse = np.zeros(total); pulse[:half] = amplitude_pA; pulse[half:] = -amplitude_pA
    window_samples = int(window_ms * 1e-3 * fs_wave)
    full = np.zeros(window_samples)
    c = window_samples // 2; s = c - total // 2
    full[s:s + total] = pulse
    t_ms = (np.arange(window_samples) - c) / fs_wave * 1000
    return t_ms, full

# -------------------- Plot & interactive GUI --------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].axhline(0, ls='--')
axs[0].set_ylabel('Vm (mV)')
axs[1].set_xlabel('Time (ms)'); axs[1].set_ylabel('Current (pA)')
current_index = 0
dragging_base = None

def find_second_baseline(idx, max_search_s=0.2, fallback_ms=5):
    b1_t, b1_v, _, _ = baseline_times[idx]
    peak_t, peak_v = fPSP_peaks[idx]
    if np.isnan(peak_t):
        fb_idx = min(int((b1_t + fallback_ms / 1000.0) * fs), L - 1)
        return abf.sweepX[fb_idx], abf_filtered[fb_idx]
    s = int(peak_t * fs); e = min(L, s + int(max_search_s * fs))
    seg = abf_filtered[s:e]; x_seg = abf.sweepX[s:e]
    diff = seg - b1_v
    zc = np.where(np.diff(np.sign(diff)))[0]
    if zc.size:
        return abf.sweepX[s + zc[0]], abf_filtered[s + zc[0]]
    fb_idx = min(s + int(fallback_ms / 1000.0 * fs), L - 1)
    return abf.sweepX[fb_idx], abf_filtered[fb_idx]

def update_peak_from_bases(idx):
    b1_t, b1_v, b2_t, b2_v = baseline_times[idx]
    if b1_t is None or b2_t is None:
        fPSP_peaks[idx] = (np.nan, np.nan); return
    i1 = max(0, min(int(b1_t * fs), L - 1)); i2 = max(0, min(int(b2_t * fs), L - 1))
    if i2 <= i1: i2 = min(i1 + 1, L - 1)
    seg = abf_filtered[i1:i2 + 1]
    if seg.size == 0:
        fPSP_peaks[idx] = (np.nan, np.nan); return
    local_idx = np.argmin(seg) if fPSP_direction == 'min' else np.argmax(seg)
    p_idx = i1 + local_idx
    fPSP_peaks[idx] = (abf.sweepX[p_idx], abf_filtered[p_idx])

def update_interp_button():
    global interp_active
    if bases_off_signal(current_index):
        interp_button.label.set_color('black')  # active
        interp_active = True
    else:
        interp_button.label.set_color('gray')   # inactive
        interp_active = False

def display_stim(idx):
    peak_time, peak_val = fPSP_peaks[idx]
    b1_t, b1_v, b2_t, b2_v = baseline_times[idx]
    if b2_t is None:
        b2_t, b2_v = find_second_baseline(idx)
        baseline_times[idx][2] = b2_t; baseline_times[idx][3] = b2_v
    stim_time = stim_times[idx]
    i0 = max(0, int((stim_time - zoom_window / 2) * fs))
    i1 = min(L, int((stim_time + zoom_window / 2) * fs))
    x_full = abf.sweepX[i0:i1]; y_full = abf_filtered[i0:i1]

    axs[0].cla()
    
    # --- Original signal (background) ---
    y_orig = abf_original[i0:i1]
    axs[0].plot(x_full, y_orig, lw=1, color='dimgrey', label='Raw')
    
    # --- FV-removed / working signal (foreground) ---
    axs[0].plot(x_full, y_full, lw=1.5, color='black', label='Edited')
    
    axs[0].axhline(0, ls='--', color='k')
    axs[0].set_ylabel('Vm (mV)'); axs[0].set_ylim(y_ax_lim)
    axs[0].set_title(f"High-Pass Filtered Vm - Stim {idx+1}/{len(stim_times)}")
    axs[0].axvline(stim_time, color='red', ls='--', lw=1)
    f_start, f_end = stim_time + delay_after_stim_s, stim_time + delay_after_stim_s + fPSP_window
    axs[0].axvspan(f_start, f_end, color='steelblue', alpha=0.2)
    if not np.isnan(peak_val): axs[0].plot(peak_time, peak_val, 'go', ms=6)
    if snap_bases:
        axs[0].plot(b1_t, b1_v, 'o', ms=6, color='blue')
        axs[0].plot(b2_t, b2_v, 'o', ms=6, color='violet')
    else:
        axs[0].plot(b1_t, b1_v, 'o', ms=8, color='red')
        axs[0].plot(b2_t, b2_v, 'o', ms=8, color='red')
    # onset/offset fits (best-effort)
    try:
        i_base1 = int(b1_t * fs); i_peak = int(peak_time * fs)
        x_on, y_on = abf.sweepX[i_base1:i_peak], abf_filtered[i_base1:i_peak]
        if x_on.size:
            axs[0].plot(x_on, np.poly1d(np.polyfit(x_on, y_on, 1))(x_on), '--', lw=1.5, color='orange')
        i_base2 = int(b2_t * fs); x_off, y_off = abf.sweepX[i_peak:i_base2], abf_filtered[i_peak:i_base2]
        if x_off.size:
            axs[0].plot(x_off, np.poly1d(np.polyfit(x_off, y_off, 1))(x_off), '--', lw=1.5, color='orangered')
    except Exception:
        pass

    # shade region between bases
    try:
        ia = max(0, min(int(b1_t * fs), L - 1)); ib = max(0, min(int(b2_t * fs), L - 1))
        ia, ib = min(ia, ib), max(ia, ib)
        x_reg = abf.sweepX[ia:ib + 1]; y_reg = abf_filtered[ia:ib + 1]
        # create slanted baseline between the two bases
        baseline_line = np.linspace(b1_v, b2_v, len(x_reg))
        
        # fill between signal and slanted baseline
        axs[0].fill_between(x_reg, y_reg, baseline_line, alpha=0.5, color='firebrick')
        
        # (optional) draw the baseline line for clarity
        axs[0].plot(x_reg, baseline_line, '--', color='black', lw=1)
    except Exception:
        pass
    
    # bottom: stimulation waveform
    axs[0].legend(loc='upper left')
    axs[1].cla()
    t_wave, y_wave = generate_biphasic_pulse(currents[idx])
    axs[1].plot(t_wave, y_wave, color = 'maroon'); axs[1].set_xlabel("Time (ms)"); axs[1].set_ylabel("Current (pA)")
    axs[1].set_title(f"Stimulation Waveform - {int(currents[idx])} pA"); axs[1].set_xlim(-0.5, 0.5); axs[1].set_ylim(-600, 600)
    update_interp_button()
    fig.canvas.draw_idle()
    

def local_base_warp(i_center, target_y, window_ms=0.9):
    """
    Smoothly warp the signal locally so it passes through (i_center, target_y)
    while preserving surrounding shape.
    """
    global abf_filtered

    half_window = int((window_ms / 1000.0) * fs)
    i_start = max(0, i_center - half_window)
    i_end   = min(L, i_center + half_window)
    push_undo_action(i_start, i_end)
    x = np.arange(i_start, i_end)

    # Current signal
    y = abf_filtered[i_start:i_end]

    # Difference needed at center
    delta = target_y - abf_filtered[i_center]

    # Gaussian weighting (peaks at center, fades outward)
    sigma = half_window / 2
    weights = np.exp(-0.5 * ((x - i_center) / sigma) ** 2)

    # Normalize so center weight = 1
    weights /= weights.max()

    # Apply smooth correction
    y_new = y + delta * weights

    abf_filtered[i_start:i_end] = y_new    


def interpolate_between_bases(event):
    if not interp_active:
        print("Interpolation not available")
        return
    b1_t, b1_v, b2_t, b2_v = baseline_times[current_index]

    i1 = int(b1_t * fs)
    i2 = int(b2_t * fs)

    # Check which base(s) are off
    y1_actual = abf_filtered[i1]
    y2_actual = abf_filtered[i2]

    if abs(y1_actual - b1_v) > interp_tolerance:
        local_base_warp(i1, b1_v)

    if abs(y2_actual - b2_v) > interp_tolerance:
        local_base_warp(i2, b2_v)

    print("Local interpolation applied to off-signal base(s)")

    update_peak_from_bases(current_index)
    display_stim(current_index)

def bases_off_signal(idx):
    b1_t, b1_v, b2_t, b2_v = baseline_times[idx]
    if b1_t is None or b2_t is None:
        return False

    i1 = int(b1_t * fs)
    i2 = int(b2_t * fs)

    if i1 < 0 or i1 >= L or i2 < 0 or i2 >= L:
        return False

    y1_actual = abf_filtered[i1]
    y2_actual = abf_filtered[i2]

    return (abs(y1_actual - b1_v) > interp_tolerance or
            abs(y2_actual - b2_v) > interp_tolerance)

# keyboard navigation
def on_key(event):
    global current_index
    if event.key == 'right': current_index = min(current_index + 1, len(stim_times) - 1)
    elif event.key == 'left': current_index = max(current_index - 1, 0)
    display_stim(current_index)
fig.canvas.mpl_connect('key_press_event', on_key)

# drag handling for baselines
def on_press(event):
    global dragging_base
    if event.inaxes != axs[0]: return
    b1_t, _, b2_t, _ = baseline_times[current_index]
    d1 = abs(event.xdata - (b1_t or 0)); d2 = abs(event.xdata - (b2_t or 1e9))
    if d1 < 0.001: dragging_base = 'base1'
    elif d2 < 0.001: dragging_base = 'base2'

# -------------------- Snap toggle --------------------

def toggle_snap(event):
    global snap_bases
    snap_bases = not snap_bases
    # Update label text
    snap_button.label.set_text(f"Snap: {'ON' if snap_bases else 'OFF'}")
    # Update color (visual feedback)
    if snap_bases:
        snap_button.color = 'lightgreen'
    else:
        snap_button.color = 'salmon'

    print(f"Snap bases {'enabled' if snap_bases else 'disabled'}")

    fig.canvas.draw_idle()

# -------------------- Update on_motion --------------------
def on_motion(event):
    global dragging_base
    if dragging_base is None or event.inaxes != axs[0] or event.xdata is None: 
        return
    xi = int(event.xdata * fs)
    if xi < 0 or xi >= L: 
        return

    # Snap logic
    b1_y = baseline_times[current_index][1]
    b2_y = baseline_times[current_index][3]

    if dragging_base == 'base1':
        baseline_times[current_index][0] = event.xdata
        if not snap_bases and b2_y is not None:
            baseline_times[current_index][1] = b2_y  # snap base1 y to base2 y
        else:
            baseline_times[current_index][1] = abf_filtered[xi]
    elif dragging_base == 'base2':
        baseline_times[current_index][2] = event.xdata
        if not snap_bases and b1_y is not None:
            baseline_times[current_index][3] = b1_y  # snap base2 y to base1 y
        else:
            baseline_times[current_index][3] = abf_filtered[xi]

    update_peak_from_bases(current_index)
    display_stim(current_index)


def on_release(event):
    global dragging_base
    dragging_base = None


# reset bases button — place both bases inside the blue shading and set their Vm to 0 mV
def reset_bases(event):
    stim_time = stim_times[current_index]
    f_start = stim_time + delay_after_stim_s
    f_end = f_start + fPSP_window

    # choose two times inside the blue shading (25% and 75% through the window)
    t1 = f_start + 0.25 * (f_end - f_start)
    t2 = f_start + 0.75 * (f_end - f_start)

    # clamp to valid sample range
    i_start = max(0, min(int(t1 * fs), L - 1))
    i_end   = max(0, min(int(t2 * fs), L - 1))

    baseline_times[current_index][0] = abf.sweepX[i_start]
    baseline_times[current_index][1] = 0.0          # place on the 0 mV y-axis line
    baseline_times[current_index][2] = abf.sweepX[i_end]
    baseline_times[current_index][3] = 0.0          # place on the 0 mV y-axis line

    # update peak and redraw
    update_peak_from_bases(current_index)
    display_stim(current_index)
    


# export CSV button
def export_csv(event):
    rows = []
    for i in range(len(stim_times)):
        b1_t, b1_v, b2_t, b2_v = baseline_times[i]
        p_t, p_v = fPSP_peaks[i]
        # --- Compute baseline at peak (slanted baseline) ---
        if (
            b1_t is not None and b2_t is not None and
            not np.isnan(p_t) and not np.isnan(p_v) and
            b2_t != b1_t
        ):
            baseline_at_peak = b1_v + (b2_v - b1_v) * ((p_t - b1_t) / (b2_t - b1_t))
            amplitude = p_v - baseline_at_peak
        else:
            amplitude = np.nan
        try:
            i_b1, i_p = int(b1_t * fs), int(p_t * fs)
            onset_slope = np.polyfit(abf.sweepX[i_b1:i_p], abf_filtered[i_b1:i_p], 1)[0]
        except Exception:
            onset_slope = np.nan
        try:
            i_b2 = int(b2_t * fs)
            offset_slope = np.polyfit(abf.sweepX[i_p:i_b2], abf_filtered[i_p:i_b2], 1)[0]
        except Exception:
            offset_slope = np.nan
        try:
            i1, i2 = sorted([int(b1_t * fs), int(b2_t * fs)])
            
            x = abf.sweepX[i1:i2]
            y = abf_filtered[i1:i2]
            
            # slanted baseline
            baseline_line = np.linspace(b1_v, b2_v, len(x))
            
            # AUC relative to slanted baseline
            area = np.trapz(y - baseline_line, x) * 1e3

        except Exception:
            area = np.nan
        rows.append([
            i + 1,
            b1_t, b1_v,
            b2_t, b2_v,
            p_t, p_v,
            amplitude,   # <-- NEW
            onset_slope,
            offset_slope,
            area,
            int(currents[i])
        ])    # Save FV-modified trace
    fv_save_path = os.path.join(output_path, f"{recording}_FV_removed.npy")
    np.save(fv_save_path, abf_filtered)
    print(f"FV-modified trace saved: {fv_save_path}")
        
    df = pd.DataFrame(rows, columns=[
        'Stim #',
        'Baseline1 Time (s)', 'Baseline1 Vm (mV)',
        'Baseline2 Time (s)', 'Baseline2 Vm (mV)',
        'Peak Time (s)', 'Peak Vm (mV)',
        'Amplitude (mV)',   # <-- NEW COLUMN
        'Onset Slope (mV/s)', 'Offset Slope (mV/s)',
        'Area (mV·ms)', 'Current (pA)'
    ])
    df.to_csv(csv_path, index=False)
    
    print(f"CSV exported: {os.path.basename(csv_path)}")
    display_stim(current_index)
    
def toggle_fv_removal(event):
    global fv_removal_active
    fv_removal_active = not fv_removal_active
    print(f"Fiber volley removal {'enabled' if fv_removal_active else 'disabled'}")

# --- FV removal variables ---
fv_removal_active = False
fv_start, fv_end = None, None

# --- FV removal click handler ---
def on_click(event):
    global fv_start, fv_end, abf_filtered

    # Only respond to LEFT mouse button inside trace axis
    if event.button != 1 or event.inaxes != axs[0] or not fv_removal_active:
        return

    if fv_start is None:
        fv_start = event.xdata
        print(f"FV start set at {fv_start*1000:.2f} ms")
    elif fv_end is None:
        fv_end = event.xdata
        print(f"FV end set at {fv_end*1000:.2f} ms")
    
        i_start = max(0, int(fv_start * fs))
        i_end = max(0, int(fv_end * fs))
        if i_end <= i_start:
            i_end = i_start + 1
    
        push_undo_action(i_start, i_end)
    
        y = abf_filtered.copy()
        y[i_start:i_end] = np.interp(
            np.arange(i_start, i_end),
            [i_start-1, i_end],
            [y[i_start-1], y[i_end] if i_end < len(y) else y[-1]]
        )
        abf_filtered[:] = y
    
        print("Fiber volley removed and interpolated")
        fv_start, fv_end = None, None
        display_stim(current_index)


def push_undo_action(start, end):
    undo_stack.append((
        start,
        end,
        abf_filtered[start:end].copy()
    ))
    
# --- Undo button ---
last_undo_time = 0

def undo_fv_removal(event):
    global abf_filtered

    if not undo_stack:
        print("Nothing to undo")
        return

    start, end, old_values = undo_stack.pop()

    abf_filtered[start:end] = old_values
    update_peak_from_bases(current_index)
    display_stim(current_index)

interp_ax = plt.axes([0.4, 0.01, 0.12, 0.05])
interp_button = Button(interp_ax, 'Interpolate')
interp_button.label.set_color('gray')  # start "inactive"
interp_active = False

# Add the button to the figure
reset_ax = plt.axes([0.7, 0.01, 0.12, 0.05])  # [left, bottom, width, height]
reset_button = Button(reset_ax, 'Reset Bases')
reset_button.on_clicked(reset_bases)


# --- Connect buttons and canvas ---
fv_ax = plt.axes([0.01, 0.01, 0.18, 0.05])
fv_button = Button(fv_ax, 'Toggle FV Removal')
fv_button.on_clicked(lambda e: toggle_fv_removal(e))
interp_button.on_clicked(interpolate_between_bases)

snap_ax = plt.axes([0.22, 0.01, 0.15, 0.05])
snap_button = Button(snap_ax, 'Snap: ON')
snap_button.on_clicked(toggle_snap)
snap_button.color = 'lightgray'
snap_button.hovercolor = 'gray'


undo_ax = plt.axes([0.55, 0.01, 0.12, 0.05])
undo_button = Button(undo_ax, 'Undo')
undo_button.on_clicked(undo_fv_removal)


export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
export_button = Button(export_ax, 'Export CSV')
export_button.on_clicked(export_csv)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('button_press_event', on_click)

# initial draw
display_stim(current_index)
fig.subplots_adjust(bottom=0.15)  # increase bottom space (default ~0.1)
plt.show()
