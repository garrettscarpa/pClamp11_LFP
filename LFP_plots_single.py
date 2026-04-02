import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pyabf

## NOTE: Errorbars are currently set within-subject. will need to be adapted for across-subject eventually

# ================= User Settings =================
root = "/Users/garrett/Desktop/analysis/lfp/HF_RSP_Tumor_1stCohort_Reanalysis copy/LFP_input"
output_path = '/Users/garrett/Desktop/analysis/lfp/HF_RSP_Tumor_1stCohort_Reanalysis copy/LFP_output'
recording = "2025_10_24_0016"
apply_highpass_filter = True
highpass_cutoff = 1.0  # Hz
highpass_order = 2

# ================= Load CSV =================
abf_path = os.path.join(root, recording + ".abf")
csv_path = os.path.join(output_path, f"LFP_results_{recording}.csv")
df = pd.read_csv(csv_path)

def highpass_filter(sig, fs, cutoff=1.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, sig)
# ================= Compute Amplitude =================
df['Amplitude (mV)'] = abs(df['Peak Vm (mV)'] - df['Baseline1 Vm (mV)'])  # take absolute

# ================= Group & Average by Stimulation Current =================
grouped = df.groupby('Current (pA)').mean().reset_index()
stim_intensity = grouped['Current (pA)']
amplitude = grouped['Amplitude (mV)']
onset_slope = grouped['Onset Slope (mV/s)']
offset_slope = grouped['Offset Slope (mV/s)']
area = abs(grouped['Area (mV·ms)'])  # take absolute


# ================= Figure 1: Summary Metrics =================
fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

axs[0].plot(stim_intensity, amplitude, marker='o', color='darkslategrey')
axs[0].set_title("Average Amplitude")
axs[0].set_xlabel("Stimulation (pA)")
axs[0].set_ylabel("Amplitude (mV)")
axs[0].grid(True)

axs[1].plot(stim_intensity, onset_slope, marker='o', color='teal')
axs[1].set_title("Average Onset Slope")
axs[1].set_xlabel("Stimulation (pA)")
axs[1].set_ylabel("Slope (mV/s)")
axs[1].grid(True)

axs[2].plot(stim_intensity, offset_slope, marker='o', color='darkred')
axs[2].set_title("Average Offset Slope")
axs[2].set_xlabel("Stimulation (pA)")
axs[2].set_ylabel("Slope (mV/s)")
axs[2].grid(True)

axs[3].plot(stim_intensity, area, marker='o', color='indigo')
axs[3].set_title("Average Area Under/Over Curve")
axs[3].set_xlabel("Stimulation (pA)")
axs[3].set_ylabel("Area (mV·ms)")
axs[3].grid(True)

plt.tight_layout()
plt.show()

# ================= Load ABF File (FV-removed if available) =================
fv_removed_path = os.path.join(output_path, f"{recording}_FV_removed.npy")

abf = pyabf.ABF(abf_path)
fs = abf.dataRate
abf.setSweep(0, channel=0)

if os.path.exists(fv_removed_path):
    # Load previously FV-removed trace
    abf_filtered = np.load(fv_removed_path)
    print(f"Loaded FV-removed trace: {fv_removed_path}")
else:
    # Load original ABF trace
    trace = abf.sweepY.copy()
    if apply_highpass_filter:
        abf_filtered = highpass_filter(trace, fs, highpass_cutoff, highpass_order)
    else:
        abf_filtered = trace.copy()
    print(f"Loaded original ABF trace: {abf_path}")

# ================= Figure 2: Average fPSPs with True Time in ms =================
unique_currents = df['Current (pA)'].unique()
avg_traces = []
sem_traces = []
time_vectors = []

for current in unique_currents:
    subset = df[df['Current (pA)'] == current]
    aligned_traces = []
    trace_times = []

    if os.path.exists(fv_removed_path):
        # Load previously FV-removed trace (with all user adjustments)
        abf_filtered = np.load(fv_removed_path)
        print(f"Loaded FV-removed trace: {fv_removed_path}")
    else:
        # Load original ABF trace
        trace = abf.sweepY.copy()
        if apply_highpass_filter:
            abf_filtered = highpass_filter(trace, fs, highpass_cutoff, highpass_order)
        else:
            abf_filtered = trace.copy()
        print(f"Loaded original ABF trace: {abf_path}")
    
    # ================= Extract segments using FV-adjusted trace =================
    for idx, row in subset.iterrows():
        # Make sure to round and clip indices
        base1_idx = int(round(row['Baseline1 Time (s)'] * fs))
        base2_idx = int(round(row['Baseline2 Time (s)'] * fs))
        base1_idx = np.clip(base1_idx, 0, len(abf_filtered)-1)
        base2_idx = np.clip(base2_idx, base1_idx+1, len(abf_filtered)-1)
    
        # Use the exact baseline value from the FV-modified trace
        baseline_val = abf_filtered[base1_idx]  
        seg = abf_filtered[base1_idx:base2_idx+1] - baseline_val
            
        aligned_traces.append(seg)
        trace_times.append(np.arange(len(seg)) / fs * 1000)  # convert to ms

    if len(aligned_traces) == 0:
        continue

    # Pad traces to the same length for averaging
    max_len = max(len(t) for t in aligned_traces)
    padded_traces = np.zeros((len(aligned_traces), max_len))
    padded_traces[:] = np.nan
    padded_times = np.zeros((len(trace_times), max_len))
    padded_times[:] = np.nan

    for i in range(len(aligned_traces)):
        padded_traces[i, :len(aligned_traces[i])] = aligned_traces[i]
        padded_times[i, :len(trace_times[i])] = trace_times[i]

    avg_trace = np.nanmean(padded_traces, axis=0)
    sem_trace = np.nanstd(padded_traces, axis=0, ddof=1) / np.sqrt(len(aligned_traces))
    avg_time = np.nanmean(padded_times, axis=0)

    avg_traces.append(avg_trace)
    sem_traces.append(sem_trace)
    time_vectors.append(avg_time)
    # Determine global x-limits
    

# Plot average fPSPs with SEM shading (ms)
fig2, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_currents)))

for i, current in enumerate(unique_currents):
    ax.plot(time_vectors[i], avg_traces[i], color=colors[i], label=f"{current} pA")
    ax.fill_between(time_vectors[i],
                    avg_traces[i]-sem_traces[i],
                    avg_traces[i]+sem_traces[i],
                    color=colors[i], alpha=0.15)

ax.set_xlabel("Time after fPSP onset (ms)")
ax.set_ylabel("Vm (Baseline-Aligned)")
ax.set_title("Average Baseline-Aligned fPSP per Stimulation Current ± SEM")
ax.legend(title="Current (pA)")
ax.grid(True)
plt.tight_layout()
plt.show()
