import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pyabf

# ================= User Settings =================
root_input = "/Users/gs075/Desktop/Data/LFP/HF_RSP_Plexxikon/LFP_input"
root_output = "/Users/gs075/Desktop/Data/LFP/HF_RSP_Plexxikon/LFP_output"
unblinding_path = "/Users/gs075/Desktop/Data/LFP/HF_RSP_Plexxikon/HF_LFP_UNBLINDING.csv"
apply_highpass_filter = True
highpass_cutoff = 1.0  # Hz
highpass_order = 2
window_ms = 3.0  # ← Signal analyzed after left base

# -------------------- Helper functions --------------------
def highpass_filter(signal, fs, cutoff=1.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high', analog=False)
    return filtfilt(b, a, signal)

# -------------------- Load condition metadata --------------------
unblind_df = pd.read_csv(unblinding_path)
unblind_df.columns = [c.strip().upper() for c in unblind_df.columns]
unblind_df['DATE'] = unblind_df['DATE'].astype(str).str.strip()
unblind_df['CONDITION'] = unblind_df['CONDITION'].str.strip()
all_conditions = sorted(unblind_df['CONDITION'].dropna().unique())

def get_condition_from_date(recording_name):
    rec_date = recording_name[:10]  # 'YYYY_MM_DD'
    match = unblind_df[unblind_df['DATE'] == rec_date]
    if len(match) == 0:
        return "Unknown"
    return match['CONDITION'].iloc[0]

# ===================== Storage =====================
recording_traces = {}       # {current: [list of avg traces per recording]}
recording_times = {}        # {current: [list of time vectors per recording]}
recording_conditions = {}   # {recording_name: condition}
all_recording_metrics = []  # overall metrics per recording
conditioned_metrics = {cond: [] for cond in all_conditions}

# ===================== Collect per-recording data =====================
# Sample size per condition (number of unique recordings)
csv_files = [f for f in os.listdir(root_output) if f.startswith("LFP_results_") and f.endswith(".csv")]

for csv_file in csv_files:
    recording_name = csv_file.replace("LFP_results_", "").replace(".csv", "")
    condition = get_condition_from_date(recording_name)
    csv_path = os.path.join(root_output, csv_file)
    abf_path = os.path.join(root_input, recording_name + ".abf")
    
    if not os.path.exists(abf_path):
        print(f"⚠️ Skipping {recording_name}: ABF file not found.")
        continue

    df = pd.read_csv(csv_path)
    df['Amplitude (mV)'] = abs(df['Peak Vm (mV)'] - df['Baseline1 Vm (mV)'])
    df['Area (mV·ms)'] = abs(df['Area (mV·ms)'])

    # ---- Load filtered ABF trace ----
    npy_path = os.path.join(root_output, f"{recording_name}_FV_removed.npy")
    abf = pyabf.ABF(abf_path)
    fs = abf.dataRate

    if os.path.exists(npy_path):
        abf_filtered = np.load(npy_path)
    else:
        abf.setSweep(0, channel=0)
        trace = abf.sweepY.copy()
        abf_filtered = highpass_filter(trace, fs, highpass_cutoff, highpass_order) if apply_highpass_filter else trace.copy()

    # ---- Average metrics per current ----
    currents = sorted(df['Current (pA)'].unique())
    per_current_metrics = {'Current (pA)': [], 'Amplitude (mV)': [], 
                           'Onset Slope (mV/s)': [], 'Offset Slope (mV/s)': [], 
                           'Area (mV·ms)': []}

    for current in currents:
        subset = df[df['Current (pA)'] == current]
        if len(subset) == 0:
            continue
        per_current_metrics['Current (pA)'].append(current)
        per_current_metrics['Amplitude (mV)'].append(subset['Amplitude (mV)'].mean())
        per_current_metrics['Onset Slope (mV/s)'].append(subset['Onset Slope (mV/s)'].mean())
        per_current_metrics['Offset Slope (mV/s)'].append(subset['Offset Slope (mV/s)'].mean())
        per_current_metrics['Area (mV·ms)'].append(subset['Area (mV·ms)'].mean())

    df_metrics = pd.DataFrame(per_current_metrics)
    df_metrics['Condition'] = condition
    all_recording_metrics.append(df_metrics)
    
    if condition in conditioned_metrics:
        conditioned_metrics[condition].append(df_metrics)

    # ---- Average fPSP traces within recording per current ----
    for current in currents:
        subset = df[df['Current (pA)'] == current]
        aligned_traces = []
        trace_times = []

        for idx, row in subset.iterrows():
            base1_idx = int(row['Baseline1 Time (s)'] * fs)
            base2_idx = base1_idx + int(window_ms / 1000 * fs)
            base2_idx = min(base2_idx, len(abf_filtered) - 1)

            if base2_idx <= base1_idx:
                base2_idx = min(base1_idx + 1, len(abf_filtered)-1)

            seg = abf_filtered[base1_idx:base2_idx+1]
            seg = seg - row['Baseline1 Vm (mV)']  # baseline alignment

            aligned_traces.append(seg)
            trace_times.append(np.arange(len(seg)) / fs * 1000)  # ms

        if len(aligned_traces) == 0:
            continue

        max_len = max(len(t) for t in aligned_traces)
        padded_traces = np.full((len(aligned_traces), max_len), np.nan)
        padded_times = np.full((len(trace_times), max_len), np.nan)
        for i in range(len(aligned_traces)):
            padded_traces[i, :len(aligned_traces[i])] = aligned_traces[i]
            padded_times[i, :len(trace_times[i])] = trace_times[i]

        avg_trace = np.nanmean(padded_traces, axis=0)  # within recording
        avg_time = np.nanmean(padded_times, axis=0)

        recording_traces.setdefault(current, []).append(avg_trace)
        recording_times.setdefault(current, []).append(avg_time)

    recording_conditions[recording_name] = condition

# ===================== Figure 1: Summary Metrics =====================
combined = pd.concat(all_recording_metrics)
numeric_cols = combined.select_dtypes(include=np.number).columns
grouped = combined.groupby('Current (pA)')[numeric_cols]
summary_mean = grouped.mean()
summary_sem = grouped.std(ddof=1) / np.sqrt(len(all_recording_metrics))
# Compute n (number of recordings) and N (number of unique dates) per condition
condition_n = {cond: len(conditioned_metrics[cond]) for cond in conditioned_metrics if len(conditioned_metrics[cond]) > 0}

# Extract recording dates
condition_n = {cond: len(df_list)
               for cond, df_list in conditioned_metrics.items()
               if len(df_list) > 0}

condition_dates = {cond: set() for cond in all_conditions}

for rec_name, cond in recording_conditions.items():
    if cond in condition_dates:
        date_str = rec_name[:10]
        condition_dates[cond].add(date_str)

condition_N = {cond: len(dates) for cond, dates in condition_dates.items()}



condition_summaries = {}

for cond, df_list in conditioned_metrics.items():
    if len(df_list) == 0:
        continue

    combined_cond = pd.concat(df_list)
    numeric_cols_cond = combined_cond.select_dtypes(include=np.number).columns
    grouped_cond = combined_cond.groupby('Current (pA)')[numeric_cols_cond]

    mean_cond = grouped_cond.mean()
    sem_cond = grouped_cond.std(ddof=1) / np.sqrt(len(df_list))

    condition_summaries[cond] = (mean_cond, sem_cond)


# ===================== Figure 1: subplots =====================
fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

metrics = [
    ('Amplitude (mV)', 'Average Amplitude', 'Amplitude (mV)'),
    ('Onset Slope (mV/s)', 'Average Onset Slope', 'Slope (mV/s)'),
    ('Offset Slope (mV/s)', 'Average Offset Slope', 'Slope (mV/s)'),
    ('Area (mV·ms)', 'Average Area Under/Over Curve', 'Area (mV·ms)')
]

# Flex colors for the second condition per subplot
flex_colors = ['darkslategrey', 'teal', 'maroon', 'indigo']

for ax_idx, (ax, (col, title, ylabel)) in enumerate(zip(range(len(axs)), metrics)):
    ax = axs[ax_idx]

    # Get list of conditions (order matters: first = black, second = flex)
    conds = list(condition_summaries.keys())
    colors = ['black', flex_colors[ax_idx]]  # black first, flex second

    for cond, color in zip(conds, colors):
        mean_cond, sem_cond = condition_summaries[cond]

        ax.errorbar(
            mean_cond.index,
            mean_cond[col],
            yerr=sem_cond[col],
            fmt='o-',       # marker + line
            color=color,
            capsize=4,
            label=cond
        )

    ax.set_title(title)
    ax.set_xlabel("Stimulation (pA)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

# Build suptitle with sample sizes
title_str = " | ".join(
    f"{cond}: n={condition_n.get(cond, 0)}, N={condition_N.get(cond, 0)}"
    for cond in condition_summaries
)

fig1.suptitle(title_str, fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



# ===================== Figure 2: subplots =====================
unique_currents = sorted(recording_traces.keys())
conditions = list(condition_summaries.keys())
n_cond = len(conditions)

fig2, axs = plt.subplots(
    1, n_cond,
    figsize=(6 * n_cond, 6),
    sharey=True
)

if n_cond == 1:
    axs = [axs]

colors = plt.cm.viridis(np.linspace(0, 1, len(unique_currents)))

for ax, cond in zip(axs, conditions):

    for i, current in enumerate(unique_currents):

        traces_cond = []
        times_cond = []

        # 🔑 THIS LOOP MUST BE INSIDE THE CURRENT LOOP
        for idx, (rec_name, rec_cond) in enumerate(recording_conditions.items()):
            if rec_cond != cond:
                continue

            traces_cond.append(recording_traces[current][idx])
            times_cond.append(recording_times[current][idx])

        if len(traces_cond) == 0:
            continue

        # Align traces to maximum length
        max_len = max(len(t) for t in traces_cond)
        traces_padded = np.full((len(traces_cond), max_len), np.nan)
        times_padded = np.full((len(times_cond), max_len), np.nan)

        for k in range(len(traces_cond)):
            traces_padded[k, :len(traces_cond[k])] = traces_cond[k]
            times_padded[k, :len(times_cond[k])] = times_cond[k]

        avg_trace = np.nanmean(traces_padded, axis=0)
        sem_trace = (
            np.nanstd(traces_padded, axis=0, ddof=1)
            / np.sqrt(traces_padded.shape[0])
        )
        avg_time = np.nanmean(times_padded, axis=0)

        ax.plot(
            avg_time,
            avg_trace,
            color=colors[i],
            label=f"{current} pA"
        )

        ax.fill_between(
            avg_time,
            avg_trace - sem_trace,
            avg_trace + sem_trace,
            color=colors[i],
            alpha=0.2
        )

    ax.set_title(f"{cond} (n={condition_n.get(cond,0)}, N={condition_N.get(cond,0)})")
    ax.set_xlabel("Time after fPSP onset (ms)")
    ax.set_ylabel("Vm (Baseline-Aligned)")
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, 0)
    ax.legend(title="Current (pA)")
    ax.grid(True)

plt.tight_layout()
plt.show()
