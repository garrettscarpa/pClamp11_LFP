import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= User Settings =================
csv_path = "/Users/gs075/Desktop/LFP/LFP_results_2025_10_01_0015.csv"
output_fig = "/Users/gs075/Desktop/LFP/LFP_summary_plot.png"

# ================= Load CSV =================
df = pd.read_csv(csv_path)

# Remove first 300 pA stim (assumes first row is 300 pA)
df = df.iloc[1:].reset_index(drop=True)

# Define x-axis (stimulation intensities)
stim_intensity = df['Current (pA)']

# Calculate amplitude (Peak Vm - Baseline1 Vm)
amplitude = df['Peak Vm (mV)'] - df['Baseline1 Vm (mV)']

# ================= Create Plots =================
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # easier indexing

# 1. Amplitude
axs[0].plot(stim_intensity, amplitude, marker='o', color='darkgreen')
axs[0].set_title("Amplitude")
axs[0].set_xlabel("Stimulation (pA)")
axs[0].set_ylabel("Amplitude (mV)")
axs[0].grid(True)

# 2. Onset Slope
axs[1].plot(stim_intensity, df['Onset Slope (mV/s)'], marker='o', color='blue')
axs[1].set_title("Onset Slope")
axs[1].set_xlabel("Stimulation (pA)")
axs[1].set_ylabel("Slope (mV/s)")
axs[1].grid(True)

# 3. Offset Slope
axs[2].plot(stim_intensity, df['Offset Slope (mV/s)'], marker='o', color='red')
axs[2].set_title("Offset Slope")
axs[2].set_xlabel("Stimulation (pA)")
axs[2].set_ylabel("Slope (mV/s)")
axs[2].grid(True)

# 4. Area
axs[3].plot(stim_intensity, df['Area (mV·ms)'], marker='o', color='purple')
axs[3].set_title("AUC/AOC")
axs[3].set_xlabel("Stimulation (pA)")
axs[3].set_ylabel("Area (mV·ms)")
axs[3].grid(True)

plt.tight_layout()
plt.savefig(output_fig, dpi=300)
plt.show()
