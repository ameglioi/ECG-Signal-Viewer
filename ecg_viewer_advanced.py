import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Load data 100
record = wfdb.rdrecord('100', channels=[0])
annotation = wfdb.rdann('100', 'atr')
ecg_signal = record.p_signal[:, 0]
sampling_rate = record.fs
expert_peaks = annotation.sample

# Show from starting point
start_second = 0
samples_to_show = 10 * sampling_rate
start_sample = start_second * sampling_rate

time_seconds = np.arange(samples_to_show) / sampling_rate
signal_zoom = ecg_signal[start_sample:start_sample + samples_to_show]

# Detect peaks
our_peaks, properties = find_peaks(
    signal_zoom,
    height=0.5,
    distance=int(0.6 * sampling_rate),
    prominence=0.3
)

# Expert peaks in window
expert_peaks_in_window = expert_peaks[(expert_peaks >= start_sample) &
                                      (expert_peaks < start_sample + samples_to_show)]
expert_peaks_adjusted = expert_peaks_in_window - start_sample

# Calculate timing difference
tolerance = int(0.05 * sampling_rate)
timing_errors = []

for expert_peak in expert_peaks_adjusted:
    # Find closest detected peak
    distances = np.abs(our_peaks - expert_peak)
    closest_idx = np.argmin(distances)
    if distances[closest_idx] <= tolerance:
        time_diff_ms = (our_peaks[closest_idx] - expert_peak) / sampling_rate * 1000
        timing_errors.append(time_diff_ms)

# Calculate metrics
our_count = len(our_peaks)
expert_count = len(expert_peaks_adjusted)
matches = len(timing_errors)
sensitivity = (matches / expert_count) * 100
false_positives = our_count - matches
mean_timing_error = np.mean(np.abs(timing_errors)) if timing_errors else 0

# Calculate heart rates
duration = samples_to_show / sampling_rate
our_heart_rate = (our_count / duration) * 60
expert_heart_rate = (expert_count / duration) * 60

# 2 subplots
fig = plt.figure(figsize=(16, 10))

# Top plot: ECG with annotations
ax1 = plt.subplot(2, 1, 1)
ax1.plot(time_seconds, signal_zoom, linewidth=0.5, label='ECG Signal', color='lightblue')
ax1.plot(our_peaks / sampling_rate, signal_zoom[our_peaks], 'ro',
         markersize=10, label=f'Our Detection (n={our_count})', alpha=0.7, zorder=3)
ax1.plot(expert_peaks_adjusted / sampling_rate, signal_zoom[expert_peaks_adjusted], 'gx',
         markersize=12, markeredgewidth=3, label=f'Expert Annotations (n={expert_count})', zorder=4)

ax1.set_title(f'Peak Detection Validation | Sensitivity: {sensitivity:.1f}% | ' +
              f'Mean Timing Error: {mean_timing_error:.1f}ms', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Amplitude (mV)', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Metrics box
textstr = f'Our HR: {our_heart_rate:.1f} BPM\nExpert HR: {expert_heart_rate:.1f} BPM\n' + \
          f'Matched: {matches}/{expert_count}\nFalse Positives: {false_positives}\n' + \
          f'Avg Timing Diff: {mean_timing_error:.1f}ms'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Bottom plot: Timing difference
ax2 = plt.subplot(2, 1, 2)
if timing_errors:
    beat_numbers = range(1, len(timing_errors) + 1)
    ax2.bar(beat_numbers, timing_errors, color='purple', alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=mean_timing_error, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_timing_error:.1f}ms')
    ax2.axhline(y=-mean_timing_error, color='red', linestyle='--', linewidth=2)

    ax2.set_title('Timing Difference: Our Detection vs Expert Annotation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Beat Number', fontsize=11)
    ax2.set_ylabel('Timing Difference (ms)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add interpretation text
    if mean_timing_error < 10:
        interpretation = "Excellent Agreement (<10ms avg difference)"
    elif mean_timing_error < 20:
        interpretation = "Good Agreement (10-20ms avg difference)"
    else:
        interpretation = "Acceptable Agreement (>20ms avg difference)"

    ax2.text(0.98, 0.95, interpretation, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.show()

# Print detailed analysis
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print(f"Time Window: {start_second} to {start_second + int(duration)} seconds")
print(f"\nDetection Performance:")
print(f"  ✓ Sensitivity: {sensitivity:.1f}% ({matches}/{expert_count} beats detected)")
print(f"  ✓ False Positives: {false_positives}")
print(f"  ✓ Heart Rate Match: {abs(our_heart_rate - expert_heart_rate):.1f} BPM difference")
print(f"\nTiming Accuracy:")
print(f"  ✓ Mean timing difference: {mean_timing_error:.2f} ms")
print(f"  ✓ Max timing difference: {max(np.abs(timing_errors)):.2f} ms" if timing_errors else "  N/A")
print(f"  ✓ Std deviation: {np.std(timing_errors):.2f} ms" if timing_errors else "  N/A")

if timing_errors:
    positive_errors = sum(1 for e in timing_errors if e > 0)
    negative_errors = sum(1 for e in timing_errors if e < 0)
    print(f"\nTiming Bias:")
    print(f"  • Our detector is LATER than expert: {positive_errors} times")
    print(f"  • Our detector is EARLIER than expert: {negative_errors} times")

    if positive_errors > negative_errors * 1.5:
        print(f"  → Our algorithm tends to detect peaks slightly AFTER expert annotations")
    elif negative_errors > positive_errors * 1.5:
        print(f"  → Our algorithm tends to detect peaks slightly BEFORE expert annotations")
    else:
        print(f"  → Our algorithm is well-balanced (no systematic bias)")

print("=" * 70)