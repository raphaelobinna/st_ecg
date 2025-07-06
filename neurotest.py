import neurokit2 as nk
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

fs=1000
# 1. Load and clean
ecg = nk.data("ecg_1000hz")
cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=fs)
rpeaks_idx = np.where(peaks["ECG_R_Peaks"] == 1)[0]
beats = nk.ecg_segment(cleaned, rpeaks_idx, sampling_rate=fs)




first_beat_df = list(beats.values())[0]
beat = first_beat_df["Signal"].values
# print(beat)
#plot single beat

j_idx = int(len(beat) * 0.4)
pr_end = int(len(beat) * 0.3)
pr_start = int(len(beat)* 0.25)

st_end_idx = j_idx + int(0.1 * fs)

# Plot the beat
plt.plot(beat, label="ECG Beat")
plt.scatter(j_idx, beat[j_idx], color='red', label="J Point")
plt.plot(range(pr_start, pr_end), beat[pr_start:pr_end], color='green', label="PR Segment")
plt.scatter(st_end_idx, beat[st_end_idx], color='orange', label="ST End (~J+80ms)")
plt.legend()
plt.title("ECG Beat with ST end, J Point and PR Segment")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()

pr_segment = beat[pr_start:pr_end]
j_point = beat[j_idx]



st_idx = j_idx + int(0.06 * fs)      # 60 ms later
st_level = beat[st_idx]  

st_offset = st_level - np.mean(pr_segment)
print(st_offset)

# def extract_features(beat):
#     j_point = beat[int(len(beat)*0.1)]
#     pr_segment = beat[:int(len(beat)*0.05)]
#     return [j_point - np.mean(pr_segment)]

# X = [extract_features(b) for b in beats]
# y = ...  # labels: 0 = normal, 1 = elevation, 2 = depression

# clf = RandomForestClassifier()
# clf.fit(X, y)

# # Predict on new heartbeat
# label = clf.predict([extract_features(new_beat)])[0]
