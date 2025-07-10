# ST ECG Analysis Toolkit

## Overview
This project provides advanced tools for the detection and analysis of the ST segment in ECG signals, with a focus on accurate J-point detection and comprehensive ST segment morphology analysis. It leverages both Continuous Wavelet Transform (CWT) and Wavelet Packet Transform (WPT) techniques for robust fiducial point detection, enabling detailed assessment of ST elevation, depression, and morphology.

## Features
- **J-point Detection:**
  - Two advanced algorithms: CWT-based and Wavelet Packet-based (WPT).
- **Comprehensive ST Segment Analysis:**
  - Baseline calculation (PR, TP, or early segment).
  - Multi-point ST elevation/depression measurement.
  - Morphology analysis (slope, curvature, shape classification).
  - Abnormality detection (elevation, depression, upsloping, downsloping).
- **Visualization:**
  - Plots for ECG beats, fiducial points, ST segment, and morphology.
- **Example and Test Scripts:**
  - Example usage and test data included.

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd st_ecg
   ```
2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Example: Running Comprehensive ST Analysis
You can run the example analysis using the provided script:

```python
from packet import ImprovedWaveletPacketJPointDetector
from main import EnhancedSTAnalyzer
import neurokit2 as nk

# Initialize detector and analyzer
j_detector = ImprovedWaveletPacketJPointDetector(sampling_rate=1000)
analyzer = EnhancedSTAnalyzer(sampling_rate=1000)
analyzer.set_j_detector(j_detector)

# Generate or load an ECG beat (example uses NeuroKit2 synthetic data)
ecq = nk.data("ecg_1000hz")
cleaned = nk.ecg_clean(ecq, sampling_rate=1000)
peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=1000)
rpeaks_idx = np.where(peaks["ECG_R_Peaks"] == 1)[0]
beats = nk.ecg_segment(cleaned, rpeaks_idx, sampling_rate=1000)
beat = list(beats.values())[0]["Signal"].values

# Run analysis
results = analyzer.analyze_st_segment_comprehensive(beat)
analyzer.print_st_analysis_results(results)
analyzer.plot_st_analysis(beat, results)
```

You can switch to the CWT-based detector by importing and using `WaveletJPointDetector` from `cwt.py`.

### Test Scripts
See `test.py` for additional test cases and usage with real ECG data (WFDB format).

## Project Structure
- `main.py`: Main analysis logic and example runner.
- `cwt.py`: CWT-based J-point detector implementation.
- `packet.py`: Wavelet Packet-based J-point detector implementation.
- `test.py`: Test and validation scripts.
- `requirements.txt`: Python dependencies.
- `test_data/`: (Optional) Directory for test ECG data.

## Dependencies
Key dependencies (see `requirements.txt` for full list):
- numpy
- matplotlib
- scipy
- pywt (PyWavelets)
- neurokit2
- wfdb
- pandas

## Difference Between Packet Wavelet (WPT) and CWT

### Continuous Wavelet Transform (CWT)
- **How it works:**
  - CWT analyzes the signal at every possible scale and translation, providing a highly detailed time-frequency representation.
  - In this project, CWT is used (see `cwt.py`) to extract features for R-peak, QRS boundaries, P and T waves, and J-point detection by targeting specific frequency bands relevant to each ECG component.
- **Strengths:**
  - Excellent for detailed, continuous analysis of frequency content.
  - Good for detecting features with variable frequency content.
- **Limitations:**
  - Computationally intensive.
  - May include redundant information due to overlapping scales.

### Wavelet Packet Transform (WPT, Packet Wavelet)
- **How it works:**
  - WPT decomposes the signal into a full binary tree of sub-bands, allowing for flexible and adaptive frequency band selection.
  - In this project, the `ImprovedWaveletPacketJPointDetector` (see `packet.py`) uses WPT to extract and reconstruct specific frequency bands (e.g., QRS, P, T) for robust detection of fiducial points.
  - The method can reconstruct signals from only the relevant nodes, providing a more targeted analysis.
- **Strengths:**
  - More efficient for isolating specific frequency bands.
  - Adaptive and can focus on the most informative sub-bands for each ECG feature.
  - Often less computationally demanding for targeted analysis.
- **Limitations:**
  - May require careful selection of decomposition level and wavelet type.
  - Less continuous than CWT; frequency resolution is limited by the tree structure.

### Summary Table
| Aspect                | CWT                                 | Wavelet Packet (WPT)                |
|-----------------------|-------------------------------------|-------------------------------------|
| Frequency Coverage    | Continuous, overlapping             | Discrete, non-overlapping           |
| Adaptivity            | Fixed wavelet, all scales           | Flexible, adaptive sub-bands        |
| Computational Cost    | Higher                              | Lower for targeted bands            |
| Use in this Project   | `WaveletJPointDetector` (cwt.py)    | `ImprovedWaveletPacketJPointDetector` (packet.py) |
| Best For              | Detailed, global analysis           | Targeted, efficient feature extraction |

## References
- [PyWavelets Documentation](https://pywavelets.readthedocs.io/)
- [NeuroKit2 Documentation](https://neurokit2.readthedocs.io/)
- [WFDB Toolbox](https://wfdb.readthedocs.io/)

## License
MIT License
