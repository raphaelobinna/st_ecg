import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import neurokit2 as nk
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import wfdb
from pathlib import Path

class ImprovedWaveletJPointDetector:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.wavelet = 'morl'  # Morlet wavelet

    def get_record(self, data_path, record_name, selected_lead="I"):
        file_path = Path(data_path) / record_name
        record_name = str(file_path)

        try:
            # Load WFDB record
            record = wfdb.rdrecord(record_name, channels=[0])
            print(record)
            
            lead_signal = record.p_signal[:, 0]

            return {
                'signal': lead_signal,        # 1D array of selected lead
                'lead_name': selected_lead,
                'unit': record.units[0],
                'record': record,
                'fs': record.fs
            }

        except Exception as e:
            print(f"❌ Error loading ECG {record_name}: {str(e)}")
            return None

    def detect_all_qrs_complexes(self, signal_data):
        """Detect all QRS complexes in the full ECG signal"""
        # Clean the signal first
        cleaned_signal = nk.ecg_clean(signal_data, sampling_rate=self.fs)
        
        # Detect R peaks using NeuroKit2
        peaks, info = nk.ecg_peaks(cleaned_signal, sampling_rate=self.fs)
        r_peaks = np.where(peaks["ECG_R_Peaks"] == 1)[0]
        
        print(f"Detected {len(r_peaks)} QRS complexes")
        
        return r_peaks, cleaned_signal

    def segment_beats(self, signal_data, r_peaks, before_samples=None, after_samples=None):
        """Segment ECG into individual beats around R peaks"""
        if before_samples is None:
            before_samples = int(0.3 * self.fs)  # 300ms before R peak
        if after_samples is None:
            after_samples = int(0.4 * self.fs)   # 400ms after R peak
            
        beats = []
        valid_r_peaks = []
        
        for i, r_peak in enumerate(r_peaks):
            start_idx = r_peak - before_samples
            end_idx = r_peak + after_samples
            
            # Check if segment is within signal bounds
            if start_idx >= 0 and end_idx < len(signal_data):
                beat = signal_data[start_idx:end_idx]
                beats.append({
                    'signal': beat,
                    'r_peak_global': r_peak,
                    'r_peak_local': before_samples,  # R peak position in local beat
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'beat_number': i
                })
                valid_r_peaks.append(r_peak)
        
        return beats, valid_r_peaks

    def _get_adaptive_scales(self, target_frequencies):
        """Get scales for specific frequency ranges"""
        scales = pywt.frequency2scale(self.wavelet, target_frequencies / self.fs)
        return scales
    
    def continuous_wavelet_transform(self, signal_data, target_frequencies):
        """Perform CWT with specific frequency targeting"""
        scales = self._get_adaptive_scales(target_frequencies)
        coefficients, _ = pywt.cwt(signal_data, scales, self.wavelet)
        
        # Compute actual frequencies
        central_freq = pywt.central_frequency(self.wavelet)
        frequencies = central_freq / (scales * (1/self.fs))
        
        return coefficients, frequencies, scales

    def detect_r_peak(self, beat, initial_r_peak=None):
        """Improved R peak detection using wavelet analysis"""
        if initial_r_peak is not None:
            # Use the provided R peak location and refine it
            r_peak_idx = initial_r_peak
        else:
            # R wave has dominant frequency around 10-40 Hz
            qrs_frequencies = np.arange(10, 45, 2)
            coeffs, freqs, scales = self.continuous_wavelet_transform(beat, qrs_frequencies)
            
            # Sum energy across QRS frequency range
            qrs_energy = np.sum(np.abs(coeffs), axis=0)
            
            # Find the maximum energy point (R peak)
            r_peak_idx = np.argmax(qrs_energy)
        
        # Refine using local maximum in original signal
        search_window = int(0.02 * self.fs)  # 20ms window
        start_search = max(0, r_peak_idx - search_window)
        end_search = min(len(beat), r_peak_idx + search_window)
        
        local_segment = beat[start_search:end_search]
        local_max_idx = np.argmax(local_segment)
        r_peak_refined = start_search + local_max_idx
        
        return r_peak_refined

    def detect_qrs_boundaries(self, beat, r_peak_idx):
        """Detect QRS onset and offset using wavelet analysis"""
        # QRS complex frequency content
        qrs_frequencies = np.arange(8, 50, 2)
        coeffs, freqs, scales = self.continuous_wavelet_transform(beat, qrs_frequencies)
        
        # Focus on mid-frequency range for QRS boundaries
        mid_coeffs = coeffs[1:-1, :]  # Remove extreme frequencies
        qrs_energy = np.sum(np.abs(mid_coeffs), axis=0)
        
        # Smooth the energy signal
        qrs_energy_smooth = gaussian_filter1d(qrs_energy, sigma=1.5)
        
        # Find boundaries using adaptive thresholding
        qrs_onset, qrs_offset = self._find_qrs_boundaries_adaptive(
            qrs_energy_smooth, r_peak_idx)
        
        return qrs_onset, qrs_offset, qrs_energy_smooth

    def _find_qrs_boundaries_adaptive(self, energy, r_peak_idx):
        """Find QRS boundaries using adaptive thresholding"""
        # Define realistic search windows
        max_qrs_width = int(0.12 * self.fs)  # 120ms max QRS width
        search_left = max(0, r_peak_idx - max_qrs_width)
        search_right = min(len(energy), r_peak_idx + max_qrs_width)
        
        # Calculate adaptive threshold
        local_energy = energy[search_left:search_right]
        baseline = np.percentile(local_energy, 20)  # 20th percentile as baseline
        peak_energy = np.max(local_energy)
        threshold = baseline + 0.15 * (peak_energy - baseline)
        
        # Find QRS onset
        qrs_onset = search_left
        for i in range(r_peak_idx, search_left, -1):
            if energy[i] < threshold:
                qrs_onset = i
                break
        
        # Find QRS offset
        qrs_offset = search_right - 1
        for i in range(r_peak_idx, search_right):
            if energy[i] < threshold:
                qrs_offset = i
                break
                
        return qrs_onset, qrs_offset

    def detect_p_wave(self, beat, qrs_onset):
        """Detect P wave using low-frequency wavelet analysis"""
        # P wave has lower frequency content (1-8 Hz)
        p_frequencies = np.arange(1, 12, 1)
        
        # Define P wave search region (before QRS)
        p_search_start = max(0, qrs_onset - int(0.25 * self.fs))  # 250ms before QRS
        p_search_end = max(0, qrs_onset - int(0.02 * self.fs))    # 20ms before QRS
        
        if p_search_start >= p_search_end:
            return p_search_start
        
        p_region = beat[p_search_start:p_search_end]
        
        if len(p_region) < 10:
            return p_search_start
        
        try:
            coeffs, freqs, scales = self.continuous_wavelet_transform(p_region, p_frequencies)
            p_energy = np.sum(np.abs(coeffs), axis=0)
            
            # Smooth P wave energy
            p_energy_smooth = gaussian_filter1d(p_energy, sigma=2)
            
            # Find P wave peak
            if len(p_energy_smooth) > 0:
                p_peak_local = np.argmax(p_energy_smooth)
                p_peak_idx = p_search_start + p_peak_local
            else:
                p_peak_idx = p_search_start
                
        except Exception as e:
            print(f"P wave detection failed: {e}")
            p_peak_idx = p_search_start
            
        return p_peak_idx

    def detect_t_wave(self, beat, j_point):
        """Detect T wave using appropriate frequency analysis"""
        # T wave has frequency content around 1-8 Hz
        t_frequencies = np.arange(1, 10, 1)
        
        # Define T wave search region (after J point)
        t_search_start = min(len(beat) - 1, j_point + int(0.04 * self.fs))  # 40ms after J
        t_search_end = min(len(beat), j_point + int(0.35 * self.fs))        # 350ms after J
        
        if t_search_start >= t_search_end or t_search_start >= len(beat):
            return min(len(beat) - 1, j_point + int(0.15 * self.fs))
        
        t_region = beat[t_search_start:t_search_end]
        
        if len(t_region) < 10:
            return min(len(beat) - 1, j_point + int(0.15 * self.fs))
        
        try:
            coeffs, freqs, scales = self.continuous_wavelet_transform(t_region, t_frequencies)
            t_energy = np.sum(np.abs(coeffs), axis=0)
            
            # Smooth T wave energy
            t_energy_smooth = gaussian_filter1d(t_energy, sigma=2)
            
            # Find T wave peak
            if len(t_energy_smooth) > 0:
                t_peak_local = np.argmax(t_energy_smooth)
                t_peak_idx = t_search_start + t_peak_local
            else:
                t_peak_idx = t_search_start
                
        except Exception as e:
            print(f"T wave detection failed: {e}")
            t_peak_idx = t_search_start
            
        return t_peak_idx

    def refine_j_point_with_derivatives(self, beat, initial_qrs_offset):
        """Refine J-point using derivative analysis"""
        # Calculate derivatives
        first_deriv = np.gradient(beat)
        second_deriv = np.gradient(first_deriv)
        
        # Search window around initial offset
        search_window = int(0.02 * self.fs)  # 20ms window
        start_idx = max(0, initial_qrs_offset - search_window)
        end_idx = min(len(beat), initial_qrs_offset + search_window)
        
        # Find inflection point (zero crossing in second derivative)
        best_j_point = initial_qrs_offset
        
        for i in range(start_idx, end_idx - 1):
            if i + 1 < len(second_deriv):
                # Look for sign change in second derivative
                if second_deriv[i] * second_deriv[i + 1] < 0:
                    # Found zero crossing - check if it's a good candidate
                    if abs(i - initial_qrs_offset) < abs(best_j_point - initial_qrs_offset):
                        best_j_point = i
        
        return best_j_point

    def detect_all_fiducial_points(self, beat_info):
        """Detect all fiducial points for a single beat"""
        beat = beat_info['signal']
        r_peak_local = beat_info['r_peak_local']
        
        # Step 1: Use known R peak location
        r_peak_idx = self.detect_r_peak(beat, r_peak_local)
        
        # Step 2: Detect QRS boundaries
        qrs_onset, qrs_offset, qrs_energy_smooth = self.detect_qrs_boundaries(beat, r_peak_idx)
        
        # Step 3: Refine J-point
        j_point = self.refine_j_point_with_derivatives(beat, qrs_offset)
        
        # Step 4: Detect P wave
        p_peak = self.detect_p_wave(beat, qrs_onset)
        
        # Step 5: Detect T wave
        t_peak = self.detect_t_wave(beat, j_point)
        
        # Step 6: Validate and adjust points
        fiducial_points = self._validate_all_points(beat, {
            'p_peak': p_peak,
            'qrs_onset': qrs_onset,
            'r_peak': r_peak_idx,
            'j_point': j_point,
            'qrs_offset': qrs_offset,
            't_peak': t_peak,
            'qrs_energy': qrs_energy_smooth,
            'beat_number': beat_info['beat_number'],
            'start_idx': beat_info['start_idx']
        })
        
        return fiducial_points

    def _validate_all_points(self, beat, points):
        """Validate all fiducial points for physiological consistency"""
        validated = points.copy()
        
        # Ensure proper ordering: P < QRS_onset < R < J < QRS_offset < T
        
        # P wave should be before QRS onset
        if validated['p_peak'] >= validated['qrs_onset']:
            validated['p_peak'] = max(0, validated['qrs_onset'] - int(0.05 * self.fs))
        
        # QRS onset should be before R peak
        if validated['qrs_onset'] >= validated['r_peak']:
            validated['qrs_onset'] = max(0, validated['r_peak'] - int(0.04 * self.fs))
        
        # J point should be after R peak but before or at QRS offset
        if validated['j_point'] <= validated['r_peak']:
            validated['j_point'] = validated['r_peak'] + int(0.02 * self.fs)
        
        # QRS offset should be at or after J point
        if validated['qrs_offset'] < validated['j_point']:
            validated['qrs_offset'] = validated['j_point']
        
        # T wave should be after J point
        if validated['t_peak'] <= validated['j_point']:
            validated['t_peak'] = min(len(beat) - 1, validated['j_point'] + int(0.15 * self.fs))
        
        # Ensure all indices are within bounds
        for key in validated:
            if key not in ['qrs_energy', 'beat_number', 'start_idx']:
                validated[key] = max(0, min(len(beat) - 1, validated[key]))
        
        return validated

    def process_full_ecg(self, ecg_data):
        """Process the full ECG signal and detect fiducial points for all beats"""
        # Step 1: Detect all QRS complexes
        r_peaks, cleaned_signal = self.detect_all_qrs_complexes(ecg_data)
        
        # Step 2: Segment into individual beats
        beats, valid_r_peaks = self.segment_beats(cleaned_signal, r_peaks)
        
        # Step 3: Detect fiducial points for each beat
        all_fiducial_points = []
        
        print(f"Processing {len(beats)} beats...")
        for i, beat_info in enumerate(beats):
            try:
                fiducial_points = self.detect_all_fiducial_points(beat_info)
                all_fiducial_points.append(fiducial_points)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(beats)} beats")
            except Exception as e:
                print(f"Error processing beat {i}: {e}")
                continue
        
        return all_fiducial_points, cleaned_signal, valid_r_peaks

    def plot_full_ecg_with_fiducial_points(self, signal_data, all_fiducial_points, title="ECG with All Fiducial Points"):
        """Plot the full ECG signal with all detected fiducial points"""
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        
        # Plot ECG signal
        time_axis = np.arange(len(signal_data)) / self.fs
        ax.plot(time_axis, signal_data, 'b-', linewidth=1.5, label='ECG Signal', alpha=0.8)
        
        # Define colors and markers for different fiducial points
        colors = {
            'p_peak': 'green',
            'qrs_onset': 'orange', 
            'r_peak': 'red',
            'j_point': 'purple',
            'qrs_offset': 'brown',
            't_peak': 'cyan'
        }
        
        markers = {
            'p_peak': 'o',
            'qrs_onset': 's',
            'r_peak': '^',
            'j_point': 'D',
            'qrs_offset': 'v',
            't_peak': 'o'
        }
        
        labels = {
            'p_peak': 'P Peak',
            'qrs_onset': 'QRS Onset',
            'r_peak': 'R Peak',
            'j_point': 'J Point',
            'qrs_offset': 'QRS Offset',
            't_peak': 'T Peak'
        }
        
        # Plot fiducial points for all beats
        for point_type in ['p_peak', 'qrs_onset', 'r_peak', 'j_point', 'qrs_offset', 't_peak']:
            x_coords = []
            y_coords = []
            
            for fiducial_points in all_fiducial_points:
                if point_type in fiducial_points:
                    # Convert local beat index to global signal index
                    global_idx = fiducial_points['start_idx'] + fiducial_points[point_type]
                    
                    if 0 <= global_idx < len(signal_data):
                        x_coords.append(global_idx / self.fs)
                        y_coords.append(signal_data[global_idx])
            
            if x_coords:
                ax.scatter(x_coords, y_coords, 
                          color=colors[point_type], 
                          marker=markers[point_type],
                          s=60, 
                          label=labels[point_type],
                          zorder=5,
                          edgecolor='black',
                          linewidth=0.5)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (mV)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        stats_text = f"Total Beats: {len(all_fiducial_points)}\n"
        stats_text += f"Sampling Rate: {self.fs} Hz\n"
        stats_text += f"Duration: {len(signal_data)/self.fs:.1f} s"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_selected_beats(self, signal_data, all_fiducial_points, beat_indices=None, max_beats=6):
        """Plot selected individual beats with their fiducial points"""
        
        if beat_indices is None:
            # Select evenly spaced beats
            total_beats = len(all_fiducial_points)
            if total_beats > max_beats:
                beat_indices = np.linspace(0, total_beats-1, max_beats, dtype=int)
            else:
                beat_indices = list(range(total_beats))
        
        n_beats = len(beat_indices)
        fig, axes = plt.subplots(n_beats, 1, figsize=(12, 3*n_beats))
        
        if n_beats == 1:
            axes = [axes]
        
        colors = {
            'p_peak': 'green',
            'qrs_onset': 'orange', 
            'r_peak': 'red',
            'j_point': 'purple',
            'qrs_offset': 'brown',
            't_peak': 'cyan'
        }
        
        markers = {
            'p_peak': 'o',
            'qrs_onset': 's',
            'r_peak': '^',
            'j_point': 'D',
            'qrs_offset': 'v',
            't_peak': 'o'
        }
        
        for i, beat_idx in enumerate(beat_indices):
            fiducial_points = all_fiducial_points[beat_idx]
            start_idx = fiducial_points['start_idx']
            
            # Extract beat segment
            beat_length = int(0.7 * self.fs)  # 700ms beat
            end_idx = start_idx + beat_length
            
            if end_idx <= len(signal_data):
                beat_signal = signal_data[start_idx:end_idx]
                time_axis = np.arange(len(beat_signal)) / self.fs * 1000  # Convert to ms
                
                # Plot beat
                axes[i].plot(time_axis, beat_signal, 'b-', linewidth=2, alpha=0.8)
                
                # Plot fiducial points
                for point_type in ['p_peak', 'qrs_onset', 'r_peak', 'j_point', 'qrs_offset', 't_peak']:
                    if point_type in fiducial_points:
                        local_idx = fiducial_points[point_type]
                        if 0 <= local_idx < len(beat_signal):
                            time_point = local_idx / self.fs * 1000
                            axes[i].scatter(time_point, beat_signal[local_idx],
                                          color=colors[point_type],
                                          marker=markers[point_type],
                                          s=80, zorder=5,
                                          edgecolor='black',
                                          linewidth=0.5)
                
                axes[i].set_title(f'Beat {beat_idx + 1}', fontsize=12)
                axes[i].set_ylabel('Amplitude (mV)', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                
                if i == len(beat_indices) - 1:
                    axes[i].set_xlabel('Time (ms)', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def print_summary_statistics(self, all_fiducial_points):
        """Print summary statistics for all detected fiducial points"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS FOR ALL BEATS")
        print("="*60)
        
        print(f"Total number of beats processed: {len(all_fiducial_points)}")
        print(f"Sampling rate: {self.fs} Hz")
        print(f"Wavelet used: {self.wavelet}")
        
        # Calculate interval statistics
        intervals = {
            'PR': [],
            'QRS': [],
            'QT': [],
            'RR': []
        }
        
        prev_r_peak = None
        
        for fiducial_points in all_fiducial_points:
            # PR interval
            if 'p_peak' in fiducial_points and 'qrs_onset' in fiducial_points:
                pr_ms = (fiducial_points['qrs_onset'] - fiducial_points['p_peak']) / self.fs * 1000
                if 50 <= pr_ms <= 300:  # Physiological range
                    intervals['PR'].append(pr_ms)
            
            # QRS duration
            if 'qrs_onset' in fiducial_points and 'qrs_offset' in fiducial_points:
                qrs_ms = (fiducial_points['qrs_offset'] - fiducial_points['qrs_onset']) / self.fs * 1000
                if 60 <= qrs_ms <= 200:  # Physiological range
                    intervals['QRS'].append(qrs_ms)
            
            # QT interval
            if 'qrs_onset' in fiducial_points and 't_peak' in fiducial_points:
                qt_ms = (fiducial_points['t_peak'] - fiducial_points['qrs_onset']) / self.fs * 1000
                if 200 <= qt_ms <= 500:  # Physiological range
                    intervals['QT'].append(qt_ms)
            
            # RR interval
            current_r_peak = fiducial_points['start_idx'] + fiducial_points['r_peak']
            if prev_r_peak is not None:
                rr_ms = (current_r_peak - prev_r_peak) / self.fs * 1000
                if 400 <= rr_ms <= 2000:  # Physiological range
                    intervals['RR'].append(rr_ms)
            prev_r_peak = current_r_peak
        
        print("\nInterval Statistics (mean ± std):")
        print("-" * 40)
        
        for interval_name, values in intervals.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{interval_name:3} interval: {mean_val:6.1f} ± {std_val:5.1f} ms (n={len(values)})")
            else:
                print(f"{interval_name:3} interval: No valid measurements")
        
        # Heart rate calculation
        if intervals['RR']:
            heart_rate = 60000 / np.mean(intervals['RR'])  # Convert from ms to bpm
            print(f"\nAverage Heart Rate: {heart_rate:.1f} bpm")
        
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Test the improved detector
    detector = ImprovedWaveletJPointDetector(sampling_rate=1000)
    
    # Load ECG record
    ecg_data = detector.get_record('./output_folder', 'test_record')
    
    if ecg_data is not None:
        # Update detector with actual sampling rate
        detector.fs = ecg_data['fs']
        
        # Process the full ECG signal
        all_fiducial_points, cleaned_signal, r_peaks = detector.process_full_ecg(ecg_data['signal'])
        
        # Plot results
        detector.plot_full_ecg_with_fiducial_points(cleaned_signal, all_fiducial_points)
        
        # Plot selected individual beats
        detector.plot_selected_beats(cleaned_signal, all_fiducial_points, max_beats=4)
        
        # Print summary statistics
        detector.print_summary_statistics(all_fiducial_points)
    else:
        print("Failed to load ECG data")

        