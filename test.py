import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import neurokit2 as nk
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import wfdb
from pathlib import Path
from packet import ImprovedWaveletPacketJPointDetector

class ImprovedWaveletJPointDetector:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.wavelet = 'morl'  # Morlet wavelet


    def get_record(self, data_path, record_name, selected_lead="I"):
        file_path = Path(data_path) / record_name
        record_name = str(file_path)

        try:
            # Load WFDB record
            record = wfdb.rdrecord(record_name)

            print(record)

            # Get index of the selected lead
            if selected_lead not in record.sig_name:
                raise ValueError(f"Lead '{selected_lead}' not found. Available leads: {record.sig_name}")

            lead_index = record.sig_name.index(selected_lead)
            lead_signal = record.p_signal[:, lead_index]

            return {
                'signal': lead_signal,        # 1D array of selected lead
                'lead_name': selected_lead,
                'unit': record.units[lead_index],
                'record': record,
                'fs': record.fs
            }

        except Exception as e:
            print(f"❌ Error loading ECG {record_name}: {str(e)}")
            return None

        
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
    
    def detect_r_peak(self, beat):
        """Improved R peak detection using wavelet analysis"""
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
        
        return r_peak_refined, qrs_energy
    
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
            return p_search_start  # Return default if no valid search region
        
        p_region = beat[p_search_start:p_search_end]
        
        if len(p_region) < 10:  # Not enough data
            return p_search_start
        
        # Apply wavelet transform to P wave region
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
            return min(len(beat) - 1, j_point + int(0.15 * self.fs))  # Default fallback
        
        t_region = beat[t_search_start:t_search_end]
        
        if len(t_region) < 10:  # Not enough data
            return min(len(beat) - 1, j_point + int(0.15 * self.fs))
        
        # Apply wavelet transform to T wave region
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
    
    def detect_all_fiducial_points(self, beat):
        """Detect all fiducial points with improved accuracy"""
        # Step 1: Detect R peak first
        r_peak_idx, qrs_energy = self.detect_r_peak(beat)
        
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
            'qrs_energy': qrs_energy_smooth
        })
        
        return fiducial_points
    
    def _validate_all_points(self, beat, points):
        """Validate all fiducial points for physiological consistency"""
        validated = points.copy()
        
        # Ensure proper ordering: P < QRS_onset < R < J < QRS_offset < T
        # And reasonable timing intervals
        
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
            if key != 'qrs_energy':
                validated[key] = max(0, min(len(beat) - 1, validated[key]))
        
        return validated
    
    def plot_comprehensive_analysis(self, beat, fiducial_points):
        """Plot comprehensive analysis with improved visualization"""
        
        fig, axes = plt.subplots(5, 1, figsize=(16, 14))
        
        # Plot 1: ECG with all fiducial points
        axes[0].plot(beat, 'b-', linewidth=2, label='ECG Beat')
        
        # Mark fiducial points with better visibility
        points_info = {
            'p_peak': ('P Peak', 'green', 'o'),
            'qrs_onset': ('QRS Onset', 'orange', 's'),
            'r_peak': ('R Peak', 'red', '^'),
            'j_point': ('J Point', 'purple', 'D'),
            'qrs_offset': ('QRS Offset', 'brown', 'v'),
            't_peak': ('T Peak', 'cyan', 'o')
        }
        
        for point, (label, color, marker) in points_info.items():
            idx = fiducial_points[point]
            if 0 <= idx < len(beat):
                axes[0].scatter(idx, beat[idx], color=color, s=120, 
                              marker=marker, label=label, zorder=5, edgecolor='black')
        
        axes[0].set_title('ECG Beat with Improved Wavelet-Based Fiducial Point Detection', fontsize=14)
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: R peak detection (QRS frequencies)
        qrs_frequencies = np.arange(10, 45, 2)
        coeffs_qrs, freqs_qrs, _ = self.continuous_wavelet_transform(beat, qrs_frequencies)
        im1 = axes[1].imshow(np.abs(coeffs_qrs), aspect='auto', cmap='jet', 
                           extent=[0, len(beat), freqs_qrs[-1], freqs_qrs[0]])
        axes[1].axvline(x=fiducial_points['r_peak'], color='white', linestyle='--', 
                       linewidth=2, label='R Peak')
        axes[1].set_title('QRS Frequency Analysis (10-45 Hz)', fontsize=12)
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].legend()
        plt.colorbar(im1, ax=axes[1], label='Magnitude')
        
        # Plot 3: P wave analysis (low frequencies)
        p_frequencies = np.arange(1, 12, 1)
        try:
            coeffs_p, freqs_p, _ = self.continuous_wavelet_transform(beat, p_frequencies)
            im2 = axes[2].imshow(np.abs(coeffs_p), aspect='auto', cmap='viridis', 
                               extent=[0, len(beat), freqs_p[-1], freqs_p[0]])
            axes[2].axvline(x=fiducial_points['p_peak'], color='white', linestyle='--', 
                           linewidth=2, label='P Peak')
            axes[2].set_title('P Wave Analysis (1-12 Hz)', fontsize=12)
            axes[2].set_ylabel('Frequency (Hz)')
            axes[2].legend()
            plt.colorbar(im2, ax=axes[2], label='Magnitude')
        except Exception as e:
            axes[2].text(0.5, 0.5, f'P wave analysis failed: {str(e)}', 
                        transform=axes[2].transAxes, ha='center', va='center')
        
        # Plot 4: T wave analysis (low frequencies)
        t_frequencies = np.arange(1, 10, 1)
        try:
            coeffs_t, freqs_t, _ = self.continuous_wavelet_transform(beat, t_frequencies)
            im3 = axes[3].imshow(np.abs(coeffs_t), aspect='auto', cmap='plasma', 
                               extent=[0, len(beat), freqs_t[-1], freqs_t[0]])
            axes[3].axvline(x=fiducial_points['t_peak'], color='white', linestyle='--', 
                           linewidth=2, label='T Peak')
            axes[3].set_title('T Wave Analysis (1-10 Hz)', fontsize=12)
            axes[3].set_ylabel('Frequency (Hz)')
            axes[3].legend()
            plt.colorbar(im3, ax=axes[3], label='Magnitude')
        except Exception as e:
            axes[3].text(0.5, 0.5, f'T wave analysis failed: {str(e)}', 
                        transform=axes[3].transAxes, ha='center', va='center')
        
        # Plot 5: Energy signals for all components
        if 'qrs_energy' in fiducial_points:
            axes[4].plot(fiducial_points['qrs_energy'], 'r-', linewidth=2, alpha=0.7, label='QRS Energy')
        
        # Mark all fiducial points
        for point, (label, color, marker) in points_info.items():
            idx = fiducial_points[point]
            axes[4].axvline(x=idx, color=color, linestyle='--', alpha=0.7, label=label)
        
        axes[4].set_title('Wavelet Energy Signals and Fiducial Points', fontsize=12)
        axes[4].set_ylabel('Energy')
        axes[4].set_xlabel('Sample')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_detection_results(self, fiducial_points):
        """Print detailed detection results"""
        print("=== Improved Wavelet-Based Fiducial Point Detection Results ===")
        print(f"Sampling Rate: {self.fs} Hz")
        print(f"Wavelet Used: {self.wavelet}")
        print()
        
        point_names = {
            'p_peak': 'P Wave Peak',
            'qrs_onset': 'QRS Onset',
            'r_peak': 'R Peak',
            'j_point': 'J Point',
            'qrs_offset': 'QRS Offset',
            't_peak': 'T Wave Peak'
        }
        
        for point, name in point_names.items():
            if point in fiducial_points:
                idx = fiducial_points[point]
                time_ms = (idx / self.fs) * 1000
                print(f"{name:15}: Sample {idx:4d}, Time: {time_ms:6.1f} ms")
        
        # Calculate intervals
        print("\n=== Calculated Intervals ===")
        if all(p in fiducial_points for p in ['p_peak', 'qrs_onset']):
            pr_interval = (fiducial_points['qrs_onset'] - fiducial_points['p_peak']) / self.fs * 1000
            print(f"PR Interval:     {pr_interval:.1f} ms")
        
        if all(p in fiducial_points for p in ['qrs_onset', 'qrs_offset']):
            qrs_duration = (fiducial_points['qrs_offset'] - fiducial_points['qrs_onset']) / self.fs * 1000
            print(f"QRS Duration:    {qrs_duration:.1f} ms")
        
        if all(p in fiducial_points for p in ['qrs_onset', 't_peak']):
            qt_interval = (fiducial_points['t_peak'] - fiducial_points['qrs_onset']) / self.fs * 1000
            print(f"QT Interval:     {qt_interval:.1f} ms")



if __name__ == "__main__":
    # Generate synthetic ECG beat for testing
    # def generate_synthetic_ecg_beat(fs=1000):
    #     """Generate a more realistic ECG beat"""
    #     ecg = nk.data("ecg_1000hz")
    #     cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    #     peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=fs)
    #     rpeaks_idx = np.where(peaks["ECG_R_Peaks"] == 1)[0]
        
    #     # Segment beats
    #     beats = nk.ecg_segment(cleaned, rpeaks_idx, sampling_rate=fs)
        
    #     # Get first beat
    #     first_beat_df = list(beats.values())[0]
    #     ecg_beat = first_beat_df["Signal"].values
        
    #     return ecg_beat, cleaned, rpeaks_idx
    
    # Test the improved detector
    detector = ImprovedWaveletJPointDetector(sampling_rate=1000)
    
    # Generate test ECG beat
    beat = detector.get_record('./test_data','JS19400')
    print(beat)

    new_detector = ImprovedWaveletPacketJPointDetector(sampling_rate=beat['fs'])
    
    # Detect all fiducial points
    fiducial_points = new_detector.detect_all_fiducial_points(beat['signal'])
    
    # # Print results
    new_detector.print_detection_results(fiducial_points)
    
    # # Plot comprehensive analysis
    new_detector.plot_comprehensive_analysis(beat['signal'], fiducial_points)