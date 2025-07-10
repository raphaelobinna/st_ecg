import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

class ImprovedWaveletPacketJPointDetector:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.wavelet = 'db4'
        self.max_level = 6
        self.target_level = 5  
        
    def map_node_path_to_frequency(self, node_path, target_level):
        """Map a node path to its corresponding frequency band"""
        # Convert path to binary representation
        binary_path = node_path.replace('a', '0').replace('d', '1')
        
        # Convert binary path to band index
        band_index = int(binary_path, 2) if binary_path else 0
        
        # Calculate frequency range
        nyquist = self.fs / 2
        num_bands = 2 ** target_level
        freq_step = nyquist / num_bands
        
        freq_low = band_index * freq_step
        freq_high = (band_index + 1) * freq_step
        
        return freq_low, freq_high
    
    def extract_frequency_band(self, signal_data, target_low_freq, target_high_freq):
        """Extract specific frequency band using wavelet packet decomposition"""
        try:
            # Perform wavelet packet decomposition
            wp_signal = pywt.WaveletPacket(signal_data, self.wavelet, maxlevel=self.max_level)
            
            # Get all nodes at target level
            nodes = wp_signal.get_level(self.target_level, 'freq')
            
            # Find nodes that match our frequency criteria
            matching_nodes = []
            for node in nodes:
                low_freq, high_freq = self.map_node_path_to_frequency(node.path, self.target_level)
                
                # Check if frequency band overlaps with target range
                if (low_freq < target_high_freq and high_freq > target_low_freq):
                    matching_nodes.append(node)
            
            if not matching_nodes:
                # Fallback to simple bandpass filter
                return self._fallback_bandpass_filter(signal_data, target_low_freq, target_high_freq)
            
            # Reconstruct signal from matching nodes
            reconstruct_wp = pywt.WaveletPacket(data=None, wavelet=self.wavelet, maxlevel=self.max_level)
            
            for node in matching_nodes:
                reconstruct_wp[node.path] = node.data
            
            # Reconstruct the filtered signal
            filtered_signal = reconstruct_wp.reconstruct(update=False)
            
            # Ensure output length matches input
            if len(filtered_signal) > len(signal_data):
                filtered_signal = filtered_signal[:len(signal_data)]
            elif len(filtered_signal) < len(signal_data):
                filtered_signal = np.pad(filtered_signal, (0, len(signal_data) - len(filtered_signal)), 'constant')
            
            return matching_nodes, filtered_signal
            
        except Exception as e:
            print(f"Wavelet packet decomposition failed: {e}")
            return self._fallback_bandpass_filter(signal_data, target_low_freq, target_high_freq)
    
    def _fallback_bandpass_filter(self, signal_data, low_freq, high_freq):
        """Fallback bandpass filter using Butterworth filter"""
        try:
            nyquist = self.fs / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Ensure frequencies are within valid range
            low = max(0.001, min(low, 0.999))
            high = max(low + 0.001, min(high, 0.999))
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            return [], filtered_signal
        except Exception as e:
            print(f"Fallback filter failed: {e}")
            return [], signal_data
    
    def detect_r_peak(self, beat):
        """Improved R peak detection"""
        # R wave has dominant frequency around 10-40 Hz
        try:
            _, qrs_signal = self.extract_frequency_band(beat, 10, 40)
        except:
            qrs_signal = beat
        
        # Calculate energy from QRS-filtered signal
        qrs_energy = np.abs(qrs_signal) ** 2
        qrs_energy_smooth = gaussian_filter1d(qrs_energy, sigma=2.0)
        
        # Find the maximum energy point
        r_peak_idx = np.argmax(qrs_energy_smooth)
        
        # Refine using local maximum in original signal
        search_window = int(0.02 * self.fs)  # 20ms window
        start_search = max(0, r_peak_idx - search_window)
        end_search = min(len(beat), r_peak_idx + search_window)
        
        local_segment = beat[start_search:end_search]
        if len(local_segment) > 0:
            local_max_idx = np.argmax(local_segment)
            r_peak_refined = start_search + local_max_idx
        else:
            r_peak_refined = r_peak_idx
        
        return r_peak_refined, qrs_energy_smooth
    
    def detect_qrs_boundaries(self, beat, r_peak_idx):
        """Improved QRS boundary detection"""
        # Use broader frequency range for QRS boundaries
        try:
            _, qrs_signal = self.extract_frequency_band(beat, 5, 40)
        except:
            qrs_signal = beat
        
        # Calculate first derivative for onset/offset detection
        first_deriv = np.gradient(qrs_signal)
        
        # Smooth the derivative
        deriv_smooth = gaussian_filter1d(first_deriv, sigma=1.5)
        
        # Find QRS onset (first significant positive slope before R peak)
        qrs_onset = self._find_qrs_onset(deriv_smooth, r_peak_idx)
        
        # Find QRS offset (return to baseline after R peak)
        qrs_offset = self._find_qrs_offset(deriv_smooth, beat, r_peak_idx)
        
        return qrs_onset, qrs_offset, deriv_smooth
    
    def _find_qrs_onset(self, deriv_smooth, r_peak_idx):
        """Find QRS onset using derivative analysis"""
        # Search backwards from R peak
        search_start = max(0, r_peak_idx - int(0.08 * self.fs))  # 80ms before R
        
        # Calculate adaptive threshold
        baseline_deriv = np.median(deriv_smooth[search_start:r_peak_idx])
        threshold = baseline_deriv + 0.3 * np.std(deriv_smooth[search_start:r_peak_idx])
        
        # Find first point where derivative exceeds threshold
        qrs_onset = search_start
        for i in range(r_peak_idx - 1, search_start, -1):
            if deriv_smooth[i] > threshold:
                qrs_onset = i
            else:
                break
        
        return qrs_onset
    
    def _find_qrs_offset(self, deriv_smooth, beat, r_peak_idx):
        """Find QRS offset using derivative and amplitude analysis"""
        # Search forward from R peak
        search_end = min(len(beat), r_peak_idx + int(0.12 * self.fs))  # 120ms after R
        
        # Calculate baseline level after R peak
        baseline_region = beat[min(len(beat)-int(0.1*self.fs), search_end-int(0.05*self.fs)):search_end]
        if len(baseline_region) > 0:
            baseline_level = np.median(baseline_region)
        else:
            baseline_level = 0
        
        # Find point where signal returns close to baseline
        qrs_offset = search_end - 1
        amplitude_threshold = 0.2 * abs(beat[r_peak_idx] - baseline_level)
        
        for i in range(r_peak_idx + int(0.03 * self.fs), search_end):  # Start 30ms after R
            if abs(beat[i] - baseline_level) < amplitude_threshold:
                qrs_offset = i
                break
        
        return qrs_offset
    
    def detect_p_wave(self, beat, qrs_onset):
        """Improved P wave detection"""
        # P wave frequency range: 0.5-15 Hz
        try:
            _, p_signal = self.extract_frequency_band(beat, 0.5, 15)
        except:
            p_signal = beat
        
        # Define P wave search region
        p_search_start = max(0, qrs_onset - int(0.3 * self.fs))  # 300ms before QRS
        p_search_end = max(0, qrs_onset - int(0.05 * self.fs))   # 50ms before QRS
        
        if p_search_start >= p_search_end:
            return p_search_start
        
        # Extract P wave region
        p_region = p_signal[p_search_start:p_search_end]
        
        if len(p_region) < 10:
            return p_search_start
        
        # Apply smoothing
        p_smooth = gaussian_filter1d(p_region, sigma=3)
        
        # Find peaks in P wave region
        try:
            peaks, _ = find_peaks(p_smooth, 
                                height=np.max(p_smooth) * 0.3,
                                distance=int(0.05 * self.fs))
            
            if len(peaks) > 0:
                # Choose the most prominent peak
                peak_heights = p_smooth[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                p_peak_idx = p_search_start + best_peak_idx
            else:
                # Fallback to maximum
                p_peak_idx = p_search_start + np.argmax(p_smooth)
                
        except Exception as e:
            print(f"P wave peak detection failed: {e}")
            p_peak_idx = p_search_start + len(p_region) // 2
            
        return p_peak_idx
    
    def detect_t_wave(self, beat, j_point):
        """Improved T wave detection"""
        # T wave frequency range: 0.5-10 Hz
        try:
            _, t_signal = self.extract_frequency_band(beat, 0.5, 10)
        except:
            t_signal = beat
        
        # Define T wave search region
        t_search_start = min(len(beat) - 1, j_point + int(0.08 * self.fs))  # 80ms after J
        t_search_end = min(len(beat), j_point + int(0.4 * self.fs))         # 400ms after J
        
        if t_search_start >= t_search_end or t_search_start >= len(beat):
            return min(len(beat) - 1, j_point + int(0.2 * self.fs))
        
        # Extract T wave region
        t_region = t_signal[t_search_start:t_search_end]
        
        if len(t_region) < 10:
            return min(len(beat) - 1, j_point + int(0.2 * self.fs))
        
        # Apply smoothing
        t_smooth = gaussian_filter1d(t_region, sigma=4)
        
        # Find peaks in T wave region
        try:
            peaks, _ = find_peaks(t_smooth, 
                                height=np.max(t_smooth) * 0.2,
                                distance=int(0.08 * self.fs))
            
            if len(peaks) > 0:
                # Choose the most prominent peak
                peak_heights = t_smooth[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                t_peak_idx = t_search_start + best_peak_idx
            else:
                # Fallback to maximum
                t_peak_idx = t_search_start + np.argmax(t_smooth)
                
        except Exception as e:
            print(f"T wave peak detection failed: {e}")
            t_peak_idx = t_search_start + len(t_region) // 2
            
        return t_peak_idx
    
    def refine_j_point_with_derivatives(self, beat, initial_qrs_offset):
        """Refined J-point detection using multiple criteria"""
        # Calculate derivatives
        first_deriv = np.gradient(beat)
        second_deriv = np.gradient(first_deriv)
        
        # Smooth derivatives
        first_deriv_smooth = gaussian_filter1d(first_deriv, sigma=1.5)
        second_deriv_smooth = gaussian_filter1d(second_deriv, sigma=1.5)
        
        # Search window
        search_window = int(0.04 * self.fs)  # 40ms window
        start_idx = max(0, initial_qrs_offset - search_window)
        end_idx = min(len(beat), initial_qrs_offset + search_window)
        
        best_j_point = initial_qrs_offset
        best_score = 0
        
        # Find J-point using multiple criteria
        for i in range(start_idx, end_idx):
            score = 0
            
            # Criterion 1: Low first derivative (flat slope)
            if abs(first_deriv_smooth[i]) < 0.1 * np.std(first_deriv_smooth):
                score += 1
            
            # Criterion 2: Zero crossing in second derivative
            if i > 0 and i < len(second_deriv_smooth) - 1:
                if second_deriv_smooth[i-1] * second_deriv_smooth[i+1] < 0:
                    score += 2
            
            # Criterion 3: Close to original estimate
            distance_penalty = abs(i - initial_qrs_offset) / search_window
            score += (1 - distance_penalty)
            
            if score > best_score:
                best_score = score
                best_j_point = i
        
        return best_j_point
    
    def detect_all_fiducial_points(self, beat):
        """Detect all fiducial points with improved algorithms"""
        # Step 1: Detect R peak first
        r_peak_idx, qrs_energy = self.detect_r_peak(beat)
        
        # Step 2: Detect QRS boundaries
        qrs_onset, qrs_offset, qrs_deriv = self.detect_qrs_boundaries(beat, r_peak_idx)
        
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
            'qrs_energy': qrs_energy,
            'qrs_deriv': qrs_deriv
        })
        
        return fiducial_points
    
    def _validate_all_points(self, beat, points):
        """Enhanced validation of all fiducial points"""
        validated = points.copy()
        
        # Ensure proper ordering and reasonable intervals
        
        # P wave: should be 50-300ms before QRS onset
        min_pr_interval = int(0.05 * self.fs)
        max_pr_interval = int(0.3 * self.fs)
        
        if (validated['qrs_onset'] - validated['p_peak']) < min_pr_interval:
            validated['p_peak'] = max(0, validated['qrs_onset'] - min_pr_interval)
        elif (validated['qrs_onset'] - validated['p_peak']) > max_pr_interval:
            validated['p_peak'] = max(0, validated['qrs_onset'] - max_pr_interval)
        
        # QRS duration: should be 60-120ms
        min_qrs_duration = int(0.06 * self.fs)
        max_qrs_duration = int(0.12 * self.fs)
        
        if (validated['qrs_offset'] - validated['qrs_onset']) < min_qrs_duration:
            validated['qrs_offset'] = validated['qrs_onset'] + min_qrs_duration
        elif (validated['qrs_offset'] - validated['qrs_onset']) > max_qrs_duration:
            validated['qrs_offset'] = validated['qrs_onset'] + max_qrs_duration
        
        # J-point: should be close to QRS offset
        if abs(validated['j_point'] - validated['qrs_offset']) > int(0.02 * self.fs):
            validated['j_point'] = validated['qrs_offset']
        
        # T wave: should be 100-400ms after J-point
        min_jt_interval = int(0.1 * self.fs)
        max_jt_interval = int(0.4 * self.fs)
        
        if (validated['t_peak'] - validated['j_point']) < min_jt_interval:
            validated['t_peak'] = min(len(beat) - 1, validated['j_point'] + min_jt_interval)
        elif (validated['t_peak'] - validated['j_point']) > max_jt_interval:
            validated['t_peak'] = min(len(beat) - 1, validated['j_point'] + max_jt_interval)
        
        # Ensure all indices are within bounds
        for key in validated:
            if key not in ['qrs_energy', 'qrs_deriv']:
                validated[key] = max(0, min(len(beat) - 1, validated[key]))
        
        return validated
    
    def plot_comprehensive_analysis(self, beat, fiducial_points):
        """Enhanced plotting with better visualization"""
        fig, axes = plt.subplots(6, 1, figsize=(16, 20))
        
        # Plot 1: Original ECG with all fiducial points
        axes[0].plot(beat, 'b-', linewidth=2, label='Original ECG Beat')
        
        # Mark fiducial points
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
                axes[0].scatter(idx, beat[idx], color=color, s=150, 
                              marker=marker, label=label, zorder=5, 
                              edgecolor='black', linewidth=2)
        
        axes[0].set_title('ECG Beat with Enhanced Fiducial Point Detection', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: QRS analysis
        try:
            _, qrs_signal = self.extract_frequency_band(beat, 10, 40)
            axes[1].plot(qrs_signal, 'r-', linewidth=2, label='QRS Band (10-40 Hz)')
            axes[1].axvline(x=fiducial_points['r_peak'], color='red', linestyle='--', 
                           linewidth=2, label='R Peak')
            axes[1].axvline(x=fiducial_points['qrs_onset'], color='orange', linestyle='--', 
                           linewidth=2, label='QRS Onset')
            axes[1].axvline(x=fiducial_points['qrs_offset'], color='brown', linestyle='--', 
                           linewidth=2, label='QRS Offset')
            axes[1].set_title('QRS Complex Analysis', fontsize=12)
            axes[1].set_ylabel('Amplitude')
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)
        except Exception as e:
            axes[1].text(0.5, 0.5, f'QRS analysis failed: {str(e)}', 
                        transform=axes[1].transAxes, ha='center', va='center')
        
        # Plot 3: P wave analysis
        try:
            _, p_signal = self.extract_frequency_band(beat, 0.5, 15)
            axes[2].plot(p_signal, 'g-', linewidth=2, label='P Wave Band (0.5-15 Hz)')
            axes[2].axvline(x=fiducial_points['p_peak'], color='green', linestyle='--', 
                           linewidth=2, label='P Peak')
            axes[2].set_title('P Wave Analysis', fontsize=12)
            axes[2].set_ylabel('Amplitude')
            axes[2].legend(fontsize=9)
            axes[2].grid(True, alpha=0.3)
        except Exception as e:
            axes[2].text(0.5, 0.5, f'P wave analysis failed: {str(e)}', 
                        transform=axes[2].transAxes, ha='center', va='center')
        
        # Plot 4: T wave analysis
        try:
            _, t_signal = self.extract_frequency_band(beat, 0.5, 10)
            axes[3].plot(t_signal, 'c-', linewidth=2, label='T Wave Band (0.5-10 Hz)')
            axes[3].axvline(x=fiducial_points['t_peak'], color='cyan', linestyle='--', 
                           linewidth=2, label='T Peak')
            axes[3].set_title('T Wave Analysis', fontsize=12)
            axes[3].set_ylabel('Amplitude')
            axes[3].legend(fontsize=9)
            axes[3].grid(True, alpha=0.3)
        except Exception as e:
            axes[3].text(0.5, 0.5, f'T wave analysis failed: {str(e)}', 
                        transform=axes[3].transAxes, ha='center', va='center')
        
        # Plot 5: Derivative analysis for J-point
        try:
            first_deriv = np.gradient(beat)
            first_deriv_smooth = gaussian_filter1d(first_deriv, sigma=1.5)
            
            axes[4].plot(first_deriv_smooth, 'm-', linewidth=2, label='First Derivative (Smoothed)')
            axes[4].axvline(x=fiducial_points['j_point'], color='purple', linestyle='--', 
                           linewidth=2, label='J Point')
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[4].set_title('Derivative Analysis for J-Point Detection', fontsize=12)
            axes[4].set_ylabel('Derivative')
            axes[4].legend(fontsize=9)
            axes[4].grid(True, alpha=0.3)
        except Exception as e:
            axes[4].text(0.5, 0.5, f'Derivative analysis failed: {str(e)}', 
                        transform=axes[4].transAxes, ha='center', va='center')
        
        # Plot 6: All fiducial points summary
        axes[5].plot(beat, 'b-', linewidth=2, alpha=0.7, label='ECG Beat')
        
        # Mark all fiducial points with vertical lines
        for point, (label, color, marker) in points_info.items():
            idx = fiducial_points[point]
            axes[5].axvline(x=idx, color=color, linestyle='--', alpha=0.8, 
                           linewidth=2, label=label)
        
        axes[5].set_title('All Fiducial Points Summary', fontsize=12)
        axes[5].set_ylabel('Amplitude')
        axes[5].set_xlabel('Sample')
        axes[5].legend(fontsize=9)
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_detection_results(self, fiducial_points):
        """Print detailed detection results"""
        print("=== Enhanced Wavelet Packet-Based Fiducial Point Detection Results ===")
        print(f"Sampling Rate: {self.fs} Hz")
        print(f"Wavelet Used: {self.wavelet}")
        print(f"Decomposition Level: {self.target_level}")
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
        
        if all(p in fiducial_points for p in ['j_point', 't_peak']):
            jt_interval = (fiducial_points['t_peak'] - fiducial_points['j_point']) / self.fs * 1000
            print(f"J-T Interval:    {jt_interval:.1f} ms")

