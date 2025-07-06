import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress
from jpoint import WaveletJPointDetector
import neurokit2 as nk

class EnhancedSTAnalyzer:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.j_detector = None  
        
    def set_j_detector(self, detector):
        """Set the J-point detector instance"""
        self.j_detector = detector
        
    def calculate_baseline(self, beat, fiducial_points):
        """Calculate baseline from PR segment or TP segment"""
        baseline_methods = []
        
        # Method 1: PR segment baseline
        if 'p_peak' in fiducial_points and 'qrs_onset' in fiducial_points:
            pr_start = max(0, fiducial_points['p_peak'] - int(0.08 * self.fs))  # 80ms before P
            pr_end = max(0, fiducial_points['qrs_onset'] - int(0.02 * self.fs))  # 20ms before QRS
            
            if pr_end > pr_start and pr_end < len(beat):
                pr_segment = beat[pr_start:pr_end]
                if len(pr_segment) > 5:  # Need minimum samples
                    pr_baseline = np.median(pr_segment)
                    baseline_methods.append(('PR', pr_baseline))
        
        # Method 2: TP segment baseline (if available)
        if 't_peak' in fiducial_points and len(beat) > fiducial_points['t_peak'] + int(0.1 * self.fs):
            tp_start = fiducial_points['t_peak'] + int(0.05 * self.fs)  # 50ms after T peak
            tp_end = min(len(beat), fiducial_points['t_peak'] + int(0.15 * self.fs))  # 150ms after T peak
            
            if tp_end > tp_start:
                tp_segment = beat[tp_start:tp_end]
                if len(tp_segment) > 5:
                    tp_baseline = np.median(tp_segment)
                    baseline_methods.append(('TP', tp_baseline))
        
        # Method 3: Early beat baseline (fallback)
        early_segment = beat[:int(0.15 * len(beat))]
        early_baseline = np.median(early_segment)
        baseline_methods.append(('Early', early_baseline))
        
        # Choose the best baseline (prefer PR, then TP, then early)
        if baseline_methods:
            # Use the first available method (they are in order of preference)
            chosen_method, baseline = baseline_methods[0]
            return baseline, chosen_method, baseline_methods
        else:
            return 0, 'Default', [('Default', 0)]
    
    def measure_st_elevation(self, beat, fiducial_points, baseline):
        """Measure ST elevation/depression at multiple time points"""
        j_point_idx = fiducial_points['j_point']
        
        # Standard measurement points (in milliseconds after J-point)
        measurement_times = [0.02, 0.06, 0.08, 0.10, 0.12]  # 20ms, 60ms, 80ms, 100ms, 120ms
        
        st_measurements = {}
        st_indices = {}
        
        for time_ms in measurement_times:
            time_samples = int(time_ms * self.fs)
            st_idx = j_point_idx + time_samples
            
            if st_idx < len(beat):
                st_level = beat[st_idx] - baseline
                st_measurements[f'st_{int(time_ms*1000)}ms'] = st_level
                st_indices[f'st_{int(time_ms*1000)}ms'] = st_idx
            else:
                st_measurements[f'st_{int(time_ms*1000)}ms'] = None
                st_indices[f'st_{int(time_ms*1000)}ms'] = None
        
        return st_measurements, st_indices
    
    def analyze_st_morphology(self, beat, fiducial_points, baseline):
        """Analyze ST segment morphology"""
        j_point_idx = fiducial_points['j_point']
        
        # Define ST segment (J-point to beginning of T-wave)
        st_start = j_point_idx
        # Use either detected T-wave start or fixed duration
        if 't_peak' in fiducial_points:
            # ST segment typically ends ~40ms before T peak
            st_end = min(len(beat), fiducial_points['t_peak'] - int(0.04 * self.fs))
        else:
            # Fixed duration: 120ms after J-point
            st_end = min(len(beat), j_point_idx + int(0.12 * self.fs))
        
        if st_end <= st_start:
            st_end = min(len(beat), st_start + int(0.08 * self.fs))  # Minimum 80ms
        
        st_segment = beat[st_start:st_end] - baseline
        
        morphology_results = {}
        
        if len(st_segment) > 3:
            # ST slope (linear regression)
            x = np.arange(len(st_segment))
            slope, intercept, r_value, p_value, std_err = linregress(x, st_segment)
            morphology_results['st_slope'] = slope * self.fs  # Convert to mV/s
            morphology_results['st_slope_r_squared'] = r_value**2
            
            # ST curvature (second derivative)
            if len(st_segment) > 5:
                # Smooth first
                st_smooth = gaussian_filter1d(st_segment, sigma=1)
                st_curvature = np.mean(np.diff(st_smooth, 2))
                morphology_results['st_curvature'] = st_curvature
            else:
                morphology_results['st_curvature'] = 0
            
            # ST shape classification
            morphology_results['st_shape'] = self._classify_st_shape(st_segment)
            
            # ST segment variability
            morphology_results['st_variability'] = np.std(st_segment)
            
        else:
            morphology_results = {
                'st_slope': 0,
                'st_slope_r_squared': 0,
                'st_curvature': 0,
                'st_shape': 'insufficient_data',
                'st_variability': 0
            }
        
        return morphology_results, st_segment, (st_start, st_end)
    
    def _classify_st_shape(self, st_segment):
        """Classify ST segment shape"""
        if len(st_segment) < 5:
            return 'insufficient_data'
        
        # Smooth the segment
        st_smooth = gaussian_filter1d(st_segment, sigma=1)
        
        # Calculate slope
        x = np.arange(len(st_smooth))
        slope, _, r_squared, _, _ = linregress(x, st_smooth)
        
        # Classify based on slope and curvature
        if abs(slope) < 0.001:  # Nearly flat
            return 'horizontal'
        elif slope > 0.002:  # Upward slope
            return 'upsloping'
        elif slope < -0.002:  # Downward slope
            return 'downsloping'
        else:
            # Check curvature for subtle patterns
            if len(st_smooth) > 5:
                curvature = np.mean(np.diff(st_smooth, 2))
                if curvature > 0.001:
                    return 'concave'
                elif curvature < -0.001:
                    return 'convex'
            return 'horizontal'
    
    def detect_st_abnormalities(self, st_measurements, morphology_results):
        """Detect ST elevation/depression abnormalities"""
        abnormalities = []
        
        # ST elevation thresholds (in mV)
        st_elevation_threshold = 0.1  # 1mm = 0.1mV
        st_depression_threshold = -0.1
        
        # Check ST measurements
        for measurement, value in st_measurements.items():
            if value is not None:
                if value >= st_elevation_threshold:
                    abnormalities.append({
                        'type': 'ST_ELEVATION',
                        'measurement': measurement,
                        'value': value,
                        'severity': 'mild' if value < 0.2 else 'moderate' if value < 0.5 else 'severe'
                    })
                elif value <= st_depression_threshold:
                    abnormalities.append({
                        'type': 'ST_DEPRESSION',
                        'measurement': measurement,
                        'value': value,
                        'severity': 'mild' if value > -0.2 else 'moderate' if value > -0.5 else 'severe'
                    })
        
        # Check morphology
        if morphology_results['st_slope'] > 0.01:  # Steep upslope
            abnormalities.append({
                'type': 'ST_UPSLOPING',
                'value': morphology_results['st_slope'],
                'severity': 'mild'
            })
        elif morphology_results['st_slope'] < -0.01:  # Steep downslope
            abnormalities.append({
                'type': 'ST_DOWNSLOPING',
                'value': morphology_results['st_slope'],
                'severity': 'mild'
            })
        
        return abnormalities
    
    def analyze_st_segment_comprehensive(self, beat, fiducial_points=None):
        """Comprehensive ST segment analysis"""
        
        # If fiducial points not provided, detect them
        if fiducial_points is None:
            if self.j_detector is None:
                raise ValueError("J-point detector not set. Use set_j_detector() first.")
            fiducial_points = self.j_detector.detect_all_fiducial_points(beat)
        
        # Calculate baseline
        baseline, baseline_method, all_baselines = self.calculate_baseline(beat, fiducial_points)
        
        # Measure ST elevation/depression
        st_measurements, st_indices = self.measure_st_elevation(beat, fiducial_points, baseline)
        
        # Analyze ST morphology
        morphology_results, st_segment, st_bounds = self.analyze_st_morphology(beat, fiducial_points, baseline)
        
        # Detect abnormalities
        abnormalities = self.detect_st_abnormalities(st_measurements, morphology_results)
        
        # Compile comprehensive results
        results = {
            'fiducial_points': fiducial_points,
            'baseline': baseline,
            'baseline_method': baseline_method,
            'all_baselines': all_baselines,
            'st_measurements': st_measurements,
            'st_indices': st_indices,
            'morphology': morphology_results,
            'st_segment': st_segment,
            'st_bounds': st_bounds,
            'abnormalities': abnormalities,
            'j_point_quality': self._assess_j_point_quality(beat, fiducial_points)
        }
        
        return results
    
    def _assess_j_point_quality(self, beat, fiducial_points):
        """Assess quality of J-point detection"""
        j_point_idx = fiducial_points['j_point']
        r_peak_idx = fiducial_points['r_peak']
        
        # Check if J-point is in reasonable position relative to R-peak
        j_r_distance = j_point_idx - r_peak_idx
        expected_distance = int(0.04 * self.fs)  # ~40ms after R-peak
        
        if abs(j_r_distance - expected_distance) > int(0.03 * self.fs):  # >30ms deviation
            quality_score = 0.5
        else:
            quality_score = 0.8
        
        # Check local signal characteristics around J-point
        window = int(0.01 * self.fs)  # 10ms window
        start_idx = max(0, j_point_idx - window)
        end_idx = min(len(beat), j_point_idx + window)
        
        local_signal = beat[start_idx:end_idx]
        if len(local_signal) > 3:
            local_variability = np.std(local_signal)
            if local_variability < 0.05:  # Low variability suggests good J-point
                quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def plot_st_analysis(self, beat, results):
        """Plot comprehensive ST analysis"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: ECG with fiducial points and ST measurements
        axes[0].plot(beat, 'b-', linewidth=2, label='ECG Beat')
        
        # Plot baseline
        axes[0].axhline(y=results['baseline'], color='gray', linestyle='--', 
                       alpha=0.7, label=f'Baseline ({results["baseline_method"]})')
        
        # Mark fiducial points
        points_info = {
            'p_peak': ('P', 'green', 'o'),
            'qrs_onset': ('QRS Start', 'orange', 's'),
            'r_peak': ('R', 'red', '^'),
            'j_point': ('J', 'purple', 'D'),
            'qrs_offset': ('QRS End', 'brown', 'v'),
            't_peak': ('T', 'cyan', 'o')
        }
        
        for point, (label, color, marker) in points_info.items():
            if point in results['fiducial_points']:
                idx = results['fiducial_points'][point]
                if 0 <= idx < len(beat):
                    axes[0].scatter(idx, beat[idx], color=color, s=100, 
                                  marker=marker, label=label, zorder=5, edgecolor='black')
        
        # Mark ST measurement points
        for measurement, idx in results['st_indices'].items():
            if idx is not None and 0 <= idx < len(beat):
                st_value = results['st_measurements'][measurement]
                if st_value is not None:
                    color = 'red' if st_value > 0.1 else 'blue' if st_value < -0.1 else 'green'
                    axes[0].scatter(idx, beat[idx], color=color, s=80, marker='x', 
                                  alpha=0.8, zorder=4)
                    # Add text annotation
                    axes[0].annotate(f'{st_value:.2f}mV', 
                                   xy=(idx, beat[idx]), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8)
        
        axes[0].set_title('ECG Beat with ST Analysis', fontsize=14)
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: ST segment detailed view
        st_start, st_end = results['st_bounds']
        st_indices_array = np.arange(st_start, st_end)
        st_segment_with_baseline = beat[st_start:st_end]
        
        axes[1].plot(st_indices_array, st_segment_with_baseline, 'b-', linewidth=2, label='ST Segment')
        axes[1].axhline(y=results['baseline'], color='gray', linestyle='--', 
                       alpha=0.7, label='Baseline')
        
        # Plot ST measurements on detailed view
        for measurement, idx in results['st_indices'].items():
            if idx is not None and st_start <= idx < st_end:
                st_value = results['st_measurements'][measurement]
                if st_value is not None:
                    color = 'red' if st_value > 0.1 else 'blue' if st_value < -0.1 else 'green'
                    axes[1].scatter(idx, beat[idx], color=color, s=100, marker='o', 
                                  alpha=0.8, zorder=4)
                    axes[1].annotate(f'{measurement}\n{st_value:.3f}mV', 
                                   xy=(idx, beat[idx]), xytext=(5, 10), 
                                   textcoords='offset points', fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        axes[1].set_title('ST Segment Detail', fontsize=12)
        axes[1].set_ylabel('Amplitude (mV)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: ST morphology analysis
        st_segment_corrected = results['st_segment']
        st_x = np.arange(len(st_segment_corrected))
        
        axes[2].plot(st_x, st_segment_corrected, 'b-', linewidth=2, label='ST Segment (Baseline Corrected)')
        axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.7, label='Zero Line')
        
        # Add slope line
        if 'st_slope' in results['morphology']:
            slope = results['morphology']['st_slope'] / self.fs  # Convert back to per sample
            slope_line = slope * st_x + st_segment_corrected[0]
            axes[2].plot(st_x, slope_line, 'r--', alpha=0.7, 
                        label=f'Slope: {results["morphology"]["st_slope"]:.6f} mV/s')
        
        axes[2].set_title('ST Morphology Analysis', fontsize=12)
        axes[2].set_ylabel('ST Deviation (mV)')
        axes[2].set_xlabel('Sample')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_st_analysis_results(self, results):
        """Print comprehensive ST analysis results"""
        print("=" * 60)
        print("COMPREHENSIVE ST SEGMENT ANALYSIS RESULTS")
        print("=" * 60)
        
        # Basic info
        print(f"J-Point Quality Score: {results['j_point_quality']:.2f}")
        print(f"Baseline: {results['baseline']:.4f} mV (Method: {results['baseline_method']})")
        print()
        
        # ST measurements
        print("ST ELEVATION/DEPRESSION MEASUREMENTS:")
        print("-" * 40)
        for measurement, value in results['st_measurements'].items():
            if value is not None:
                status = "ELEVATED" if value > 0.1 else "DEPRESSED" if value < -0.1 else "NORMAL"
                print(f"{measurement:12}: {value:8.4f} mV ({status})")
            else:
                print(f"{measurement:12}: N/A (out of range)")
        print()
        
        # Morphology
        print("ST MORPHOLOGY ANALYSIS:")
        print("-" * 40)
        morphology = results['morphology']
        print(f"ST Slope:        {morphology['st_slope']:8.6f} mV/s")
        print(f"ST Shape:        {morphology['st_shape']}")
        print(f"ST Curvature:    {morphology['st_curvature']:8.6f}")
        print(f"ST Variability:  {morphology['st_variability']:8.6f} mV")
        if 'st_slope_r_squared' in morphology:
            print(f"Slope R²:        {morphology['st_slope_r_squared']:8.4f}")
        print()
        
        # Abnormalities
        print("DETECTED ABNORMALITIES:")
        print("-" * 40)
        if results['abnormalities']:
            for abnormality in results['abnormalities']:
                print(f"• {abnormality['type']}: {abnormality['value']:.4f} mV ({abnormality['severity']})")
        else:
            print("• No significant abnormalities detected")
        print()
        
        # All baseline methods
        print("BASELINE CALCULATION METHODS:")
        print("-" * 40)
        for method, baseline in results['all_baselines']:
            marker = "★" if method == results['baseline_method'] else " "
            print(f"{marker} {method:8}: {baseline:.4f} mV")



def run_st_analysis_example():
    """Run example ST analysis"""

    
    # Create detector and analyzer
    detector = WaveletJPointDetector(sampling_rate=1000)
    analyzer = EnhancedSTAnalyzer(sampling_rate=1000)
    analyzer.set_j_detector(detector)
    
    # Generate test ECG beat (you can replace this with your own data)
    ecg = nk.data("ecg_1000hz")
    cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=1000)
    rpeaks_idx = np.where(peaks["ECG_R_Peaks"] == 1)[0]
    beats = nk.ecg_segment(cleaned, rpeaks_idx, sampling_rate=1000)
    beat = list(beats.values())[0]["Signal"].values
    
    # Run comprehensive ST analysis
    results = analyzer.analyze_st_segment_comprehensive(beat)
    
    # Print results
    analyzer.print_st_analysis_results(results)
    
    # Plot analysis
    analyzer.plot_st_analysis(beat, results)
    
    return results


results = run_st_analysis_example()