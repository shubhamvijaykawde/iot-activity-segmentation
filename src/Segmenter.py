import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from claspy.segmentation import ClaSPEnsemble, BinaryClaSPSegmentation
import ruptures as rpt 
class Segmenter:
    """ClaSP-based segmentation with baseline comparison."""
    
    def __init__(self):
        self.segments = {}
        self.baseline_segments = {}
    
    def _recursive_binary_segmentation(self, signal_array: np.ndarray, n_segments: int, 
                                      window_size: str = "suss") -> np.ndarray:
        """
        Recursively apply binary segmentation using ClaSPEnsemble to get multiple segments.
        This is a fallback method.
        """
        if ClaSPEnsemble is None:
            raise ImportError("ClaSPEnsemble not available")
        
        change_points = []
        segments_to_find = n_segments - 1
        
        # Recursively split segments
        segments_to_split = [(0, len(signal_array))]
        
        while len(change_points) < segments_to_find and len(segments_to_split) > 0:
            start, end = segments_to_split.pop(0)
            if end - start < 20:  # Too short to split
                continue
            
            segment = signal_array[start:end]
            try:
                ensemble = ClaSPEnsemble(window_size=window_size)
                ensemble.fit(segment)
                cp = ensemble.split(sparse=True)
                if cp is not None and 0 < cp < len(segment):
                    global_cp = start + cp
                    change_points.append(global_cp)
                    # Add new segments to split
                    segments_to_split.append((start, global_cp))
                    segments_to_split.append((global_cp, end))
            except:
                continue
        
        return np.array(sorted(change_points))
    
    def segment_clasp_ensemble(self, signal: pd.Series, n_segments: Optional[int] = None,
                               window_size: str = "suss") -> np.ndarray:
        """
        Apply ClaSP Ensemble segmentation.
        
        Args:
            signal: 1D time series
            n_segments: Number of segments (if None, auto-detect)
            window_size: Window size parameter ("suss" for default)
        
        Returns:
            Array of change point indices
        """
        if ClaSPEnsemble is None:
            raise ImportError("claspy library not available. Install with: pip install claspy==0.2.7")
        
        if len(signal) < 10:
            # Too short for segmentation, return single segment
            return np.array([0, len(signal)])
        
        # Check for constant signal
        if signal.nunique() <= 1:
            # Constant signal, return single segment
            return np.array([0, len(signal)])
        
        signal_array = signal.values.reshape(-1, 1)  # ClaSP expects 2D
        
        if n_segments is None:
            # Auto-detect number of segments using binary segmentation
            try:
                # Use BinaryClaSPSegmentation with n_segments='learn' for auto-detection
                binary_seg = BinaryClaSPSegmentation(n_segments='learn', window_size=window_size)
                # Use fit_predict (standard method)
                change_points = binary_seg.fit_predict(signal_array)
                
                if isinstance(change_points, (int, np.integer)):
                    change_points = [change_points]
                change_points = np.array(change_points).flatten()
                n_segments = len(change_points) + 1
                print(f"     Auto-detected {n_segments} segments")
            except Exception as e:
                # Fallback: use heuristic
                n_segments = max(2, min(10, len(signal) // 100))
                print(f"     Using heuristic: {n_segments} segments (auto-detect failed: {e})")
        
        # Apply ClaSP segmentation
        # Use BinaryClaSPSegmentation for multiple segments (it accepts n_segments)
        # ClaSPEnsemble is only for binary segmentation (2 segments)
        change_points = None
        try:
            # Use BinaryClaSPSegmentation which supports n_segments parameter
            binary_seg = BinaryClaSPSegmentation(n_segments=n_segments, window_size=window_size)
            
            # Try fit_predict (standard method)
            try:
                change_points = binary_seg.fit_predict(signal_array)
            except (TypeError, AttributeError):
                # Fallback: try fit then predict
                try:
                    binary_seg.fit(signal_array)
                    change_points = binary_seg.predict()
                except:
                    # Last resort: try with 1D signal
                    change_points = binary_seg.fit_predict(signal.values)
                    
        except Exception as e1:
            # If BinaryClaSPSegmentation fails, try recursive binary segmentation with ClaSPEnsemble
            try:
                print(f"    ️ BinaryClaSPSegmentation failed, using recursive binary segmentation")
                change_points = self._recursive_binary_segmentation(signal_array, n_segments, window_size)
            except Exception as e2:
                raise RuntimeError(f"ClaSP segmentation failed: {e1}, {e2}")
        
        # Ensure change_points is a list/array
        if change_points is None:
            raise ValueError("Failed to get change points from ClaSP")
        
        if isinstance(change_points, (int, np.integer)):
            change_points = [change_points]
        change_points = np.array(change_points).flatten()
        
        # Remove any invalid indices
        change_points = change_points[(change_points > 0) & (change_points < len(signal))]
        change_points = np.unique(change_points)  # Remove duplicates
        
        # Convert to indices (ClaSP returns change points, we need segment boundaries)
        boundaries = [0] + sorted(change_points.tolist()) + [len(signal)]
        boundaries = np.unique(boundaries)  # Ensure unique boundaries
        
        return np.array(boundaries)
    
    def segment_baseline_pelt(self, signal: pd.Series, n_segments: Optional[int] = None) -> np.ndarray:
        """
        Baseline segmentation using PELT (ruptures library).
        
        Args:
            signal: 1D time series
            n_segments: Number of segments (if None, auto-detect)
        
        Returns:
            Array of segment boundary indices
        """
        if rpt is None:
            raise ImportError("ruptures library not available. Install with: pip install ruptures")
        
        signal_array = signal.values.reshape(-1, 1)
        
        # Use PELT algorithm (faster than Dynp for many segments)
        if n_segments is None:
            # Auto-detect using penalty (PELT is fast)
            algo = rpt.Pelt(model="rbf", min_size=2).fit(signal_array)
            change_points = algo.predict(pen=10)
        else:
            # For fixed number of segments, use Dynp but limit to reasonable number
            if n_segments > 20:
                # Too many segments, use PELT instead
                algo = rpt.Pelt(model="rbf", min_size=2).fit(signal_array)
                change_points = algo.predict(pen=10)
            else:
                # Use Dynp for small number of segments
                algo = rpt.Dynp(model="rbf", min_size=2, jump=5).fit(signal_array)  # jump=5 for speed
                change_points = algo.predict(n_bkps=n_segments - 1)
        
        boundaries = [0] + sorted(change_points) + [len(signal)]
        
        return np.array(boundaries)
    
    def evaluate_boundaries(self, boundaries: np.ndarray, state_changes: np.ndarray,
                           tolerance: int = 5) -> Dict:
        """
        Evaluate segmentation boundaries against state changes.
        
        Args:
            boundaries: Segment boundary indices
            state_changes: Indices where current_state changes
            tolerance: Tolerance window for matching (in samples)
        
        Returns:
            Dictionary with precision, recall, F1, and other metrics
        """
        if len(state_changes) == 0:
            pred_count = len(boundaries) - 2 if len(boundaries) > 2 else 0
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'true_positives': 0,
                'false_positives': pred_count,
                'false_negatives': 0,
                'n_predicted': pred_count,
                'n_true': 0
            }
        
        # Remove first boundary (always 0) and last (always end)
        pred_boundaries = set(boundaries[1:-1])
        true_boundaries = set(state_changes)
        
        # Match boundaries within tolerance
        true_positives = 0
        matched_pred = set()
        matched_true = set()
        
        for pred_bp in pred_boundaries:
            for true_bp in true_boundaries:
                if abs(pred_bp - true_bp) <= tolerance and pred_bp not in matched_pred and true_bp not in matched_true:
                    true_positives += 1
                    matched_pred.add(pred_bp)
                    matched_true.add(true_bp)
                    break
        
        false_positives = len(pred_boundaries) - len(matched_pred)
        false_negatives = len(true_boundaries) - len(matched_true)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'n_predicted': len(pred_boundaries),
            'n_true': len(true_boundaries)
        }
    def prune_boundaries(self, boundaries: np.ndarray, min_gap: int = 10):

        inner = sorted(boundaries[1:-1])
        pruned = []
        for b in inner:
          if not pruned or b - pruned[-1] > min_gap:
              pruned.append(b)
        return np.array([0] + pruned + [boundaries[-1]])
    def compute_true_boundaries(self, signal, min_jump=0.08):

      if not isinstance(signal, pd.Series):
          raise ValueError("Signal must be numeric pandas Series")

      sig = signal.values.astype(float)

      boundaries = []

      for i in range(1, len(sig)):
          jump = abs(sig[i] - sig[i-1])

          if jump > min_jump:
              boundaries.append(i)

      return np.array(boundaries)

    def compute_regime_aligned_gt(self, df, signal,smooth_window=25,jump_threshold=0.05,min_regime_len=100):
      """
      Ground truth aligned with signal regime structure.
      """

      # --- Smooth signal (remove micro noise) ---
      sig = pd.Series(signal).astype(float)
      smooth = sig.rolling(smooth_window, center=True).mean()
      smooth = smooth.fillna(method="bfill").fillna(method="ffill")

      # --- Detect major regime jumps ---
      diff = smooth.diff().abs().fillna(0)

      candidates = np.where(diff > jump_threshold)[0]

      # --- Enforce minimum regime duration ---
      boundaries = []
      last = 0

      for c in candidates:
          if c - last > min_regime_len:
              boundaries.append(c)
              last = c

      return np.array(boundaries)
    def validate_segment_statistics(self,signal, boundaries):

      sig = signal.values.astype(float)
      segments = []

      for i in range(len(boundaries) - 1):
          start = boundaries[i]
          end = boundaries[i+1]
          segments.append(sig[start:end])

      stats = []

      for i, seg in enumerate(segments):
          stats.append({
              "segment": i,
              "mean": np.mean(seg),
              "std": np.std(seg),
              "var": np.var(seg)
          })

      return pd.DataFrame(stats)
    def segment_separation_score(self, signal, boundaries):
      sig = signal.astype(float).values

      segments = [
          sig[boundaries[i]:boundaries[i+1]]
          for i in range(len(boundaries)-1)
      ]

      scores = []

      for i in range(len(segments)-1):
          m1, m2 = np.mean(segments[i]), np.mean(segments[i+1])
          v1, v2 = np.var(segments[i]), np.var(segments[i+1])

          separation = abs(m2 - m1) / np.sqrt(v1 + v2 + 1e-8)
          scores.append(separation)

      return np.mean(scores)

    def segment_stability_score(self, signal, boundaries):
      sig = signal.astype(float).values

      variances = []

      for i in range(len(boundaries)-1):
          seg = sig[boundaries[i]:boundaries[i+1]]
          variances.append(np.var(seg))

      return np.mean(variances)
    def boundary_strength_score(self, signal, boundaries):
      sig = signal.astype(float).values

      strengths = []

      for b in boundaries[1:-1]:
          jump = abs(sig[b] - sig[b-1])
          strengths.append(jump)

      return np.mean(strengths)

    def segmentation_explained_variance(self, signal, boundaries):
      sig = signal.astype(float).values

      total_var = np.var(sig)

      seg_vars = []

      for i in range(len(boundaries)-1):
          seg = sig[boundaries[i]:boundaries[i+1]]
          seg_vars.append(np.var(seg))

      within_var = np.mean(seg_vars)

      return 1 - (within_var / (total_var + 1e-8))
    def get_segment_time_ranges(self, df, boundaries):
      """
      Convert ClaSP boundaries into timestamp intervals.
      """

      timestamps = pd.to_datetime(df["timestamp"])

      segments = []

      for i in range(len(boundaries) - 1):
          start_idx = boundaries[i]
          end_idx = boundaries[i + 1] - 1

          segments.append({
              "segment": i,
              "start_index": start_idx,
              "end_index": end_idx,
              "start_time": timestamps.iloc[start_idx],
              "end_time": timestamps.iloc[end_idx]
          })

      return pd.DataFrame(segments)
    def enhance_signal_for_segmentation(self, signal,smooth_window=15,diff_weight=2.0):
      """
      Enhance regime structure before ClaSP.
      """

      sig = pd.Series(signal).astype(float)

      # Smooth noise
      smooth = sig.rolling(smooth_window, center=True).mean()
      smooth = smooth.fillna(method="bfill").fillna(method="ffill")

      # First derivative (change intensity)
      diff = smooth.diff().fillna(0)

      # Combine signal + change emphasis
      enhanced = smooth + diff_weight * diff

      return enhanced
    def segment_station(self, df: pd.DataFrame, signal: pd.Series, station: str,
                       use_baseline: bool = False) -> Dict:
        """
        Segment a station's time series.
        
        Returns:
            Dictionary with segments and evaluation metrics
        """
        print(f"\n   Segmenting {station}...")
        print(f"     Signal length: {len(signal)}")
        
        # Detect state changes for evaluation
        '''if 'current_state' in df.columns:
            state_changes = []
            prev_state = None
            for idx, state in enumerate(df['current_state']):
                if prev_state is not None and state != prev_state:
                    state_changes.append(idx)
                prev_state = state
            state_changes = np.array(state_changes)
            print(f"     State changes detected: {len(state_changes)}")
        else:
            state_changes = np.array([])'''
        state_changes = self.compute_regime_aligned_gt(df, signal)
        #state_changes = self.compute_true_boundaries(signal)
        print(f"     Signal-aligned GT boundaries: {len(state_changes)}")
        # Estimate number of segments based on state changes
        n_segments = max(2, len(state_changes) + 1) if len(state_changes) > 0 else None
        
        # ClaSP segmentation
        try:
            enhanced_signal = self.enhance_signal_for_segmentation(signal)
            boundaries = self.segment_clasp_ensemble(enhanced_signal, n_segments=None)

            boundaries = self.prune_boundaries(boundaries, min_gap=3)
            print(f"     ClaSP boundaries: {len(boundaries) - 1} segments")

            stats = self.validate_segment_statistics(signal, boundaries)
            print("\n Segment statistics:")
            print(stats)
            segment_times = self.get_segment_time_ranges(df, boundaries)
            print("\n Segment time ranges:")
            print(segment_times)
        except Exception as e:
            print(f"     ClaSP failed: {e}")
            # Fallback: simple equal-length segments
            n_segments = max(2, min(10, len(signal) // 100))
            boundaries = np.linspace(0, len(signal), n_segments + 1, dtype=int)
            print(f"    ️ Using fallback segmentation: {n_segments} segments")

        # Baseline segmentation (if requested) - skip if too many segments (too slow)
        baseline_boundaries = None
        if use_baseline and rpt is not None:
            # Skip baseline if n_segments is too large (Dynp is very slow)
            if n_segments is not None and n_segments > 20:
                print(f"    ️ Skipping baseline segmentation (too many segments: {n_segments}, would be too slow)")
            else:
                try:
                    print(f"     Running baseline segmentation (PELT)...")
                    baseline_boundaries = self.segment_baseline_pelt(signal, n_segments=n_segments)
                    print(f"     Baseline (PELT) boundaries: {len(baseline_boundaries) - 1} segments")
                except Exception as e:
                    print(f"     Baseline segmentation failed: {e}")
        
        # Evaluate boundaries
        print(f"     Evaluating boundaries...")
        try:
            '''metrics = self.evaluate_boundaries_timestamp(
                boundaries,
                state_changes,
                df["timestamp"],
                tolerance_seconds=5.0
            )'''

            metrics = {
                  "separation_score": self.segment_separation_score(signal, boundaries),
                  "stability_score": self.segment_stability_score(signal, boundaries),
                  "boundary_strength": self.boundary_strength_score(signal, boundaries),
                  "explained_variance": self.segmentation_explained_variance(signal, boundaries)
            }
            #metrics = self.evaluate_boundaries(boundaries, state_changes,tolerance=20)
            #print(f"     Evaluation complete: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
            print("\nSegmentation credibility metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.3f}")

            #print('Specially focussed metrics section',metrics)
        except Exception as e:
            print(f"     Error in evaluation: {e}")
            # Use default metrics on error
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'n_predicted': len(boundaries) - 2,
                'n_true': len(state_changes)
            }

            metrics = {
                "separation_score": 0,
                "stability_score": 0,
                "boundary_strength": 0,
                "explained_variance":0
            }

        result = {
            'station': station,
            'boundaries': boundaries,
            'baseline_boundaries': baseline_boundaries,
            'state_changes': state_changes,
            'metrics': metrics,
            'n_segments': len(boundaries) - 1
        }
        print(f"     Segmentation complete for {station}")
        return result