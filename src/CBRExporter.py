# ============================================================================
# TASK 4: PIPELINE INTEGRATION & EXPORT
# ============================================================================
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

class CBRExporter:
    """Export segments in CBR-ready format."""
    
    @staticmethod
    def create_segment_dict(df: pd.DataFrame, station: str, seg_id: int,
                           start_idx: int, end_idx: int,
                           aggregated_signal: pd.Series) -> Dict:
        """
        Create a single segment dictionary in CBR format.
        
        Args:
            df: Full DataFrame for the station
            station: Station name
            seg_id: Segment ID
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            aggregated_signal: Aggregated signal for feature extraction
        
        Returns:
            Segment dictionary
        """
        segment_df = df.iloc[start_idx:end_idx].copy()
        
        # Extract features (aggregated statistics)
        sensor_cols = [col for col in segment_df.columns if col not in 
                      ['id', 'station', 'timestamp', 'current_state', 'current_task',
                       'current_sub_task', 'source_file', 'hbw1_current_stock']]
        
        features = {}
        for col in sensor_cols:
            if segment_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                mean_val = segment_df[col].mean()
                std_val = segment_df[col].std() if len(segment_df) > 1 else 0.0
                # Handle NaN values
                if pd.notna(mean_val):
                    features[f"mean_{col}"] = float(mean_val)
                if pd.notna(std_val):
                    features[f"std_{col}"] = float(std_val)
        
        # Dominant state
        if 'current_state' in segment_df.columns and len(segment_df) > 0:
            mode_result = segment_df['current_state'].mode()
            dominant_state = mode_result.iloc[0] if len(mode_result) > 0 else "unknown"
            state_changes = max(0, int((segment_df['current_state'] != segment_df['current_state'].shift()).sum() - 1))
        else:
            dominant_state = "unknown"
            state_changes = 0
        
        # Timestamps
        if 'timestamp' in segment_df.columns and len(segment_df) > 0:
            start_time = segment_df['timestamp'].iloc[0]
            end_time = segment_df['timestamp'].iloc[-1]
        else:
            start_time = None
            end_time = None
        
        segment_dict = {
            "station": station,
            "segment_id": seg_id,
            "start_index": int(start_idx),
            "end_index": int(end_idx),
            "start_time": str(start_time) if start_time else None,
            "end_time": str(end_time) if end_time else None,
            "features": features,
            "dominant_state": str(dominant_state),
            "state_changes": int(max(0, state_changes)),
            "length": int(end_idx - start_idx)
        }
        
        return segment_dict
    
    @staticmethod
    def export_segments(segmentation_result: Dict, df: pd.DataFrame,
                       aggregated_signal: pd.Series, output_file: str = None) -> List[Dict]:
        """
        Export all segments for a station.
        
        Returns:
            List of segment dictionaries
        """
        station = segmentation_result['station']
        boundaries = segmentation_result['boundaries']
        
        segments = []
        for seg_id in range(len(boundaries) - 1):
            start_idx = boundaries[seg_id]
            end_idx = boundaries[seg_id + 1]
            
            segment_dict = CBRExporter.create_segment_dict(
                df, station, seg_id + 1, start_idx, end_idx, aggregated_signal
            )
            segments.append(segment_dict)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(segments, f, indent=2, default=str)
            print(f"   Exported {len(segments)} segments to {output_file}")
        
        return segments