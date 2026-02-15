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
# Segmentation libraries
try:
    from claspy.segmentation import ClaSPEnsemble, BinaryClaSPSegmentation
except ImportError:
    try:
        from claspy import ClaSPEnsemble, BinaryClaSPSegmentation
    except ImportError:
        print("Warning: claspy not found. Install with: pip install claspy==0.2.7")
        ClaSPEnsemble = None
        BinaryClaSPSegmentation = None

try:
    import ruptures as rpt
except ImportError:
    print("Warning: ruptures not found. Install with: pip install ruptures")
    rpt = None

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# For Jupyter display
# ============================================================================
# TASK 1: DATA LOADING & EXPLORATION
# ============================================================================

class IoTDataLoader:
    """Load and preprocess IoT JSONL files with mixed station data."""
    
    def __init__(self, data_dir: str = "data/data-radiant-eval-paper/factory/evaluation/iot_logs"):
        self.data_dir = Path(data_dir)
        self.raw_data = None
        self.processed_data = None
        self.station_data = {}
        
    def load_all_files(self, file_pattern: str = "*.jsonl") -> pd.DataFrame:
        """
        Load all JSONL files from data directory.
        
        Returns:
            Unified DataFrame with all records
        """
        jsonl_files = sorted(self.data_dir.glob(file_pattern))
        
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {self.data_dir}")
        
        print(f" Loading {len(jsonl_files)} JSONL files...")
        
        all_records = []
        for file_path in jsonl_files:
            print(f"   Loading {file_path.name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        record['source_file'] = file_path.name
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON at line {line_num} in {file_path.name}: {e}")
        
        self.raw_data = pd.DataFrame(all_records)
        print(f" Loaded {len(self.raw_data)} total records")
        
        return self.raw_data
    
    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp strings to datetime objects."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            # Sort by timestamp for temporal analysis
            #df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def handle_mixed_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert string numbers to numeric types.
        Handles fields like "0.00" that should be numeric.
        """
        df = df.copy()
        
        for col in df.columns:
            if col in ['id', 'station', 'timestamp', 'current_state', 'current_task', 
                      'current_sub_task', 'source_file', 'hbw1_current_stock']:
                continue
            
            # Try to convert to numeric
            if df[col].dtype == 'object':
                # Check if values are numeric strings
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    try:
                        # Try converting first non-null value
                        test_val = sample.iloc[0]
                        if isinstance(test_val, str) and test_val.replace('.', '').replace('-', '').isdigit():
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
        
        return df
    
    def expand_nested_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand nested JSON structures (e.g., hbw1_current_stock).
        For now, we'll extract key information or flatten if needed.
        """
        df = df.copy()
        
        # Handle hbw1_current_stock nested structure
        if 'hbw1_current_stock' in df.columns:
            # Count non-empty stock positions
            def count_stock_items(stock_dict):
                if isinstance(stock_dict, dict):
                    return sum(1 for v in stock_dict.values() if v and str(v).strip())
                return 0
            
            df['hbw1_stock_count'] = df['hbw1_current_stock'].apply(count_stock_items)
            # Drop the nested column for now (can be expanded later if needed)
            df = df.drop(columns=['hbw1_current_stock'], errors='ignore')
        
        return df
    
    def split_by_station(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split unified DataFrame by station for individual analysis.
        
        Returns:
            Dictionary mapping station names to DataFrames
        """
        if 'station' not in df.columns:
            raise ValueError("DataFrame must have 'station' column")
        
        station_data = {}
        for station in df['station'].unique():
            station_df = df[df['station'] == station].copy()
            station_df = station_df.sort_values('timestamp').reset_index(drop=True)
            station_data[station] = station_df
            print(f"   Station {station}: {len(station_df)} records")
        
        self.station_data = station_data
        return station_data
    
    def get_diagnostics(self, df: pd.DataFrame) -> Dict:
        """
        Provide basic diagnostics: time ranges, missing values, sensor distributions.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            'total_records': len(df),
            'stations': df['station'].value_counts().to_dict() if 'station' in df.columns else {},
            'time_range': {},
            'missing_values': {},
            'sensor_info': {}
        }
        
        # Time range per station
        if 'timestamp' in df.columns and 'station' in df.columns:
            for station in df['station'].unique():
                station_df = df[df['station'] == station]
                if not station_df['timestamp'].isna().all():
                    diagnostics['time_range'][station] = {
                        'start': station_df['timestamp'].min(),
                        'end': station_df['timestamp'].max(),
                        'duration_hours': (station_df['timestamp'].max() - station_df['timestamp'].min()).total_seconds() / 3600
                    }
        
        # Missing values
        sensor_cols = [col for col in df.columns if col not in 
                      ['id', 'station', 'timestamp', 'current_state', 'current_task', 
                       'current_sub_task', 'source_file', 'hbw1_current_stock']]
        
        for col in sensor_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                diagnostics['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        # Sensor distributions (basic stats)
        for col in sensor_cols:
            if df[col].dtype in ['int64', 'float64']:
                diagnostics['sensor_info'][col] = {
                    'dtype': str(df[col].dtype),
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'unique_values': int(df[col].nunique())
                }
        
        return diagnostics
    
    def process_all(self) -> pd.DataFrame:
        """Complete data loading and preprocessing pipeline."""
        print("=" * 60)
        print("TASK 1: DATA LOADING & EXPLORATION")
        print("=" * 60)
        
        # Load files
        df = self.load_all_files()
        
        # Parse timestamps
        print("\n Parsing timestamps...")
        df = self.parse_timestamps(df)
        
        # Handle mixed types
        print(" Handling mixed numeric types...")
        df = self.handle_mixed_types(df)
        
        # Expand nested structures
        print(" Expanding nested structures...")
        df = self.expand_nested_structures(df)
        
        # Split by station
        print("\n Splitting data by station...")
        self.split_by_station(df)
        
        # Diagnostics
        print("\n Generating diagnostics...")
        diagnostics = self.get_diagnostics(df)
        
        print("\n" + "=" * 60)
        print("DIAGNOSTICS SUMMARY")
        print("=" * 60)
        print(f" Total records: {diagnostics['total_records']}")
        print(f"\n Stations: {list(diagnostics['stations'].keys())}")
        for station, count in diagnostics['stations'].items():
            print(f"  {station}: {count} records")
        
        print(f"\n Time ranges:")
        for station, tr in diagnostics['time_range'].items():
            print(f"  {station}:")
            print(f"    Start: {tr['start']}")
            print(f"    End: {tr['end']}")
            print(f"    Duration: {tr['duration_hours']:.2f} hours")
        
        if diagnostics['missing_values']:
            print(f"\n Missing values (top 10):")
            sorted_missing = sorted(diagnostics['missing_values'].items(), 
                                  key=lambda x: x[1]['count'], reverse=True)[:10]
            for col, info in sorted_missing:
                print(f"  {col}: {info['count']} ({info['percentage']:.2f}%)")
        else:
            print("\n No missing values detected in sensor columns.")
        
        self.processed_data = df
        return df
