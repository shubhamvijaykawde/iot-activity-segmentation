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
class FeatureEngineer:
    """Feature engineering for station-specific sensor data."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_configs = {}
    
    def get_station_sensors(self, df: pd.DataFrame, station: str) -> List[str]:
        """Extract station-specific sensor column names."""
        station_prefix = station.lower().replace('_', '')
        sensor_cols = []
        
        for col in df.columns:
            # Skip metadata columns
            if col in ['id', 'station', 'timestamp', 'current_state', 'current_task',
                      'current_sub_task', 'source_file', 'hbw1_current_stock']:
                continue
            
            # Check if column belongs to this station
            col_lower = col.lower()
            if col_lower.startswith(station_prefix) or col_lower.startswith(station_prefix[:3]):
                sensor_cols.append(col)
        
        return sensor_cols
    
    def classify_sensor_type(self, df: pd.DataFrame, col: str) -> str:
        """
        Classify sensor as binary, continuous, or position.
        
        Returns:
            'binary', 'continuous', 'position', or 'other'
        """
        if df[col].isna().all():
            return 'other'
        
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        
        # Binary sensors (0/1 or True/False)
        if n_unique <= 2 and all(v in [0, 1, True, False, 0.0, 1.0] for v in unique_vals):
            return 'binary'
        
        # Position coordinates
        if 'pos' in col.lower() and ('x' in col.lower() or 'y' in col.lower() or 'z' in col.lower()):
            return 'position'
        
        # Continuous sensors (speeds, photoresistor, color sensors, etc.)
        return 'continuous'
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features: time since last event, rate of change.
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Time since last event (in seconds)
        if 'timestamp' in df.columns:
            df['time_since_last'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
            df['time_since_last'] = df['time_since_last'].clip(upper=3600)  # Cap at 1 hour
        
        return df
    def build_position_features(self, values):
      vel = np.diff(values, prepend=values[0])
      threshold = np.percentile(np.abs(vel), 10)
      vel[np.abs(vel) < threshold] = 0
      motion = np.abs(vel)
      scale = np.percentile(motion, 95)
      if scale > 0:
        motion = motion / scale

      return np.clip(motion, 0, 1)


    def aggregate_signals(self, df: pd.DataFrame, sensor_cols: List[str],
                         station: str) -> pd.Series:
        """
        Create 1D aggregated signal for segmentation.
        
        Strategy:
        - Binary sensors: Keep as 0/1, use mean
        - Continuous sensors: Normalize and use mean
        - Position sensors: Use distance from origin or change detection
        
        Returns:
            1D time series for segmentation
        """
        if len(sensor_cols) == 0:
            raise ValueError(f"No sensor columns found for station {station}")
        
        # Classify sensors
        sensor_types = {col: self.classify_sensor_type(df, col) for col in sensor_cols}
        
        # Prepare normalized features
        normalized_features = []
        feature_weights = []
        for col in sensor_cols:
            if df[col].isna().all():
                continue
            
            sensor_type = sensor_types[col]
            values = df[col].fillna(0).values
            
            if sensor_type == 'binary':
                # Binary sensors: keep as 0/1
                stable_binary = (
                    pd.Series(values)
                    .rolling(3, min_periods=1)
                    .mean()
                    .fillna(0)
                    .values
                )
                normalized_features.append(stable_binary)
                feature_weights.append(1.0)


            elif sensor_type == 'continuous':
                # Continuous sensors: normalize to [0, 1]
                '''if values.max() > values.min():
                    normalized = (values - values.min()) / (values.max() - values.min())
                else:
                    normalized = values'''
                rng = values.max() - values.min()
                normalized = (values - values.min()) / rng if rng > 0 else np.zeros_like(values)
                normalized_features.append(normalized)
                feature_weights.append(1.0)


            elif sensor_type == 'position':
                # Position sensors: use absolute value or distance
                # For simplicity, use absolute value normalized
                motion_feature = self.build_position_features(values)
                '''if abs_values.max() > abs_values.min():
                    normalized = (abs_values - abs_values.min()) / (abs_values.max() - abs_values.min())
                else:
                    normalized = abs_values'''
                normalized_features.append(motion_feature)
                feature_weights.append(1.0)

        if len(normalized_features) == 0:
            # Fallback: use all numeric columns
            numeric_cols = df[sensor_cols].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    values = df[col].fillna(0).values
                    if values.max() > values.min():
                        normalized = (values - values.min()) / (values.max() - values.min())
                    else:
                        normalized = values
                    normalized_features.append(normalized)
            else:
                raise ValueError(f"No valid sensor data for station {station}")
        
        if len(normalized_features) == 0:
            raise ValueError(f"No valid normalized features for station {station}")
        
        # Aggregate: mean of normalized features
        aggregated = np.average(normalized_features, axis=0, weights=feature_weights)
        #aggregated = np.mean(normalized_features, axis=0)

        # Ensure no NaN or Inf values
        #aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=1.0, neginf=0.0)
        aggregated = pd.Series(aggregated, index=df.index)
        aggregated = (
            aggregated
            .rolling(3, min_periods=1)
            .mean()
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )


        


        return aggregated#pd.Series(aggregated, index=df.index)

    def engineer_features(self, df: pd.DataFrame, station: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete feature engineering for a station.
        
        Returns:
            (processed_df, aggregated_signal)
        """
        df = df.copy()
        
        # Get station-specific sensors
        sensor_cols = self.get_station_sensors(df, station)
        print(f"   Found {len(sensor_cols)} sensor columns for {station}")
        
        # Ensure all sensors are numeric
        for col in sensor_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)  # Fill missing with 0
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create aggregated 1D signal
        aggregated_signal = self.aggregate_signals(df, sensor_cols, station)
        
        return df, aggregated_signal