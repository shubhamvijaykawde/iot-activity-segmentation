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
from IOTLoader import IoTDataLoader
from FeatureEngineer import FeatureEngineer
from Segmenter import Segmenter
from CBRExporter import CBRExporter


def visualize_segments_inline(df: pd.DataFrame, signal: pd.Series, segmentation_result: Dict,
                             station: str, show_plot: bool = True):
    """
    Visualize segmentation results with state changes overlaid - inline display.
    """
    if len(signal) == 0:
        print(f"  ️ Warning: Empty signal for {station}, skipping visualization")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    boundaries = segmentation_result['boundaries']
    state_changes = segmentation_result['state_changes']
    
    # Plot 1: Aggregated signal with boundaries
    ax1 = axes[0]
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(signal))
    
    ax1.plot(timestamps, signal.values, 'b-', alpha=0.7, label='Aggregated Signal', linewidth=1)
    
    # Draw segment boundaries
    if len(boundaries) > 2:
        for i, bp in enumerate(boundaries[1:-1]):
            if bp < len(timestamps):
                ax1.axvline(timestamps[bp], color='r', linestyle='--', linewidth=2, alpha=0.7, 
                           label='Segment Boundary' if i == 0 else '')
    
    # Draw state changes
    if len(state_changes) > 0:
        for i, sc in enumerate(state_changes):
            if sc < len(timestamps):
                ax1.axvline(timestamps[sc], color='g', linestyle=':', linewidth=1.5, alpha=0.5, 
                           label='State Change' if i == 0 else '')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Aggregated Signal')
    ax1.set_title(f'{station} - Segmentation Results (ClaSP)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis for timestamps
    if 'timestamp' in df.columns:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Segments with different colors
    ax2 = axes[1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(boundaries)-1))
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        if start < len(timestamps) and end <= len(timestamps):
            ax2.axvspan(timestamps[start], timestamps[end-1] if end < len(timestamps) else timestamps[-1], 
                       alpha=0.3, color=colors[i % len(colors)], label=f'Segment {i+1}' if i < 5 else '')
    
    ax2.plot(timestamps, signal.values, 'b-', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Aggregated Signal')
    ax2.set_title(f'{station} - Segments (colored by segment)')
    if len(boundaries) - 1 <= 5:
        ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    if 'timestamp' in df.columns:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Current state over time
    ax3 = axes[2]
    if 'current_state' in df.columns:
        # Encode states as numbers for plotting
        states = df['current_state'].values
        unique_states = pd.Series(states).unique()
        state_map = {state: idx for idx, state in enumerate(unique_states)}
        state_encoded = [state_map[s] for s in states]
        
        ax3.plot(timestamps, state_encoded, 'o-', markersize=3, alpha=0.6, label='Current State')
        ax3.set_yticks(range(len(unique_states)))
        ax3.set_yticklabels(unique_states)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('State')
        ax3.set_title(f'{station} - State Transitions')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        if 'timestamp' in df.columns:
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig

# ============================================================================
# COMPLETE PIPELINE WITH INLINE VISUALIZATION
# ============================================================================

def run_complete_pipeline(data_dir: str = "data/data-radiant-eval-paper/factory/evaluation/iot_logs", 
                         output_dir: str = "segmentation_output",
                         stations_to_process: List[str] = None,
                         visualize_inline: bool = True):
    """
    Run complete segmentation pipeline with inline visualization.
    
    Args:
        data_dir: Directory containing JSONL files
        output_dir: Directory for output files
        stations_to_process: List of stations to process (None = all)
        visualize_inline: Whether to show visualizations inline
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # TASK 1: Data Loading
    print("=" * 60)
    print("TASK 1: DATA LOADING & EXPLORATION")
    print("=" * 60)
    
    loader = IoTDataLoader(data_dir)
    df = loader.process_all()
    
    # TASK 2: Feature Engineering
    print("\n" + "=" * 60)
    print("TASK 2: FEATURE ENGINEERING & STANDARDIZATION")
    print("=" * 60)
    
    engineer = FeatureEngineer()
    station_signals = {}
    station_dfs = {}
    
    stations = stations_to_process if stations_to_process else list(loader.station_data.keys())
    
    for station in stations:
        print(f"\n Processing {station}...")
        station_df = loader.station_data[station].copy()
        
        try:
            processed_df, aggregated_signal = engineer.engineer_features(station_df, station)
            station_signals[station] = aggregated_signal
            station_dfs[station] = processed_df
            print(f"   Successfully engineered features for {station}")
        except Exception as e:
            print(f"   Error processing {station}: {e}")
            continue
    
    # TASK 3: Segmentation
    print("\n" + "=" * 60)
    print("TASK 3: SEGMENTATION IMPLEMENTATION")
    print("=" * 60)
    
    segmenter = Segmenter()
    segmentation_results = {}
    station_visualizations = {}
    
    for station in stations:
        if station not in station_signals:
            continue
        
        try:
            print(f"\n Segmenting {station}...")
            result = segmenter.segment_station(
                station_dfs[station],
                station_signals[station],
                station,
                use_baseline=False
            )
            segmentation_results[station] = result
            
            # Print metrics
            metrics = result['metrics']
            '''print(f"\n Evaluation metrics for {station}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1 Score: {metrics['f1']:.3f}")
            print(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
            print(f"    Predicted boundaries: {metrics.get('n_predicted', 'N/A')}, True boundaries: {metrics.get('n_true', 'N/A')}")'''
            for k, v in metrics.items():
              print(f"{k}: {v:.3f}")

            # Create inline visualization
            if visualize_inline:
                print(f"\n Generating inline visualization for {station}...")
                fig = visualize_segments_inline(
                    station_dfs[station],
                    station_signals[station],
                    result,
                    station,
                    show_plot=True
                )
                station_visualizations[station] = fig
            
        except Exception as e:
            print(f"   Error segmenting {station}: {e}")
            continue
    
    # TASK 4: Export
    print("\n" + "=" * 60)
    print("TASK 4: PIPELINE INTEGRATION & EXPORT")
    print("=" * 60)
    
    all_segments = {}
    for station in stations:
        if station in segmentation_results and station in station_signals:
            try:
                segments = CBRExporter.export_segments(
                    segmentation_results[station],
                    station_dfs[station],
                    station_signals[station],
                    output_file=str(output_path / f"{station}_segments.json")
                )
                all_segments[station] = segments
                print(f"   Exported {len(segments)} segments for {station}")
                
                # Show sample segment
                if len(segments) > 0:
                    print(f"   Sample segment for {station}:")
                    sample_segment = segments[0]
                    print(f"    Segment ID: {sample_segment['segment_id']}")
                    print(f"    Length: {sample_segment['length']} samples")
                    print(f"    Dominant State: {sample_segment['dominant_state']}")
                    print(f"    Features: {len(sample_segment['features'])} sensor statistics")
                
            except Exception as e:
                print(f"   Error exporting {station}: {e}")
    
    # Summary export
    summary = {
        'total_stations': len(all_segments),
        'stations': list(all_segments.keys()),
        'total_segments': sum(len(segs) for segs in all_segments.values()),
        'segments_per_station': {station: len(segs) for station, segs in all_segments.items()},
        'evaluation_metrics': {
            station: result['metrics'] 
            for station, result in segmentation_results.items()
        }
    }
    
    with open(output_path / "segmentation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print(" PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f" Processed {len(all_segments)} stations")
    print(f" Total segments: {summary['total_segments']}")
    print(f" Output directory: {output_path}")
    
    # Display final summary
    print(f"\n FINAL SUMMARY:")
    print("-" * 40)
    for station, segs in all_segments.items():
        metrics = summary['evaluation_metrics'].get(station, {})
        print(f"{station}: {len(segs)} segments | separation_score: {metrics.get('separation_score', 0):.3f} | "
              f"stability_score: {metrics.get('stability_score', 0):.3f} | boundary_strength: {metrics.get('boundary_strength', 0):.3f} | "
              f"explained_variance: {metrics.get('explained_variance', 0):.3f}")

    return {
        'loader': loader,
        'engineer': engineer,
        'segmenter': segmenter,
        'segmentation_results': segmentation_results,
        'all_segments': all_segments,
        'summary': summary,
        'visualizations': station_visualizations
    }

if __name__ == "__main__":
    print("Starting IoT Segmentation Pipeline...")
    print("=" * 60)
    
    try:
        # Run the complete pipeline
        results = run_complete_pipeline(
            data_dir="data/data-radiant-eval-paper/factory/evaluation/iot_logs",
            output_dir="segmentation_output",
            stations_to_process=None  # Process all stations
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS: Pipeline completed successfully!")
        print("=" * 60)
        print("\nOutput files are in the 'segmentation_output' directory:")
        print("  - segmentation_summary.json: Overall summary")
        print("  - {station}_segments.json: CBR-ready segments per station")
        print("  - visualizations/{station}_segmentation.png: Visualizations")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()