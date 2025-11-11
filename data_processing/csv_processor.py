
import pandas as pd
import numpy as np
from pathlib import Path

class CSVProcessor:
    """Process CSV files from oscilloscopes, simulators, etc."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.txt', '.dat']
    
    def load_waveform(self, file_path, time_col=None, value_cols=None):
        """Load waveform data from CSV file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not read CSV file with any encoding: {file_path}")
        
        # Auto-detect columns if not specified
        if time_col is None:
            time_col = self._detect_time_column(df)
        
        if value_cols is None:
            value_cols = self._detect_value_columns(df, time_col)
        
        # Extract data
        time_data = df[time_col].values
        value_data = {}
        
        for col in value_cols:
            value_data[col] = df[col].values
        
        metadata = {
            'file_path': str(file_path),
            'time_column': time_col,
            'value_columns': value_cols,
            'num_points': len(time_data),
            'sampling_rate': self._estimate_sampling_rate(time_data)
        }
        
        return time_data, value_data, metadata
    
    def _detect_time_column(self, df):
        """Auto-detect time column"""
        time_keywords = ['time', 't', 'x', 'sample', 'point']
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in time_keywords:
                if keyword in col_lower:
                    return col
        
        # Default to first column
        return df.columns[0]
    
    def _detect_value_columns(self, df, time_col):
        """Auto-detect value columns"""
        value_cols = []
        for col in df.columns:
            if col != time_col:
                value_cols.append(col)
        return value_cols
    
    def _estimate_sampling_rate(self, time_data):
        """Estimate sampling rate from time data"""
        if len(time_data) < 2:
            return 0
        
        time_diffs = np.diff(time_data)
        if np.all(time_diffs > 0):  # Increasing time
            avg_interval = np.mean(time_diffs)
            return 1.0 / avg_interval if avg_interval > 0 else 0
        return 0
    
    def validate_waveform(self, time_data, value_data):
        """Validate waveform data for consistency"""
        issues = []
        
        # Check time data
        if len(time_data) == 0:
            issues.append("Time data is empty")
        
        if len(time_data) > 1:
            time_diffs = np.diff(time_data)
            if np.any(time_diffs <= 0):
                issues.append("Time data is not strictly increasing")
        
        # Check value data
        for col_name, values in value_data.items():
            if len(values) != len(time_data):
                issues.append(f"Column {col_name} has different length than time data")
            
            if np.all(np.isnan(values)):
                issues.append(f"Column {col_name} contains only NaN values")
        
        return issues