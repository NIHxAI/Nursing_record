"""
data_preprocessing.py
Nursing Record Data Preprocessing Pipeline
- Medical staff de-identification  
- Temporal segmentation (12-hour windows)
"""

import pandas as pd
import re

# ==========================================
# Configuration
# ==========================================
INPUT_FILE = 'nursing_records_raw.csv'
OUTPUT_FILE = 'nursing_records_preprocessed.csv'
WINDOW_HOURS = 12

# ==========================================
# 1. Medical Staff De-identification
# ==========================================
def deidentify_medical_staff(df):
    """
    Replace medical staff names with role placeholders
    """
    # Intern
    df.loc[
        df['text'].str.contains(r'(?i)\b(intern|int)\b', na=False) &
        ~df['text'].str.contains(r'(?i)\binternal\b', na=False),
        'text'
    ] = df['text'].str.replace(
        r'(?i)\b(intern|int)\b[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?',
        ' 인턴 OOO',
        regex=True
    )
    
    # Korean intern
    df['text'] = df['text'].str.replace(
        r'(?i)\b(?:\w*인턴)[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?', 
        ' 인턴 OOO', 
        regex=True
    )
    
    # Attending physician
    df['text'] = df['text'].str.replace(
        r'(?i)\b(?:\w*주치의)[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?', 
        ' 주치의 OOO', 
        regex=True
    )
    
    # Professor
    df['text'] = df['text'].str.replace(
        r'(?i)\b(professor|prof|pf)[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?',
        ' prof OOO', 
        regex=True
    )
    
    # Korean professor
    df['text'] = df['text'].str.replace(
        r'(?i)\b(?:\w*교수님|교수)[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?', 
        ' OOO 교수님', 
        regex=True
    )
    
    # Doctor
    df.loc[
        df['text'].str.contains(r'(?i)\bdr\b', na=False) &
        ~df['text'].str.contains(r'(?i)\bdressing\b', na=False),
        'text'
    ] = df['text'].str.replace(
        r'(?i)\bdr\b[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?',
        ' dr OOO',
        regex=True
    )
    
    # On-call doctor
    df['text'] = df['text'].str.replace(
        r'(?i)\b(?:\w*당직의|당직)[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?', 
        ' 당직의 OOO', 
        regex=True
    )
    
    # Residents (R1-R4)
    df['text'] = df['text'].str.replace(
        r'(?i)\br[1-4]\.?\s*[\s,\.]*(\(\s*[^)]+\s*\)|\w+)?',
        'R OOO',
        regex=True
    )
    
    # Other roles
    df['text'] = df['text'].str.replace(r'\bCI\s+(\w+)', r'CI OOO', regex=True)
    df['text'] = df['text'].str.replace(r'\bstaff\s+(\w+)', r'staff OOO', regex=True)
    df['text'] = df['text'].str.replace(r'\bPA\s+(\w+)', r'PA OOO', regex=True)
    
    return df

# ==========================================
# 2. Temporal Segmentation
# ==========================================
def create_temporal_windows(df, window_hours=12):
    """
    Create temporal windows for nursing records
    """
    # Ensure record_no is string type
    df['record_no'] = df['record_no'].astype(str)
    
    # Parse datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['time'] = df['time'].astype(str).str.strip().replace({'': '00:00'})
    df['dt'] = pd.to_datetime(
        df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['time'], 
        errors='coerce'
    )
    
    # Keep only valid datetime rows
    df = df[df['dt'].notna()].copy()
    
    # Calculate anchor time (first timestamp) for each admission
    anchor = df.groupby('record_no')['dt'].transform('min')
    
    # Calculate window index
    df['window_idx'] = (
        (df['dt'] - anchor).dt.total_seconds() // (window_hours * 3600)
    ).astype('int64')
    
    # Create window ID
    df['window_id_12h'] = (
        df['record_no'] + '_W' + 
        (df['window_idx'] + 1).astype(str).str.zfill(3)
    )
    
    return df

# ==========================================
# 3. Main Pipeline
# ==========================================
def main():
    """
    Main preprocessing pipeline
    """
    # Load data
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    
    # De-identify medical staff
    df = deidentify_medical_staff(df)
    
    # Create temporal windows
    df = create_temporal_windows(df, window_hours=WINDOW_HOURS)
    
    # Save processed data
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()