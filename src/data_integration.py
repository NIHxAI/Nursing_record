"""
data_integration.py
Data Integration and Drug-Symptom Pairing
- Consolidates NER results by admission
- Creates drug-symptom pairs for causality assessment
"""

import os
import json
import glob
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
NER_RESULTS_DIR = 'ner_results'
OUTPUT_DIR = 'integrated_data'

# Time window parameters
PREFERRED_LOOKBACK_HOURS = 72
MAX_LOOKBACK_HOURS = 168
MAX_MEDICATIONS_PER_SYMPTOM = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Helper Functions
# ==========================================
def safe_filename(s: str) -> str:
    """Create safe filename"""
    import re
    s = str(s)
    s = re.sub(r'[^0-9A-Za-z_\-\.]+', '_', s)
    return s.strip('_') or "NA"

def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse datetime string"""
    if not dt_str:
        return None
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def calculate_time_diff(med_time: datetime, symptom_time: datetime) -> str:
    """Calculate time difference between medication and symptom"""
    diff = symptom_time - med_time
    
    if diff.days > 0:
        hours = diff.days * 24 + diff.seconds // 3600
        return f"{hours} hours ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60
        if minutes > 0:
            return f"{hours} hours {minutes} minutes ago"
        else:
            return f"{hours} hours ago"
    else:
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"

# ==========================================
# Data Loading
# ==========================================
def load_ner_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load and group NER results by patient admission"""
    records = defaultdict(list)
    
    pattern = os.path.join(results_dir, "ptno_*__record_*__win_*.json")
    files = glob.glob(pattern)
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            meta = data.get('meta', {})
            ptno = meta.get('ptno', 'NA')
            record_no = meta.get('record_no', 'NA')
            
            key = f"{ptno}_{record_no}"
            records[key].append(data)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return records

# ==========================================
# Entity Extraction
# ==========================================
def extract_medications_and_symptoms(record_windows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Extract medications and symptoms from NER results"""
    medications = []
    symptoms = []
    
    for window_data in record_windows:
        model_output = window_data.get('model_output', {})
        drugs = model_output.get('drugs', [])
        sympts = model_output.get('symptoms', [])
        
        # Extract medications
        for drug in drugs:
            abs_time = drug.get('abs_time')
            if abs_time:
                med_dt = parse_datetime(abs_time)
                if med_dt:
                    medications.append({
                        'name': drug.get('name', ''),
                        'surface': drug.get('surface', ''),
                        'action': drug.get('action', 'unknown'),
                        'dose_text': drug.get('dose_text'),
                        'rate_text': drug.get('rate_text'),
                        'abs_time': abs_time,
                        'datetime': med_dt
                    })
        
        # Extract symptoms
        for symptom in sympts:
            abs_time = symptom.get('abs_time')
            if abs_time:
                symp_dt = parse_datetime(abs_time)
                if symp_dt:
                    symptoms.append({
                        'code': symptom.get('code', ''),
                        'surface': symptom.get('surface', ''),
                        'values': symptom.get('values', {}),
                        'abs_time': abs_time,
                        'datetime': symp_dt
                    })
    
    # Sort by datetime
    medications.sort(key=lambda x: x['datetime'])
    symptoms.sort(key=lambda x: x['datetime'])
    
    return medications, symptoms

# ==========================================
# Drug-Symptom Pairing
# ==========================================
def find_prior_medications(medications: List[Dict], symptom_time: datetime) -> Tuple[List[Dict], str]:
    """Find medications prior to symptom within time windows"""
    
    # Try preferred window (72 hours)
    preferred_cutoff = symptom_time - timedelta(hours=PREFERRED_LOOKBACK_HOURS)
    preferred_meds = [
        med for med in medications 
        if preferred_cutoff <= med['datetime'] < symptom_time
    ]
    preferred_meds.sort(key=lambda x: x['datetime'], reverse=True)
    
    if preferred_meds:
        return preferred_meds[:MAX_MEDICATIONS_PER_SYMPTOM], f"{PREFERRED_LOOKBACK_HOURS}h"
    
    # Extend to max window (168 hours)
    max_cutoff = symptom_time - timedelta(hours=MAX_LOOKBACK_HOURS)
    extended_meds = [
        med for med in medications 
        if max_cutoff <= med['datetime'] < symptom_time
    ]
    extended_meds.sort(key=lambda x: x['datetime'], reverse=True)
    
    if extended_meds:
        return extended_meds[:MAX_MEDICATIONS_PER_SYMPTOM], f"{MAX_LOOKBACK_HOURS}h"
    
    # Take any prior medications
    all_prior_meds = [
        med for med in medications 
        if med['datetime'] < symptom_time
    ]
    all_prior_meds.sort(key=lambda x: x['datetime'], reverse=True)
    
    if all_prior_meds:
        time_diff = symptom_time - all_prior_meds[0]['datetime']
        hours = int(time_diff.total_seconds() / 3600)
        return all_prior_meds[:MAX_MEDICATIONS_PER_SYMPTOM], f"{hours}h (all)"
    
    return [], "no medications"

def create_drug_symptom_pairs(ptno: str, record_no: str, medications: List[Dict], symptoms: List[Dict]) -> List[Dict]:
    """Create drug-symptom pairs for each symptom"""
    
    if not symptoms:
        return []
    
    pairs = []
    
    for symptom in symptoms:
        prior_meds, window = find_prior_medications(medications, symptom['datetime'])
        
        # Build medication data (always 3 slots)
        medication_data = {}
        for i in range(3):
            if i < len(prior_meds):
                med = prior_meds[i]
                medication_data[f"medication_{i+1}"] = {
                    "name": med['name'],
                    "time": med['abs_time'],
                    "time_diff": calculate_time_diff(med['datetime'], symptom['datetime']),
                    "action": med.get('action', 'unknown'),
                    "dose": med.get('dose_text', ''),
                    "rate": med.get('rate_text', ''),
                    "surface": med.get('surface', '') 
                }
            else:
                medication_data[f"medication_{i+1}"] = {
                    "name": "",
                    "time": "",
                    "time_diff": "",
                    "action": "",
                    "dose": "",
                    "rate": "",
                    "surface": "" 
                }
        
        # Create pair entry
        pair = {
            "patient_id": ptno,
            "record_no": record_no,
            "symptom": {
                "code": symptom['code'],
                "time": symptom['abs_time'],
                "surface": symptom['surface'],
                "values": symptom.get('values', {})
            },
            **medication_data,
            "search_window": window,
            "medications_found": len(prior_meds)
        }
        
        pairs.append(pair)
    
    return pairs

# ==========================================
# Main Processing
# ==========================================
def process_admission(ptno: str, record_no: str, windows: List[Dict]) -> Dict:
    """Process single admission record"""
    
    # Extract entities
    medications, symptoms = extract_medications_and_symptoms(windows)
    
    # Create drug-symptom pairs
    pairs = create_drug_symptom_pairs(ptno, record_no, medications, symptoms)
    
    # Build output data
    output = {
        "meta": {
            "patient_id": ptno,
            "record_no": record_no,
            "total_medications": len(medications),
            "total_symptoms": len(symptoms),
            "total_pairs": len(pairs)
        },
        "drug_symptom_pairs": pairs
    }
    
    return output

def main():
    """Main integration pipeline"""
    
    # Load NER results
    print("Loading NER results...")
    records = load_ner_results(NER_RESULTS_DIR)
    print(f"Found {len(records)} admission records")
    
    total_pairs = 0
    
    # Process each admission
    for record_key, windows in tqdm(records.items(), desc="Processing admissions"):
        ptno, record_no = record_key.split('_', 1)
        
        # Process admission
        result = process_admission(ptno, record_no, windows)
        
        # Save result
        filename = f"integrated_{safe_filename(ptno)}_{safe_filename(record_no)}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        total_pairs += result['meta']['total_pairs']
    
    print(f"\nIntegration completed:")
    print(f"- Processed admissions: {len(records)}")
    print(f"- Total drug-symptom pairs: {total_pairs}")
    print(f"- Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()