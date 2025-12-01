"""
prepare_causality_data.py
Prepare data for causality labeling by creating structured prompts
- Transforms integrated drug-symptom pairs into prompts for OpenAI API
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, Optional, List

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = 'integrated_data'
OUTPUT_DIR = 'prepared_prompts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Enhanced Prompts with Clear Scoring
# ==========================================
SYSTEM_PROMPT = """You are an expert clinical pharmacologist specializing in adverse drug reaction (ADR) causality assessment.

Your task is to analyze clinical data and provide a detailed causality assessment for each medication. Focus on the causality category, temporal relationship, and supporting evidence.

Base your judgments on established criteria similar to the Naranjo Scale and WHO-UMC causality categories.

IMPORTANT SCORING SYSTEM:
- Certain = 5 points (highest confidence in causal relationship)
- Probable = 4 points (high likelihood of causal relationship)  
- Possible = 3 points (moderate likelihood, but alternatives exist)
- Unlikely = 2 points (low likelihood, better alternatives exist)
- Conditional = 1 point (assessment pending additional data)
- Unassessable = 0 points (insufficient information for assessment)

The score should directly correspond to the category selected."""

USER_TEMPLATE = """**CLINICAL CASE FOR ADR CAUSALITY ASSESSMENT**

**Patient Information:**
{patient_info}

**Clinical Context:**
{clinical_context}

**ADR Symptom Event:**
- Symptom Code: {symptom_code}
- Onset Time: {symptom_time}
- Clinical Description: {symptom_surface}
- Measured Values: {symptom_values}

**Number of Medications to Assess: {num_medications}**

{medications_section}

**CAUSALITY ASSESSMENT CRITERIA AND SCORING:**

1. **Certain (Score: 5)**
   - Definitive evidence with strong temporal relationship
   - No alternative explanations
   - Positive rechallenge/dechallenge if applicable
   
2. **Probable (Score: 4)**
   - Reasonable temporal sequence
   - Known pharmacological mechanism
   - No equally likely alternatives
   
3. **Possible (Score: 3)**
   - Reasonable temporal relationship
   - Pharmacologically plausible
   - Alternative explanations also possible
   
4. **Unlikely (Score: 2)**
   - Poor temporal relationship OR
   - Clear alternative explanations more likely OR
   - Pharmacologically implausible
   
5. **Conditional (Score: 1)**
   - More data needed for assessment
   - Critical information missing but potentially available
   
6. **Unassessable (Score: 0)**
   - Insufficient information for any assessment
   - Critical data permanently unavailable

**ASSESSMENT TASK:**
For EACH medication listed above (total: {num_medications}), provide:
1. Causality category from the 6 options above
2. Causality score (0-5) matching the category
3. Temporal relationship assessment
4. Supporting evidence (2-4 key clinical observations)

**REQUIRED JSON OUTPUT FORMAT:**
```json
{output_json_format}
```

**CRITICAL INSTRUCTIONS:**
1. Assess exactly {num_medications} medication(s)
2. Each medication must have a score that matches its category (Certain=5, Probable=4, Possible=3, Unlikely=2, Conditional=1, Unassessable=0)
3. Do not include medications not listed in the case
4. Keep supporting_evidence as an array of 2-4 concise statements"""

MEDICATION_SECTION_TEMPLATE = """**Medication {index}:**
- Name: {name}
- Timestamp: {time}
- Time Relationship: {time_diff}
- Clinical Details: {details}
- Original Text: "{surface}" """

ASSESSMENT_TEMPLATE = """    {{
      "medication_index": {index},
      "medication_name": "{name}",
      "causality_category": "Certain|Probable|Possible|Unlikely|Conditional|Unassessable",
      "causality_score": 0,
      "temporal_plausibility": {{
        "relationship": "appropriate|too_early|too_late|unclear",
        "onset_time_hours": "calculated_value",
        "expected_range_hours": "based_on_drug_profile"
      }},
      "supporting_evidence": [
        "Evidence point 1",
        "Evidence point 2"
      ]
    }}"""

# ==========================================
# Helper Functions
# ==========================================
def safe_filename(s: str) -> str:
    """Create safe filename from string"""
    import re
    s = str(s) if s else ""
    s = re.sub(r'[^0-9A-Za-z_\-\.]+', '_', s)
    return s.strip('_') or "NA"

def load_json(path: str) -> Optional[Dict]:
    """Load JSON file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def save_json(path: str, data: Dict):
    """Save JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")

# ==========================================
# Data Preparation Functions
# ==========================================
def prepare_single_case(pair_data: Dict, search_window: str = "") -> Optional[Dict]:
    """Prepare prompt for a single drug-symptom pair"""
    
    # Extract patient information
    patient_id = pair_data.get('patient_id', 'Unknown')
    record_no = pair_data.get('record_no', 'Unknown')
    
    # Build patient info and clinical context
    patient_info = f"Patient ID: {patient_id}, Record: {record_no}"
    
    if search_window:
        clinical_context = f"Nursing record-based ADR assessment - Search window: {search_window}"
    else:
        clinical_context = f"Nursing record-based ADR assessment - Search window: {pair_data.get('search_window', 'Not specified')}"
    
    # Extract symptom information
    symptom = pair_data.get('symptom', {})
    symptom_code = symptom.get('code', '')
    symptom_time = symptom.get('time', '')
    symptom_surface = symptom.get('surface', '')
    symptom_values = symptom.get('values', {})
    
    # Build medications section and assessment templates
    medications_text = ""
    assessment_templates = []
    valid_medications = []
    
    for i in range(1, 4):
        med_key = f"medication_{i}"
        if med_key in pair_data:
            med = pair_data[med_key]
            if med.get('name'):  # Only include if medication has a name
                valid_medications.append(med)
                med_index = len(valid_medications)
                
                # Build clinical details
                details_parts = []
                if med.get('action') and med.get('action') != 'unknown':
                    details_parts.append(f"Action: {med['action']}")
                if med.get('dose'):
                    details_parts.append(f"Dose: {med['dose']}")
                if med.get('rate'):
                    details_parts.append(f"Rate: {med['rate']}")
                
                details = ", ".join(details_parts) if details_parts else "No additional details"
                
                # Add medication section
                medications_text += MEDICATION_SECTION_TEMPLATE.format(
                    index=med_index,
                    name=med.get('name', 'Unknown'),
                    time=med.get('time', 'Unknown'),
                    time_diff=med.get('time_diff', 'Unknown'),
                    details=details,
                    surface=med.get('surface', '') if 'surface' in med else ""
                )
                
                if med_index < 3:  # Add newline except for last medication
                    medications_text += "\n"
                
                # Add assessment template
                assessment_templates.append(
                    ASSESSMENT_TEMPLATE.format(
                        index=med_index,
                        name=med.get('name', 'Unknown')
                    )
                )
    
    # Skip if no valid medications
    if len(valid_medications) == 0:
        return None
    
    # Generate case ID
    case_id = f"{patient_id}_{record_no}_{safe_filename(symptom_code)}"
    
    # Format symptom values
    symptom_values_str = json.dumps(symptom_values, ensure_ascii=False) if symptom_values else "{}"
    
    # Build assessment templates string
    assessments_str = ",\n".join(assessment_templates)
    
    # Build output JSON format
    output_json = f'''{{
  "case_id": "{case_id}",
  "assessment_date": "{get_current_date()}",
  "symptom_code": "{symptom_code}",
  "total_medications_assessed": {len(valid_medications)},
  "medication_assessments": [
{assessments_str}
  ]
}}'''
    
    # Build user prompt
    user_prompt = USER_TEMPLATE.format(
        patient_info=patient_info,
        clinical_context=clinical_context,
        symptom_code=symptom_code,
        symptom_time=symptom_time,
        symptom_surface=symptom_surface,
        symptom_values=symptom_values_str,
        num_medications=len(valid_medications),
        medications_section=medications_text,
        output_json_format=output_json
    )
    
    return {
        "prompt": {
            "system": SYSTEM_PROMPT,
            "user": user_prompt
        },
        "metadata": {
            "case_id": case_id,
            "patient_id": patient_id,
            "record_no": record_no,
            "symptom_code": symptom_code,
            "symptom_time": symptom_time,
            "num_medications": len(valid_medications),
            "medications": [med.get('name') for med in valid_medications],
            "assessment_date": get_current_date()
        },
        "raw_data": {
            "symptom": symptom,
            "medications": valid_medications
        }
    }

def process_integrated_file(filepath: str) -> List[Dict]:
    """Process integrated data file and prepare all cases"""
    
    data = load_json(filepath)
    if not data:
        return []
    
    prepared_cases = []
    pairs = data.get('drug_symptom_pairs', [])
    
    for pair_idx, pair in enumerate(pairs):
        case_data = prepare_single_case(pair)
        
        if case_data:
            case_data['source_file'] = os.path.basename(filepath)
            case_data['pair_index'] = pair_idx
            prepared_cases.append(case_data)
    
    return prepared_cases

# ==========================================
# Main Processing Pipeline
# ==========================================
def main():
    """Main data preparation pipeline"""
    
    print("=" * 60)
    print("PREPARE DATA FOR CAUSALITY LABELING")
    print("=" * 60)
    
    # Find all integrated data files
    pattern = os.path.join(INPUT_DIR, "integrated_*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No integrated data files found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(files)} integrated data files")
    
    # Process each file
    total_cases = 0
    
    for file_idx, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        print(f"\n[{file_idx}/{len(files)}] Processing {filename}")
        
        # Prepare cases from file
        cases = process_integrated_file(filepath)
        
        if not cases:
            print(f"  No valid cases found")
            continue
        
        # Save each case as separate file
        for case in cases:
            # Generate output filename
            case_id = case['metadata']['case_id']
            pair_index = case['pair_index']
            output_file = f"{case_id}_{pair_index:03d}_prepared.json"
            output_path = os.path.join(OUTPUT_DIR, output_file)
            
            # Save prepared data
            save_json(output_path, case)
            total_cases += 1
        
        print(f"  Prepared {len(cases)} cases")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal cases prepared: {total_cases}")
    print(f"Output directory: {OUTPUT_DIR}/")
    

if __name__ == "__main__":
    main()