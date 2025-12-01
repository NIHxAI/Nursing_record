"""
dataset_builder.py
Build instruction tuning dataset from labeled causality data
- Creates dataset with proper format including original text
- Splits into train/validation/test sets
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = 'integrated_data'
LABELS_DIR = 'causality_labels'
OUTPUT_DIR = 'instruction_dataset'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Instruction Template
# ==========================================
INSTRUCTION = """You are an expert clinical pharmacologist specializing in adverse drug reaction (ADR) causality assessment.
Analyze the temporal relationship between medications and the ADR symptom, then classify causality for EACH medication using WHO-UMC criteria.
**CAUSALITY CATEGORIES:**
- **Certain**: Definitive evidence with strong temporal relationship and no alternative explanations
- **Probable**: Reasonable temporal sequence with known mechanism and no equally likely alternatives
- **Possible**: Reasonable temporal relationship with some plausibility but alternatives possible
- **Unlikely**: Poor temporal relationship or clear alternative explanations
- **Conditional**: Assessment pending additional data
- **Unassessable**: Insufficient information
**CRITICAL INSTRUCTIONS:**
1. Assess ALL medications listed in the input
2. Return exactly the same number of assessments as medications provided
3. Maintain the order of medications as given in the input
**OUTPUT FORMAT:**
Return a JSON array with one object per medication:
[
  {
    "medication_name": "drug name",
    "causality_category": "Certain|Probable|Possible|Unlikely|Conditional|Unassessable"
  }
]"""

# ==========================================
# Helper Functions
# ==========================================
def load_json(path: str) -> Dict:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path: str, data: Any):
    """Save JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_medication_block(med_data: Dict, index: int) -> str:
    """Build medication text block"""
    # Build action details
    action_parts = []
    if med_data.get('action') and med_data.get('action') != 'unknown':
        action_parts.append(f"Action: {med_data['action']}")
    if med_data.get('dose'):
        action_parts.append(f"Dose: {med_data['dose']}")
    if med_data.get('rate'):
        action_parts.append(f"Rate: {med_data['rate']}")
    
    action_str = ", ".join(action_parts) if action_parts else "No additional details"
    
    return f"""**Medication {index}:**
- Name: {med_data.get('name', 'Unknown')}
- Timestamp: {med_data.get('time', 'Unknown')}
- Time Difference: {med_data.get('time_diff', 'Unknown')}
- Action: {action_str}
- Original Text: {med_data.get('surface', '')}"""

def create_input_text(pair_data: Dict, patient_id: str, record_suffix: str) -> str:
    """Create input text from drug-symptom pair"""
    
    # Symptom info
    symptom = pair_data.get('symptom', {})
    symptom_code = symptom.get('code', '')
    symptom_time = symptom.get('time', '')
    symptom_surface = symptom.get('surface', '')
    symptom_values = symptom.get('values', {})
    
    # Collect medications
    med_blocks = []
    num_medications = 0
    
    for i in range(1, 4):
        med_key = f"medication_{i}"
        if med_key in pair_data:
            med = pair_data[med_key]
            if med.get('name'):
                num_medications += 1
                med_blocks.append(build_medication_block(med, num_medications))
    
    # Build input text - Onset Time을 Timestamp로 변경
    input_text = f"""**Patient ID:** {patient_id} | **Record:** {record_suffix}
**ADR Symptom Event:**
- Symptom Code: {symptom_code}
- Timestamp: {symptom_time}
- Clinical Description: {symptom_surface}
- Measured Values: {json.dumps(symptom_values, ensure_ascii=False)}
**Number of Medications to Assess:** {num_medications}
{chr(10).join(med_blocks)}"""
    
    return input_text

def create_output_json(label_data: Dict) -> str:
    """Create output JSON from label"""
    
    assessments = label_data.get('gpt_response', {}).get('medication_assessments', [])
    
    output_array = [
        {
            "medication_name": assess.get('medication_name', 'Unknown'),
            "causality_category": assess.get('causality_category', 'Unassessable')
        }
        for assess in assessments
    ]
    
    return json.dumps(output_array, ensure_ascii=False, indent=2)

def extract_patient_record_info(metadata: Dict) -> tuple:
    """Extract patient ID and record suffix from metadata"""
    # From metadata like patient_id: "ADG877_2023AA_ADG877"
    patient_full = metadata.get('patient_id', 'Unknown')
    record_no = metadata.get('record_no', 'Unknown')
    
    # Extract base patient ID (first part before underscore)
    if '_' in str(patient_full):
        patient_id = str(patient_full).split('_')[0]
    else:
        patient_id = str(patient_full)
    
    # Extract record suffix (simplify record number)
    # If record_no is like "2023AA_ADG877_002", we want "002"
    if '_' in str(record_no):
        parts = str(record_no).split('_')
        record_suffix = parts[-1] if parts[-1].isdigit() else record_no
    else:
        record_suffix = str(record_no)
    
    return patient_id, record_suffix

# ==========================================
# Main Processing
# ==========================================
def process_labeled_file(label_file: str) -> Dict:
    """Process single labeled file"""
    
    # Load label data
    label_data = load_json(label_file)
    
    # Load source data
    source_file = label_data.get('source_file', '')
    if not source_file or not Path(source_file).exists():
        return None
    
    source_data = load_json(source_file)
    
    # Extract patient and record info
    metadata = source_data.get('metadata', {})
    patient_id, record_suffix = extract_patient_record_info(metadata)
    
    # Get raw data
    raw_data = source_data.get('raw_data', {})
    symptom = raw_data.get('symptom', {})
    medications = raw_data.get('medications', [])
    
    # Build pair data structure
    pair_data = {
        'symptom': symptom
    }
    
    # Add medications with all their fields
    for i, med in enumerate(medications, 1):
        # Ensure we have all necessary fields including surface text
        med_complete = {
            'name': med.get('name', 'Unknown'),
            'time': med.get('time', 'Unknown'),
            'time_diff': med.get('time_diff', 'Unknown'),
            'action': med.get('action', 'unknown'),
            'dose': med.get('dose', ''),
            'rate': med.get('rate', ''),
            'surface': med.get('surface', '')  # Original text
        }
        pair_data[f'medication_{i}'] = med_complete
    
    # Create training example
    example = {
        "instruction": INSTRUCTION,
        "input": create_input_text(pair_data, patient_id, record_suffix),
        "output": create_output_json(label_data)
    }
    
    return example

def main():
    """Main dataset building pipeline"""
    
    print("=" * 60)
    print("BUILDING INSTRUCTION TUNING DATASET")
    print("=" * 60)
    
    # Find all labeled files
    label_files = sorted(Path(LABELS_DIR).glob("*_labeled.json"))
    
    if not label_files:
        print(f"No labeled files found in {LABELS_DIR}")
        return
    
    print(f"\nFound {len(label_files)} labeled files")
    
    # Process each file
    dataset = []
    
    for label_file in label_files:
        example = process_labeled_file(str(label_file))
        if example:
            dataset.append(example)
    
    if not dataset:
        print("No valid examples created")
        return
    
    # Split into train/validation/test (70/15/15)
    total = len(dataset)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]
    
    # Save datasets in JSON format
    train_file = os.path.join(OUTPUT_DIR, 'train.json')
    val_file = os.path.join(OUTPUT_DIR, 'validation.json')
    test_file = os.path.join(OUTPUT_DIR, 'test.json')
    
    save_json(train_file, train_data)
    save_json(val_file, val_data)
    save_json(test_file, test_data)
    
    # Also save in JSONL format for fine-tuning
    train_jsonl = os.path.join(OUTPUT_DIR, 'train.jsonl')
    val_jsonl = os.path.join(OUTPUT_DIR, 'validation.jsonl')
    test_jsonl = os.path.join(OUTPUT_DIR, 'test.jsonl')
    
    # Write JSONL files
    for jsonl_file, data in [(train_jsonl, train_data), 
                             (val_jsonl, val_data), 
                             (test_jsonl, test_data)]:
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": example['instruction']},
                        {"role": "user", "content": example['input']},
                        {"role": "assistant", "content": example['output']}
                    ]
                }, ensure_ascii=False) + "\n")
    
    print(f"\nDataset created successfully!")
    print(f"  Training examples: {len(train_data)}")
    print(f"  Validation examples: {len(val_data)}")
    print(f"  Test examples: {len(test_data)}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    
    # Show sample
    if train_data:
        print("\n" + "=" * 60)
        print("Sample from training dataset:")
        print("=" * 60)
        print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()