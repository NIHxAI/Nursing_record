"""
causality_labeling.py
Perform causality labeling using OpenAI API
- Reads prepared prompts and generates WHO-UMC causality labels
- Creates instruction tuning dataset
"""

import os
import json
import glob
import time
import re
from datetime import datetime
from typing import Dict, Optional, List
from openai import OpenAI

# ==========================================
# Configuration
# ==========================================
API_KEY = 'your-api-key'
MODEL_NAME = 'gpt-4.1'

INPUT_DIR = 'prepared_prompts'
LABELS_DIR = 'causality_labels'
DATASET_DIR = 'instruction_dataset'

os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Rate limiting
BATCH_SIZE = 10
DELAY_BETWEEN_CALLS = 1
DELAY_BETWEEN_BATCHES = 5

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# ==========================================
# Helper Functions
# ==========================================
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

def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().isoformat()

def extract_json_from_response(content: str) -> Optional[Dict]:
    """Extract JSON from GPT response"""
    # Try to find JSON in the response
    json_match = re.search(r'\{[\s\S]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try removing code blocks
    content = content.strip()
    if content.startswith('```'):
        content = re.sub(r'^```(?:json)?', '', content, re.IGNORECASE).strip()
        content = re.sub(r'```$', '', content).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    
    # Direct parse attempt
    try:
        return json.loads(content)
    except:
        return None

# ==========================================
# OpenAI API Functions
# ==========================================
def call_openai_for_labeling(system_msg: str, user_msg: str, retries: int = 3) -> Optional[Dict]:
    """Call OpenAI API with retry logic"""
    
    for attempt in range(retries):
        try:
            print(f"    Calling {MODEL_NAME} (attempt {attempt + 1})...", end="")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content or ""
            result = extract_json_from_response(content)
            
            if result:
                print(" ✓")
                return result
            else:
                print(" ✗ (no valid JSON)")
                if attempt < retries - 1:
                    time.sleep(1.5 * (2 ** attempt))
                    
        except Exception as e:
            print(f" ✗ ({str(e)[:50]})")
            if attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))
    
    return None

def validate_response(response: Dict, expected_med_count: int) -> bool:
    """Validate and fix GPT response"""
    
    if 'medication_assessments' not in response:
        return False
    
    assessments = response['medication_assessments']
    if not isinstance(assessments, list):
        return False
    
    # Check medication count
    if len(assessments) != expected_med_count:
        print(f"    Warning: Expected {expected_med_count} medications, got {len(assessments)}")
        return False
    
    # Fix scores to match categories
    category_to_score = {
        "Certain": 5,
        "Probable": 4,
        "Possible": 3,
        "Unlikely": 2,
        "Conditional": 1,
        "Unassessable": 0
    }
    
    for assessment in assessments:
        category = assessment.get('causality_category', '')
        if category in category_to_score:
            expected_score = category_to_score[category]
            actual_score = assessment.get('causality_score', -1)
            if actual_score != expected_score:
                assessment['causality_score'] = expected_score
    
    return True

# ==========================================
# Processing Functions
# ==========================================
def process_prepared_file(filepath: str) -> Optional[Dict]:
    """Process a single prepared prompt file"""
    
    # Load prepared prompt
    prompt_data = load_json(filepath)
    if not prompt_data:
        return None
    
    prompt = prompt_data.get('prompt', {})
    metadata = prompt_data.get('metadata', {})
    
    system_prompt = prompt.get('system', '')
    user_prompt = prompt.get('user', '')
    
    if not system_prompt or not user_prompt:
        print(f"    Error: Missing prompts")
        return None
    
    # Call GPT
    gpt_response = call_openai_for_labeling(system_prompt, user_prompt)
    if not gpt_response:
        print(f"    Failed to get valid response")
        return None
    
    # Validate response
    expected_meds = metadata.get('num_medications', 0)
    is_valid = validate_response(gpt_response, expected_meds)
    
    # Prepare output
    output = {
        "generated_at": get_timestamp(),
        "source_file": filepath,
        "metadata": metadata,
        "gpt_response": gpt_response,
        "model": MODEL_NAME,
        "validation_passed": is_valid
    }
    
    return output

def create_training_dataset(label_files: List[str]) -> Dict:
    """Create instruction tuning dataset from labels"""
    
    training_examples = []
    validation_examples = []
    
    # 90/10 train/validation split
    split_index = int(len(label_files) * 0.9)
    
    for idx, label_file in enumerate(label_files):
        # Load labeled data
        labeled_data = load_json(label_file)
        if not labeled_data or not labeled_data.get('validation_passed'):
            continue
        
        # Load original prompt
        source_file = labeled_data.get('source_file', '')
        prompt_data = load_json(source_file) if source_file else None
        if not prompt_data:
            continue
        
        # Create training example
        prompt = prompt_data.get('prompt', {})
        example = {
            "instruction": prompt.get('system', ''),
            "input": prompt.get('user', ''),
            "output": json.dumps(labeled_data['gpt_response'], ensure_ascii=False),
            "metadata": labeled_data['metadata']
        }
        
        # Split into train/validation
        if idx < split_index:
            training_examples.append(example)
        else:
            validation_examples.append(example)
    
    return {
        "version": "1.0",
        "created_at": get_timestamp(),
        "model_used": MODEL_NAME,
        "train": training_examples,
        "validation": validation_examples,
        "statistics": {
            "total_examples": len(training_examples) + len(validation_examples),
            "train_examples": len(training_examples),
            "validation_examples": len(validation_examples)
        }
    }

# ==========================================
# Main Pipeline
# ==========================================
def main():
    """Main labeling and dataset creation pipeline"""
    
    print("=" * 60)
    print("CAUSALITY LABELING WITH OPENAI API")
    print("=" * 60)
    
    # Find all prepared files
    pattern = os.path.join(INPUT_DIR, "*_prepared.json")
    prepared_files = sorted(glob.glob(pattern))
    
    if not prepared_files:
        print(f"No prepared files found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(prepared_files)} prompt files to process")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Process statistics
    succeeded = 0
    failed = 0
    label_files = []
    category_stats = {}
    
    # Process in batches
    for batch_start in range(0, len(prepared_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(prepared_files))
        batch_files = prepared_files[batch_start:batch_end]
        
        print(f"\n[Batch {batch_start//BATCH_SIZE + 1}] Processing files {batch_start+1}-{batch_end}/{len(prepared_files)}")
        
        for idx, filepath in enumerate(batch_files):
            file_num = batch_start + idx + 1
            print(f"\n[{file_num}/{len(prepared_files)}] {os.path.basename(filepath)}")
            
            # Process file
            result = process_prepared_file(filepath)
            
            if result:
                # Save label
                base_name = os.path.basename(filepath).replace('_prepared.json', '')
                output_file = os.path.join(LABELS_DIR, f"{base_name}_labeled.json")
                save_json(output_file, result)
                label_files.append(output_file)
                
                print(f"    Saved: {os.path.basename(output_file)}")
                succeeded += 1
                
                # Update category statistics
                if 'gpt_response' in result and 'medication_assessments' in result['gpt_response']:
                    for assessment in result['gpt_response']['medication_assessments']:
                        category = assessment.get('causality_category', 'Unknown')
                        category_stats[category] = category_stats.get(category, 0) + 1
            else:
                failed += 1
                print(f"    Failed to process")
            
            # Rate limiting
            if idx < len(batch_files) - 1:
                time.sleep(DELAY_BETWEEN_CALLS)
        
        # Delay between batches
        if batch_end < len(prepared_files):
            print(f"\nWaiting {DELAY_BETWEEN_BATCHES} seconds before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    # Print summary
    print("\n" + "=" * 60)
    print("LABELING SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(prepared_files)}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    
    print("\n[Distribution by causality category]")
    total_assessments = sum(category_stats.values())
    for category in ["Certain", "Probable", "Possible", "Unlikely", "Conditional", "Unassessable"]:
        if category in category_stats:
            count = category_stats[category]
            percentage = (count / total_assessments * 100) if total_assessments > 0 else 0
            print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print(f"\nLabels saved to: {LABELS_DIR}/")
    
    # Create training dataset
    if label_files:
        print("\n" + "=" * 60)
        print("CREATING INSTRUCTION TUNING DATASET")
        print("=" * 60)
        
        dataset = create_training_dataset(label_files)
        
        # Save JSON format
        dataset_file = os.path.join(DATASET_DIR, 'training_dataset.json')
        save_json(dataset_file, dataset)
        
        # Save JSONL format for fine-tuning
        train_jsonl = os.path.join(DATASET_DIR, 'train.jsonl')
        val_jsonl = os.path.join(DATASET_DIR, 'validation.jsonl')
        
        with open(train_jsonl, 'w', encoding='utf-8') as f:
            for example in dataset['train']:
                f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": example['instruction']},
                        {"role": "user", "content": example['input']},
                        {"role": "assistant", "content": example['output']}
                    ]
                }, ensure_ascii=False) + "\n")
        
        with open(val_jsonl, 'w', encoding='utf-8') as f:
            for example in dataset['validation']:
                f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": example['instruction']},
                        {"role": "user", "content": example['input']},
                        {"role": "assistant", "content": example['output']}
                    ]
                }, ensure_ascii=False) + "\n")
        
        print(f"\nDataset created:")
        print(f"  Training examples: {len(dataset['train'])}")
        print(f"  Validation examples: {len(dataset['validation'])}")
        print(f"  Files saved in: {DATASET_DIR}/")
    
    print("\n Labeling and dataset creation complete!")

if __name__ == "__main__":
    main()