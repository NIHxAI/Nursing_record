### Clinical Phenotype Classification/Prediction Algorithm

# Nursing Record-based Drug-Symptom Causality Assessment Algorithm

## Overview
Instruction-tuned large language model for drug–symptom causality classification in inpatient nursing records. 
The algorithm classifies causal relationships between drug–symptom pairs documented in nursing records according to the World Health Organization – Uppsala Monitoring Centre (WHO–UMC) causality assessment system.

## Pipeline

**1. Data Preprocessing** (`data_preprocessing.py`)
   - De-identification of clinician identifiers (e.g., names)
   - Temporal segmentation (12-hour windows)

**2. Named Entity Recognition & Information Extraction** (`ner_extraction.py`)
   - Drug entity extraction (medication names and dosage-related information)
   - ADR symptom identification (10 predefined categories)
   - Structured output in JSON format generated using the [OpenAI API](https://openai.com/ko-KR/index/openai-api/)


**3. Data Integration & Pair Generation** (`data_integration.py`)
   - Consolidation of segmented records by admission
   - Drug-symptom pair generation (up to 3 prior medications per symptom)
   - Temporal context calculation between medications and symptoms

**4. Causality Assessment & Dataset Construction** (`prepare_causality_data.py`,`causality_labeling.py`, `dataset_builder.py`)
   - WHO–UMC causality categories (Certain, Probable, Possible, Unlikely, Conditional, Unassessable)
   - Causality labeling of drug–symptom pairs using the [OpenAI API](https://openai.com/ko-KR/index/openai-api/)
   - Instruction tuning dataset generation with structured prompts and responses


**5. Model Development** (`train.py`)

- Instruction-tuned on [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) (12B)
- Fine-tuning with LoRA (Low-Rank Adaptation)
- WHO–UMC-based causality classification with structured JSON outputs


## ADR Categories
10 symptom categories: Fever, Hypotension, Liver dysfunction, Leukopenia, Thrombocytopenia, Acute kidney injury, Hypoglycemia, Rhabdomyolysis, Mucocutaneous reactions, Respiratory symptoms

## Requirements
### Environment
- Python 3.10
- CUDA 12.4
- PyTorch 2.5.1

### Installation
```bash
pip install torch transformers peft datasets accelerate openai pandas numpy
```
