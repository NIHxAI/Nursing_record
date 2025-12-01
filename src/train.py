"""
train.py
Instruction tuning for drug-symptom causality assessment
- Fine-tunes Mistral-Nemo-Instruct model using LoRA
- Implements WHO-UMC causality classification
"""

import os
import torch
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict
from tqdm import tqdm
import re

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset


# ==========================================
# Logging Setup
# ==========================================

def setup_logging(output_dir):
    """Initialize logging configuration"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


# ==========================================
# Configuration
# ==========================================

@dataclass
class ModelArguments:
    """Model configuration arguments"""
    model_name_or_path: str = field(
        default="mistralai/Mistral-Nemo-Instruct-2407"
    )
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    """Data configuration arguments"""
    train_data_path: str = field(default="instruction_dataset/train.json")
    val_data_path: str = field(default="instruction_dataset/validation.json")
    test_data_path: str = field(default="instruction_dataset/test.json")
    max_seq_length: int = field(default=2048)


# ==========================================
# Training Arguments
# ==========================================

def get_training_args():
    """Get training arguments with memory optimization"""

    args_dict = {
        "output_dir": "models/mistral-nemo-causality-tuned",
        "overwrite_output_dir": True,

        # Training hyperparameters
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "num_train_epochs": 5,

        # Save and evaluation
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": 50,

        # Memory optimization
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": False,
        "bf16": True,
        "tf32": True,

        # Other settings
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "report_to": "none",
        "seed": 42,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
    }

    # Handle transformers version compatibility
    try:
        args_dict["eval_strategy"] = "epoch"
        args_dict["save_strategy"] = "epoch"
        args_dict["logging_strategy"] = "steps"
        return TrainingArguments(**args_dict)
    except TypeError:
        args_dict.pop("eval_strategy", None)
        args_dict.pop("save_strategy", None)
        args_dict.pop("logging_strategy", None)
        args_dict["evaluation_strategy"] = "epoch"
        args_dict["save_strategy"] = "epoch"
        args_dict["logging_strategy"] = "steps"
        return TrainingArguments(**args_dict)


# ==========================================
# Dataset Class
# ==========================================

class CausalityDataset:
    """Dataset for causality assessment instruction tuning"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logging.info(f"Loaded {len(self.data)} samples from {data_path}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _build_prompt_text(self, instruction: str, input_text: str) -> str:
        """Build prompt using chat template"""
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{input_text}"}
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )
        return prompt_text

    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        prompts, targets = [], []

        for i in range(len(examples['instruction'])):
            prompt_text = self._build_prompt_text(
                examples['instruction'][i],
                examples['input'][i]
            )
            target_text = examples['output'][i] + self.tokenizer.eos_token
            prompts.append(prompt_text)
            targets.append(target_text)

        # Tokenize full texts
        full_texts = [p + t for p, t in zip(prompts, targets)]
        model_inputs = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )

        # Create labels with prompt masked
        prompt_inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )

        labels = []
        for input_ids, prompt_ids in zip(model_inputs['input_ids'], prompt_inputs['input_ids']):
            label = input_ids.copy()
            prompt_len = len(prompt_ids)
            label[:prompt_len] = [-100] * prompt_len
            labels.append(label)

        model_inputs['labels'] = labels
        return model_inputs

    def get_dataset(self):
        """Get tokenized dataset"""
        dataset = Dataset.from_list(self.data)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        return tokenized_dataset


# ==========================================
# Custom Trainer
# ==========================================

class CausalityTrainer(Trainer):
    """Custom trainer with causality-specific evaluation"""

    def __init__(self, logger=None, *args, **kwargs):
        self.logger = logger or logging.getLogger(__name__)
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate with causality metrics"""
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if eval_dataset is not None:
            self.logger.info(f"Generating predictions for validation set ({len(eval_dataset)} samples)...")
            predictions = self.generate_all_predictions(eval_dataset)

            # Save predictions
            prediction_file = os.path.join(
                self.args.output_dir,
                f"val_predictions_epoch_{int(self.state.epoch)}.json"
            )
            with open(prediction_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Validation predictions saved: {prediction_file}")

            # Compute metrics
            metrics = self.compute_causality_metrics(predictions)
            eval_result.update(metrics)

            self.logger.info("=== Causality Metrics ===")
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"{key}: {value:.4f}")
                else:
                    self.logger.info(f"{key}: {value}")

        return eval_result

    def generate_all_predictions(self, eval_dataset):
        """Generate predictions for all samples"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for idx in tqdm(range(len(eval_dataset)), desc="Generating predictions"):
                sample = eval_dataset[idx]
                input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(self.model.device)
                labels = sample['labels']
                prompt_length = next((i for i, lab in enumerate(labels) if lab != -100), len(labels))

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        input_ids[:, :prompt_length],
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                predicted_text = self.tokenizer.decode(
                    outputs[0][prompt_length:],
                    skip_special_tokens=True
                )
                true_text = self.tokenizer.decode(
                    [lab for lab in labels[prompt_length:] if lab != -100],
                    skip_special_tokens=True
                )

                predictions.append({
                    'sample_id': idx,
                    'predicted': predicted_text.strip(),
                    'actual': true_text.strip()
                })

        return predictions

    def compute_causality_metrics(self, predictions):
        """Compute causality classification metrics"""

        def extract_relations(text):
            relations = set()
            try:
                if text.strip().startswith('[') and text.strip().endswith(']'):
                    data = json.loads(text)
                    for item in data:
                        if isinstance(item, dict):
                            med = item.get('medication_name', '')
                            cat = item.get('causality_category', '')
                            if med and cat:
                                relations.add((med.lower(), cat))
            except Exception:
                med_pattern = r'"medication_name":\s*"([^"]+)"'
                cat_pattern = r'"causality_category":\s*"([^"]+)"'
                meds = re.findall(med_pattern, text)
                cats = re.findall(cat_pattern, text)
                for med, cat in zip(meds, cats):
                    relations.add((med.lower(), cat))
            return relations

        all_predicted, all_actual, correct = set(), set(), set()

        for pred in predictions:
            pr = extract_relations(pred['predicted'])
            gt = extract_relations(pred['actual'])
            all_predicted.update(pr)
            all_actual.update(gt)
            correct.update(pr.intersection(gt))

        precision = len(correct) / len(all_predicted) if all_predicted else 0.0
        recall = len(correct) / len(all_actual) if all_actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
            'eval_predicted_relations': len(all_predicted),
            'eval_actual_relations': len(all_actual),
            'eval_correct_relations': len(correct),
        }

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Custom logging"""
        super().log(logs, start_time)
        if self.state.global_step % self.args.logging_steps == 0:
            log_str = f"Step {self.state.global_step}: "
            log_str += ", ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in logs.items()
            ])
            self.logger.info(log_str)


# ==========================================
# Model Setup
# ==========================================

def setup_model_and_tokenizer(model_args: ModelArguments, logger):
    """Initialize model and tokenizer"""

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    if model_args.use_lora:
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")

    logger.info("Model setup complete")
    return model, tokenizer


# ==========================================
# Test Evaluation
# ==========================================

def evaluate_test_set(model, tokenizer, test_data_path, output_dir, logger):
    """Evaluate on test set"""

    logger.info("=== Test Set Evaluation ===")
    logger.info(f"Test data: {test_data_path}")

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    logger.info(f"Test samples: {len(test_data)}")

    test_output_dir = os.path.join(output_dir, "test_predictions")
    os.makedirs(test_output_dir, exist_ok=True)

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_data, desc="Generating test predictions")):
            input_text = sample['input']
            patient_match = re.search(r'\*\*Patient ID:\*\* (\w+)', input_text)
            record_match = re.search(r'\*\*Record:\*\* (\w+)', input_text)
            patient_id = patient_match.group(1) if patient_match else f"unknown_{idx}"
            record_no = record_match.group(1) if record_match else f"unknown_{idx}"

            messages = [{"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"}]
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            predicted_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            result = {
                'patient_id': patient_id,
                'record_no': record_no,
                'input': sample['input'],
                'instruction': sample['instruction'],
                'predicted': predicted_text.strip(),
                'actual': sample['output'],
            }
            all_predictions.append(result)

            # Save individual prediction
            individual_file = os.path.join(
                test_output_dir,
                f"{patient_id}_{record_no}_test_result.json"
            )
            with open(individual_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # Save all predictions
    all_results_file = os.path.join(output_dir, "test_all_predictions.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    logger.info(f"Test predictions complete")
    logger.info(f"Individual files: {test_output_dir}/")
    logger.info(f"All results: {all_results_file}")

    # Compute metrics
    logger.info("Computing test metrics...")
    metrics = compute_test_metrics(all_predictions)

    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")

    with open(os.path.join(output_dir, "test_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return all_predictions, metrics


def compute_test_metrics(predictions):
    """Compute test set metrics"""

    def extract_relations(text):
        relations = set()
        try:
            if text.strip().startswith('[') and text.strip().endswith(']'):
                data = json.loads(text)
                for item in data:
                    if isinstance(item, dict):
                        med = item.get('medication_name', '')
                        cat = item.get('causality_category', '')
                        if med and cat:
                            relations.add((med.lower(), cat))
        except Exception:
            med_pattern = r'"medication_name":\s*"([^"]+)"'
            cat_pattern = r'"causality_category":\s*"([^"]+)"'
            meds = re.findall(med_pattern, text)
            cats = re.findall(cat_pattern, text)
            for med, cat in zip(meds, cats):
                relations.add((med.lower(), cat))
        return relations

    all_predicted, all_actual, correct = set(), set(), set()

    for pred in predictions:
        pr = extract_relations(pred['predicted'])
        gt = extract_relations(pred['actual'])
        all_predicted.update(pr)
        all_actual.update(gt)
        correct.update(pr.intersection(gt))

    precision = len(correct) / len(all_predicted) if all_predicted else 0.0
    recall = len(correct) / len(all_actual) if all_actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_predicted_relations': len(all_predicted),
        'test_actual_relations': len(all_actual),
        'test_correct_relations': len(correct),
    }


# ==========================================
# Main Training Pipeline
# ==========================================

def main():
    """Main training pipeline"""

    # Initialize arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = get_training_args()

    # Setup output directory and logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger = setup_logging(training_args.output_dir)

    logger.info("=" * 60)
    logger.info("Starting Causality Assessment Instruction Tuning")
    logger.info("=" * 60)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args, logger)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = CausalityDataset(
        data_args.train_data_path, tokenizer, data_args.max_seq_length
    ).get_dataset()
    val_dataset = CausalityDataset(
        data_args.val_data_path, tokenizer, data_args.max_seq_length
    ).get_dataset()

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # Setup callbacks
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.0,
    )

    # Initialize trainer
    trainer = CausalityTrainer(
        logger=logger,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    # Start training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    train_result = trainer.train()

    # Save model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    logger.info("=" * 60)
    logger.info("Training completed")
    logger.info("=" * 60)
    logger.info(f"Final loss: {train_result.training_loss:.4f}")
    logger.info(f"Total steps: {train_result.global_step}")

    # Save training metrics
    with open(os.path.join(training_args.output_dir, "training_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)

    # Test evaluation with best model
    if os.path.exists(data_args.test_data_path):
        logger.info("=" * 60)
        logger.info("Loading best model for test evaluation...")
        logger.info("=" * 60)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )

        # Load LoRA weights and merge
        test_model = PeftModel.from_pretrained(base_model, training_args.output_dir)
        test_model = test_model.merge_and_unload()

        logger.info("Best model loaded successfully")

        # Run test evaluation
        evaluate_test_set(
            model=test_model,
            tokenizer=tokenizer,
            test_data_path=data_args.test_data_path,
            output_dir=training_args.output_dir,
            logger=logger,
        )

    logger.info("=" * 60)
    logger.info("All tasks completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()