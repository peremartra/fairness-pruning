"""
Fairness Pruning Research - Utility Functions
=============================================

This module provides the core infrastructure for the "Fairness Pruning" experiments,
focusing on bias mitigation in Llama-3.2 and Salamandra models through activation-guided
width pruning.

Key Responsibilities:
--------------------
1. Experiment Orchestration: Manages configuration for Llama-3.2 (1B/3B) and Salamandra-2B.
2. Robust Evaluation: Wraps `lm-evaluation-harness` with a fault-tolerant checkpoint system,
   allowing experiments to resume automatically after disconnections (essential for Colab).
3. Dynamic Pruning: Integrates with `OptiPFair` to apply pruning masks on-the-fly before evaluation.
4. Cross-Lingual Support: Configured to handle both English (MMLU, HellaSwag) and 
   Spanish (Belebele, XCOPA) benchmarks.
5. Resource Management: Handles GPU memory cleanup and optional carbon profiling.

Usage:
------
    from utils import run_robust_evaluation, load_or_create_model, ALL_TASKS
    
    # Load model and apply pruning if specified in config
    model, tokenizer, stats = load_or_create_model(config_entry)
    
    # Run benchmarks with automatic state saving
    results = run_robust_evaluation(
        model, tokenizer, 
        tasks=ALL_TASKS, 
        checkpoint_path="./checkpoints/llama_1b_fairness.json"
    )
    
author: Pere Martra
Repository: https://github.com/peremartra/fairness-pruning
"""

try:
    import lm_eval
    from lm_eval.tasks import TaskManager
    import transformers
    import optipfair
    # Additional imports for carbon profiling
    import time
    import numpy as np
    from datetime import datetime
    import os
except ImportError as e:
    raise ImportError(
        f"Missing required library: {e.name}\n"
        "Install all dependencies with:\n"
        "  pip install optipfair lm-eval transformers torch langdetect"
    )
# /content/veritas_qa_ca.yaml
os.environ["LMEVAL_INCLUDE_PATH"] = "/content/"

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG = [
    {
        "base_model": "BSC-LT/salamandra-2b",
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
    },
]

# =============================================================================
# BENCHMARK CONFIGURATIONS
# =============================================================================

# Base models benchmark suite (10 benchmarks)
# Default: 0-shot unless specified otherwise
BENCHMARKS_BASE = [
    # --- English Core Capabilities ---
    {"name": "wikitext", "num_fewshot": 0},       # Perplexity / Modeling base
    {"name": "lambada_openai", "num_fewshot": 0}, # Next-token prediction
    {"name": "ifeval", "num_fewshot": 0},         # Instruction Following
    
    # --- English Reasoning & Knowledge ---
    {"name": "gsm8k", "num_fewshot": 5},          # Multi-step Reasoning (Fragile metric)
    {"name": "mmlu", "num_fewshot": 5},           # General Knowledge (Standard 5-shot)
    {"name": "arc_challenge", "num_fewshot": 0},  # Reasoning
    {"name": "hellaswag", "num_fewshot": 0},      # Commonsense
    {"name": "truthfulqa_mc2", "num_fewshot": 0}, # Hallucinations (MC2 is standard)

    # --- Spanish / Cross-Lingual Capabilities (Symmetrical) ---
    {"name": "global_mmlu_es", "num_fewshot": 5}, # Spanish Knowledge
    {"name": "arc_es", "num_fewshot": 0},         # Spanish Reasoning
    {"name": "hellaswag_es", "num_fewshot": 0},   # Spanish Commonsense
    {"name": "belebele_spa_Latn", "num_fewshot": 0}, # Native Reading Comprehension
   {"name": "veritas_qa_es", "num_fewshot": 0}, # Search veritas_qa_es.yaml
    {"name": "veritas_qa_ca", "num_fewshot": 0}, # Search veritas_qa_ca.yaml
]

# MMLU Category Groupings for detailed analysis
MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social_Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_aging", "human_sexuality", "management", "marketing",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy"
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "miscellaneous", "nutrition", "professional_accounting", "professional_medicine",
        "virology"
    ]
}

# MMLU_ES uses similar structure - lm-eval handles it automatically
# Categories apply to both mmlu and global_mmlu (mmlu_es)

# TruthfulQA category groupings
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv
TRUTHFULQA_CATEGORIES = {
    "high_stakes": {
        "health": ["Health", "Nutrition", "Medicine"],
        "law": ["Law", "Legal"],
        "finance": ["Finance", "Economics"],
        "politics": ["Politics", "Government"]
    },
    "misinformation": {
        "science": ["Science", "Physics", "Biology"],
        "history": ["History"],
        "conspiracies": ["Conspiracies", "Paranormal"],
        "myths": ["Myths and Fairytales", "Superstitions"]
    },
    "other": ["Advertising", "Fiction", "Indexical Error", "Language", "Logical Fallacies",
              "Misconceptions", "Proverbs", "Stereotypes", "Subjective", "Weather"]
}


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Device detection
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default checkpoint directory (Google Drive recommended for Colab)
DEFAULT_CHECKPOINT_DIR = "/content/drive/MyDrive/glu_pruning/checkpoints"

# Library versions for reproducibility (to be filled during development)
LIBRARY_VERSIONS = {
    "optipfair": None,  # Will be populated at runtime
    "transformers": None,
    "lm-eval": None,
    "torch": None,
}

# =============================================================================
# CORE EVALUATION FUNCTIONS
# =============================================================================

def _process_mmlu_subcategories(results_dict, task_prefix="mmlu"):
    """
    Process MMLU subcategories and group them by academic domain.

    Args:
        results_dict: Dictionary from lm-eval with all task results
        task_prefix: "mmlu", "mmlu_es", or "global_mmlu_es"

    Returns:
        dict: Grouped accuracy by category plus overall average

    Example output:
        {
            "accuracy": "0.3111",  # Overall average
            "STEM": "0.2950",
            "Humanities": "0.3200",
            "Social_Sciences": "0.3150",
            "Other": "0.3080",
            "subcategories": {
                "abstract_algebra": "0.2800",
                "anatomy": "0.3200",
                ...
            }
        }
    """
    # Find all MMLU subcategory results
    subcategory_results = {}
    overall_scores = []

    for task_name, metrics in results_dict.items():
        # Check if this is an MMLU subcategory
        # lm-eval names them like "mmlu_abstract_algebra", "mmlu_es_abstract_algebra", or "global_mmlu_es_abstract_algebra"
        if task_name.startswith(task_prefix) and task_name != task_prefix:
            # Extract subcategory name (remove prefix)
            if task_prefix == "mmlu":
                subcat = task_name.replace("mmlu_", "")
            elif task_prefix == "global_mmlu_es":
                subcat = task_name.replace("global_mmlu_es_", "")
            else:  # mmlu_es
                subcat = task_name.replace("mmlu_es_", "")

            # Get accuracy score
            if 'acc,none' in metrics:
                score = metrics['acc,none']
                subcategory_results[subcat] = score
                overall_scores.append(score)

    # Group by category
    category_scores = {}
    for category, subcats in MMLU_CATEGORIES.items():
        category_results = [subcategory_results[sc] for sc in subcats if sc in subcategory_results]
        if category_results:
            category_scores[category] = sum(category_results) / len(category_results)

    # Calculate overall average
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    # Format result
    result = {
        "accuracy": f"{overall_avg:.4f}",
        "acc_norm": "N/A"
    }

    # Add category breakdowns
    for category, score in category_scores.items():
        result[f"category_{category}"] = f"{score:.4f}"

    # Add detailed subcategory scores (optional, for debugging/analysis)
    result["subcategories"] = {
        k: f"{v:.4f}" for k, v in subcategory_results.items()
    }

    return result


def _get_truthfulqa_category(question_metadata):
    """
    Classify TruthfulQA question into high-stakes or other categories.
    
    Args:
        question_metadata (dict): Metadata from TruthfulQA dataset
        
    Returns:
        tuple: (risk_level, category)
        
    Example:
        >>> _get_truthfulqa_category({"Category": "Health"})
        ("high_stakes", "health")
    """
    category_raw = question_metadata.get("Category", "").strip()
    
    if not category_raw:
        return "other", "miscellaneous"
    
    # Check high-stakes categories first (PRIORITY)
    for risk_category, keywords in TRUTHFULQA_CATEGORIES["high_stakes"].items():
        if any(keyword.lower() in category_raw.lower() or category_raw.lower() in keyword.lower() 
               for keyword in keywords):
            return "high_stakes", risk_category
    
    # Check misinformation categories
    for info_category, keywords in TRUTHFULQA_CATEGORIES["misinformation"].items():
        if any(keyword.lower() in category_raw.lower() or category_raw.lower() in keyword.lower() 
               for keyword in keywords):
            return "misinformation", info_category
    
    # Check if in "other" list
    if any(keyword.lower() in category_raw.lower() or category_raw.lower() in keyword.lower() 
           for keyword in TRUTHFULQA_CATEGORIES["other"]):
        return "other", "miscellaneous"
    
    # Default to "other" for unknown categories
    return "other", "miscellaneous"


def _save_raw_result(raw_results, model_name, task_name, base_dir):
    """
    Save raw lm-eval results to JSON file with robust type handling.
    """
    import json
    import os
    import numpy as np
    
    # Clase auxiliar para convertir tipos no serializables (NumPy, Torch, etc.)
    class RobustEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Convertir cualquier otro tipo extraño a string (ej. torch.dtype)
            return str(obj)

    # Sanitize model name for filename
    safe_model = model_name.replace('/', '_').replace('-', '_').lower()
    safe_task = task_name.lower().replace(' ', '_')

    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Build filename
    filename = f"{safe_model}_{safe_task}.json"
    filepath = os.path.join(base_dir, filename)

    # Save with atomic write using the custom encoder
    temp_path = filepath
    with open(temp_path, 'w') as f:
        # Usamos cls=RobustEncoder para evitar el error de "dtype is not JSON serializable"
        json.dump(raw_results, f, indent=2, ensure_ascii=False, cls=RobustEncoder)

    # Atomic rename (si estás en Windows esto puede dar error si existe, pero en Colab/Linux va bien)
    os.replace(temp_path, filepath)

    return filepath

def parse_veritasqa_choices(doc):
    """Parse and clean VeritasQA answer choices.
    
    VeritasQA stores answers as semicolon-separated strings.
    This function splits them, cleans whitespace, and returns
    correct answers first (so index 0 is always correct).
    
    Args:
        doc: Dataset example with 'correct_answers' and 'incorrect_answers'
        
    Returns:
        list: All answer choices (correct + incorrect)
    """
    correct = [ans.strip() for ans in doc['correct_answers'].split(';')]
    incorrect = [ans.strip() for ans in doc['incorrect_answers'].split(';')]
    
    # Return correct answers first (doc_to_target=0 expects this)
    return correct + incorrect

def model_evaluation(model_obj, tokenizer, tasks, limit=None, save_raw_results=False, raw_results_dir=None):
    """
    Runs lm-eval on a model and tokenizer already in memory.
    NOW with TruthfulQA category breakdown for safety analysis.

    Args:
        model_obj: PyTorch model object to evaluate
        tokenizer: Tokenizer object for the model
        tasks (list): List of task dicts with format:
                     [{"name": "wikitext", "num_fewshot": 0}, ...]
                     OR simple list of strings: ["wikitext", "boolq"]
        limit (int, optional): Number of samples per task for quick testing
        save_raw_results (bool): If True, save raw lm-eval output to JSON files
        raw_results_dir (str): Directory to save raw results (if save_raw_results=True)

    Returns:
        dict: Formatted results with metrics per task and TruthfulQA subcategories
        
    Raises:
        ImportError: If lm-eval is not installed
        Exception: If evaluation fails for all tasks
        
    Example:
        >>> results = model_evaluation(
        ...     model, tokenizer, 
        ...     tasks=BENCHMARKS_BASE,
        ...     limit=100  # Quick test
        ... )
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    # Extract model name for logging
    model_name = getattr(model_obj.config, '_name_or_path', 'unknown')
    limit_str = f"(limit={limit})" if limit else "(full dataset)"
    
    # Parse tasks to handle both dict and string formats
    task_names = []
    task_fewshot_map = {}
    
    for task in tasks:
        if isinstance(task, dict):
            task_name = task["name"]
            task_names.append(task_name)
            task_fewshot_map[task_name] = task["num_fewshot"]
        else:
            # Backward compatibility: simple string list
            task_names.append(task)
            task_fewshot_map[task] = 0
    
    print(f"\n{'='*70}")
    print(f"Starting lm-eval on model '{model_name}'")
    print(f"Tasks: {task_names} {limit_str}")
    print(f"Few-shot config: {task_fewshot_map}")
    print(f"{'='*70}\n")
    
    # Wrap model for lm-eval
    model_wrapper = HFLM(
        pretrained=model_obj,
        batch_size=8,
        tokenizer=tokenizer,
        device=str(DEVICE)
    )
    
    # Run evaluation with per-task few-shot configuration
    if len(set(task_fewshot_map.values())) == 1:
        fewshot_value = list(task_fewshot_map.values())[0]
    else:
        fewshot_value = 0

    # tm = TaskManager(include_path="/content/")
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=task_names,
        num_fewshot=fewshot_value,
        limit=limit,
        device=str(DEVICE), 
        log_samples=True
    )

    # Save raw results if requested
    if save_raw_results and raw_results_dir:
        import os
        print(f"\n💾 Saving raw results to: {raw_results_dir}")
        for task_name in task_names:
            try:
                filepath = _save_raw_result(
                    raw_results=results,
                    model_name=model_name,
                    task_name=task_name,
                    base_dir=raw_results_dir
                )
                print(f"   ✅ Saved: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"   ⚠️ Failed to save raw results for {task_name}: {e}")

    # Initialize results structure
    formatted_results = {}
    
    # Track TruthfulQA samples for category breakdown
    truthfulqa_samples = {"mc1": [], "mc2": []}

    # Check if MMLU or MMLU_ES was evaluated (has subcategories)
    has_mmlu = "mmlu" in task_names
    has_mmlu_es = "mmlu_es" in task_names or "global_mmlu_es" in task_names
    has_truthfulqa = any(t.startswith("truthfulqa") for t in task_names)

    # Process MMLU subcategories if present
    if has_mmlu:
        mmlu_detailed = _process_mmlu_subcategories(results["results"], "mmlu")
        formatted_results["mmlu"] = mmlu_detailed
        print(f"\n✅ MMLU processed with {len(mmlu_detailed.get('subcategories', {}))} subcategories")
        print(f"   Overall: {mmlu_detailed['accuracy']}")
        print(f"   STEM: {mmlu_detailed.get('category_STEM', 'N/A')}")
        print(f"   Humanities: {mmlu_detailed.get('category_Humanities', 'N/A')}")
        print(f"   Social Sciences: {mmlu_detailed.get('category_Social_Sciences', 'N/A')}")
        print(f"   Other: {mmlu_detailed.get('category_Other', 'N/A')}")

    if has_mmlu_es:
        # Detect which prefix to use (global_mmlu_es or mmlu_es)
        task_prefix = "global_mmlu_es" if "global_mmlu_es" in task_names else "mmlu_es"
        mmlu_es_detailed = _process_mmlu_subcategories(results["results"], task_prefix)
        # Store with the actual task name used
        formatted_results[task_prefix] = mmlu_es_detailed
        print(f"\n✅ {task_prefix.upper()} processed with {len(mmlu_es_detailed.get('subcategories', {}))} subcategories")
        print(f"   Overall: {mmlu_es_detailed['accuracy']}")
        print(f"   STEM: {mmlu_es_detailed.get('category_STEM', 'N/A')}")
        print(f"   Humanities: {mmlu_es_detailed.get('category_Humanities', 'N/A')}")
        print(f"   Social Sciences: {mmlu_es_detailed.get('category_Social_Sciences', 'N/A')}")
        print(f"   Other: {mmlu_es_detailed.get('category_Other', 'N/A')}")

    # Process all tasks
    for task_name, res in results["results"].items():
        # Skip MMLU subcategories (already processed above)
        if task_name.startswith("mmlu_") or task_name.startswith("mmlu_es_") or task_name.startswith("global_mmlu_es_"):
            continue

        # Skip main MMLU/MMLU_ES/GLOBAL_MMLU_ES if already processed
        if task_name in ["mmlu", "mmlu_es", "global_mmlu_es"] and task_name in formatted_results:
            continue

        # Handle TruthfulQA variants specially
        if task_name in ["truthfulqa_mc1", "truthfulqa_mc2"]:
            variant = "mc1" if "mc1" in task_name else "mc2"
            
            # Store global result
            formatted_results[task_name] = {
                'accuracy': f"{res.get('acc,none', 0):.4f}",
                'acc_norm': f"{res.get('acc_norm,none', 0):.4f}" if 'acc_norm,none' in res else "N/A"
            }
            
            # Collect samples if available for category breakdown
            if 'samples' in res:
                for sample in res['samples']:
                    metadata = sample.get('doc', {})
                    risk_level, category = _get_truthfulqa_category(metadata)
                    truthfulqa_samples[variant].append({
                        'risk_level': risk_level,
                        'category': category,
                        'accuracy': float(sample.get('acc', 0)),
                        'acc_norm': float(sample.get('acc_norm', sample.get('acc', 0)))
                    })
            continue

        # Extract relevant metrics based on task type
        if 'perplexity,none' in res:
            # Perplexity tasks (wikitext, lambada)
            formatted_results[task_name] = {
                'perplexity': f"{res.get('perplexity,none', 0):.2f}",
                'word_perplexity': f"{res.get('word_perplexity,none', 0):.2f}",
                'bits_per_byte': f"{res.get('bits_per_byte,none', 0):.4f}"
            }
        elif 'acc,none' in res:
            # Accuracy tasks (boolq, arc, hellaswag, etc.)
            formatted_results[task_name] = {
                'accuracy': f"{res.get('acc,none', 0):.4f}",
                'acc_norm': f"{res.get('acc_norm,none', 0):.4f}" if 'acc_norm,none' in res else "N/A"
            }
        else:
            # Fallback: store all numeric metrics
            formatted_results[task_name] = {
                k: f"{v:.4f}" for k, v in res.items()
                if isinstance(v, (int, float))
            }

    # =========================================================================
    # POST-PROCESS: Aggregate TruthfulQA by categories
    # =========================================================================
    if has_truthfulqa:
        for variant in ["mc1", "mc2"]:
            if not truthfulqa_samples[variant]:
                continue
            
            # Group by risk level and category
            category_groups = {}
            for sample in truthfulqa_samples[variant]:
                risk_level = sample['risk_level']
                category = sample['category']
                
                key = (risk_level, category)
                if key not in category_groups:
                    category_groups[key] = []
                category_groups[key].append(sample)
            
            # Calculate averages and add to results
            for (risk_level, category), samples in category_groups.items():
                avg_acc = sum(s['accuracy'] for s in samples) / len(samples)
                avg_acc_norm = sum(s['acc_norm'] for s in samples) / len(samples)
                
                # Create synthetic task name for the subcategory
                task_key = f"truthfulqa_{variant}_{risk_level}_{category}"
                
                formatted_results[task_key] = {
                    'accuracy': f"{avg_acc:.4f}",
                    'acc_norm': f"{avg_acc_norm:.4f}",
                    'num_samples': len(samples),
                    'risk_level': risk_level,
                    'category': category
                }
        
        # Print TruthfulQA breakdown
        print(f"\n✅ TruthfulQA category breakdown:")
        for variant in ["mc1", "mc2"]:
            variant_tasks = [k for k in formatted_results.keys() if k.startswith(f"truthfulqa_{variant}_")]
            if variant_tasks:
                print(f"\n   {variant.upper()}:")
                for task in sorted(variant_tasks):
                    if task == f"truthfulqa_{variant}":  # Skip global
                        continue
                    data = formatted_results[task]
                    risk_marker = "🚨" if data['risk_level'] == 'high_stakes' else "📊"
                    print(f"      {risk_marker} {data['category']}: {data['accuracy']} ({data['num_samples']} samples)")

    return formatted_results

# =============================================================================
# INTERNAL HELPER FUNCTIONS FOR CARBON PROFILING
# =============================================================================

def _get_checkpoint_dir(base_dir, model_size, mode="evaluation"):
    """
    Internal helper: Construct checkpoint directory based on mode.
    
    Args:
        base_dir: Base checkpoint directory
        model_size: "1b", "3b", "1b_instruct", etc.
        mode: "evaluation" (default) or "carbon"
    
    Returns:
        str: Full checkpoint directory path (created if doesn't exist)
    """
    import os
    
    if mode == "evaluation":
        subdir = model_size
    elif mode == "carbon":
        subdir = f"{model_size}_carbon"
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'evaluation' or 'carbon'")
    
    checkpoint_dir = os.path.join(base_dir, subdir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return checkpoint_dir


def _get_results_filename(model_size, mode="evaluation", version="latest"):
    """
    Internal helper: Construct results filename based on mode.
    
    Args:
        model_size: "1b", "3b", etc.
        mode: "evaluation" (default) or "carbon"
        version: "latest" or timestamp string
    
    Returns:
        str: Results filename
    """
    prefix = "carbon_" if mode == "carbon" else ""
    return f"{prefix}llama_{model_size}_results_{version}.csv"


def _load_workload_prompts(workload):
    """
    Internal helper: Load prompts from specified dataset.
    
    Args:
        workload (dict): Workload specification with keys:
            - dataset: "gsm8k", "mmlu", etc.
            - subset: "test", "train", etc.
            - num_prompts: Number of prompts to load
            - random_seed: (optional) Seed for reproducible sampling
    
    Returns:
        list[str]: List of text prompts
    """
    from datasets import load_dataset
    import random
    
    dataset_name = workload["dataset"]
    num_prompts = workload["num_prompts"]
    subset = workload.get("subset", "test")
    random_seed = workload.get("random_seed", None)  # ← NUEVO
    
    try:
        if dataset_name == "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split=subset)
            
            # Selección determinística con seed
            if random_seed is not None:
                indices = list(range(len(dataset)))
                random.Random(random_seed).shuffle(indices)
                indices = indices[:num_prompts]
                selected_items = [dataset[i] for i in indices]
            else:
                selected_items = dataset.select(range(min(num_prompts, len(dataset))))
            
            # Construir prompts (necesario porque combinamos múltiples campos)
            prompts = []
            for item in selected_items:
                context = item["ctx"]
                endings = item["endings"]
                
                prompt = f"{context}\n\nWhat happens next?\n"
                for i, ending in enumerate(endings):
                    prompt += f"{chr(65+i)}. {ending}\n"
                prompt += "\nAnswer:"
                
                prompts.append(prompt)
        
        elif dataset_name == "mmlu":
            dataset = load_dataset("cais/mmlu", "all", split=subset)
            # ← NUEVO: Selección determinística con seed
            if random_seed is not None:
                indices = list(range(len(dataset)))
                random.Random(random_seed).shuffle(indices)
                indices = indices[:num_prompts]
                prompts = [dataset[i]["question"] for i in indices]
            else:
                prompts = [item["question"] for item in dataset.select(range(min(num_prompts, len(dataset))))]
                
        elif dataset_name == "IFEval":
            actual_split = "train" if subset in ["default", "test"] else subset
            dataset = load_dataset("google/IFEval", split=actual_split)
            # ← NUEVO: Selección determinística con seed
            if random_seed is not None:
                indices = list(range(len(dataset)))
                random.Random(random_seed).shuffle(indices)
                indices = indices[:num_prompts]
                prompts = [dataset[i]["prompt"] for i in indices]
            else:
                prompts = [item["prompt"] for item in dataset.select(range(min(num_prompts, len(dataset))))]
        else:
            # Fallback: generic prompts
            prompts = [f"Test prompt {i+1}" for i in range(num_prompts)]
        
        return prompts
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        # Fallback prompts
        return [f"Fallback prompt {i+1}" for i in range(num_prompts)]


def _get_memory_stats(model, device="cuda"):
    """
    Internal helper: Get memory usage statistics.
    
    Args:
        model: PyTorch model
        device (str): Device placement
    
    Returns:
        dict: Memory statistics in GB
    """
    stats = {}
    
    if device == "cuda" and torch.cuda.is_available():
        stats["memory_allocated_gb"] = float(torch.cuda.memory_allocated() / (1024**3))
        stats["memory_reserved_gb"] = float(torch.cuda.memory_reserved() / (1024**3))
        stats["max_memory_allocated_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    
    # Model size (works for both CPU and CUDA)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    stats["model_size_gb"] = float(model_size_bytes / (1024**3))
    
    return stats

def run_robust_evaluation(model, tokenizer, tasks, checkpoint_path, model_name=None):
    """
    Run evaluation with checkpoint/resume support for Colab disconnections.
    
    This function saves progress after each benchmark, allowing recovery from
    interruptions. Checkpoint files are stored as JSON with task completion status.
    
    Args:
        model: PyTorch model object to evaluate
        tokenizer: Tokenizer object for the model
        tasks (list): List of task dicts with format:
                     [{"name": "wikitext", "num_fewshot": 0}, ...]
        checkpoint_path (str): Path to checkpoint JSON file
                              (e.g., "/content/drive/MyDrive/glu_pruning/llama_1b_20pct.json")
        model_name (str, optional): Human-readable model name for logging
        
    Returns:
        dict: Complete results with all benchmark metrics
        
    Example:
        >>> results = run_robust_evaluation(
        ...     model, tokenizer,
        ...     tasks=BENCHMARKS_BASE,
        ...     checkpoint_path="/content/drive/MyDrive/checkpoints/model.json"
        ... )
        >>> # If interrupted, re-run the same command - it will resume
    """
    import json
    import os
    from datetime import datetime
    from pathlib import Path
    
    # Extract model name for metadata
    if model_name is None:
        model_name = getattr(model.config, '_name_or_path', 'unknown')
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Parse tasks to get task names
    task_names = [t["name"] if isinstance(t, dict) else t for t in tasks]
    
    # Load or create checkpoint
    if os.path.exists(checkpoint_path):
        print(f"📂 Found existing checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Validate checkpoint structure
        if "results" not in checkpoint or "pending_tasks" not in checkpoint:
            print("⚠️  Invalid checkpoint structure. Starting fresh.")
            checkpoint = _create_new_checkpoint(model_name, task_names)
        else:
            print(f"✅ Loaded checkpoint. Completed: {len(checkpoint['results'])}/{len(task_names)} tasks")
            print(f"   Pending: {checkpoint['pending_tasks']}")
            if checkpoint.get('failed_tasks'):
                print(f"   ⚠️  Previously failed: {checkpoint['failed_tasks']}")
    else:
        print(f"🆕 Creating new checkpoint: {checkpoint_path}")
        checkpoint = _create_new_checkpoint(model_name, task_names)
    
    # Identify tasks to run (pending + failed to retry)
    completed_tasks = set(checkpoint["results"].keys())
    tasks_to_run = [t for t in tasks if (t["name"] if isinstance(t, dict) else t) not in completed_tasks]
    
    if not tasks_to_run:
        print("🎉 All tasks already completed!")
        return checkpoint["results"]

    # Determine raw results directory
    raw_results_dir = os.path.join(
        os.path.dirname(os.path.dirname(checkpoint_path)),  # Go up 2 levels from checkpoint
        "results", "lm_evals"
    )

    print(f"\n{'='*70}")
    print(f"🚀 Starting evaluation: {len(tasks_to_run)} tasks remaining")
    print(f"{'='*70}\n")

    # Run each pending task
    for i, task in enumerate(tasks_to_run, 1):
        task_name = task["name"] if isinstance(task, dict) else task

        print(f"\n[{i}/{len(tasks_to_run)}] Evaluating: {task_name}")
        print(f"{'─'*70}")

        try:
            # Run evaluation for single task
            result = model_evaluation(
                model, tokenizer,
                tasks=[task],
                limit=None,
                save_raw_results=True,  # Enable raw result saving
                raw_results_dir=raw_results_dir
            )
            
            # Store result in checkpoint
            checkpoint["results"][task_name] = result[task_name]
            if task_name in checkpoint["pending_tasks"]:
                checkpoint["pending_tasks"].remove(task_name)
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Remove from failed tasks if it was there
            if task_name in checkpoint.get("failed_tasks", []):
                checkpoint["failed_tasks"].remove(task_name)
            
            # Save checkpoint after each task
            _save_checkpoint(checkpoint_path, checkpoint)
            
            print(f"✅ {task_name} completed and saved to checkpoint")
            print(f"   Results: {result[task_name]}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {task_name} FAILED: {error_msg}")
            
            # Track failed task but continue
            if "failed_tasks" not in checkpoint:
                checkpoint["failed_tasks"] = []
            if task_name not in checkpoint["failed_tasks"]:
                checkpoint["failed_tasks"].append(task_name)
            
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            _save_checkpoint(checkpoint_path, checkpoint)
            
            print(f"⚠️  Continuing with next task...")
            continue
    
    # Mark as completed if all tasks done
    if not checkpoint["pending_tasks"]:
        checkpoint["metadata"]["completed"] = True
        checkpoint["metadata"]["completed_at"] = datetime.now().isoformat()
        _save_checkpoint(checkpoint_path, checkpoint)
        
        print(f"\n{'='*70}")
        print("🎉 ALL TASKS COMPLETED!")
        if checkpoint.get("failed_tasks"):
            print(f"⚠️  Some tasks failed: {checkpoint['failed_tasks']}")
        print(f"{'='*70}\n")
    
    return checkpoint["results"]


def _create_new_checkpoint(model_name, task_names):
    """Create a new checkpoint structure."""
    from datetime import datetime
    return {
        "metadata": {
            "model_name": model_name,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed": False
        },
        "results": {},
        "pending_tasks": task_names.copy(),
        "failed_tasks": []
    }


def _save_checkpoint(checkpoint_path, checkpoint):
    """Save checkpoint to file with error handling."""
    import json
    import shutil
    from pathlib import Path
    
    try:
        # Write to temporary file first (atomic write)
        temp_path = f"{checkpoint_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Move to final location
        shutil.move(temp_path, checkpoint_path)
        
        # Sync with Google Drive if path contains 'drive'
        if 'drive' in checkpoint_path.lower():
            try:
                # Force sync by touching the file
                Path(checkpoint_path).touch()
            except:
                pass  # Drive sync is best-effort
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint: {e}")
        # Don't crash the evaluation if checkpoint save fails


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_gpu_cache():
    """
    Clear GPU memory cache and run garbage collection.
    
    Essential for Colab environments to prevent OOM errors when
    switching between models or after pruning operations.
    """
    import gc
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    if torch.cuda.is_available():
        print(f"🧹 GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def get_model_stats(model):
    """
    Calculate model statistics: total parameters, trainable parameters, size.
    
    Args:
        model: PyTorch model object
        
    Returns:
        dict: Statistics including parameter counts and model size
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "size_mb": size_mb,
        "size_gb": size_mb / 1024
    }


def load_or_create_model(config_entry, device="auto"):
    """
    Load model from HF Hub (if star) or create via pruning (if not star).
    
    Args:
        config_entry (dict): Entry from EXPERIMENT_CONFIG
        device (str): Device placement ("auto", "cuda", "cpu")
        
    Returns:
        tuple: (model, tokenizer, stats_dict)
        
    Example:
        >>> config = EXPERIMENT_CONFIG[1]  # 1B-40% (star)
        >>> model, tokenizer, stats = load_or_create_model(config)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from optipfair import prune_model
    
    base_model_id = config_entry["base_model"]
    hf_repo_id = config_entry["hf_repo_id"]
    is_star = config_entry["is_star"]
    pruning_pct = config_entry["pruning_pct"]
    
    print(f"\n{'='*70}")
    print(f"Loading model: {hf_repo_id}")
    print(f"  Base: {base_model_id}")
    print(f"  Pruning: {pruning_pct}%")
    print(f"  Star model: {'⭐ Yes' if is_star else 'No (on-the-fly)'}")
    print(f"{'='*70}\n")
    
    # Load tokenizer (always from base model)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_star:
        # Try loading from HF Hub first
        try:
            print(f"📥 Attempting to load from HF Hub: {hf_repo_id}")
            model = AutoModelForCausalLM.from_pretrained(
                hf_repo_id,
                #torch_dtype=torch.float16, #L4
                torch_dtype=torch.bfloat16, #A100
                device_map=device
            )
            print(f"✅ Loaded from HF Hub")
            stats = {"source": "hf_hub", **get_model_stats(model)}
            return model, tokenizer, stats
            
        except Exception as e:
            print(f"⚠️  HF Hub load failed: {e}")
            print(f"   Falling back to on-the-fly pruning...")
    
    # Create via pruning (fallback or non-star)
    print(f"🔧 Creating model via on-the-fly pruning...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        #torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    print(f"✂️  Pruning with MAW method ({pruning_pct}%)...")
    pruned_model, prune_stats = prune_model(
        model=base_model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=pruning_pct,
        show_progress=True,
        return_stats=True
    )
    
    print(f"✅ Model created")
    print(f"   Original params: {prune_stats['original_parameters']:,}")
    print(f"   Pruned params: {prune_stats['pruned_parameters']:,}")
    print(f"   Reduction: {prune_stats['percentage_reduction']:.2f}%")
    
    stats = {
        "source": "on_the_fly_pruning",
        "pruning_stats": prune_stats,
        **get_model_stats(pruned_model)
    }
    
    return pruned_model, tokenizer, stats


def format_results_table(results_dict):
    """
    Format evaluation results as a pretty table for display.
    
    Args:
        results_dict (dict): Results from run_robust_evaluation()
        
    Returns:
        str: Formatted table string
    """
    import pandas as pd
    
    # Flatten nested metrics
    rows = []
    for task_name, metrics in results_dict.items():
        row = {"task": task_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_string(index=False)
