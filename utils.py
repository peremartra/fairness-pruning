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
    import transformers
    import optipfair
    # Additional imports for carbon profiling
    import time
    import numpy as np
    from datetime import datetime
    import codecarbon
except ImportError as e:
    raise ImportError(
        f"Missing required library: {e.name}\n"
        "Install all dependencies with:\n"
        "  pip install optipfair lm-eval transformers torch langdetect"
    )

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

MODELS_CONFIG = [
    {
        "base_model": "meta-llama/Llama-3.2-1B",
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
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
    {"name": "belebele_spa_eur", "num_fewshot": 0},
    {"name": "xcopa_es", "num_fewshot": 0},
    {"name": "mmlu_es", "num_fewshot": 5},
    {"name": "wikitext", "num_fewshot": 0},
    {"name": "boolq", "num_fewshot": 0},
    {"name": "lambada_openai", "num_fewshot": 0},
    {"name": "mmlu", "num_fewshot": 5},  # Standard is 5-shot for MMLU
    {"name": "arc_challenge", "num_fewshot": 0},
    {"name": "hellaswag", "num_fewshot": 0},
    {"name": "winogrande", "num_fewshot": 0},
    {"name": "piqa", "num_fewshot": 0},
    {"name": "truthfulqa_mc1", "num_fewshot": 0},
    {"name": "truthfulqa_mc2", "num_fewshot": 0},
    {"name": "gsm8k", "num_fewshot": 5},  # Chain-of-thought requires few-shot
    {"name": "ifeval", "num_fewshot": 0},
    {"name": "leaderboard_musr", "num_fewshot": 0},
]


# =============================================================================
# CARBON PROFILING CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG_CARBON = [
    # -------------------------------------------------------------------------
    # Llama-3.2-1B: Baseline + Star Model Only
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 0,
        "hf_repo_id": None,  # Baseline
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 10,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-10pct",
        "is_star": False, 
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 20,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-20pct",
        "is_star": False, 
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 30,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-30pct",
        "is_star": False, 
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 40,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-40pct",
        "is_star": True,  # Star model (140% expansion)
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 50,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-50pct",
        "is_star": False, 
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 60,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-60pct",
        "is_star": False, 
    },
    
    # -------------------------------------------------------------------------
    # Llama-3.2-3B: Baseline + Star Model Only
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 0,
        "hf_repo_id": None,  # Baseline
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 10,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-10pct",
        "is_star": True,  # Star model
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 20,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-20pct",
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 30,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-30pct",
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 40,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-40pct",
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 50,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-50pct",
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 60,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-60pct",
        "is_star": False,
    },
    
    
    # -------------------------------------------------------------------------
    # Llama-3.2-1B-Instruct: Baseline + Star Model Only
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "pruning_pct": 0,
        "hf_repo_id": None,  # Baseline
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "pruning_pct": 40,
        "hf_repo_id": "oopere/Llama-3.2-1B-I-pruned-40pct",
        "is_star": True,  # Star model
    },
]

BENCHMARKS_CARBON = [
    {
        "name": "hellaswag_latency_b1",
        "num_prompts": 100,
        "max_new_tokens": 20,
        "dataset": "hellaswag",
        "subset": "validation",  # ← Importante
        "description": "Short responses (Latency, TTFT, bsz=1)",
        "batch_size": 1
    },
    {
        "name": "mmlu_latency_b1",
        "num_prompts": 100,
        "max_new_tokens": 50,
        "dataset": "mmlu",
        "subset": "test",
        "description": "Knowledge QA (Latency, TTFT, bsz=1)",
        "batch_size": 1
    },
    {
        "name": "ifeval_latency_b1",
        "num_prompts": 30,  
        "max_new_tokens": 150, 
        "dataset": "IFEval",
        "subset": "train", 
        "description": "Instruction (Latency, TTFT, bsz=1)", 
        "batch_size": 1
    },
    {
        "name": "hellaswag_throughput_b8",
        "num_prompts": 100,
        "max_new_tokens": 20,
        "dataset": "hellaswag",
        "subset": "validation",
        "description": "Short responses (Throughput, bsz=8)",
        "batch_size": 8
    },
    {
        "name": "mmlu_throughput_b8",
        "num_prompts": 100, 
        "max_new_tokens": 50,
        "dataset": "mmlu",
        "subset": "test",
        "description": "Knowledge QA (Throughput, bsz=8)",
        "batch_size": 8
    },
    {
        "name": "ifeval_throughput_b8",
        "num_prompts": 30,  
        "max_new_tokens": 150, 
        "dataset": "IFEval",
        "subset": "train", 
        "description": "Instruction (Throughput, bsz=8)",
        "batch_size": 8
    },
]

# Instruct models additional benchmark (+1)
BENCHMARKS_INSTRUCT = [
    {"name": "leaderboard_musr", "num_fewshot": 0},  
    {"name": "truthfulqa_mc1", "num_fewshot": 0},
    {"name": "truthfulqa_mc2", "num_fewshot": 0},
    {"name": "lambada_openai", "num_fewshot": 0},
    {"name": "mmlu", "num_fewshot": 5},  # Standard is 5-shot for MMLU
    {"name": "gsm8k", "num_fewshot": 5},  # Chain-of-thought requires few-shot
    {"name": "ifeval", "num_fewshot": 0},
]


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

def calibrate_idle_power(device="cuda", duration_seconds=30, verbose=True):
    """
    Measure GPU idle power consumption to establish baseline.
    
    This should be run ONCE at the start of the notebook, before any model loading.
    
    Args:
        device (str): Device to calibrate ("cuda" or "cpu")
        duration_seconds (int): How long to measure (30s recommended)
        verbose (bool): Print progress messages
    
    Returns:
        dict: {
            "idle_power_watts": float,
            "idle_energy_kwh": float,
            "measurement_duration_s": float,
            "gpu_temp_celsius": float,
            "gpu_name": str,
            "timestamp": str
        }
    """
    import torch
    import time
    from codecarbon import EmissionsTracker
    from datetime import datetime
    
    if verbose:
        print(f"🔋 Starting idle power calibration ({duration_seconds}s)...")
        print(f"   Clearing GPU cache...")
    
    # Clear GPU to true idle state
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize tracker
    tracker = EmissionsTracker(
        project_name="idle_calibration",
        measure_power_secs=1,  # Sample every second
        save_to_file=False,
        log_level="error"  # Suppress warnings
    )
    
    # Measure idle consumption
    tracker.start()
    start_time = time.time()
    
    # Just wait (GPU should be completely idle)
    if verbose:
        print(f"   Measuring idle power for {duration_seconds}s...")
    time.sleep(duration_seconds)
    
    emissions_kwh = tracker.stop()
    actual_duration = time.time() - start_time
    
    # Calculate average power
    idle_power_watts = (emissions_kwh * 1000) / (actual_duration / 3600)  # kWh -> W
    
    # Capture GPU state
    gpu_info = {}
    if device == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_temp_celsius": torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else None,
            "gpu_power_limit_w": torch.cuda.get_device_properties(0).total_memory / (1024**3) * 10  # Rough estimate
        }
    
    calibration_result = {
        "idle_power_watts": idle_power_watts,
        "idle_energy_kwh": emissions_kwh,
        "measurement_duration_s": actual_duration,
        "timestamp": datetime.now().isoformat(),
        **gpu_info
    }
    
    if verbose:
        print(f"✅ Calibration complete!")
        print(f"   Idle Power: {idle_power_watts:.2f} W")
        print(f"   Idle Energy (30s): {emissions_kwh:.6f} kWh")
        if gpu_info.get("gpu_temp_celsius"):
            print(f"   GPU Temperature: {gpu_info['gpu_temp_celsius']:.1f}°C")
    
    return calibration_result

def model_evaluation(model_obj, tokenizer, tasks, limit=None):
    """
    Runs lm-eval on a model and tokenizer already in memory.
    
    Args:
        model_obj: PyTorch model object to evaluate
        tokenizer: Tokenizer object for the model
        tasks (list): List of task dicts with format:
                     [{"name": "wikitext", "num_fewshot": 0}, ...]
                     OR simple list of strings: ["wikitext", "boolq"]
        limit (int, optional): Number of samples per task for quick testing
        
    Returns:
        dict: Formatted results with metrics per task
        
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
        tokenizer=tokenizer,
        device=str(DEVICE)
    )
    
    # Run evaluation with per-task few-shot configuration
    # Note: lm-eval handles num_fewshot per task if tasks are configured properly
    if len(set(task_fewshot_map.values())) == 1:
        # Todas las tareas usan el mismo num_fewshot
        fewshot_value = list(task_fewshot_map.values())[0]
    else:
        # Múltiples valores diferentes (no debería pasar, pero por si acaso)
        fewshot_value = 0
        
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=task_names,
        num_fewshot=fewshot_value,  # Let task configs handle this
        batch_size="auto",
        limit=limit,
        device=str(DEVICE)
    )
    
    # Format results for clean display
    formatted_results = {}
    for task_name, res in results["results"].items():
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


def _measure_inference_performance(model, tokenizer, prompts, max_new_tokens, batch_size, device="cuda", random_seed=None):

    """
    Measure inference performance with warm-up period.
    
    Handles two modes:
    1. batch_size = 1: Measures TTFT (Time To First Token) and latency.
    2. batch_size > 1: Measures batched throughput. TTFT metrics will be None.
    
    The first 5 prompts/batches are used for GPU warm-up and excluded from metrics.
    """
    if batch_size == 1:
        total_iterations = len(prompts)
    else:
        total_iterations = (len(prompts) + batch_size - 1) // batch_size  # Ceiling division

    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            
    WARMUP_STEPS = min(5, max(1, int(total_iterations * 0.2)))
    print(f"   Using {WARMUP_STEPS} warmup iterations (out of {total_iterations} total)")
    
    ttft_times_ms = []
    total_new_tokens = 0
    total_inference_time_sec = 0
    
    # Helper para crear lotes
    def create_batches(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    # --- LÓGICA PARA BATCH SIZE = 1 (LATENCIA / TTFT) ---
    if batch_size == 1:
        print(f"   Running in LATENCY mode (bsz=1). Measuring TTFT...")
        loop_start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            gen_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_time_sec = time.time() - gen_start
            
            # Solo contar métricas después del calentamiento
            if i >= WARMUP_STEPS:
                ttft_times_ms.append(gen_time_sec * 1000) # TTFT es el tiempo total de generación para bsz=1
                total_new_tokens += (outputs.shape[1] - inputs.input_ids.shape[1])
                total_inference_time_sec += gen_time_sec
        
        total_time_sec = time.time() - loop_start_time # Tiempo total del bucle
        num_measured_prompts = len(ttft_times_ms)
        
        return {
            "mode": "latency",
            "avg_ttft_ms": float(np.mean(ttft_times_ms)) if num_measured_prompts > 0 else None,
            "std_ttft_ms": float(np.std(ttft_times_ms)) if num_measured_prompts > 0 else None,
            "throughput_tokens_per_sec": float(total_new_tokens / total_inference_time_sec) if total_inference_time_sec > 0 else 0.0,
            "total_loop_time_sec": float(total_time_sec),
            "total_new_tokens": int(total_new_tokens),
            "avg_new_tokens_per_prompt": float(total_new_tokens / num_measured_prompts) if num_measured_prompts > 0 else 0.0,
            "num_measured_prompts": num_measured_prompts,
            "num_warmup_prompts": WARMUP_STEPS
        }

    # --- LÓGICA PARA BATCH SIZE > 1 (THROUGHPUT) ---
    else:
        print(f"   Running in THROUGHPUT mode (bsz={batch_size}). TTFT will not be measured.")
        # El tokenizer necesita padding para los lotes
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left" # Para decodificación
        
        prompt_batches = list(create_batches(prompts, batch_size))
        loop_start_time = time.time()
        
        for i, batch_prompts in enumerate(prompt_batches):
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512 # Añadir max_length para seguridad
            ).to(device)
            
            gen_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_time_sec = time.time() - gen_start
            
            if i >= WARMUP_STEPS:
                total_inference_time_sec += gen_time_sec
                
                # Contar TODOS los tokens generados (outputs - inputs)
                input_length = inputs.input_ids.shape[1]
                num_new_tokens_in_batch = (outputs.shape[1] - input_length) * len(batch_prompts)
                total_new_tokens += num_new_tokens_in_batch
                
                ttft_times_ms.append(gen_time_sec * 1000)
                    
                total_time_sec = time.time() - loop_start_time
                num_measured_batches = len(ttft_times_ms)
                num_measured_prompts = num_measured_batches * batch_size

        return {
            "mode": "throughput",
            "avg_ttft_ms": None, 
            "std_ttft_ms": None,
            "avg_batch_time_ms": float(np.mean(ttft_times_ms)) if num_measured_batches > 0 else None,
            "throughput_tokens_per_sec": float(total_new_tokens / total_inference_time_sec) if total_inference_time_sec > 0 else 0.0,
            "total_loop_time_sec": float(total_time_sec),
            "total_new_tokens": int(total_new_tokens),
            "avg_new_tokens_per_prompt": float(total_new_tokens / num_measured_prompts) if num_measured_prompts > 0 else 0.0,
            "num_measured_prompts": num_measured_prompts,
            "num_warmup_batches": WARMUP_STEPS
        }


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
                limit=None
            )
            
            # Store result in checkpoint
            checkpoint["results"][task_name] = result[task_name]
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

# =============================================================================
# CARBON PROFILING MAIN FUNCTION
# =============================================================================
def _capture_hardware_metadata(tracker, device="cuda", idle_power_watts=None, inference_duration_s=None, energy_raw_kwh=None, energy_net_kwh=None):
    """
    Helper function to capture hardware metadata from CodeCarbon tracker.
    Extended to include GPU state, energy breakdown, and idle power correction.
    
    Args:
        tracker: CodeCarbon EmissionsTracker instance (after stop())
        device (str): Device used ("cuda" or "cpu")
        idle_power_watts (float, optional): Calibrated idle power for correction
        inference_duration_s (float, optional): Actual inference time
        energy_raw_kwh (float, optional): Raw energy before idle correction
        energy_net_kwh (float, optional): Net energy after idle correction
    
    Returns:
        dict: Comprehensive hardware and energy metadata
    """
    import torch
    import codecarbon
    from datetime import datetime
    
    # ============================================================
    # CODECARBON METADATA (original logic)
    # ============================================================
    try:
        gpu_model = tracker._gpu.model_
        gpu_count = tracker._gpu.gpu_count_
        gpu_power_W = tracker._gpu.power_
    except Exception:
        gpu_model = "N/A"
        gpu_count = 0
        gpu_power_W = "N/A"

    try:
        cpu_model = tracker._cpu.model_
        cpu_count = tracker._cpu.cpu_count_
        cpu_power_W = tracker._cpu.power_
    except Exception:
        cpu_model = "N/A"
        cpu_count = 0
        cpu_power_W = "N/A"
        
    try:
        location = tracker.location_
        country_name = location['country_name'] if location else "N/A"
        country_iso = location['country_iso_code'] if location else "N/A"
        region = location['region'] if location else "N/A"
        cloud_provider = location['cloud_provider'] if location else "N/A"
        cloud_region = location['cloud_region'] if location else "N/A"
    except Exception:
        country_name = "N/A"
        country_iso = "N/A"
        region = "N/A"
        cloud_provider = "N/A"
        cloud_region = "N/A"

    base_metadata = {
        "timestamp": getattr(tracker, 'timestamp_', datetime.now().isoformat()),
        "project_name": getattr(tracker, 'project_name_', "N/A"),
        "duration_sec": getattr(tracker, 'duration_', inference_duration_s or "N/A"),
        "energy_kwh": getattr(tracker, 'energy_consumed_', energy_net_kwh or "N/A"),
        "co2_g": getattr(tracker, 'emissions_', "N/A"),
        "carbon_intensity_gCO2_kWh": getattr(tracker, 'carbon_intensity_', "N/A"),
        "country_name": country_name,
        "country_iso_code": country_iso,
        "region": region,
        "cloud_provider": cloud_provider,
        "cloud_region": cloud_region,
        "os": getattr(tracker, 'os_', "N/A"),
        "python_version": getattr(tracker, 'python_version_', "N/A"),
        "codecarbon_version": getattr(codecarbon, '__version__', "N/A"),
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "cpu_power_usage_W": cpu_power_W,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "gpu_power_usage_W": gpu_power_W
    }
    
    # ============================================================
    # EXTENDED METADATA (new functionality)
    # ============================================================
    extended_metadata = {}
    
    # GPU state information (PyTorch)
    if device == "cuda" and torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            extended_metadata.update({
                "gpu_name_torch": torch.cuda.get_device_name(0),
                "gpu_total_memory_gb": gpu_props.total_memory / (1024**3),
                "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            })
            
            # Try to get temperature (may not work on all GPUs/environments)
            try:
                extended_metadata["gpu_temperature_celsius"] = torch.cuda.temperature()
            except:
                extended_metadata["gpu_temperature_celsius"] = None
                
        except Exception as e:
            extended_metadata["gpu_metadata_error"] = str(e)
    
    # Energy breakdown (if idle correction was applied)
    if idle_power_watts is not None and inference_duration_s is not None:
        idle_energy_kwh = (idle_power_watts * inference_duration_s) / (1000 * 3600)
        extended_metadata.update({
            "idle_power_watts": idle_power_watts,
            "idle_power_applied": True,
            "energy_raw_kwh": energy_raw_kwh if energy_raw_kwh is not None else "N/A",
            "energy_idle_contribution_kwh": idle_energy_kwh,
            "energy_net_kwh": energy_net_kwh if energy_net_kwh is not None else "N/A",
            "idle_correction_note": "Net energy = Raw energy - (Idle power × Duration)"
        })
    else:
        extended_metadata.update({
            "idle_power_applied": False,
            "idle_correction_note": "No idle power correction applied (calibration not provided)"
        })
    
    # CodeCarbon overhead documentation
    extended_metadata["codecarbon_overhead_estimated_pct"] = 2.5  # Conservative estimate
    extended_metadata["pue_assumption"] = "unknown (datacenter PUE not measured)"
    
    # Try to extract CodeCarbon's internal energy breakdown (CPU vs GPU)
    try:
        if hasattr(tracker, '_total_energy'):
            extended_metadata["codecarbon_total_energy_internal_kwh"] = tracker._total_energy.kWh
        if hasattr(tracker, '_total_cpu_energy'):
            extended_metadata["codecarbon_cpu_energy_kwh"] = tracker._total_cpu_energy.kWh
        if hasattr(tracker, '_total_gpu_energy'):
            extended_metadata["codecarbon_gpu_energy_kwh"] = tracker._total_gpu_energy.kWh
    except:
        pass  # Not critical if these attributes don't exist
    
    # Merge base and extended metadata
    return {**base_metadata, **extended_metadata}

def run_carbon_profiling(
    model,
    tokenizer,
    workloads,
    checkpoint_path,
    model_name="unknown",
    idle_power_calibration=None,
    device="cuda",
    random_seed=None  
):
    """
    Run carbon and performance profiling on a model (parallel to run_robust_evaluation).
    
    Measures:
        - Energy consumption (CodeCarbon)
        - Throughput (tokens/second)
        - Latency (Time To First Token)
        - Memory footprint
        - Detailed hardware & emissions metadata
    
    Uses the same checkpoint/resume system as run_robust_evaluation for reliability.
    
    Args:
        model: PyTorch model object
        tokenizer: Tokenizer for the model
        workloads (list): List of workload dicts from BENCHMARKS_CARBON
        checkpoint_path (str): Path to checkpoint JSON file
        model_name (str, optional): Human-readable model name
        device (str): Device placement ("cuda" or "cpu")
    
    Returns:
        dict: Complete profiling results with all metrics
    """
    import json
    import os
    from codecarbon import EmissionsTracker
    from datetime import datetime
    
    # Extract model name
    if model_name is None:
        model_name = getattr(model.config, '_name_or_path', 'unknown')
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load or create checkpoint
    if os.path.exists(checkpoint_path):
        print(f"📂 Found existing checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        if "results" not in checkpoint:
            checkpoint["results"] = {}
        if "metadata" not in checkpoint:
            checkpoint["metadata"] = {
                "model_name": model_name,
                "started_at": datetime.now().isoformat(),
                "mode": "carbon_profiling"
            }
        if "completed_workloads" not in checkpoint:
            checkpoint["completed_workloads"] = []
        if "failed_workloads" not in checkpoint:
            checkpoint["failed_workloads"] = []
    else:
        print(f"🆕 Creating new checkpoint: {checkpoint_path}")
        checkpoint = {
            "metadata": {
                "model_name": model_name,
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "mode": "carbon_profiling"
            },
            "results": {},
            "completed_workloads": [],
            "failed_workloads": []
        }
    
    completed = checkpoint.get("completed_workloads", [])
    failed = checkpoint.get("failed_workloads", [])
    
    # Determine pending workloads
    pending = [w for w in workloads 
               if w["name"] not in completed and w["name"] not in failed]
    
    print(f"✅ Loaded checkpoint. Completed: {len(completed)}/{len(workloads)} workloads")
    if failed:
        print(f"⚠️ Previously failed: {failed}")
    
    if not pending:
        print("🎉 All workloads completed!")
        return checkpoint["results"]
    
    # Validate idle power calibration
    if idle_power_calibration is None:
        print("⚠️  WARNING: No idle power calibration provided!")
        print("   Energy measurements will include GPU idle consumption.")
        print("   Run calibrate_idle_power() before evaluation for accurate results.\n")
        idle_power_watts = 0.0  # No correction
    else:
        idle_power_watts = idle_power_calibration.get("idle_power_watts", 0.0)
        print(f"✅ Using idle power calibration: {idle_power_watts:.2f} W")
        print(f"   (Measured at: {idle_power_calibration.get('timestamp', 'unknown')})\n")
    
    print(f"\n🚀 Starting profiling: {len(pending)} workloads remaining")
    print("="*70 + "\n")
    
    
    # Process each workload
    for i, workload in enumerate(pending, 1):
        workload_name = workload["name"]
        
        print(f"[{i}/{len(pending)}] Profiling: {workload_name}")
        print("-"*70)
        print("   Clearing GPU cache...")
        clear_gpu_cache()
        
        # ==========================================================
        # CODECARBON WARM UP
        # ==========================================================
        print("   Running GPU warm-up for CodeCarbon sensors...")
        tracker = EmissionsTracker(
            project_name=f"glu_pruning_{model_name}",
            measure_power_secs=1,
            save_to_file=False,
            log_level="warning"
        )
        
        try:
            dummy_prompt = "Warm-up GPU"
            dummy_inputs = tokenizer(dummy_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model.generate(**dummy_inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            print("   GPU warm-up complete.")
        except Exception as warmup_e:
            print(f"   ⚠️ GPU warm-up failed (continuing anyway): {warmup_e}")
        
        # ==========================================================
        # MAIN PROFILING LOGIC
        # ==========================================================
        workload_success = False
        
        try:
            # Start tracker
            import time
            inference_start_time = time.time()  # ← AÑADIR ESTA LÍNEA
            tracker.start()
            
            # 
            workload_with_seed = workload.copy()
            if random_seed is not None:
                workload_with_seed["random_seed"] = random_seed
            
            # Load prompts
            print(f"   Loading {workload['num_prompts']} prompts from {workload['dataset']}...")
            if random_seed is not None:
                print(f"   Using random seed: {random_seed} for reproducible sampling")
            prompts = _load_workload_prompts(workload_with_seed)  # ← CAMBIO AQUÍ
            
            # Measure inference performance
            print(f"   Running inference...")
            batch_size = workload.get("batch_size", 1)
            print(f"   (Batch Size: {batch_size})")
            
            perf_metrics = _measure_inference_performance(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=workload["max_new_tokens"],
                batch_size=batch_size,
                device=device,
                random_seed=random_seed  
            )
            
            # Stop tracker (SOLO si llegamos aquí sin excepciones)
            emissions_raw = tracker.stop()
            inference_duration_s = time.time() - inference_start_time
            print(f"   Tracker stopped. Raw emissions: {emissions_raw:.6f} kWh")
            
            # Calculate net energy (subtract idle power)
            idle_energy_kwh = (idle_power_watts * inference_duration_s) / (1000 * 3600)
            emissions_net = max(0.0, emissions_raw - idle_energy_kwh)  # Never negative
            
            print(f"   Net emissions (idle-corrected): {emissions_net:.6f} kWh")
            print(f"   (Idle contribution removed: {idle_energy_kwh:.6f} kWh)")
            
            # Capturar metadata de hardware (ahora con parámetros extendidos)
            hardware_metadata = _capture_hardware_metadata(
                tracker=tracker,
                device=device,
                idle_power_watts=idle_power_watts,
                inference_duration_s=inference_duration_s,
                energy_raw_kwh=emissions_raw,
                energy_net_kwh=emissions_net
            )
            
            # Get memory stats
            memory_stats = _get_memory_stats(model, device)
            
            # --- Joules por Token ---
            total_new_tokens = perf_metrics.get("total_new_tokens", 0)
            joules_per_token = 0.0
            if total_new_tokens > 0:
                # 1 kWh = 3,600,000 Joules
                total_joules = emissions_net * 3_600_000
                joules_per_token = total_joules / total_new_tokens

            # Consolidate results
            result = {
                **perf_metrics,
                **memory_stats,
                "energy_kwh": float(emissions_net),
                "energy_raw_kwh": float(emissions_raw),
                "energy_idle_kwh": float(idle_energy_kwh),
                "joules_per_token": float(joules_per_token), # <-- MÉTRICA AÑADIDA
                "joules_per_token_calculation": {
                "total_joules": float(emissions_net * 3_600_000),
                    "total_tokens": total_new_tokens,
                    "formula": "total_joules / total_tokens"
                },
                "hardware_metadata": hardware_metadata,
                "batch_size": workload.get("batch_size", 1),
                "num_prompts": len(prompts),
                "max_new_tokens": workload["max_new_tokens"],
                "workload_description": workload.get("description", "")
            }
            
            # Save to checkpoint
            checkpoint["results"][workload_name] = result
            checkpoint["completed_workloads"].append(workload_name)
            workload_success = True
            
            # Print summary
            print(f"✅ {workload_name} completed")
            print(f"   Energy: {result['energy_kwh']:.6f} kWh")
            print(f"   Throughput: {result['throughput_tokens_per_sec']:.2f} tok/s")
            if result.get("avg_ttft_ms"):
                print(f"   Avg TTFT: {result['avg_ttft_ms']:.2f} ms")
            if result.get("avg_batch_time_ms"):
                print(f"   Avg Batch Time: {result['avg_batch_time_ms']:.2f} ms")
            print(f"   Memory: {result['model_size_gb']:.2f} GB\n")
            
        except Exception as e:
            print(f"❌ {workload_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            checkpoint["failed_workloads"].append(workload_name)
            workload_success = False
            
            # Intentar detener el tracker si está corriendo
            try:
                if hasattr(tracker, '_scheduler') and tracker._scheduler:
                    tracker.stop()
                    print("   Tracker stopped after error.")
            except:
                pass
        
        finally:
            # Siempre actualizar checkpoint (incluso si falló)
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            if not workload_success:
                print("   Continuing with next workload...\n")
    
    print("="*70)
    print("🎉 ALL WORKLOADS COMPLETED!")
    if checkpoint.get("failed_workloads"):
        print(f"⚠️ Some workloads failed: {checkpoint['failed_workloads']}")
    print("="*70 + "\n")
    
    return checkpoint["results"]
