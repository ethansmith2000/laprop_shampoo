import os
import tempfile
import inspect as _inspect
from pathlib import Path

# Scratch root for HF caches and compiler temp dirs. Override with SCRATCH_ROOT
# to point at a fast local disk or a shared cluster path.
_SCRATCH = os.environ.setdefault("SCRATCH_ROOT", "/workspace/.scratch")

# Ensure Hugging Face datasets cache is set early to a user-writable path.
os.environ.setdefault("HF_HOME", os.path.join(_SCRATCH, "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

# Make Triton and TorchInductor use stable, user-writable cache/tmp dirs
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_SCRATCH, "triton"))
os.environ.setdefault("TMPDIR", os.path.join(_SCRATCH, "tmp"))
for _d in (os.environ["HF_HOME"], os.environ["HF_DATASETS_CACHE"],
           os.environ["TRITON_CACHE_DIR"], os.environ["TMPDIR"]):
    os.makedirs(_d, exist_ok=True)
tempfile.tempdir = os.environ["TMPDIR"]

# When using datasets.map(num_proc=...) we rely on multiprocessing for parallelism,
# so we disable internal tokenizer threading to avoid oversubscription.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _safe_getsourcelines(obj):
    """
    Work around Triton+Python 3.11 \"source code not available\" issues by
    returning a dummy source snippet instead of raising, so torch.compile
    can proceed. This is only meant for tooling and does not affect numerics.
    """
    try:
        return _inspect._orig_getsourcelines(obj)  # type: ignore[attr-defined]
    except OSError as e:
        if "source code not available" in str(e):
            return ["# source code not available\n"], 0
        raise


if not hasattr(_inspect, "_orig_getsourcelines"):
    _inspect._orig_getsourcelines = _inspect.getsourcelines  # type: ignore[attr-defined]
    _inspect.getsourcelines = _safe_getsourcelines


import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.v8_api_enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # 
torch.backends.cuda.allow_tensor_float_32 = True


# torch._inductor.config.triton.cudagraphs = True   # or False if capture causes overhead
# torch._inductor.config.use_mixed_mm = True        # enables faster matmul codegen
# torch._inductor.config.triton.cudagraphs = True
# torch._inductor.config.triton.cudagraph_trees = True  # More aggressive cudagraph usage
# torch._inductor.config.triton.autotune_pointwise = True
# # torch._inductor.config.triton.dense_indexing = True
# torch._inductor.config.triton.max_tiles = 8  # Increase tiling options
# torch._inductor.config.aggressive_fusion = True
# torch._inductor.config.pattern_matcher = True
# torch._inductor.config.permute_fusion = True
# torch._inductor.config.max_autotune = True
# torch._inductor.config.max_autotune_gemm = True

torch.set_num_threads(12)
torch.set_num_interop_threads(2)

# torch._inductor.config.autotune_in_subproc = True            # instead of exporting TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC
# torch._inductor.config.autotune_multi_device = True          # mirrors TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE
# # torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"



import argparse
import copy
import json
import logging
import math
import os
import sys
from itertools import chain
from typing import Dict, Any

import datasets
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from types import SimpleNamespace

import torch
import time
from transformer import Transformer
from ordered_shampoo import OrderedShampoo

def warmup_to_steps(warmup_value, total_steps):
    """
    Convert warmup config to actual steps.
    
    Args:
        warmup_value: None (no warmup), 0-1 (ratio of total_steps), or >1 (direct step count)
        total_steps: Total training steps for ratio calculation
    
    Returns:
        int or None: Warmup steps, or None if no warmup
    """
    if warmup_value is None:
        return None
    if warmup_value <= 1.0:
        # Interpret as ratio
        return int(warmup_value * total_steps)
    else:
        # Interpret as direct step count
        return int(warmup_value)

logger = get_logger(__name__)


def _load_overrides():
    """
    Lightweight override loader to keep the script largely argument-free.
    Priority: CLI flag --override_json, then env OVERRIDE_JSON.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--override_json", type=str, default=os.environ.get("OVERRIDE_JSON"))
    parsed, _ = parser.parse_known_args()
    override_path = parsed.override_json
    if override_path:
        with open(override_path, "r") as f:
            data = json.load(f)
        # Stash metadata so downstream code can use the sweep entry name.
        data["_override_json_path"] = override_path
        data["_override_name"] = Path(override_path).stem
        return data
    return {}


def _build_activation_probe_batch(batch, limit):
    """Return a shallow copy of `batch` with tensors truncated along batch dim."""
    if limit is None:
        return batch
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) > limit:
            sliced[key] = value[:limit]
        else:
            sliced[key] = value
    return sliced


logger = get_logger(__name__)


def _compact_kwargs(kwargs: Dict[str, Any]) -> str:
    """Return a short, deterministic string for kwargs."""
    if not kwargs:
        return ""
    bits = []
    for k in sorted(kwargs.keys()):
        bits.append(f"{k}{kwargs[k]}")
    return "-".join(bits)


class SyntheticTokenDataset(Dataset):
    """Small deterministic token dataset for offline trainer smoke tests."""

    def __init__(self, num_samples: int, block_size: int, vocab_size: int, seed: int):
        if num_samples < 1:
            raise ValueError("synthetic_num_samples must be >= 1")
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        if vocab_size < 2:
            raise ValueError("synthetic_vocab_size must be >= 2")
        generator = torch.Generator().manual_seed(int(seed))
        self.input_ids = torch.randint(
            0,
            int(vocab_size),
            (int(num_samples), int(block_size)),
            generator=generator,
        )

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, index):
        # No labels column: targets are input_ids shifted by one, produced in
        # the training loop. Storing them would double the bytes moved.
        return {"input_ids": self.input_ids[index]}


class CSVLogger:
    """Append-only CSV sink with a fixed header discovered from the first row.

    Metrics arrive from several call sites with different key sets, so unknown
    keys seen after the header is written are dropped rather than silently
    shifting columns; log the union on the first call if you need them all.
    """

    def __init__(self, path, flush_every: int = 100):
        self.path = path
        self.flush_every = max(1, int(flush_every))
        self._fields = None
        self._pending = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._writer = None

    def log(self, metrics: Dict[str, Any]):
        import csv as _csv

        if self._fields is None:
            self._fields = ["step"] + sorted(k for k in metrics if k != "step")
            self._writer = _csv.DictWriter(
                self._fh, fieldnames=self._fields, extrasaction="ignore"
            )
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._pending += 1
        if self._pending >= self.flush_every:
            self._fh.flush()
            self._pending = 0

    def close(self):
        if self._fh is not None and not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def optimizer_state_bytes(optimizer) -> int:
    """Total bytes held in optimizer state tensors.

    This is the headline memory number for the AdamW-vs-Shampoo comparison:
    Adam keeps 2 param-sized buffers, Shampoo additionally keeps Kronecker
    factors and their inverse roots, which dominate for wide layers.
    """
    opt = getattr(optimizer, "optimizer", optimizer)  # unwrap Accelerate wrapper
    total = 0
    seen = set()

    def _walk(obj):
        nonlocal total
        if torch.is_tensor(obj):
            if obj.data_ptr() not in seen:
                seen.add(obj.data_ptr())
                total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(opt.state)
    return total


def build_optimizer(args, grouped_parameters):
    """Construct the optimizer named by args.mode.

    The three headline arms are 'adamw', 'shampoo' (momentum on the raw
    gradient, then whiten) and 'laprop_shampoo' (whiten, then momentum).
    'adopt_shampoo' is the stale-preconditioner variant.
    """
    mode = args.mode
    betas = (args.beta1, args.beta2)

    if mode in ("adamw", "base"):
        return torch.optim.AdamW(
            grouped_parameters,
            lr=args.learning_rate,
            betas=betas,
            eps=1e-8,
            weight_decay=args.weight_decay,
            fused=True,
        )

    shampoo_orders = {
        "shampoo": "post",
        "laprop_shampoo": "pre",
        "adopt_shampoo": "pre_stale",
    }
    if mode in shampoo_orders:
        return OrderedShampoo(
            grouped_parameters,
            lr=args.learning_rate,
            betas=betas,
            eps=1e-8,
            weight_decay=args.weight_decay,
            order=shampoo_orders[mode],
            mode="shampoo",
            precondition_frequency=args.precondition_frequency,
            start_preconditioning_step=args.start_preconditioning_step,
            max_precond_dim=args.max_precond_dim,
            graft=args.graft,
            nesterov=args.nesterov,
            one_sided=args.one_sided,
            root_method=args.root_method,
            ndb_iters=args.ndb_iters,
            ndb_damp=args.ndb_damp,
        )

    raise ValueError(
        f"Unknown optimizer mode: {mode}. Choose from: "
        f"adamw, {', '.join(shampoo_orders)}"
    )


def build_run_name(args: Dict[str, Any]) -> str:
    """Compact, deterministic name for manual (non-sweep) runs."""
    bits = [
        str(args["mode"]),
        f"lr{args['learning_rate']:g}",
        f"wd{args['weight_decay']:g}",
        f"b{args['beta1']:g}-{args['beta2']:g}",
        f"d{args['hidden_size']}x{args['depth']}",
        f"bs{args['per_device_train_batch_size']}",
    ]
    if args["mode"] not in ("adamw", "base"):
        bits.append(f"pf{args['precondition_frequency']}")
        bits.append(f"graft-{args['graft']}")
        bits.append("1sided" if args["one_sided"] else "2sided")
        if args["root_method"] != "eigh":
            bits.append(args["root_method"])
    return "_".join(bits)


def main():
    override_args = _load_overrides()

    args = {
        "num_validation_batches": 25,
        "validate_every": 1000,
        # `huggingface_hub` now enforces `namespace/name` for `hf://...` URIs.
        # The OpenWebText dataset is hosted at `Skylion007/openwebtext`.
        "dataset_name": "Skylion007/openwebtext",
        "dataset_config_name": None,
        "synthetic_num_samples": 16,
        "synthetic_validation_samples": 4,
        "synthetic_vocab_size": 128,
        # "dataset_name": "wikitext",
        # "dataset_config_name": "wikitext-103-v1",
        "train_file": None,
        "validation_file": None,
        "validation_split_percentage": 5,
        "model_name_or_path": "openai-community/gpt2-medium",
        # "model_name_or_path": "openai-community/gpt2",
        "config_name": None,
        "tokenizer_name": None,
        "use_slow_tokenizer": False,
        "per_device_train_batch_size": 32,

        "num_train_epochs": 2,
        # "max_train_steps": 500_000,
        "max_train_steps": 100_000,
        # "max_train_steps": 125_000,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 100,
        "seed": 123,
        "model_type": None,
        "block_size": 1024,
        "preprocessing_num_workers": 180,
        "overwrite_cache": False,
        "no_keep_linebreaks": False,
        "trust_remote_code": False,
        "checkpointing_steps": None,
        "resume_from_checkpoint": None,
        "save_final_model": False,  # paper runs evaluate live; skip ~1GB weight dumps
        "with_tracking": True,
        "report_to": "wandb",
        "low_cpu_mem_usage": False,
        "max_grad_norm": 1.0,
        "hf_path": None,
        "base_output_dir": "model-output",



        "compile": True,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,

        "gradient_checkpointing": True,

        # Only deepcopy a second model when explicitly enabled (saves a full
        # model worth of VRAM on the default path).
        "activation_probe": False,

        "num_workers": 8,
        "prefetch_factor": 2,
        "profile_timers": False,
        # When profile_timers is on, record CUDA-event breakdown every N optimizer steps
        # (not every step — per-step synchronize destroys throughput).
        "profile_every": 10,
        "log_every_n": 10,
        "csv_logging": False,
        "csv_flush_every": 100,

        "log_params_every_n": 100,

        # optimizer stats logging
        "track_stats_every_n": 200,
        
        "diagnostics_every": 50,
        # torch.compile SNRAdam.step (default on when CUDA is available)
        "compile_optimizer": True,

        # model parameters
        "hidden_size": 1024,
        "depth": 12,
        "n_head": 8,

        # optimizer selection: adamw | shampoo | laprop_shampoo | adopt_shampoo
        "mode": "adamw",

        "beta1": 0.9,
        "beta2": 0.999,

        "learning_rate": 4e-4,
        "weight_decay": 0.01,

        # --- OrderedShampoo knobs (ignored when mode == adamw) ---
        # beta2 above doubles as the Kronecker-factor decay. Shampoo usually
        # wants a shorter window than Adam's 0.999; override per arm.
        "precondition_frequency": 10,
        "start_preconditioning_step": 250,
        "max_precond_dim": 8192,
        "graft": "rms",
        "nesterov": False,
        # Precondition only the smaller axis of each 2-D parameter. Cuts the
        # full-model refresh from 17.5s to 1.33s and optimizer state from
        # 5.1GB to 0.43GB (22% overhead at freq=10 vs 282%), at the cost of
        # being one-sided Shampoo rather than Shampoo. Off by default; it is a
        # config option, not the baseline.
        "one_sided": False,
        # Route embeddings / lm_head / 1-D params to plain AdamW, as Muon and
        # most matrix-preconditioned optimizers do.
        "exclude_embeddings_from_precond": True,
        # Precondition fused weights (QKV, GeGLU value+gate) as independent
        # blocks rather than one stacked matrix. More faithful to Shampoo's
        # Kronecker assumption and ~1.8x cheaper, since eigh is superlinear
        # in n. Does NOT change the model's forward pass.
        "split_fused_precond": True,
        # 'eigh' (fp64, exact) or 'ndb' (Newton-Denman-Beavers, matmul-only).
        # NDB is NOT recommended here: with one_sided the root is 2, where NDB
        # plateaus at 5-13 deg of update error -- at or above the effect size
        # this study is trying to resolve. Kept for two-sided experiments.
        "root_method": "eigh",
        "ndb_iters": 15,
        "ndb_damp": 1e-6,

        "qk_norm": True,
        # 1/sqrt(2*depth) on the two projections writing into the residual
        # stream. Moves the baseline (it is not a bug fix), so it is a flag.
        "depth_scaled_residual_init": True,

        "wandb_project": "laprop-shampoo",

        "hf_cache_dir": os.path.join(os.environ["HF_HOME"], "hub"),
        # Pre-tokenized OpenWebText already on this box: 8.37M rows x 1024
        # tokens (~8.6B tokens).
        "tokenized_dataset_path": "/workspace/tokenized/gpt2-medium_openwebtext_1024",

        "lr_warmup_mult": 1.0,
    }

    # Optional overrides from a JSON file for sweep scripts or manual runs
    if override_args:
        unknown = sorted(
            k for k in override_args
            if k not in args and not k.startswith("_")
        )
        if unknown:
            raise ValueError(
                f"Unknown override key(s): {', '.join(unknown)}. "
                "Add them to the defaults dict in main() first."
            )
        args.update(override_args)


    synthetic_data = args["dataset_name"] == "synthetic"
    if synthetic_data:
        config = None
        vocab_size = int(args["synthetic_vocab_size"])
    else:
        config = AutoConfig.from_pretrained(
            args['model_name_or_path'],
            trust_remote_code=args['trust_remote_code'],
        )
        vocab_size = config.vocab_size

    if args.get("_override_name"):
        # Generated sweep configs use the filename as the reviewed run identity.
        run_name = args["_override_name"]
    else:
        # For manual runs, build a compact descriptive name from the hyperparams.
        run_name = build_run_name(args)

    args["run_name"] = run_name
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"
    args["wandb_run_name"] = run_name

    # Drop metadata helpers before namespacing.
    args.pop("_override_json_path", None)
    args.pop("_override_name", None)

    args = SimpleNamespace(**args)

    print("Running with the following arguments:", flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.output_dir is None:
        args.output_dir = time.strftime("run_%Y%m%d_%H%M%S")

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        # Accelerate's DataLoaderShard does send_to_device inside next(). Default
        # non_blocking=False makes that a device sync, so host timers around
        # next() look like "100ms data loading" when they are mostly waiting on
        # the previous step's GPU work. Keep H2D async; we still stage tensors
        # below with non_blocking=.to().
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = None
    if not synthetic_data:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )


    print("Creating model...", flush=True)
    model = Transformer(
        dim=args.hidden_size,
        depth=args.depth,
        heads=args.n_head,
        ff_mult=4,
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        gradient_checkpointing=args.gradient_checkpointing,
        qk_norm=args.qk_norm,
        depth_scaled_residual_init=args.depth_scaled_residual_init,
    )

    print("num parameters", sum(p.numel() for p in model.parameters()), flush=True)
    model = model.to(accelerator.device)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.token_embedding.weight.shape[0]
    if tokenizer is not None and len(tokenizer) > embedding_size:
        print("resizing token embeddings", len(tokenizer), embedding_size)
        model.resize_token_embeddings(len(tokenizer))

    # ---- Load or build tokenized dataset ----
    tok_path = getattr(args, "tokenized_dataset_path", None)

    if synthetic_data:
        lm_datasets = {
            "train": SyntheticTokenDataset(
                args.synthetic_num_samples,
                args.block_size,
                vocab_size,
                args.seed,
            ),
            "validation": SyntheticTokenDataset(
                args.synthetic_validation_samples,
                args.block_size,
                vocab_size,
                args.seed + 1,
            ),
        }
    elif tok_path and os.path.isdir(tok_path):
        logger.info(f"Loading pre-tokenized dataset from {tok_path}")
        print(f"Loading pre-tokenized dataset from {tok_path}", flush=True)
        lm_datasets = datasets.load_from_disk(tok_path)
        # Older caches carry `labels` (int64, a copy of input_ids) and
        # `attention_mask` (int8, unused -- attention is is_causal). The loop
        # derives targets by shifting input_ids, so keeping them costs 13
        # bytes/token instead of 4. Dropping is a view change, not a rewrite.
        for split in lm_datasets:
            drop = [c for c in lm_datasets[split].column_names if c != "input_ids"]
            if drop:
                lm_datasets[split] = lm_datasets[split].remove_columns(drop)
                print(f"  {split}: dropped unused columns {drop}", flush=True)
    else:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split={
                "train": f"train[{args.validation_split_percentage}%:]",
                "validation": f"train[:{args.validation_split_percentage}%]",
            },
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.hf_cache_dir,
            num_proc=args.preprocessing_num_workers,
        )

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        def tokenize_and_group(examples):
            tokenized = tokenize_function(examples)
            concatenated_examples = {k: list(chain(*tokenized[k])) for k in tokenized.keys()}
            total_length = len(concatenated_examples[list(tokenized.keys())[0]])
            total_length = (total_length // block_size) * block_size
            # No labels column: the training loop derives targets by shifting
            # input_ids, so materializing a duplicate would double both the
            # on-disk dataset and the bytes the dataloader moves per step.
            return {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }

        print("Starting dataset tokenization...", flush=True)
        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                tokenize_and_group,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Tokenize + group into {block_size}",
            )
        print("Dataset tokenization complete.", flush=True)

        if tok_path and accelerator.is_main_process:
            logger.info(f"Saving tokenized dataset to {tok_path}")
            lm_datasets.save_to_disk(tok_path)

    if getattr(args, "prepare_dataset_only", False):
        print(f"Dataset preparation complete at {tok_path}", flush=True)
        return

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    loader_common = {
        "collate_fn": default_data_collator,
        "batch_size": args.per_device_train_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = max(1, int(getattr(args, "prefetch_factor", 2)))

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_common,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    # Optimizer
    # Weight decay exclusions using 1D param strategy (robust to naming conventions)
    # - 1D params: catches all biases and all normalization weights (LayerNorm, RMSNorm, etc.)
    # - Embeddings: 2D but shouldn't have weight decay
    # Coalesce into one group per (weight_decay, lr, preconditioned) key.
    # Per-parameter groups work but make fused AdamW launch ~150 tiny kernels
    # per step; optimizer state is per-parameter either way.
    #
    # Embeddings, the lm_head and every 1-D parameter are routed to plain
    # AdamW, the usual convention for matrix-preconditioned optimizers (Muon
    # does the same). Those tensors have one axis the size of the vocabulary,
    # so the Kronecker factor on the other axis mixes unrelated tokens and buys
    # little. Excluding them also keeps the arms honest: they get an *identical*
    # rule in every arm (order='post' + graft='none' is exactly AdamW), so the
    # only thing that differs between AdamW / Shampoo / LaProp-Shampoo is how
    # the preconditioned matrices are handled.
    def is_excluded(name, param):
        if param.dim() == 1:
            return True
        lowered = name.lower()
        return "embed" in lowered or "out_proj.1" in name

    def fused_blocks(name, param):
        """How many independent projections a fused weight actually stacks.

        `to_qkv` is [3*dim, dim] (Q, K, V) and GeGLU `proj_in` is
        [2*hidden, dim] (value, gate). Shampoo's Kronecker approximation
        assumes a coherent [out, in] map, which a stack of unrelated
        projections is not: G^T G sums blocks with very different scales.
        Preconditioning them as separate blocks is both more faithful and
        cheaper -- one 5504^2 eigh costs far more than two 2752^2 ones.
        """
        if not args.split_fused_precond or param.dim() < 2:
            return 1
        if name.endswith("attn.to_qkv.weight"):
            return 3
        if name.endswith("ff.proj_in.weight"):
            return 2
        return 1

    buckets = {}
    n_decay, n_no_decay, n_excluded = 0, 0, 0
    split_counts = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_mult = getattr(param, "lr_mult", 1.0)
        # 1D params (biases, norm weights) or embeddings -> no weight decay
        if param.dim() == 1 or "embed" in name.lower():
            wd = 0.0
            n_no_decay += 1
        else:
            wd = args.weight_decay
            n_decay += 1
        excluded = args.exclude_embeddings_from_precond and is_excluded(name, param)
        n_excluded += bool(excluded)
        split = 1 if excluded else fused_blocks(name, param)
        if split > 1:
            split_counts[split] = split_counts.get(split, 0) + 1
        key = (wd, args.learning_rate * lr_mult, excluded, split)
        buckets.setdefault(key, {"params": [], "names": []})
        buckets[key]["params"].append(param)
        buckets[key]["names"].append(name)

    optimizer_grouped_parameters = []
    for (wd, lr, excluded, split), v in sorted(buckets.items()):
        group = {"params": v["params"], "weight_decay": wd, "lr": lr}
        if args.mode not in ("adamw", "base"):
            if excluded:
                # AdamW semantics inside OrderedShampoo: no preconditioner, Adam
                # ordering, no grafting. Identical across every arm.
                group.update(precondition=False, order="post", graft="none")
            elif split > 1:
                group["precond_split"] = split
        optimizer_grouped_parameters.append(group)

    print(f"Params with weight decay: {n_decay}, without: {n_no_decay}")
    print(f"Params excluded from preconditioning (AdamW path): {n_excluded}")
    if split_counts:
        print(f"Fused params preconditioned block-wise: {split_counts}")
    print(f"Optimizer param groups: {len(optimizer_grouped_parameters)}")

    optimizer = build_optimizer(args, optimizer_grouped_parameters)

    # Print optimizer param group settings (excluding raw parameter tensors).
    print("Optimizer param groups:", flush=True)
    for idx, group in enumerate(optimizer.param_groups):
        group_view = {k: v for k, v in group.items() if k != "params"}
        print(f"  Group {idx}: {group_view}")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Allow optimizer to scale LR scheduler warmup without touching beta/alpha warmups.
    # e.g. num_warmup_steps=2000 and optimizer lr_warmup_mult=5.0 -> 10_000 warmup steps.
    lr_warmup_mult = float(optimizer.param_groups[0].get("lr_warmup_mult", 1.0))
    base_num_warmup_steps = int(args.num_warmup_steps or 0) * accelerator.num_processes
    scaled_num_warmup_steps = int(round(base_num_warmup_steps * lr_warmup_mult))

    # Clamp to avoid pathological schedules where warmup exceeds total training steps.
    total_sched_steps = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    scaled_num_warmup_steps = min(scaled_num_warmup_steps, int(total_sched_steps))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=scaled_num_warmup_steps,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator` first, then compile the
    # device-placed (and DDP-wrapped, if any) module. Compiling after prepare
    # generally yields a better graph for reduce-overhead / cudagraphs.
    print("Preparing accelerator...", flush=True)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    print("Accelerator ready.", flush=True)

    if args.compile:
        if (
            int(getattr(args, "gradient_accumulation_steps", 1)) > 1
            and args.compile_mode == "reduce-overhead"
        ):
            # CUDA Graphs reuse static buffers across captures. With grad-accum,
            # a later microbatch forward can overwrite activations still needed
            # by an earlier backward → RuntimeError on tok_emb / graph outputs.
            print(
                "[compile] gradient_accumulation_steps>1 is incompatible with "
                "reduce-overhead CUDAGraphs; falling back to compile_mode=default",
                flush=True,
            )
            args.compile_mode = "default"

        if args.compile_mode == "reduce-overhead":
            # The transformer lazily allocates RoPE tables on the first forward.
            # Build them before torch.compile so CUDAGraphs do not capture those
            # cache tensors as graph outputs that are reused on later steps.
            seq_len = max(1, int(args.block_size) - 1)
            dummy_input = torch.zeros((1, seq_len), dtype=torch.long, device=accelerator.device)
            dummy_targets = torch.zeros((1, seq_len), dtype=torch.long, device=accelerator.device)
            was_training = model.training
            model.eval()
            with torch.no_grad(), accelerator.autocast():
                _ = model(input_ids=dummy_input, targets=dummy_targets)
            if was_training:
                model.train()
            del dummy_input, dummy_targets
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(
            f"Compiling model (mode={args.compile_mode}, fullgraph={args.compile_fullgraph})...",
            flush=True,
        )
        model = torch.compile(
            model, mode=args.compile_mode, fullgraph=args.compile_fullgraph
        )

    cudagraph_mark_step_begin = None
    if args.compile and args.compile_mode == "reduce-overhead":
        cudagraph_mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        init_kwargs = {
            "wandb": {
                "name": args.wandb_run_name,
            }
        }
        project_name = getattr(args, "wandb_project", "laprop-shampoo")
        print(f"Initializing wandb tracker ({project_name})...", flush=True)
        accelerator.init_trackers(project_name, experiment_config, init_kwargs=init_kwargs)
        print("Wandb initialized.", flush=True)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # Allocated and reserved memory (zero for CPU smoke tests).
    allocated_memory = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    reserved_memory = (
        torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    )
    progress_bar.set_postfix(vram=f"{reserved_memory / (1024 ** 3):.2f} GB")

    log_every_n = max(1, int(getattr(args, "log_every_n", 10)))
    profile_timers = bool(getattr(args, "profile_timers", False)) and torch.cuda.is_available()
    profile_every = max(1, int(getattr(args, "profile_every", log_every_n)))
    csv_log = (
        CSVLogger(
            os.path.join(args.output_dir, "metrics.csv"),
            flush_every=max(1, int(getattr(args, "csv_flush_every", 100))),
        )
        if accelerator.is_main_process and bool(getattr(args, "csv_logging", False))
        else None
    )
    train_wall_start = time.perf_counter()
    throughput_window_start = train_wall_start
    throughput_window_step = completed_steps
    training_start_step = completed_steps
    optimizer_state_bytes_cache = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # CUDA events only fire on profile_every steps; data load uses host time
    # because next(dataloader) is CPU-side and invisible to the GPU clock.
    if profile_timers:
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_start = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        if accelerator.is_main_process:
            print(
                f"[profile] every {profile_every} optimizer steps: "
                f"input_stall=host (sync then next), "
                f"h2d/fwd/bwd/opt=cuda events. "
                f"Note: accelerate next() includes device H2D; "
                f"input_stall is the true post-GPU bubble.",
                flush=True,
            )

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
            # Denominator for the running train-loss mean. The eval loop below
            # rebinds `step`, so this cannot be derived from the loop variable.
            loss_steps = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        dataloader_iter = iter(active_dataloader)
        for step in range(len(active_dataloader)):
            model.train()

            # Profile the upcoming optimizer step (completed_steps increments after sync).
            do_profile = profile_timers and ((completed_steps + 1) % profile_every == 0)
            input_stall_ms = None
            if do_profile:
                # Drain prior GPU work first. Otherwise accelerate's (even async)
                # next()/H2D can look like "data loading" while it is really
                # waiting on the previous backward/optimizer kernels.
                torch.cuda.synchronize()
                stall_t0 = time.perf_counter()
            batch = next(dataloader_iter)
            if do_profile:
                input_stall_ms = (time.perf_counter() - stall_t0) * 1000.0
            
            with accelerator.accumulate(model):
                # Only time full update iterations (skip grad-accum microbatches).
                do_profile = do_profile and accelerator.sync_gradients

                if do_profile:
                    h2d_start.record()
                # Batch may already be on-device via accelerate; .to(non_blocking)
                # is cheap then. Slice stays on device.
                tokens = batch["input_ids"].to(accelerator.device, non_blocking=True)
                input_ids, targets = tokens[:, :-1], tokens[:, 1:]
                if do_profile:
                    h2d_end.record()
                    forward_start.record()

                if cudagraph_mark_step_begin is not None:
                    cudagraph_mark_step_begin()
                loss = model(input_ids=input_ids, targets=targets)
                if do_profile:
                    forward_end.record()
                
                # Sync loss across GPUs for accurate metrics (needed for multi-GPU training)
                synced_loss = loss.detach().float()
                if accelerator.num_processes > 1:
                    synced_loss = accelerator.gather(synced_loss).mean()
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += synced_loss
                    loss_steps += 1
                
                if do_profile:
                    backward_start.record()
                accelerator.backward(loss)
                if do_profile:
                    backward_end.record()
                
                # clip the gradients
                mini_logs = {
                    "step_loss": synced_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }

                # Detect scout optimizer and whether scouting happens this step
                underlying_opt = getattr(optimizer, 'optimizer', optimizer)

                if do_profile:
                    optimizer_start.record()

                if args.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    mini_logs["grad_norm"] = grad_norm

                optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Moment-only snapshots are cheap. Held-out probes additionally
                # run one validation backward pass at the updated weights.
                diagnostic_step = completed_steps + 1

                diagnostic_records = []
                diagnostic_loss = None

                if do_profile:
                    optimizer_end.record()
                    # One sync per profiled step — never on the hot path.
                    torch.cuda.synchronize()
                    h2d_ms = h2d_start.elapsed_time(h2d_end)
                    fwd_ms = forward_start.elapsed_time(forward_end)
                    bwd_ms = backward_start.elapsed_time(backward_end)
                    opt_ms = optimizer_start.elapsed_time(optimizer_end)
                    compute_ms = h2d_ms + fwd_ms + bwd_ms + opt_ms
                    mini_logs.update(
                        {
                            "timer/input_stall_ms": float(input_stall_ms),
                            # Keep old key as alias so existing CSV/W&B panels keep working.
                            "timer/data_ms": float(input_stall_ms),
                            "timer/h2d_ms": float(h2d_ms),
                            "timer/forward_ms": float(fwd_ms),
                            "timer/backward_ms": float(bwd_ms),
                            "timer/optimizer_ms": float(opt_ms),
                            "timer/compute_ms": float(compute_ms),
                            "timer/step_ms": float(input_stall_ms) + float(compute_ms),
                        }
                    )
                    if accelerator.is_main_process:
                        print(
                            f"[profile] step={completed_steps + 1} "
                            f"stall={input_stall_ms:.1f}ms h2d={h2d_ms:.1f}ms "
                            f"fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms "
                            f"opt={opt_ms:.1f}ms compute={compute_ms:.1f}ms "
                            f"step={input_stall_ms + compute_ms:.1f}ms",
                            flush=True,
                        )
                
                # Log update norm if available
                opt = getattr(optimizer, "optimizer", optimizer)  # unwrap Accelerate wrapper if present
                if hasattr(opt, "last_update_norm"):
                    mini_logs["optim/update_norm"] = opt.last_update_norm
                
                log_step = completed_steps + (1 if accelerator.sync_gradients else 0)
                should_log_step = (
                    accelerator.sync_gradients
                    and log_step > 0
                    and log_step % log_every_n == 0
                )
                if should_log_step:
                    now = time.perf_counter()
                    elapsed = max(now - throughput_window_start, 1e-9)
                    update_steps = max(log_step - throughput_window_step, 1)
                    sequences = update_steps * total_batch_size
                    tokens_per_sequence = max(1, int(args.block_size) - 1)
                    if optimizer_state_bytes_cache is None:
                        optimizer_state_bytes_cache = optimizer_state_bytes(optimizer)
                    sequences_per_second = sequences / elapsed
                    tokens_per_second = (
                        sequences * tokens_per_sequence / elapsed
                    )
                    peak_allocated = (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else 0
                    )
                    peak_reserved = (
                        torch.cuda.max_memory_reserved()
                        if torch.cuda.is_available()
                        else 0
                    )
                    mini_logs.update(
                        {
                            "wall_time_seconds": now - train_wall_start,
                            "throughput/sequences_per_second": sequences_per_second,
                            "throughput/tokens_per_second": tokens_per_second,
                            "memory/max_allocated_gb": peak_allocated / (1024 ** 3),
                            "memory/max_reserved_gb": peak_reserved / (1024 ** 3),
                            "efficiency/elapsed_wall_seconds": now - train_wall_start,
                            "efficiency/sequences_per_second": sequences_per_second,
                            "efficiency/tokens_per_second": tokens_per_second,
                            "efficiency/peak_cuda_memory_bytes": float(peak_allocated),
                            "efficiency/optimizer_state_bytes": float(
                                optimizer_state_bytes_cache
                            ),
                        }
                    )
                    throughput_window_start = now
                    throughput_window_step = log_step
                if args.with_tracking and should_log_step:
                    accelerator.log(
                        mini_logs,
                        step=log_step,
                    )
                elif (
                    do_profile
                    and args.with_tracking
                    and accelerator.sync_gradients
                ):
                    # profile_every may not divide log_every_n — still emit timers.
                    accelerator.log(
                        {k: v for k, v in mini_logs.items() if str(k).startswith("timer/")},
                        step=log_step,
                    )

                if csv_log is not None and should_log_step:
                    csv_metrics = {"step": log_step}
                    for key, value in mini_logs.items():
                        if isinstance(value, (int, float)):
                            csv_metrics[key] = float(value)
                        elif torch.is_tensor(value) and value.numel() == 1:
                            csv_metrics[key] = float(value.detach().item())
                    csv_log.log(csv_metrics)
                elif csv_log is not None and do_profile and accelerator.sync_gradients:
                    csv_metrics = {"step": log_step}
                    for key, value in mini_logs.items():
                        if str(key).startswith("timer/") and isinstance(value, (int, float)):
                            csv_metrics[key] = float(value)
                    csv_log.log(csv_metrics)

                # Log optimizer stats when gathered this step
                opt = getattr(optimizer, "optimizer", optimizer)  # unwrap Accelerate wrapper if present
                if getattr(opt, "just_gathered_stats", False):
                    stats_logs = {}
                    for name, stats in getattr(opt, "latest_stats", {}).items():
                        clean = name.replace(".", "/")
                        for k, v in stats.items():
                            if k == "name" or v is None:
                                continue
                            key = f"optim/{clean}/{k}"
                            if torch.is_tensor(v):
                                v = v.item()
                            if isinstance(v, (int, float)):
                                stats_logs[key] = float(v)
                    if stats_logs and args.with_tracking:
                        accelerator.log(stats_logs, step=completed_steps)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps % args.validate_every == 0:
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        tokens = batch["input_ids"].to(accelerator.device, non_blocking=True)
                        input_ids, targets = tokens[:, :-1], tokens[:, 1:]
                        if cudagraph_mark_step_begin is not None:
                            cudagraph_mark_step_begin()
                        loss = model(input_ids=input_ids, targets=targets)
                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_train_batch_size)))
                    if args.num_validation_batches is not None:
                        if step >= args.num_validation_batches:
                            break

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

                if args.with_tracking:
                    eval_metrics = {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": (
                            total_loss.item() / loss_steps if loss_steps else float("nan")
                        ),
                        "epoch": epoch,
                        "step": completed_steps,
                    }
                    accelerator.log(eval_metrics, step=completed_steps)

                    if csv_log is not None:
                        csv_log.log({"step": completed_steps, **{k: float(v) for k, v in eval_metrics.items()}})

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    # One terminal synchronization makes the final paper metric include all GPU
    # work without putting a synchronization on the per-step hot path.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    final_wall_seconds = max(time.perf_counter() - train_wall_start, 1e-9)
    run_update_steps = max(completed_steps - training_start_step, 0)
    run_sequences = run_update_steps * total_batch_size
    run_tokens = run_sequences * max(1, int(args.block_size) - 1)
    if optimizer_state_bytes_cache is None:
        optimizer_state_bytes_cache = optimizer_state_bytes(optimizer)
    peak_cuda_memory_bytes = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    final_efficiency_metrics = {
        "efficiency/elapsed_wall_seconds": final_wall_seconds,
        "efficiency/sequences_per_second": run_sequences / final_wall_seconds,
        "efficiency/tokens_per_second": run_tokens / final_wall_seconds,
        "efficiency/peak_cuda_memory_bytes": float(peak_cuda_memory_bytes),
        "efficiency/optimizer_state_bytes": float(optimizer_state_bytes_cache),
    }
    if args.with_tracking:
        accelerator.log(final_efficiency_metrics, step=completed_steps)
    if accelerator.is_main_process:
        print(
            "[efficiency] "
            f"tokens/s={final_efficiency_metrics['efficiency/tokens_per_second']:.2f} "
            f"sequences/s={final_efficiency_metrics['efficiency/sequences_per_second']:.2f} "
            f"wall={final_wall_seconds:.2f}s "
            f"peak_cuda={peak_cuda_memory_bytes}B "
            f"optimizer_state={optimizer_state_bytes_cache}B",
            flush=True,
        )
    if csv_log is not None:
        csv_log.log({"step": completed_steps, **final_efficiency_metrics})
        csv_log.close()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        save_final_model = bool(getattr(args, "save_final_model", False))
        if save_final_model:
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_orig_mod"):
                unwrapped_model = unwrapped_model._orig_mod
            print("Saving model to", args.output_dir)
            save_pretrained = getattr(unwrapped_model, "save_pretrained", None)
            if callable(save_pretrained):
                save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
            else:
                # Local Transformer is a plain nn.Module, not a Hugging Face PreTrainedModel.
                state_dict = accelerator.get_state_dict(model)
                if accelerator.is_main_process:
                    accelerator.save(
                        state_dict,
                        os.path.join(args.output_dir, "pytorch_model.bin"),
                    )
                    model_config = {
                        "model_class": "Transformer",
                        "hidden_size": args.hidden_size,
                        "depth": args.depth,
                        "n_head": args.n_head,
                        "ff_mult": 4,
                        "vocab_size": vocab_size,
                        "max_seq_len": args.block_size,
                        "gradient_checkpointing": args.gradient_checkpointing,
                        "qk_norm": args.qk_norm,
                    }
                    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
                        json.dump(model_config, f, indent=2)
            if accelerator.is_main_process and tokenizer is not None:
                tokenizer.save_pretrained(args.output_dir)
        elif accelerator.is_main_process:
            print(
                "Skipping final model/tokenizer save "
                f"(save_final_model={save_final_model}); writing results only to",
                args.output_dir,
            )
            # Still write a tiny model_config for provenance when weights are skipped.
            model_config_path = os.path.join(args.output_dir, "model_config.json")
            if not os.path.exists(model_config_path):
                with open(model_config_path, "w") as f:
                    json.dump(
                        {
                            "model_class": "Transformer",
                            "hidden_size": args.hidden_size,
                            "depth": args.depth,
                            "n_head": args.n_head,
                            "ff_mult": 4,
                            "vocab_size": vocab_size,
                            "max_seq_len": args.block_size,
                            "gradient_checkpointing": args.gradient_checkpointing,
                            "qk_norm": args.qk_norm,
                            "weights_saved": False,
                        },
                        f,
                        indent=2,
                    )

        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(
                    {
                        "perplexity": locals().get("perplexity"),
                        **final_efficiency_metrics,
                    },
                    f,
                    indent=2,
                )

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
