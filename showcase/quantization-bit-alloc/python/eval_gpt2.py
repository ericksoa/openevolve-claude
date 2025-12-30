#!/usr/bin/env python3
"""
Real quantization evaluator for GPT-2 family models.

Applies a bit allocation plan to GPT-2/DistilGPT2 and measures perplexity.
Outputs JSON with perplexity, model size, and bit histogram.

Usage:
    python eval_gpt2.py --plan allocation.json --mode fast
    python eval_gpt2.py --plan allocation.json --mode verify
    python eval_gpt2.py --plan allocation.json --mode verify --model distilgpt2
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Fixed seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Eval corpus paths (committed to repo)
DATA_DIR = Path(__file__).parent.parent / "data"
FAST_CORPUS = DATA_DIR / "eval_fast.txt"
VERIFY_CORPUS = DATA_DIR / "eval_verify.txt"

# Model constants
SUPPORTED_MODELS = ["gpt2", "distilgpt2"]
MAX_LENGTH = 256
FAST_TOKENS = 2048
VERIFY_TOKENS = 10240

# Bit width sizes (bytes per param)
BIT_SIZES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def load_model_and_tokenizer(model_name: str = "gpt2") -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load GPT-2 family model from local cache or download once."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.float()  # Ensure FP32
    model.eval()
    return model, tokenizer


def quantize_to_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to INT8 (weight-only, symmetric)."""
    if tensor.numel() == 0:
        return tensor
    scale = tensor.abs().max() / 127.0
    if scale == 0:
        return tensor
    quantized = torch.round(tensor / scale).clamp(-128, 127)
    return (quantized * scale).to(tensor.dtype)


def quantize_to_int4(tensor: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Quantize tensor to INT4 (weight-only, group-wise)."""
    if tensor.numel() == 0:
        return tensor

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    tensor_flat = tensor.flatten().float()

    # Pad to multiple of group_size
    pad_len = (group_size - len(tensor_flat) % group_size) % group_size
    if pad_len > 0:
        tensor_flat = torch.nn.functional.pad(tensor_flat, (0, pad_len))

    tensor_groups = tensor_flat.view(-1, group_size)
    scales = tensor_groups.abs().max(dim=1, keepdim=True).values / 7.0
    scales = scales.clamp(min=1e-8)

    quantized = torch.round(tensor_groups / scales).clamp(-8, 7)
    dequantized = (quantized * scales).flatten()

    # Remove padding
    if pad_len > 0:
        dequantized = dequantized[:-pad_len]

    return dequantized.view(original_shape).to(original_dtype)


def apply_bit_allocation(
    model: GPT2LMHeadModel,
    allocation: Dict[str, str]
) -> Dict[str, int]:
    """
    Apply bit allocation to model weights in-place.

    Args:
        model: GPT-2 model
        allocation: { layer_pattern: bit_width }
                   where bit_width in ["fp32", "fp16", "int8", "int4"]

    Returns:
        Bit histogram { "fp32": n, "fp16": n, "int8": n, "int4": n }
    """
    histogram = {"fp32": 0, "fp16": 0, "int8": 0, "int4": 0}

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Find matching allocation (support partial matches)
            bit_width = "fp32"  # default

            for alloc_pattern, alloc_width in allocation.items():
                # Check if pattern matches this parameter name
                if alloc_pattern in name:
                    bit_width = alloc_width.lower()
                    break

            # Apply quantization
            if bit_width == "fp32":
                # Already fp32, no change needed
                pass
            elif bit_width == "fp16":
                # Simulate FP16 precision loss
                param.data = param.data.half().float()
            elif bit_width == "int8":
                param.data = quantize_to_int8(param.data)
            elif bit_width == "int4":
                param.data = quantize_to_int4(param.data)
            else:
                # Unknown bit width, default to fp32
                bit_width = "fp32"

            histogram[bit_width] += param.numel()

    return histogram


def calculate_model_size(histogram: Dict[str, int]) -> int:
    """Calculate model size in bytes from bit histogram."""
    total_bytes = 0
    for bit_width, count in histogram.items():
        total_bytes += int(count * BIT_SIZES[bit_width])
    return total_bytes


def evaluate_perplexity(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    corpus_path: Path,
    max_tokens: int,
) -> float:
    """
    Evaluate perplexity on a text corpus.

    Uses sliding window with stride = max_length // 2 for efficiency.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0][:max_tokens]

    if len(input_ids) < 2:
        raise ValueError(f"Corpus too short: {len(input_ids)} tokens")

    # Sliding window evaluation
    stride = MAX_LENGTH // 2
    nlls = []
    total_tokens = 0

    for i in range(0, len(input_ids) - 1, stride):
        begin = max(0, i - MAX_LENGTH + stride)
        end = min(i + stride, len(input_ids))

        target_begin = max(0, i)
        target_end = end
        target_len = target_end - target_begin

        if target_len <= 0:
            continue

        input_chunk = input_ids[begin:end].unsqueeze(0)
        target_chunk = input_chunk.clone()

        # Mask tokens before target_begin (we don't count their loss)
        mask_len = target_begin - begin
        if mask_len > 0:
            target_chunk[0, :mask_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            # Loss is averaged over non-masked tokens
            nll = outputs.loss.item() * target_len
            nlls.append(nll)
            total_tokens += target_len

        if end >= len(input_ids):
            break

    if total_tokens == 0:
        raise ValueError("No tokens evaluated")

    # Calculate perplexity
    avg_nll = sum(nlls) / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return perplexity


def main():
    parser = argparse.ArgumentParser(description="GPT-2 quantization evaluator")
    parser.add_argument("--plan", type=str, required=True,
                        help="Path to JSON bit allocation plan")
    parser.add_argument("--mode", choices=["fast", "verify"], default="fast",
                        help="Evaluation mode: fast (~2k tokens) or verify (~10k tokens)")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="gpt2",
                        help="Model to evaluate: gpt2 (12 layers) or distilgpt2 (6 layers)")
    args = parser.parse_args()

    start_time = time.time()

    # Load allocation plan
    with open(args.plan) as f:
        allocation = json.load(f)

    # Load model (fresh copy for each evaluation)
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Apply quantization
    histogram = apply_bit_allocation(model, allocation)
    model_size = calculate_model_size(histogram)

    # Select corpus and token limit
    if args.mode == "fast":
        corpus_path = FAST_CORPUS
        max_tokens = FAST_TOKENS
    else:
        corpus_path = VERIFY_CORPUS
        max_tokens = VERIFY_TOKENS

    # Evaluate perplexity
    perplexity = evaluate_perplexity(model, tokenizer, corpus_path, max_tokens)

    eval_time = time.time() - start_time

    # Output result as single JSON line
    result = {
        "perplexity": round(perplexity, 4),
        "model_size_bytes": model_size,
        "bit_histogram": histogram,
        "eval_time_seconds": round(eval_time, 2),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
