//! Baseline quantization bit allocation strategies
//!
//! These represent known approaches from the literature:
//! - Uniform quantization (TensorRT default)
//! - First/Last FP16 (common practice)
//! - Sensitivity-based (HAWQ-style)
//! - Layer-type aware (heuristic rules)
//!
//! Also includes real quantization baselines for GPT-2 evaluation.

use crate::{BitAllocationHeuristic, BitWidth, LayerInfo, LayerType};
use crate::eval_bridge::{AllocationPlan, GPT2_LAYER_PATTERNS};

/// Uniform INT8 quantization - the simplest baseline
/// This is what TensorRT does by default
pub struct UniformInt8;

impl BitAllocationHeuristic for UniformInt8 {
    fn allocate(&self, _layer: &LayerInfo) -> BitWidth {
        BitWidth::INT8
    }
}

/// Uniform INT4 - aggressive compression
pub struct UniformInt4;

impl BitAllocationHeuristic for UniformInt4 {
    fn allocate(&self, _layer: &LayerInfo) -> BitWidth {
        BitWidth::INT4
    }
}

/// First and Last layers at FP16, rest INT8
/// Common practice because first/last layers are often most sensitive
pub struct FirstLastFP16;

impl BitAllocationHeuristic for FirstLastFP16 {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        if layer.layer_idx == 0 || layer.layer_idx == layer.num_layers - 1 {
            BitWidth::FP16
        } else {
            BitWidth::INT8
        }
    }
}

/// Sensitivity-based allocation (HAWQ-style)
/// Uses gradient sensitivity to decide bit width
/// High sensitivity -> more bits
pub struct SensitivityBased;

impl BitAllocationHeuristic for SensitivityBased {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        // Use gradient sensitivity as the primary signal
        // Thresholds calibrated for typical models
        let sensitivity = layer.gradient_sensitivity;

        if sensitivity > 0.8 {
            BitWidth::FP16
        } else if sensitivity > 0.5 {
            BitWidth::INT8
        } else if sensitivity > 0.2 {
            BitWidth::INT4
        } else {
            BitWidth::INT4  // Very insensitive layers can go low
        }
    }
}

/// Layer-type aware allocation
/// Different layer types have different quantization tolerance
pub struct LayerTypeAware;

impl BitAllocationHeuristic for LayerTypeAware {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        match layer.layer_type {
            // Embeddings are often sensitive (discrete lookups)
            LayerType::Embedding => BitWidth::FP16,
            // LayerNorm is very sensitive (small values, division)
            LayerType::LayerNorm => BitWidth::FP16,
            // Classifier head affects final predictions directly
            LayerType::Classifier => BitWidth::FP16,
            // Attention layers are moderately sensitive
            LayerType::Attention => BitWidth::INT8,
            // Conv and Linear are usually robust
            LayerType::Conv => BitWidth::INT8,
            LayerType::Linear => BitWidth::INT8,
        }
    }
}

/// Mixed-precision with position awareness
/// First 10% and last 10% get more bits
pub struct PositionAware;

impl BitAllocationHeuristic for PositionAware {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        let pos = layer.relative_position();

        if pos < 0.1 || pos > 0.9 {
            // First and last 10% of layers
            BitWidth::FP16
        } else if pos < 0.25 || pos > 0.75 {
            // Next tier
            BitWidth::INT8
        } else {
            // Middle layers - can be more aggressive
            BitWidth::INT8
        }
    }
}

/// Hybrid: combines layer type and sensitivity
/// This is a strong baseline that mimics expert rules
pub struct HybridBaseline;

impl BitAllocationHeuristic for HybridBaseline {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        // Always keep certain layer types at FP16
        match layer.layer_type {
            LayerType::LayerNorm => return BitWidth::FP16,
            LayerType::Embedding if layer.layer_idx == 0 => return BitWidth::FP16,
            LayerType::Classifier => return BitWidth::FP16,
            _ => {}
        }

        // For other layers, use sensitivity
        let sensitivity = layer.gradient_sensitivity;

        // Combine with position (first/last are often sensitive)
        let pos = layer.relative_position();
        let position_boost = if pos < 0.1 || pos > 0.9 { 0.2 } else { 0.0 };
        let effective_sensitivity = sensitivity + position_boost;

        if effective_sensitivity > 0.7 {
            BitWidth::FP16
        } else if effective_sensitivity > 0.3 {
            BitWidth::INT8
        } else {
            BitWidth::INT4
        }
    }
}

/// Greedy sensitivity-based allocation
/// For each layer, pick the lowest bit width that keeps sensitivity below threshold
pub struct GreedySensitivity {
    pub max_layer_loss: f64,
}

impl Default for GreedySensitivity {
    fn default() -> Self {
        Self { max_layer_loss: 0.02 }  // Max 2% accuracy loss per layer
    }
}

impl BitAllocationHeuristic for GreedySensitivity {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        // Try bit widths from lowest to highest
        // Pick first one where sensitivity is below threshold
        for &bw in &[BitWidth::INT2, BitWidth::INT4, BitWidth::INT8, BitWidth::FP16, BitWidth::FP32] {
            if layer.sensitivity_at(bw) <= self.max_layer_loss {
                return bw;
            }
        }
        BitWidth::FP32  // Fallback
    }
}

/// Pareto-optimal selection
/// Balances compression vs accuracy using the sensitivity curves
pub struct ParetoOptimal {
    pub compression_weight: f64,  // 0-1, higher = prefer compression
}

impl Default for ParetoOptimal {
    fn default() -> Self {
        Self { compression_weight: 0.5 }
    }
}

impl BitAllocationHeuristic for ParetoOptimal {
    fn allocate(&self, layer: &LayerInfo) -> BitWidth {
        let mut best_bw = BitWidth::FP32;
        let mut best_score = f64::NEG_INFINITY;

        for &bw in BitWidth::all() {
            let compression = 32.0 / bw.bits() as f64;  // Higher = more compression
            let accuracy = 1.0 - layer.sensitivity_at(bw);  // Higher = better

            // Weighted combination
            let score = self.compression_weight * compression.ln()
                + (1.0 - self.compression_weight) * accuracy.ln().max(-10.0);

            if score > best_score {
                best_score = score;
                best_bw = bw;
            }
        }

        best_bw
    }
}

// ============================================================================
// Real Quantization Baselines for GPT-2
// ============================================================================

/// All FP32 - unquantized baseline
pub fn gpt2_all_fp32() -> AllocationPlan {
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .map(|&p| (p.to_string(), "fp32".to_string()))
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// All FP16 - standard mixed precision baseline
pub fn gpt2_all_fp16() -> AllocationPlan {
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .map(|&p| (p.to_string(), "fp16".to_string()))
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// All INT8 - weight-only INT8 quantization
pub fn gpt2_all_int8() -> AllocationPlan {
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .map(|&p| (p.to_string(), "int8".to_string()))
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// LayerNorm FP32, rest INT8
/// LayerNorm layers are known to be sensitive to quantization
pub fn gpt2_layernorm_fp32_rest_int8() -> AllocationPlan {
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .map(|&p| {
            let bw = if p.contains("ln_") || p == "ln_f" {
                "fp32"
            } else {
                "int8"
            };
            (p.to_string(), bw.to_string())
        })
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// Embeddings + LayerNorm FP32, rest INT8
/// Embeddings are also sensitive
pub fn gpt2_emb_ln_fp32_rest_int8() -> AllocationPlan {
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .map(|&p| {
            let bw = if p.contains("ln_") || p == "ln_f" || p == "wte" || p == "wpe" {
                "fp32"
            } else {
                "int8"
            };
            (p.to_string(), bw.to_string())
        })
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// All INT4 - aggressive compression
pub fn gpt2_all_int4() -> AllocationPlan {
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .map(|&p| (p.to_string(), "int4".to_string()))
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// LayerNorm FP32, rest INT4
pub fn gpt2_layernorm_fp32_rest_int4() -> AllocationPlan {
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .map(|&p| {
            let bw = if p.contains("ln_") || p == "ln_f" {
                "fp32"
            } else {
                "int4"
            };
            (p.to_string(), bw.to_string())
        })
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// Position-aware: edges FP16, middle INT8
pub fn gpt2_edges_fp16_middle_int8() -> AllocationPlan {
    let total = GPT2_LAYER_PATTERNS.len();
    let allocations: Vec<_> = GPT2_LAYER_PATTERNS
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let pos = i as f64 / (total - 1) as f64;
            let bw = if pos < 0.1 || pos > 0.9 {
                "fp16"
            } else if p.contains("ln_") || p == "ln_f" {
                "fp32"  // Always protect layer norms
            } else {
                "int8"
            };
            (p.to_string(), bw.to_string())
        })
        .collect();
    AllocationPlan::from_iter(allocations)
}

/// Get all real quantization baselines with names
pub fn all_gpt2_baselines() -> Vec<(&'static str, AllocationPlan)> {
    vec![
        ("all_fp32", gpt2_all_fp32()),
        ("all_fp16", gpt2_all_fp16()),
        ("all_int8", gpt2_all_int8()),
        ("ln_fp32_rest_int8", gpt2_layernorm_fp32_rest_int8()),
        ("emb_ln_fp32_rest_int8", gpt2_emb_ln_fp32_rest_int8()),
        ("all_int4", gpt2_all_int4()),
        ("ln_fp32_rest_int4", gpt2_layernorm_fp32_rest_int4()),
        ("edges_fp16_middle_int8", gpt2_edges_fp16_middle_int8()),
    ]
}
