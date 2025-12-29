//! Quantization Bit Allocation Benchmark
//!
//! Evolve a heuristic that decides how many bits to allocate to each layer
//! of a neural network for mixed-precision quantization.
//!
//! This is relevant to NVIDIA TensorRT and inference optimization.
//!
//! Two modes of operation:
//! 1. Synthetic mode: Uses generated sensitivity curves (fast, for prototyping)
//! 2. Real mode: Uses actual GPT-2 quantization with perplexity measurement

pub mod baselines;
pub mod evolved;
pub mod eval_bridge;

use serde::{Deserialize, Serialize};

/// Available bit widths for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitWidth {
    INT2 = 2,
    INT4 = 4,
    INT8 = 8,
    FP16 = 16,
    FP32 = 32,
}

impl BitWidth {
    pub fn bits(&self) -> u32 {
        *self as u32
    }

    pub fn all() -> &'static [BitWidth] {
        &[BitWidth::INT2, BitWidth::INT4, BitWidth::INT8, BitWidth::FP16, BitWidth::FP32]
    }

    pub fn from_bits(bits: u32) -> Option<BitWidth> {
        match bits {
            2 => Some(BitWidth::INT2),
            4 => Some(BitWidth::INT4),
            8 => Some(BitWidth::INT8),
            16 => Some(BitWidth::FP16),
            32 => Some(BitWidth::FP32),
            _ => None,
        }
    }
}

/// Layer type in a neural network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    Embedding,      // First layer, often sensitive
    Conv,           // Convolutional layer
    Linear,         // Fully connected / Dense
    Attention,      // Self-attention (QKV + projection)
    LayerNorm,      // Normalization layers, very sensitive
    Classifier,     // Final classification head, sensitive
}

/// Information about a layer for bit allocation decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer index (0-indexed)
    pub layer_idx: usize,
    /// Total number of layers in the model
    pub num_layers: usize,
    /// Type of layer
    pub layer_type: LayerType,
    /// Number of parameters in this layer
    pub num_params: u64,
    /// Total parameters in the model
    pub total_params: u64,
    /// Weight statistics
    pub weight_mean: f64,
    pub weight_std: f64,
    pub weight_range: f64,  // max - min
    /// Activation range (typical output magnitude)
    pub activation_range: f64,
    /// Gradient sensitivity (higher = more sensitive to quantization)
    /// Approximates Hessian trace / Fisher information
    pub gradient_sensitivity: f64,
    /// Quantization sensitivity scores (accuracy drop at each bit width)
    /// Index: 0=INT2, 1=INT4, 2=INT8, 3=FP16, 4=FP32
    pub sensitivity: [f64; 5],
}

impl LayerInfo {
    /// Relative position in the model (0.0 = first, 1.0 = last)
    pub fn relative_position(&self) -> f64 {
        if self.num_layers <= 1 {
            0.5
        } else {
            self.layer_idx as f64 / (self.num_layers - 1) as f64
        }
    }

    /// Fraction of total parameters in this layer
    pub fn param_fraction(&self) -> f64 {
        self.num_params as f64 / self.total_params as f64
    }

    /// Get sensitivity for a specific bit width
    pub fn sensitivity_at(&self, bw: BitWidth) -> f64 {
        match bw {
            BitWidth::INT2 => self.sensitivity[0],
            BitWidth::INT4 => self.sensitivity[1],
            BitWidth::INT8 => self.sensitivity[2],
            BitWidth::FP16 => self.sensitivity[3],
            BitWidth::FP32 => self.sensitivity[4],
        }
    }
}

/// Trait for bit allocation heuristics
pub trait BitAllocationHeuristic {
    /// Given layer information, return the recommended bit width.
    fn allocate(&self, layer: &LayerInfo) -> BitWidth;
}

/// Result of applying a bit allocation strategy to a model
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Bit width assigned to each layer
    pub allocations: Vec<BitWidth>,
    /// Estimated accuracy retention (1.0 = no loss)
    pub accuracy_retention: f64,
    /// Compression ratio vs FP32 baseline
    pub compression_ratio: f64,
    /// Average bits per parameter
    pub avg_bits: f64,
}

/// Evaluate a bit allocation heuristic on a model
pub fn evaluate_allocation<H: BitAllocationHeuristic>(
    heuristic: &H,
    layers: &[LayerInfo],
) -> AllocationResult {
    let allocations: Vec<BitWidth> = layers.iter()
        .map(|layer| heuristic.allocate(layer))
        .collect();

    // Calculate accuracy retention (product of per-layer retentions)
    // Each layer contributes: 1.0 - sensitivity_at(bit_width)
    let mut accuracy_retention = 1.0;
    for (layer, &bw) in layers.iter().zip(allocations.iter()) {
        let layer_loss = layer.sensitivity_at(bw);
        // Use multiplicative model: total_accuracy = product of (1 - layer_loss)
        accuracy_retention *= 1.0 - layer_loss;
    }

    // Calculate compression ratio
    let mut total_fp32_bits: u64 = 0;
    let mut total_quantized_bits: u64 = 0;
    for (layer, &bw) in layers.iter().zip(allocations.iter()) {
        total_fp32_bits += layer.num_params * 32;
        total_quantized_bits += layer.num_params * bw.bits() as u64;
    }
    let compression_ratio = total_fp32_bits as f64 / total_quantized_bits as f64;
    let avg_bits = total_quantized_bits as f64 / layers.iter().map(|l| l.num_params).sum::<u64>() as f64;

    AllocationResult {
        allocations,
        accuracy_retention,
        compression_ratio,
        avg_bits,
    }
}

/// Fitness function: maximize compression while maintaining accuracy
///
/// fitness = compression_ratio * accuracy_bonus
/// where accuracy_bonus rewards staying above threshold
pub fn compute_fitness(result: &AllocationResult, accuracy_threshold: f64) -> f64 {
    // Accuracy must be above threshold
    if result.accuracy_retention < accuracy_threshold {
        // Heavy penalty for dropping below threshold
        // Scale by how far below we are
        return result.accuracy_retention * 0.1;
    }

    // Above threshold: reward compression with accuracy bonus
    // accuracy_bonus ranges from 1.0 (at threshold) to 2.0 (at 100%)
    let accuracy_margin = result.accuracy_retention - accuracy_threshold;
    let max_margin = 1.0 - accuracy_threshold;
    let accuracy_bonus = 1.0 + (accuracy_margin / max_margin);

    result.compression_ratio * accuracy_bonus
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UniformInt8;
    impl BitAllocationHeuristic for UniformInt8 {
        fn allocate(&self, _layer: &LayerInfo) -> BitWidth {
            BitWidth::INT8
        }
    }

    #[test]
    fn test_bit_width_conversion() {
        assert_eq!(BitWidth::INT8.bits(), 8);
        assert_eq!(BitWidth::from_bits(8), Some(BitWidth::INT8));
        assert_eq!(BitWidth::from_bits(7), None);
    }

    #[test]
    fn test_layer_info() {
        let layer = LayerInfo {
            layer_idx: 5,
            num_layers: 10,
            layer_type: LayerType::Linear,
            num_params: 1000,
            total_params: 10000,
            weight_mean: 0.0,
            weight_std: 0.02,
            weight_range: 0.1,
            activation_range: 5.0,
            gradient_sensitivity: 0.5,
            sensitivity: [0.5, 0.2, 0.05, 0.001, 0.0],
        };

        assert!((layer.relative_position() - 0.555).abs() < 0.01);
        assert!((layer.param_fraction() - 0.1).abs() < 0.001);
    }
}
