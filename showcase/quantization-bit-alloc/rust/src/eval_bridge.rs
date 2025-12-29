//! Bridge between Rust evolution and Python evaluator.
//!
//! Emits bit allocation as JSON, invokes Python, parses results.

use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, Stdio};
use serde::{Deserialize, Serialize};

/// Bit allocation plan for Python evaluator
#[derive(Debug, Clone, Serialize)]
pub struct AllocationPlan {
    #[serde(flatten)]
    pub allocations: HashMap<String, String>,
}

/// Result from Python evaluator
#[derive(Debug, Clone, Deserialize)]
pub struct EvalResult {
    pub perplexity: f64,
    pub model_size_bytes: u64,
    pub bit_histogram: HashMap<String, u64>,
    pub eval_time_seconds: f64,
}

impl AllocationPlan {
    /// Create plan from layer name -> bit width mappings
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
        }
    }

    /// Add a layer allocation
    pub fn add(&mut self, layer_pattern: &str, bit_width: &str) {
        self.allocations.insert(layer_pattern.to_string(), bit_width.to_string());
    }

    /// Create from iterator of (layer_pattern, bit_width) pairs
    pub fn from_iter<I, S1, S2>(iter: I) -> Self
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        let allocations: HashMap<String, String> = iter
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        Self { allocations }
    }

    /// Write plan to JSON file
    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.allocations)?;
        std::fs::write(path, json)
    }
}

impl Default for AllocationPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Mode for evaluation
#[derive(Debug, Clone, Copy)]
pub enum EvalMode {
    /// Fast evaluation (~2k tokens, ~1-3s)
    Fast,
    /// Verify evaluation (~10k tokens, ~10-15s)
    Verify,
}

impl EvalMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvalMode::Fast => "fast",
            EvalMode::Verify => "verify",
        }
    }
}

/// Evaluate a bit allocation using the Python evaluator
pub fn evaluate(
    plan: &AllocationPlan,
    mode: EvalMode,
    python_script: &Path,
    work_dir: &Path,
) -> Result<EvalResult, String> {
    // Write plan to temp file in work directory
    let plan_path = work_dir.join("_temp_plan.json");
    plan.write_to_file(&plan_path)
        .map_err(|e| format!("Failed to write plan: {}", e))?;

    // Invoke Python evaluator
    let output = Command::new("python3")
        .arg(python_script)
        .arg("--plan")
        .arg(&plan_path)
        .arg("--mode")
        .arg(mode.as_str())
        .current_dir(work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run evaluator: {}", e))?;

    // Clean up temp file
    let _ = std::fs::remove_file(&plan_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Evaluator failed: {}", stderr));
    }

    // Parse result
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse result: {} (output: {})", e, stdout))
}

/// Calculate fitness from evaluation results.
///
/// Fitness = compression_ratio - λ * max(0, (P - P_baseline) / P_baseline)
///
/// Where:
/// - P = perplexity
/// - P_baseline = FP16 baseline perplexity
/// - λ = 10.0 (makes >10% perplexity degradation dominate compression)
pub fn calculate_fitness(
    result: &EvalResult,
    baseline_perplexity: f64,
    baseline_size: u64,
) -> f64 {
    const LAMBDA: f64 = 10.0;

    let compression_ratio = baseline_size as f64 / result.model_size_bytes as f64;
    let perplexity_degradation = (result.perplexity - baseline_perplexity) / baseline_perplexity;
    let penalty = LAMBDA * perplexity_degradation.max(0.0);

    (compression_ratio - penalty).max(0.0)
}

/// Check if VERIFY result is consistent with FAST result
///
/// Returns true if perplexity is within tolerance (2% by default)
pub fn verify_consistency(fast: &EvalResult, verify: &EvalResult, tolerance: f64) -> bool {
    let diff = (verify.perplexity - fast.perplexity).abs() / fast.perplexity;
    diff <= tolerance
}

/// GPT-2 small layer patterns for bit allocation
///
/// These patterns match against parameter names in the model
pub const GPT2_LAYER_PATTERNS: &[&str] = &[
    // Token and position embeddings
    "wte",      // token embeddings (50257 x 768)
    "wpe",      // position embeddings (1024 x 768)

    // Transformer blocks 0-11 (each has attention + MLP + layer norms)
    "h.0.ln_1", "h.0.attn", "h.0.ln_2", "h.0.mlp",
    "h.1.ln_1", "h.1.attn", "h.1.ln_2", "h.1.mlp",
    "h.2.ln_1", "h.2.attn", "h.2.ln_2", "h.2.mlp",
    "h.3.ln_1", "h.3.attn", "h.3.ln_2", "h.3.mlp",
    "h.4.ln_1", "h.4.attn", "h.4.ln_2", "h.4.mlp",
    "h.5.ln_1", "h.5.attn", "h.5.ln_2", "h.5.mlp",
    "h.6.ln_1", "h.6.attn", "h.6.ln_2", "h.6.mlp",
    "h.7.ln_1", "h.7.attn", "h.7.ln_2", "h.7.mlp",
    "h.8.ln_1", "h.8.attn", "h.8.ln_2", "h.8.mlp",
    "h.9.ln_1", "h.9.attn", "h.9.ln_2", "h.9.mlp",
    "h.10.ln_1", "h.10.attn", "h.10.ln_2", "h.10.mlp",
    "h.11.ln_1", "h.11.attn", "h.11.ln_2", "h.11.mlp",

    // Final layer norm
    "ln_f",
];

/// Bit width options for real quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealBitWidth {
    FP32,
    FP16,
    INT8,
    INT4,
}

impl RealBitWidth {
    pub fn as_str(&self) -> &'static str {
        match self {
            RealBitWidth::FP32 => "fp32",
            RealBitWidth::FP16 => "fp16",
            RealBitWidth::INT8 => "int8",
            RealBitWidth::INT4 => "int4",
        }
    }

    pub fn all() -> &'static [RealBitWidth] {
        &[RealBitWidth::FP32, RealBitWidth::FP16, RealBitWidth::INT8, RealBitWidth::INT4]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_plan() {
        let mut plan = AllocationPlan::new();
        plan.add("ln_f", "fp32");
        plan.add("h.0.mlp", "int8");

        assert_eq!(plan.allocations.len(), 2);
        assert_eq!(plan.allocations.get("ln_f"), Some(&"fp32".to_string()));
    }

    #[test]
    fn test_fitness_calculation() {
        let result = EvalResult {
            perplexity: 32.0,
            model_size_bytes: 250_000_000,  // ~250MB
            bit_histogram: HashMap::new(),
            eval_time_seconds: 1.5,
        };

        // Baseline: perplexity 30, size 500MB
        let fitness = calculate_fitness(&result, 30.0, 500_000_000);

        // Compression = 500/250 = 2.0
        // Degradation = (32-30)/30 = 0.0667
        // Penalty = 10 * 0.0667 = 0.667
        // Fitness = 2.0 - 0.667 = 1.333
        assert!((fitness - 1.333).abs() < 0.01);
    }

    #[test]
    fn test_verify_consistency() {
        let fast = EvalResult {
            perplexity: 30.0,
            model_size_bytes: 250_000_000,
            bit_histogram: HashMap::new(),
            eval_time_seconds: 1.5,
        };

        let verify_good = EvalResult {
            perplexity: 30.5,  // 1.67% diff
            model_size_bytes: 250_000_000,
            bit_histogram: HashMap::new(),
            eval_time_seconds: 10.0,
        };

        let verify_bad = EvalResult {
            perplexity: 33.0,  // 10% diff
            model_size_bytes: 250_000_000,
            bit_histogram: HashMap::new(),
            eval_time_seconds: 10.0,
        };

        assert!(verify_consistency(&fast, &verify_good, 0.02));
        assert!(!verify_consistency(&fast, &verify_bad, 0.02));
    }
}
