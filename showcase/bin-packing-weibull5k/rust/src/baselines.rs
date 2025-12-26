use crate::BinPackingHeuristic;

/// First Fit: Place in first bin that has space
/// Simple but not optimal
pub struct FirstFit;

impl BinPackingHeuristic for FirstFit {
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64> {
        // Earlier bins get higher priority (negative index)
        bins.iter()
            .enumerate()
            .map(|(i, &remaining)| {
                if remaining >= item {
                    -(i as f64)
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }
}

/// Best Fit: Place in bin with least remaining space after placement
/// This is the FunSearch baseline: `-(bins - item)`
pub struct BestFit;

impl BinPackingHeuristic for BestFit {
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64> {
        // Prefer bins where item fits tightly (least remaining space)
        // FunSearch formula: -(bins - item) = item - bins
        bins.iter()
            .map(|&remaining| {
                if remaining >= item {
                    item as f64 - remaining as f64
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }
}

/// Worst Fit: Place in bin with most remaining space
/// Generally performs worse but included for comparison
pub struct WorstFit;

impl BinPackingHeuristic for WorstFit {
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64> {
        bins.iter()
            .map(|&remaining| {
                if remaining >= item {
                    remaining as f64
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }
}

/// FunSearch Weibull Heuristic (from their paper)
/// The discovered heuristic that achieved 0.68% excess
pub struct FunSearchWeibull;

impl BinPackingHeuristic for FunSearchWeibull {
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64> {
        // FunSearch discovered heuristic for Weibull dataset:
        // max_bin_cap = max(bins)
        // score = (bins - max_bin_cap)**2 / item + bins**2 / (item**2)
        // score += bins**2 / item**3
        // score[bins > item] = -score[bins > item]
        // score[1:] -= score[:-1]

        let max_bin_cap = *bins.iter().max().unwrap_or(&0) as f64;
        let item_f = item as f64;

        let mut scores: Vec<f64> = bins.iter()
            .map(|&b| {
                let b_f = b as f64;
                let mut score = (b_f - max_bin_cap).powi(2) / item_f
                    + b_f.powi(2) / item_f.powi(2)
                    + b_f.powi(2) / item_f.powi(3);

                if b > item {
                    score = -score;
                }
                score
            })
            .collect();

        // Apply difference: score[1:] -= score[:-1]
        // This is done in reverse to avoid index issues
        for i in (1..scores.len()).rev() {
            scores[i] -= scores[i - 1];
        }

        scores
    }
}
