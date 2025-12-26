use crate::BinPackingHeuristic;

pub struct Evolved;

impl BinPackingHeuristic for Evolved {
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64> {
        if bins.is_empty() { return vec![]; }

        let max_bin_cap = *bins.iter().max().unwrap_or(&0) as f64;
        let item_f = item as f64;

        let mut scores: Vec<f64> = bins.iter()
            .map(|&b| {
                let b_f = b as f64;
                let waste = b_f - item_f;

                // Log transformations - capture relationships in log space
                let log_waste = (waste + 1.0).ln();
                let log_item = (item_f + 1.0).ln();
                let log_bin = (b_f + 1.0).ln();
                let log_ratio = log_bin - log_item; // ln(bin/item)

                // Keep proven quadratic max difference term from FunSearch
                let max_diff_term = (b_f - max_bin_cap).powi(2) / item_f;

                // Log-based utilization emphasizes tight fits
                let log_util_term = log_waste / log_item;

                // Ratio-based log term
                let log_ratio_term = log_ratio / log_item;

                let mut score = max_diff_term + log_util_term * 2.0 + log_ratio_term;

                if b > item { score = -score; }
                score
            })
            .collect();

        for i in (1..scores.len()).rev() {
            scores[i] -= scores[i - 1];
        }
        scores
    }
}
