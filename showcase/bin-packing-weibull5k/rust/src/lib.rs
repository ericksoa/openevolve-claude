//! Online 1D Bin Packing Benchmark
//!
//! Evolve a priority function to minimize bins used.
//! This is the FunSearch Weibull 5k benchmark.

pub mod baselines;
pub mod evolved;

/// Trait for bin packing priority heuristics.
///
/// Given an item size and array of bin remaining capacities,
/// return a priority score for each bin. Higher priority = prefer that bin.
/// Bins with insufficient capacity will be filtered out by the packer.
pub trait BinPackingHeuristic {
    /// Returns priority scores for placing `item` into each bin.
    /// `bins[i]` = remaining capacity of bin i
    /// Returns array of same length as `bins` with priority scores.
    fn priority(&self, item: u32, bins: &[u32]) -> Vec<f64>;
}

/// Online bin packing using a priority heuristic (FunSearch-compatible).
/// Pre-allocates one bin per item, passes ONLY valid bins to priority function.
/// Returns the number of bins actually used.
pub fn online_bin_pack<H: BinPackingHeuristic>(
    heuristic: &H,
    items: &[u32],
    capacity: u32,
) -> usize {
    // FunSearch style: pre-allocate one bin per item
    let mut bins: Vec<u32> = vec![capacity; items.len()];

    for &item in items {
        // Get indices of bins that can fit the item
        let valid_indices: Vec<usize> = bins.iter()
            .enumerate()
            .filter(|(_, &remaining)| remaining >= item)
            .map(|(i, _)| i)
            .collect();

        if valid_indices.is_empty() {
            panic!("No bin can fit item {} - should never happen with pre-allocated bins", item);
        }

        // Extract only the valid bins for priority calculation
        let valid_bins: Vec<u32> = valid_indices.iter().map(|&i| bins[i]).collect();

        // Get priorities for valid bins only
        let priorities = heuristic.priority(item, &valid_bins);

        // Find best among valid bins
        let mut best_idx = 0;
        let mut best_priority = priorities[0];
        for (i, &p) in priorities.iter().enumerate() {
            if p > best_priority {
                best_priority = p;
                best_idx = i;
            }
        }

        // Map back to original bin index and update
        let actual_bin = valid_indices[best_idx];
        bins[actual_bin] -= item;
    }

    // Count bins that were actually used (capacity changed from initial)
    bins.iter().filter(|&&remaining| remaining != capacity).count()
}

/// Calculate L1 lower bound (theoretical minimum bins)
pub fn l1_lower_bound(items: &[u32], capacity: u32) -> usize {
    let total: u64 = items.iter().map(|&x| x as u64).sum();
    ((total + capacity as u64 - 1) / capacity as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_heuristic<H: BinPackingHeuristic>(h: &H) {
        // Simple test: items that perfectly fill bins
        let items = vec![50, 50, 50, 50]; // 4 items of size 50
        let bins = online_bin_pack(h, &items, 100);
        assert!(bins >= 2, "Should use at least 2 bins for 4x50 in capacity 100");
        assert!(bins <= 4, "Should use at most 4 bins");

        // Test with varied sizes
        let items = vec![30, 70, 40, 60, 20, 80];
        let bins = online_bin_pack(h, &items, 100);
        assert!(bins >= 3, "L1 bound is 3");
        assert!(bins <= 6, "Should not use more bins than items");
    }

    #[test]
    fn test_first_fit() { test_heuristic(&baselines::FirstFit); }

    #[test]
    fn test_best_fit() { test_heuristic(&baselines::BestFit); }

    #[test]
    fn test_evolved() { test_heuristic(&evolved::Evolved); }
}
