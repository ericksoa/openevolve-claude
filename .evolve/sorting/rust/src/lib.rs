//! Sorting Algorithm Benchmark
//!
//! Compares evolved sorting algorithms against standard implementations.

pub mod baselines;
pub mod evolved;

/// Trait for sorting implementations
pub trait Sorter {
    /// Sort the slice in-place in ascending order
    fn sort(&self, data: &mut [u64]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn is_sorted(data: &[u64]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    fn test_sorter<S: Sorter>(sorter: &S) {
        // Empty
        let mut empty: Vec<u64> = vec![];
        sorter.sort(&mut empty);
        assert!(is_sorted(&empty));

        // Single element
        let mut single = vec![42u64];
        sorter.sort(&mut single);
        assert!(is_sorted(&single));

        // Already sorted
        let mut sorted: Vec<u64> = (0..100).collect();
        sorter.sort(&mut sorted);
        assert!(is_sorted(&sorted));

        // Reverse sorted
        let mut reverse: Vec<u64> = (0..100).rev().collect();
        sorter.sort(&mut reverse);
        assert!(is_sorted(&reverse));

        // Random
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        let mut random: Vec<u64> = (0..1000).map(|_| rng.gen()).collect();
        sorter.sort(&mut random);
        assert!(is_sorted(&random));

        // All same
        let mut same = vec![7u64; 100];
        sorter.sort(&mut same);
        assert!(is_sorted(&same));

        // Two elements
        let mut two = vec![5u64, 3u64];
        sorter.sort(&mut two);
        assert!(is_sorted(&two));
    }

    #[test]
    fn test_std_sort() {
        test_sorter(&baselines::StdSort);
    }

    #[test]
    fn test_std_unstable() {
        test_sorter(&baselines::StdUnstable);
    }

    #[test]
    fn test_heap_sort() {
        test_sorter(&baselines::HeapSort);
    }

    #[test]
    fn test_evolved() {
        test_sorter(&evolved::EvolvedSorter);
    }
}
