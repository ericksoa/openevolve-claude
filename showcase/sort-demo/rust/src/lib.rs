//! Sort Algorithm Benchmark
//!
//! Demonstrates evolving a naive bubble sort into something faster.

pub mod baselines;
pub mod evolved;

/// Trait for sorting implementations
pub trait Sorter {
    fn sort(&self, data: &mut [i32]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_sorted(data: &[i32]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    fn test_sorter<S: Sorter>(s: &S) {
        // Empty
        let mut data: Vec<i32> = vec![];
        s.sort(&mut data);
        assert!(is_sorted(&data));

        // Single element
        let mut data = vec![42];
        s.sort(&mut data);
        assert!(is_sorted(&data));

        // Already sorted
        let mut data = vec![1, 2, 3, 4, 5];
        s.sort(&mut data);
        assert!(is_sorted(&data));

        // Reverse sorted
        let mut data = vec![5, 4, 3, 2, 1];
        s.sort(&mut data);
        assert!(is_sorted(&data));

        // Random
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        s.sort(&mut data);
        assert!(is_sorted(&data));

        // Duplicates
        let mut data = vec![1, 1, 1, 1, 1];
        s.sort(&mut data);
        assert!(is_sorted(&data));

        // Large random
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut data: Vec<i32> = (0..1000).map(|_| rng.gen_range(-10000..10000)).collect();
        s.sort(&mut data);
        assert!(is_sorted(&data));
    }

    #[test]
    fn test_bubble() { test_sorter(&baselines::BubbleSorter); }

    #[test]
    fn test_std() { test_sorter(&baselines::StdSorter); }

    #[test]
    fn test_evolved() { test_sorter(&evolved::EvolvedSorter); }
}
