use crate::Sorter;

/// Classic O(n²) bubble sort - the "bad" algorithm we want to beat
pub struct BubbleSorter;

impl Sorter for BubbleSorter {
    fn sort(&self, data: &mut [i32]) {
        let n = data.len();
        if n <= 1 { return; }

        for i in 0..n {
            for j in 0..n - 1 - i {
                if data[j] > data[j + 1] {
                    data.swap(j, j + 1);
                }
            }
        }
    }
}

/// Standard library sort - the gold standard
pub struct StdSorter;

impl Sorter for StdSorter {
    fn sort(&self, data: &mut [i32]) {
        data.sort();
    }
}

/// Unstable sort - often faster
pub struct StdUnstableSorter;

impl Sorter for StdUnstableSorter {
    fn sort(&self, data: &mut [i32]) {
        data.sort_unstable();
    }
}
