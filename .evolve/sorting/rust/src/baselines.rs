use crate::Sorter;

/// Standard library stable sort
pub struct StdSort;

impl Sorter for StdSort {
    #[inline]
    fn sort(&self, data: &mut [u64]) {
        data.sort();
    }
}

/// Standard library unstable sort (faster, no stability guarantee)
pub struct StdUnstable;

impl Sorter for StdUnstable {
    #[inline]
    fn sort(&self, data: &mut [u64]) {
        data.sort_unstable();
    }
}

/// Heap sort - O(n log n) worst case, in-place
pub struct HeapSort;

impl Sorter for HeapSort {
    fn sort(&self, data: &mut [u64]) {
        if data.len() <= 1 {
            return;
        }

        // Build max heap
        let n = data.len();
        for i in (0..n / 2).rev() {
            heapify(data, n, i);
        }

        // Extract elements from heap
        for i in (1..n).rev() {
            data.swap(0, i);
            heapify(data, i, 0);
        }
    }
}

#[inline]
fn heapify(data: &mut [u64], n: usize, i: usize) {
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;

    if left < n && data[left] > data[largest] {
        largest = left;
    }
    if right < n && data[right] > data[largest] {
        largest = right;
    }

    if largest != i {
        data.swap(i, largest);
        heapify(data, n, largest);
    }
}

/// Simple quicksort with median-of-three pivot
pub struct QuickSort;

impl Sorter for QuickSort {
    fn sort(&self, data: &mut [u64]) {
        quicksort(data);
    }
}

fn quicksort(data: &mut [u64]) {
    if data.len() <= 1 {
        return;
    }

    // Insertion sort for small arrays
    if data.len() <= 16 {
        insertion_sort(data);
        return;
    }

    let pivot_idx = partition(data);
    let (left, right) = data.split_at_mut(pivot_idx);
    quicksort(left);
    quicksort(&mut right[1..]);
}

#[inline]
fn partition(data: &mut [u64]) -> usize {
    let len = data.len();
    let mid = len / 2;

    // Median of three
    if data[0] > data[mid] {
        data.swap(0, mid);
    }
    if data[mid] > data[len - 1] {
        data.swap(mid, len - 1);
    }
    if data[0] > data[mid] {
        data.swap(0, mid);
    }

    let pivot = data[mid];
    data.swap(mid, len - 1);

    let mut i = 0;
    for j in 0..len - 1 {
        if data[j] <= pivot {
            data.swap(i, j);
            i += 1;
        }
    }
    data.swap(i, len - 1);
    i
}

#[inline]
fn insertion_sort(data: &mut [u64]) {
    for i in 1..data.len() {
        let mut j = i;
        while j > 0 && data[j - 1] > data[j] {
            data.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Radix sort - O(n * k) where k is number of digits
/// Very fast for integers but uses O(n) extra space
pub struct RadixSort;

impl Sorter for RadixSort {
    fn sort(&self, data: &mut [u64]) {
        if data.len() <= 1 {
            return;
        }

        // LSD radix sort with 8-bit digits (8 passes for u64)
        let mut output = vec![0u64; data.len()];
        let mut count = [0usize; 256];

        for shift in (0..64).step_by(8) {
            // Count occurrences
            count.fill(0);
            for &val in data.iter() {
                let digit = ((val >> shift) & 0xFF) as usize;
                count[digit] += 1;
            }

            // Compute cumulative count
            for i in 1..256 {
                count[i] += count[i - 1];
            }

            // Build output (iterate in reverse for stability)
            for &val in data.iter().rev() {
                let digit = ((val >> shift) & 0xFF) as usize;
                count[digit] -= 1;
                output[count[digit]] = val;
            }

            // Copy back
            data.copy_from_slice(&output);
        }
    }
}
