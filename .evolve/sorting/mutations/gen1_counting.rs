use crate::Sorter;

pub struct EvolvedSorter;

impl Sorter for EvolvedSorter {
    fn sort(&self, data: &mut [u64]) {
        let len = data.len();
        if len <= 1 {
            return;
        }

        if len <= 32 {
            insertion_sort(data);
            return;
        }

        // Find min/max in single pass
        let (mut min, mut max) = (data[0], data[0]);
        for &val in &data[1..] {
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
        }

        // If all elements are the same
        if min == max {
            return;
        }

        let range = max - min;

        // Use counting sort for small ranges
        if range < 10_000_000 {
            counting_sort(data, min, max);
        } else {
            radix_sort_11bit(data);
        }
    }
}

#[inline]
fn insertion_sort(data: &mut [u64]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;
        while j > 0 && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

#[inline]
fn counting_sort(data: &mut [u64], min: u64, max: u64) {
    let range = (max - min + 1) as usize;
    let mut count = vec![0u32; range];

    // Count occurrences
    for &val in data.iter() {
        count[(val - min) as usize] += 1;
    }

    // Write back sorted values
    let mut idx = 0;
    for (i, &cnt) in count.iter().enumerate() {
        let val = min + i as u64;
        for _ in 0..cnt {
            data[idx] = val;
            idx += 1;
        }
    }
}

#[inline]
fn radix_sort_11bit(data: &mut [u64]) {
    let len = data.len();
    let mut output = vec![0u64; len];
    let mut count = vec![0u32; 2048]; // 2^11 = 2048

    // 6 passes with 11-bit digits (covers 66 bits, enough for u64)
    for shift in [0, 11, 22, 33, 44, 55] {
        // Reset counts
        unsafe {
            std::ptr::write_bytes(count.as_mut_ptr(), 0, 2048);
        }

        // Count occurrences
        for &val in data.iter() {
            let digit = ((val >> shift) & 0x7FF) as usize;
            count[digit] += 1;
        }

        // Compute cumulative count
        for i in 1..2048 {
            count[i] += count[i - 1];
        }

        // Build output (reverse iteration for stability)
        for &val in data.iter().rev() {
            let digit = ((val >> shift) & 0x7FF) as usize;
            count[digit] -= 1;
            unsafe {
                *output.get_unchecked_mut(count[digit] as usize) = val;
            }
        }

        // Swap buffers
        std::mem::swap(data, &mut output);
    }
}
