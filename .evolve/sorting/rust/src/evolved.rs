use crate::Sorter;

pub struct EvolvedSorter;

impl Sorter for EvolvedSorter {
    fn sort(&self, data: &mut [u64]) {
        evolved_sort(data);
    }
}

fn evolved_sort(data: &mut [u64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len() <= 64 {
        insertion_sort(data);
        return;
    }
    radix_sort_lsd(data);
}

#[inline]
fn radix_sort_lsd(data: &mut [u64]) {
    let len = data.len();
    let mut buffer = vec![0u64; len];

    const BITS: u32 = 11;
    const RADIX: usize = 1 << BITS;
    const MASK: u64 = (RADIX - 1) as u64;
    const PASSES: u32 = (64 + BITS - 1) / BITS;

    for pass in 0..PASSES {
        let shift = pass * BITS;
        let mut counts = [0usize; RADIX];

        // Count occurrences
        for &value in data.iter() {
            let digit = ((value >> shift) & MASK) as usize;
            counts[digit] += 1;
        }

        // Compute prefix sums
        let mut sum = 0;
        for count in counts.iter_mut() {
            let temp = *count;
            *count = sum;
            sum += temp;
        }

        // Distribute elements
        for &value in data.iter() {
            let digit = ((value >> shift) & MASK) as usize;
            buffer[counts[digit]] = value;
            counts[digit] += 1;
        }

        // Copy back
        data.copy_from_slice(&buffer);
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
