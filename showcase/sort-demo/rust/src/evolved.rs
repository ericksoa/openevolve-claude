use crate::Sorter;

pub struct EvolvedSorter;

impl Sorter for EvolvedSorter {
    fn sort(&self, data: &mut [i32]) {
        evolved_sort(data);
    }
}

fn evolved_sort(data: &mut [i32]) {
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
fn radix_sort_lsd(data: &mut [i32]) {
    let len = data.len();
    let mut buffer = vec![0i32; len];

    const BITS: u32 = 11;
    const RADIX: usize = 1 << BITS;
    const MASK: u32 = (RADIX - 1) as u32;
    const PASSES: u32 = (32 + BITS - 1) / BITS;

    for pass in 0..PASSES {
        let shift = pass * BITS;
        let mut counts = [0usize; RADIX];

        // Count occurrences - handle negatives by flipping sign bit
        for &value in data.iter() {
            let key = flip_sign_bit(value as u32);
            let digit = ((key >> shift) & MASK) as usize;
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
            let key = flip_sign_bit(value as u32);
            let digit = ((key >> shift) & MASK) as usize;
            buffer[counts[digit]] = value;
            counts[digit] += 1;
        }

        // Copy back
        data.copy_from_slice(&buffer);
    }
}

#[inline]
fn flip_sign_bit(val: u32) -> u32 {
    val ^ 0x80000000
}

#[inline]
fn insertion_sort(data: &mut [i32]) {
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
