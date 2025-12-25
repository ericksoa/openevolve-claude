use crate::Sorter;

pub struct EvolvedSorter;

impl Sorter for EvolvedSorter {
    fn sort(&self, data: &mut [u64]) {
        evolved_sort(data);
    }
}

fn evolved_sort(data: &mut [u64]) {
    if data.len() <= 1 { return; }
    if data.len() <= 64 { insertion_sort(data); return; }
    radix_sort_lsd(data);
}

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
fn radix_sort_lsd(data: &mut [u64]) {
    let len = data.len();

    // Find max value to determine required passes
    let max_val = *data.iter().max().unwrap();
    if max_val == 0 { return; }

    let mut buffer = vec![0u64; len];
    const BITS: u32 = 11;
    const RADIX: usize = 1 << BITS;
    const MASK: u64 = (RADIX - 1) as u64;

    // Calculate actual number of passes needed
    let max_passes = (64 - max_val.leading_zeros() + BITS - 1) / BITS;

    let mut src = data;
    let mut dst = &mut buffer[..];

    for pass in 0..max_passes {
        let shift = pass * BITS;
        let mut counts = [0usize; RADIX];

        // Count phase using unsafe for speed
        unsafe {
            let src_ptr = src.as_ptr();
            for i in 0..len {
                let value = *src_ptr.add(i);
                let digit = ((value >> shift) & MASK) as usize;
                *counts.get_unchecked_mut(digit) += 1;
            }
        }

        // Check if all values have same digit (already sorted for this pass)
        let first_nonzero = counts.iter().position(|&c| c != 0);
        if let Some(idx) = first_nonzero {
            if counts[idx] == len {
                continue;
            }
        }

        // Prefix sum
        let mut sum = 0;
        for count in counts.iter_mut() {
            let temp = *count;
            *count = sum;
            sum += temp;
        }

        // Distribute phase using unsafe for speed
        unsafe {
            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            for i in 0..len {
                let value = *src_ptr.add(i);
                let digit = ((value >> shift) & MASK) as usize;
                let pos = counts.get_unchecked_mut(digit);
                *dst_ptr.add(*pos) = value;
                *pos += 1;
            }
        }

        // Swap buffers
        std::mem::swap(&mut src, &mut dst);
    }

    // If odd number of passes, copy back to data
    if max_passes % 2 == 1 {
        data.copy_from_slice(src);
    }
}
