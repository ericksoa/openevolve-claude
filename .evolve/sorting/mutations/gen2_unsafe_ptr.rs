use crate::Sorter;
use std::ptr;

pub struct EvolvedSorter;

impl Sorter for EvolvedSorter {
    fn sort(&self, data: &mut [u64]) {
        if data.len() <= 1 {
            return;
        }

        unsafe {
            radix_sort_11bit_unsafe(data);
        }
    }
}

#[inline(always)]
unsafe fn radix_sort_11bit_unsafe(data: &mut [u64]) {
    const BITS: u32 = 11;
    const BUCKETS: usize = 1 << BITS;
    const MASK: u64 = (BUCKETS - 1) as u64;
    const PASSES: u32 = (64 + BITS - 1) / BITS;

    let len = data.len();
    let mut buffer = Vec::with_capacity(len);
    buffer.set_len(len);

    let mut counts = vec![0usize; BUCKETS];
    let mut src = data.as_mut_ptr();
    let mut dst = buffer.as_mut_ptr();

    for pass in 0..PASSES {
        let shift = pass * BITS;

        // Zero counts using pointer arithmetic
        let counts_ptr = counts.as_mut_ptr();
        for i in 0..BUCKETS {
            *counts_ptr.add(i) = 0;
        }

        // Count with unrolling
        let mut i = 0;
        let unroll_limit = len & !7;

        while i < unroll_limit {
            let v0 = *src.add(i);
            let v1 = *src.add(i + 1);
            let v2 = *src.add(i + 2);
            let v3 = *src.add(i + 3);
            let v4 = *src.add(i + 4);
            let v5 = *src.add(i + 5);
            let v6 = *src.add(i + 6);
            let v7 = *src.add(i + 7);

            let b0 = ((v0 >> shift) & MASK) as usize;
            let b1 = ((v1 >> shift) & MASK) as usize;
            let b2 = ((v2 >> shift) & MASK) as usize;
            let b3 = ((v3 >> shift) & MASK) as usize;
            let b4 = ((v4 >> shift) & MASK) as usize;
            let b5 = ((v5 >> shift) & MASK) as usize;
            let b6 = ((v6 >> shift) & MASK) as usize;
            let b7 = ((v7 >> shift) & MASK) as usize;

            *counts_ptr.add(b0) += 1;
            *counts_ptr.add(b1) += 1;
            *counts_ptr.add(b2) += 1;
            *counts_ptr.add(b3) += 1;
            *counts_ptr.add(b4) += 1;
            *counts_ptr.add(b5) += 1;
            *counts_ptr.add(b6) += 1;
            *counts_ptr.add(b7) += 1;

            i += 8;
        }

        while i < len {
            let bucket = ((*src.add(i) >> shift) & MASK) as usize;
            *counts_ptr.add(bucket) += 1;
            i += 1;
        }

        // Prefix sum
        let mut sum = 0;
        for i in 0..BUCKETS {
            let tmp = *counts_ptr.add(i);
            *counts_ptr.add(i) = sum;
            sum += tmp;
        }

        // Distribute
        for i in 0..len {
            let value = *src.add(i);
            let bucket = ((value >> shift) & MASK) as usize;
            let pos = counts_ptr.add(bucket);
            let idx = *pos;
            *dst.add(idx) = value;
            *pos += 1;
        }

        // Swap buffers
        ptr::swap(&mut src, &mut dst);
    }

    // Copy back if needed
    if PASSES % 2 == 1 {
        ptr::copy_nonoverlapping(buffer.as_ptr(), data.as_mut_ptr(), len);
    }
}
