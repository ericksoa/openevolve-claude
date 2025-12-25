use crate::Sorter;

pub struct EvolvedSorter;

impl Sorter for EvolvedSorter {
    fn sort(&self, data: &mut [u64]) {
        let len = data.len();
        if len <= 1 {
            return;
        }

        let mut aux = vec![0u64; len];
        let mut counts = [0usize; 256];

        // Process each byte (8 passes for u64)
        for shift in (0..64).step_by(8) {
            // Clear counts
            unsafe {
                std::ptr::write_bytes(counts.as_mut_ptr(), 0, 256);
            }

            // Count occurrences
            unsafe {
                let mut ptr = data.as_ptr();
                let end = ptr.add(len);
                while ptr < end {
                    let byte = ((*ptr >> shift) & 0xFF) as usize;
                    *counts.get_unchecked_mut(byte) += 1;
                    ptr = ptr.add(1);
                }
            }

            // Cumulative sum
            let mut sum = 0;
            for i in 0..256 {
                let tmp = counts[i];
                counts[i] = sum;
                sum += tmp;
            }

            // Distribute to aux
            unsafe {
                let mut ptr = data.as_ptr();
                let end = ptr.add(len);
                let aux_ptr = aux.as_mut_ptr();
                while ptr < end {
                    let val = *ptr;
                    let byte = ((val >> shift) & 0xFF) as usize;
                    let pos = counts.get_unchecked_mut(byte);
                    *aux_ptr.add(*pos) = val;
                    *pos += 1;
                    ptr = ptr.add(1);
                }
            }

            // Copy back
            unsafe {
                std::ptr::copy_nonoverlapping(aux.as_ptr(), data.as_mut_ptr(), len);
            }
        }
    }
}
