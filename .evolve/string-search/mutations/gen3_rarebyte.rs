use crate::SubstringSearcher;

pub struct EvolvedSearcher;

const BYTE_FREQUENCIES: [u8; 256] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 200, 200, 255, 255, 200, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    100, 150, 140, 180, 180, 180, 180, 150, 140, 140, 180, 180, 130, 140, 130, 170,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 150, 140, 180, 180, 180, 180,
    180, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170,
    170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 140, 180, 140, 180, 150,
    180, 120, 150, 130, 140, 110, 150, 150, 140, 115, 180, 160, 130, 140, 115, 115,
    145, 180, 125, 115, 110, 125, 150, 155, 170, 145, 170, 180, 180, 180, 180, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

#[inline(always)]
fn find_rarest_byte(needle: &[u8]) -> usize {
    let mut rarest_idx = 0;
    let mut rarest_freq = BYTE_FREQUENCIES[needle[0] as usize];

    for i in 1..needle.len() {
        let freq = BYTE_FREQUENCIES[needle[i] as usize];
        if freq > rarest_freq {
            rarest_freq = freq;
            rarest_idx = i;
        }
    }

    rarest_idx
}

impl SubstringSearcher for EvolvedSearcher {
    #[inline(always)]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }

        let needle_len = needle.len();

        if needle_len == 1 {
            return memchr::memchr(needle[0], haystack);
        }

        if needle_len == 2 {
            let first = needle[0];
            let second = needle[1];
            let mut pos = 0;
            while pos + 1 < haystack.len() {
                if let Some(offset) = memchr::memchr(first, &haystack[pos..]) {
                    let idx = pos + offset;
                    if idx + 1 < haystack.len() && haystack[idx + 1] == second {
                        return Some(idx);
                    }
                    pos = idx + 1;
                } else {
                    return None;
                }
            }
            return None;
        }

        let rarest_idx = find_rarest_byte(needle);
        let rarest_byte = needle[rarest_idx];

        let mut pos = 0;
        let search_end = haystack.len() - needle_len + 1;

        while pos < search_end {
            if let Some(offset) = memchr::memchr(rarest_byte, &haystack[pos + rarest_idx..]) {
                let candidate_start = pos + offset;

                if candidate_start >= search_end {
                    return None;
                }

                unsafe {
                    let hay_ptr = haystack.as_ptr().add(candidate_start);
                    let needle_ptr = needle.as_ptr();

                    let mut matches = true;
                    for i in 0..needle_len {
                        if *hay_ptr.add(i) != *needle_ptr.add(i) {
                            matches = false;
                            break;
                        }
                    }

                    if matches {
                        return Some(candidate_start);
                    }
                }

                pos = candidate_start + 1;
            } else {
                return None;
            }
        }

        None
    }
}
