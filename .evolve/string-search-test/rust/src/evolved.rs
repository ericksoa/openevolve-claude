use crate::SubstringSearcher;

pub struct EvolvedSearcher;

impl SubstringSearcher for EvolvedSearcher {
    #[inline]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() { return Some(0); }
        if needle.len() > haystack.len() { return None; }

        let needle_len = needle.len();
        let haystack_len = haystack.len();

        // Single byte: delegate directly to memchr
        if needle_len == 1 {
            return memchr::memchr(needle[0], haystack);
        }

        // Two bytes: memchr + verify
        if needle_len == 2 {
            let first = needle[0];
            let second = needle[1];
            let mut pos = 0;
            while pos + 1 < haystack_len {
                if let Some(offset) = memchr::memchr(first, &haystack[pos..haystack_len - 1]) {
                    let idx = pos + offset;
                    if haystack[idx + 1] == second {
                        return Some(idx);
                    }
                    pos = idx + 1;
                } else {
                    return None;
                }
            }
            return None;
        }

        // Find the rarest byte in the needle for anchoring
        let mut freq = [0u16; 256];
        for &b in needle {
            freq[b as usize] += 1;
        }

        let mut rarest_byte = needle[0];
        let mut rarest_pos = 0;
        let mut min_freq = freq[needle[0] as usize];

        for (i, &b) in needle.iter().enumerate() {
            if freq[b as usize] < min_freq {
                min_freq = freq[b as usize];
                rarest_byte = b;
                rarest_pos = i;
            }
        }

        // Use memchr on the rarest byte for candidate positions
        let first = needle[0];
        let last = needle[needle_len - 1];
        let search_limit = haystack_len - needle_len + 1;

        let mut pos = 0;
        while pos < search_limit {
            // Find rarest byte candidate
            let search_slice = &haystack[pos + rarest_pos..];
            if let Some(offset) = memchr::memchr(rarest_byte, search_slice) {
                let candidate = pos + offset;

                if candidate + needle_len > haystack_len {
                    return None;
                }

                // Quick filter: check first and last bytes
                unsafe {
                    if *haystack.get_unchecked(candidate) == first
                        && *haystack.get_unchecked(candidate + needle_len - 1) == last
                    {
                        // Full verification
                        if &haystack[candidate..candidate + needle_len] == needle {
                            return Some(candidate);
                        }
                    }
                }

                pos = candidate + 1;
            } else {
                return None;
            }
        }

        None
    }
}
