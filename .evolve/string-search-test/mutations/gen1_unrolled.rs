use crate::SubstringSearcher;

pub struct EvolvedSearcher;

impl SubstringSearcher for EvolvedSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() { return Some(0); }
        if needle.len() > haystack.len() { return None; }

        let needle_len = needle.len();
        let haystack_len = haystack.len();

        // Single byte: use memchr directly
        if needle_len == 1 {
            return memchr::memchr(needle[0], haystack);
        }

        // Fast path for short patterns
        if needle_len == 2 {
            let n0 = needle[0];
            let n1 = needle[1];
            let ptr = haystack.as_ptr();
            let end_ptr = unsafe { ptr.add(haystack_len - 1) };
            let mut p = ptr;
            while p < end_ptr {
                unsafe {
                    if *p == n0 && *p.add(1) == n1 {
                        return Some(p.offset_from(ptr) as usize);
                    }
                    p = p.add(1);
                }
            }
            return None;
        }

        if needle_len == 3 {
            let n0 = needle[0];
            let n1 = needle[1];
            let n2 = needle[2];
            let ptr = haystack.as_ptr();
            let end_ptr = unsafe { ptr.add(haystack_len - 2) };
            let mut p = ptr;
            while p < end_ptr {
                unsafe {
                    if *p == n0 && *p.add(1) == n1 && *p.add(2) == n2 {
                        return Some(p.offset_from(ptr) as usize);
                    }
                    p = p.add(1);
                }
            }
            return None;
        }

        // For longer needles: find rarest byte to minimize memchr calls
        let mut freq = [0u8; 256];
        for &b in needle.iter() {
            freq[b as usize] = freq[b as usize].saturating_add(1);
        }

        let mut min_freq = 256u8;
        let mut rarest_byte = needle[0];
        let mut rarest_pos = 0usize;

        for (i, &b) in needle.iter().enumerate() {
            let f = freq[b as usize];
            if f < min_freq {
                min_freq = f;
                rarest_byte = b;
                rarest_pos = i;
            }
        }

        let first_byte = needle[0];
        let last_byte = needle[needle_len - 1];

        // Quick first+last byte check for unrolled verification
        let mut search_start = 0;
        while search_start + needle_len <= haystack_len {
            match memchr::memchr(rarest_byte, &haystack[search_start..]) {
                Some(offset) => {
                    let rarest_found_pos = search_start + offset;

                    // Calculate where needle would start
                    if rarest_found_pos >= rarest_pos {
                        let potential_start = rarest_found_pos - rarest_pos;

                        // Bounds check
                        if potential_start + needle_len <= haystack_len {
                            // Quick check: first and last bytes before full comparison
                            if haystack[potential_start] == first_byte && haystack[potential_start + needle_len - 1] == last_byte {
                                // Full verification
                                if unsafe { haystack.get_unchecked(potential_start..potential_start + needle_len) == &needle[..] } {
                                    return Some(potential_start);
                                }
                            }
                        }
                    }

                    // Move past this position to find next candidate
                    search_start = rarest_found_pos + 1;
                }
                None => return None,
            }
        }

        None
    }
}
