use crate::SubstringSearcher;

pub struct EvolvedSearcher;

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

        // For very short needles, use optimized direct search
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

        // For longer needles, use memchr + SIMD-friendly verification
        let first = needle[0];
        let last = needle[needle_len - 1];
        let second = needle[1];

        let mut pos = 0;
        let search_end = haystack.len() - needle_len + 1;

        while pos < search_end {
            // Use memchr to find first byte candidates
            if let Some(offset) = memchr::memchr(first, &haystack[pos..search_end]) {
                let idx = pos + offset;

                // Quick reject: check last and second bytes
                unsafe {
                    if *haystack.get_unchecked(idx + needle_len - 1) == last
                        && *haystack.get_unchecked(idx + 1) == second
                    {
                        // Full comparison
                        if haystack.get_unchecked(idx..idx + needle_len) == needle {
                            return Some(idx);
                        }
                    }
                }

                pos = idx + 1;
            } else {
                return None;
            }
        }

        None
    }
}
