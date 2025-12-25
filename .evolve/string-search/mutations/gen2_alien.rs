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

        // Rabin-Karp rolling hash approach
        let first = needle[0];
        let last = needle[needle_len - 1];

        const BASE: u64 = 257;
        let mut needle_hash: u64 = 0;
        let mut pow: u64 = 1;

        unsafe {
            for i in 0..needle_len {
                needle_hash = needle_hash.wrapping_mul(BASE).wrapping_add(*needle.get_unchecked(i) as u64);
                if i < needle_len - 1 {
                    pow = pow.wrapping_mul(BASE);
                }
            }
        }

        let mut pos = 0;
        let search_end = haystack.len() - needle_len + 1;

        while pos < search_end {
            if let Some(offset) = memchr::memchr(first, &haystack[pos..search_end]) {
                let idx = pos + offset;

                unsafe {
                    if *haystack.get_unchecked(idx + needle_len - 1) != last {
                        pos = idx + 1;
                        continue;
                    }

                    let mut hash: u64 = 0;
                    for i in 0..needle_len {
                        hash = hash.wrapping_mul(BASE).wrapping_add(*haystack.get_unchecked(idx + i) as u64);
                    }

                    if hash == needle_hash {
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
