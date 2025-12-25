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

        let first = needle[0];
        let last = needle[needle_len - 1];
        let second = needle[1];

        let mut pos = 0;
        let search_end = haystack.len() - needle_len + 1;

        while pos < search_end {
            if let Some(offset) = memchr::memchr(first, &haystack[pos..search_end]) {
                let idx = pos + offset;

                unsafe {
                    if *haystack.get_unchecked(idx + needle_len - 1) == last
                        && *haystack.get_unchecked(idx + 1) == second
                    {
                        if verify_match_vectorized(haystack, needle, idx, needle_len) {
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

#[inline(always)]
unsafe fn verify_match_vectorized(haystack: &[u8], needle: &[u8], idx: usize, needle_len: usize) -> bool {
    let hay_ptr = haystack.as_ptr().add(idx);
    let needle_ptr = needle.as_ptr();

    let mut offset = 0;

    // Compare 8 bytes at a time using u64
    while offset + 8 <= needle_len {
        let hay_word = (hay_ptr.add(offset) as *const u64).read_unaligned();
        let needle_word = (needle_ptr.add(offset) as *const u64).read_unaligned();
        if hay_word != needle_word {
            return false;
        }
        offset += 8;
    }

    // Compare 4 bytes at a time using u32
    if offset + 4 <= needle_len {
        let hay_word = (hay_ptr.add(offset) as *const u32).read_unaligned();
        let needle_word = (needle_ptr.add(offset) as *const u32).read_unaligned();
        if hay_word != needle_word {
            return false;
        }
        offset += 4;
    }

    // Compare 2 bytes at a time using u16
    if offset + 2 <= needle_len {
        let hay_word = (hay_ptr.add(offset) as *const u16).read_unaligned();
        let needle_word = (needle_ptr.add(offset) as *const u16).read_unaligned();
        if hay_word != needle_word {
            return false;
        }
        offset += 2;
    }

    // Compare remaining byte
    if offset < needle_len {
        if *hay_ptr.add(offset) != *needle_ptr.add(offset) {
            return false;
        }
    }

    true
}
