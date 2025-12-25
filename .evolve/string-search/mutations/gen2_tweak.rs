use crate::SubstringSearcher;

pub struct EvolvedSearcher;

impl SubstringSearcher for EvolvedSearcher {
    #[inline(always)]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        let needle_len = needle.len();
        let haystack_len = haystack.len();
        if needle_len > haystack_len {
            return None;
        }

        // For very short needles, use optimized direct search
        if needle_len == 1 {
            return memchr::memchr(needle[0], haystack);
        }

        if needle_len == 2 {
            return memchr::memmem::find(haystack, needle);
        }

        // For longer needles, use raw pointer-based comparison
        let first = needle[0];
        let second = needle[1];
        let last = needle[needle_len - 1];

        let needle_ptr = needle.as_ptr();
        let haystack_ptr = haystack.as_ptr();
        let search_end = haystack_len - needle_len;

        let mut pos = 0;

        while pos <= search_end {
            // Use memchr to find first byte candidates
            if let Some(offset) = memchr::memchr(first, &haystack[pos..=search_end]) {
                let idx = pos + offset;

                unsafe {
                    let hay_ptr = haystack_ptr.add(idx);

                    // Quick reject: check last and second bytes
                    if *hay_ptr.add(needle_len - 1) == last && *hay_ptr.add(1) == second {
                        // Use raw pointer comparison for the middle section
                        let mut matches = true;
                        let mut i = 2;
                        while i < needle_len - 1 {
                            if *hay_ptr.add(i) != *needle_ptr.add(i) {
                                matches = false;
                                break;
                            }
                            i += 1;
                        }

                        if matches {
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
