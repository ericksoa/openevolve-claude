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
            return memchr::memmem::find(haystack, needle);
        }

        let first = needle[0];
        let second = needle[1];
        let last = needle[needle_len - 1];
        let mid = needle[needle_len / 2];

        let mut pos = 0;
        let search_end = haystack.len() - needle_len + 1;

        // Build bad character skip table for last byte
        let mut skip_table = [needle_len; 256];
        for i in 0..needle_len - 1 {
            skip_table[needle[i] as usize] = needle_len - 1 - i;
        }

        while pos < search_end {
            // Check from the end first (Boyer-Moore style)
            let check_pos = pos + needle_len - 1;

            unsafe {
                let h_last = *haystack.get_unchecked(check_pos);

                if h_last == last {
                    // Quick filters before full comparison
                    if *haystack.get_unchecked(pos) == first
                        && *haystack.get_unchecked(pos + 1) == second
                        && *haystack.get_unchecked(pos + needle_len / 2) == mid
                    {
                        // Full comparison with manual unrolling hint
                        let hay_slice = haystack.get_unchecked(pos..pos + needle_len);
                        if hay_slice == needle {
                            return Some(pos);
                        }
                    }
                    pos += 1;
                } else {
                    // Use skip table for large jumps
                    let skip = skip_table[h_last as usize];
                    pos += skip.max(1);
                }
            }
        }

        None
    }
}
