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

        // Fast path for single byte
        if needle_len == 1 {
            return haystack.iter().position(|&b| b == needle[0]);
        }

        // Build bad character skip table (Boyer-Moore)
        let mut skip_table = [needle_len; 256];
        for i in 0..needle_len - 1 {
            skip_table[needle[i] as usize] = needle_len - 1 - i;
        }

        let last_needle = needle[needle_len - 1];
        let first_needle = needle[0];
        let end = haystack.len() - needle_len;

        let mut i = 0;

        unsafe {
            while i <= end {
                let h_ptr = haystack.as_ptr().add(i);

                // Check last character first (Boyer-Moore style)
                let last_byte = *h_ptr.add(needle_len - 1);

                if last_byte == last_needle {
                    // Check first character
                    if *h_ptr == first_needle {
                        // Check middle bytes
                        let mut match_found = true;
                        for j in 1..needle_len - 1 {
                            if *h_ptr.add(j) != *needle.as_ptr().add(j) {
                                match_found = false;
                                break;
                            }
                        }
                        if match_found {
                            return Some(i);
                        }
                    }
                    i += 1;
                } else {
                    // Skip using bad character table
                    i += skip_table[last_byte as usize];
                }
            }
        }

        None
    }
}
