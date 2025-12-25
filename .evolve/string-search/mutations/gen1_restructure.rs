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
        let haystack_len = haystack.len();

        if needle_len == 1 {
            return haystack.iter().position(|&b| b == needle[0]);
        }

        // Boyer-Moore-Horspool bad character shift table
        let mut bad_char_skip = [needle_len; 256];
        for i in 0..needle_len - 1 {
            bad_char_skip[needle[i] as usize] = needle_len - 1 - i;
        }

        let last = needle[needle_len - 1];
        let mut pos = 0;

        unsafe {
            let haystack_ptr = haystack.as_ptr();
            let needle_ptr = needle.as_ptr();

            while pos <= haystack_len - needle_len {
                let hay_ptr = haystack_ptr.add(pos);

                if *hay_ptr.add(needle_len - 1) == last {
                    let mut matched = true;
                    for i in 0..needle_len - 1 {
                        if *hay_ptr.add(i) != *needle_ptr.add(i) {
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        return Some(pos);
                    }
                }

                pos += bad_char_skip[*hay_ptr.add(needle_len - 1) as usize];
            }
        }

        None
    }
}
