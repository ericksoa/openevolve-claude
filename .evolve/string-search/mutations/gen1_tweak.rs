use crate::SubstringSearcher;

pub struct EvolvedSearcher;

impl SubstringSearcher for EvolvedSearcher {
    #[inline(always)]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        let needle_len = needle.len();

        if needle_len == 0 {
            return Some(0);
        }

        let haystack_len = haystack.len();

        if needle_len > haystack_len {
            return None;
        }

        unsafe {
            let haystack_ptr = haystack.as_ptr();
            let needle_ptr = needle.as_ptr();
            let first = *needle_ptr;
            let last = *needle_ptr.add(needle_len - 1);
            let end = haystack_len - needle_len + 1;

            for i in 0..end {
                let h_ptr = haystack_ptr.add(i);

                if *h_ptr == first && *h_ptr.add(needle_len - 1) == last {
                    let mut match_found = true;
                    for j in 1..needle_len - 1 {
                        if *h_ptr.add(j) != *needle_ptr.add(j) {
                            match_found = false;
                            break;
                        }
                    }
                    if match_found {
                        return Some(i);
                    }
                }
            }
        }

        None
    }
}
