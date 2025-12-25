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
            for idx in memchr::memchr_iter(first, haystack) {
                if idx + 1 < haystack.len() && haystack[idx + 1] == second {
                    return Some(idx);
                }
            }
            return None;
        }

        let first = needle[0];
        let second = needle[1];
        let last = needle[needle_len - 1];
        let search_end = haystack.len() - needle_len + 1;

        for idx in memchr::memchr_iter(first, &haystack[..search_end]) {
            unsafe {
                if *haystack.get_unchecked(idx + needle_len - 1) == last
                    && *haystack.get_unchecked(idx + 1) == second
                {
                    let mut match_found = true;
                    for i in 2..needle_len - 1 {
                        if *haystack.get_unchecked(idx + i) != *needle.get_unchecked(i) {
                            match_found = false;
                            break;
                        }
                    }
                    if match_found {
                        return Some(idx);
                    }
                }
            }
        }

        None
    }
}
