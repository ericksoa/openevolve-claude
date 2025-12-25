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
                if idx + 1 < haystack.len() && unsafe { *haystack.get_unchecked(idx + 1) } == second {
                    return Some(idx);
                }
            }
            return None;
        }

        let first = needle[0];
        let last = needle[needle_len - 1];

        let first_count = haystack.iter().filter(|&&b| b == first).count();
        let last_count = haystack.iter().filter(|&&b| b == last).count();

        let search_end = haystack.len() - needle_len + 1;

        if first_count <= last_count {
            for idx in memchr::memchr_iter(first, &haystack[..search_end]) {
                unsafe {
                    if *haystack.get_unchecked(idx + needle_len - 1) == last {
                        if haystack.get_unchecked(idx..idx + needle_len) == needle {
                            return Some(idx);
                        }
                    }
                }
            }
        } else {
            for i in memchr::memchr_iter(last, &haystack[needle_len - 1..]) {
                let idx = i + needle_len - 1 - (needle_len - 1);
                if idx < search_end {
                    unsafe {
                        if *haystack.get_unchecked(idx) == first {
                            if haystack.get_unchecked(idx..idx + needle_len) == needle {
                                return Some(idx);
                            }
                        }
                    }
                }
            }
        }

        None
    }
}
