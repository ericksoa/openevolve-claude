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
        let last = needle[needle_len - 1];
        let second = needle[1];

        let search_end = haystack.len() - needle_len + 1;
        let haystack_ptr = haystack.as_ptr();
        let needle_ptr = needle.as_ptr();

        for idx in memchr::memchr_iter(first, &haystack[..search_end]) {
            unsafe {
                if *haystack_ptr.add(idx + needle_len - 1) == last
                    && *haystack_ptr.add(idx + 1) == second
                {
                    let mut i = 2;
                    while i < needle_len {
                        if *haystack_ptr.add(idx + i) != *needle_ptr.add(i) {
                            break;
                        }
                        i += 1;
                    }
                    if i == needle_len {
                        return Some(idx);
                    }
                }
            }
        }

        None
    }
}
