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
            return memchr::memchr_iter(first, haystack)
                .find(|&idx| idx + 1 < haystack.len() && unsafe { *haystack.get_unchecked(idx + 1) == second })
                .or(None);
        }

        if needle_len == 3 {
            let first = needle[0];
            let second = needle[1];
            let third = needle[2];
            return memchr::memchr_iter(first, haystack)
                .find(|&idx| {
                    idx + 2 < haystack.len() && unsafe {
                        *haystack.get_unchecked(idx + 1) == second
                            && *haystack.get_unchecked(idx + 2) == third
                    }
                })
                .or(None);
        }

        if needle_len == 4 {
            let needle_word = unsafe { *(needle.as_ptr() as *const u32) };
            return memchr::memchr_iter(needle[0], haystack)
                .find(|&idx| {
                    idx + 3 < haystack.len() && unsafe {
                        let hay_word = *(haystack.as_ptr().add(idx) as *const u32);
                        hay_word == needle_word
                    }
                })
                .or(None);
        }

        if needle_len <= 8 {
            let first = needle[0];
            let last = needle[needle_len - 1];
            let second = needle[1];
            let mid = needle[needle_len / 2];
            let mid_idx = needle_len / 2;

            return memchr::memchr_iter(first, haystack)
                .find(|&idx| {
                    idx + needle_len <= haystack.len() && unsafe {
                        *haystack.get_unchecked(idx + needle_len - 1) == last
                            && *haystack.get_unchecked(idx + 1) == second
                            && *haystack.get_unchecked(idx + mid_idx) == mid
                            && haystack.get_unchecked(idx..idx + needle_len) == needle
                    }
                })
                .or(None);
        }

        let first = needle[0];
        let last = needle[needle_len - 1];
        let second = needle[1];
        let mid = needle[needle_len / 2];
        let mid_idx = needle_len / 2;

        let search_end = haystack.len() - needle_len + 1;

        memchr::memchr_iter(first, &haystack[..search_end])
            .find(|&idx| unsafe {
                *haystack.get_unchecked(idx + needle_len - 1) == last
                    && *haystack.get_unchecked(idx + 1) == second
                    && *haystack.get_unchecked(idx + mid_idx) == mid
                    && compare_words(haystack.as_ptr().add(idx), needle.as_ptr(), needle_len)
            })
            .or(None)
    }
}

#[inline(always)]
unsafe fn compare_words(hay: *const u8, needle: *const u8, len: usize) -> bool {
    let mut offset = 0;

    while offset + 8 <= len {
        let hay_word = *(hay.add(offset) as *const u64);
        let needle_word = *(needle.add(offset) as *const u64);
        if hay_word != needle_word {
            return false;
        }
        offset += 8;
    }

    if offset + 4 <= len {
        let hay_word = *(hay.add(offset) as *const u32);
        let needle_word = *(needle.add(offset) as *const u32);
        if hay_word != needle_word {
            return false;
        }
        offset += 4;
    }

    while offset < len {
        if *hay.add(offset) != *needle.add(offset) {
            return false;
        }
        offset += 1;
    }

    true
}
