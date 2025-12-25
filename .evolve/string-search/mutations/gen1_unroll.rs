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

        let first = needle[0];
        let last = needle[needle.len() - 1];
        let needle_len = needle.len();
        let end = haystack.len() - needle_len + 1;

        let mut i = 0;

        // Unroll by 4: check 4 positions at once
        while i + 4 <= end {
            unsafe {
                let h0 = *haystack.get_unchecked(i);
                let h1 = *haystack.get_unchecked(i + 1);
                let h2 = *haystack.get_unchecked(i + 2);
                let h3 = *haystack.get_unchecked(i + 3);

                if h0 == first && *haystack.get_unchecked(i + needle_len - 1) == last {
                    if haystack.get_unchecked(i..i + needle_len) == needle {
                        return Some(i);
                    }
                }

                if h1 == first && *haystack.get_unchecked(i + needle_len) == last {
                    if haystack.get_unchecked(i + 1..i + 1 + needle_len) == needle {
                        return Some(i + 1);
                    }
                }

                if h2 == first && *haystack.get_unchecked(i + needle_len + 1) == last {
                    if haystack.get_unchecked(i + 2..i + 2 + needle_len) == needle {
                        return Some(i + 2);
                    }
                }

                if h3 == first && *haystack.get_unchecked(i + needle_len + 2) == last {
                    if haystack.get_unchecked(i + 3..i + 3 + needle_len) == needle {
                        return Some(i + 3);
                    }
                }
            }

            i += 4;
        }

        // Handle remaining positions
        while i < end {
            unsafe {
                if *haystack.get_unchecked(i) == first
                    && *haystack.get_unchecked(i + needle_len - 1) == last
                {
                    if haystack.get_unchecked(i..i + needle_len) == needle {
                        return Some(i);
                    }
                }
            }
            i += 1;
        }

        None
    }
}
