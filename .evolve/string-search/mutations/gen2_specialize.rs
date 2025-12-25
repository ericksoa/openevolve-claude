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
            let mut pos = 0;
            while pos + 1 < haystack.len() {
                if let Some(offset) = memchr::memchr(first, &haystack[pos..]) {
                    let idx = pos + offset;
                    if idx + 1 < haystack.len() && haystack[idx + 1] == second {
                        return Some(idx);
                    }
                    pos = idx + 1;
                } else {
                    return None;
                }
            }
            return None;
        }

        if needle_len == 3 {
            let first = needle[0];
            let second = needle[1];
            let third = needle[2];
            let mut pos = 0;
            let end = haystack.len() - 2;
            while pos < end {
                if let Some(offset) = memchr::memchr(first, &haystack[pos..end]) {
                    let idx = pos + offset;
                    unsafe {
                        if *haystack.get_unchecked(idx + 1) == second
                            && *haystack.get_unchecked(idx + 2) == third
                        {
                            return Some(idx);
                        }
                    }
                    pos = idx + 1;
                } else {
                    return None;
                }
            }
            return None;
        }

        if needle_len == 4 {
            let b0 = needle[0];
            let b1 = needle[1];
            let b2 = needle[2];
            let b3 = needle[3];
            let mut pos = 0;
            let end = haystack.len() - 3;
            while pos < end {
                if let Some(offset) = memchr::memchr(b0, &haystack[pos..end]) {
                    let idx = pos + offset;
                    unsafe {
                        if *haystack.get_unchecked(idx + 3) == b3
                            && *haystack.get_unchecked(idx + 1) == b1
                            && *haystack.get_unchecked(idx + 2) == b2
                        {
                            return Some(idx);
                        }
                    }
                    pos = idx + 1;
                } else {
                    return None;
                }
            }
            return None;
        }

        if needle_len <= 8 {
            let first = needle[0];
            let last = needle[needle_len - 1];
            let mut pos = 0;
            let search_end = haystack.len() - needle_len + 1;

            while pos < search_end {
                if let Some(offset) = memchr::memchr2(first, last, &haystack[pos..search_end]) {
                    let idx = pos + offset;
                    unsafe {
                        let byte = *haystack.get_unchecked(idx);
                        if byte == first {
                            if *haystack.get_unchecked(idx + needle_len - 1) == last {
                                let mut matched = true;
                                for i in 1..needle_len - 1 {
                                    if *haystack.get_unchecked(idx + i) != *needle.get_unchecked(i) {
                                        matched = false;
                                        break;
                                    }
                                }
                                if matched {
                                    return Some(idx);
                                }
                            }
                            pos = idx + 1;
                        } else {
                            pos = idx + 1;
                        }
                    }
                } else {
                    return None;
                }
            }
            return None;
        }

        if needle_len <= 16 {
            let first = needle[0];
            let second = needle[1];
            let last = needle[needle_len - 1];
            let mut pos = 0;
            let search_end = haystack.len() - needle_len + 1;

            while pos < search_end {
                if let Some(offset) = memchr::memchr(first, &haystack[pos..search_end]) {
                    let idx = pos + offset;
                    unsafe {
                        if *haystack.get_unchecked(idx + needle_len - 1) == last
                            && *haystack.get_unchecked(idx + 1) == second
                        {
                            let mut matched = true;
                            for i in 2..needle_len - 1 {
                                if *haystack.get_unchecked(idx + i) != *needle.get_unchecked(i) {
                                    matched = false;
                                    break;
                                }
                            }
                            if matched {
                                return Some(idx);
                            }
                        }
                    }
                    pos = idx + 1;
                } else {
                    return None;
                }
            }
            return None;
        }

        let first = needle[0];
        let last = needle[needle_len - 1];
        let second = needle[1];

        let mut pos = 0;
        let search_end = haystack.len() - needle_len + 1;

        while pos < search_end {
            if let Some(offset) = memchr::memchr(first, &haystack[pos..search_end]) {
                let idx = pos + offset;

                unsafe {
                    if *haystack.get_unchecked(idx + needle_len - 1) == last
                        && *haystack.get_unchecked(idx + 1) == second
                    {
                        if haystack.get_unchecked(idx..idx + needle_len) == needle {
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
