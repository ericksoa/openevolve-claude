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
                        // Unrolled comparison for 8-byte chunks
                        let haystack_ptr = haystack.as_ptr().add(idx);
                        let needle_ptr = needle.as_ptr();
                        let mut offset = 0;
                        let chunks = needle_len / 8;

                        // Process 8 bytes at a time
                        let mut matches = true;
                        for _ in 0..chunks {
                            let h = (haystack_ptr.add(offset) as *const u64).read_unaligned();
                            let n = (needle_ptr.add(offset) as *const u64).read_unaligned();
                            if h != n {
                                matches = false;
                                break;
                            }
                            offset += 8;
                        }

                        // Handle remaining bytes
                        if matches {
                            while offset < needle_len {
                                if *haystack_ptr.add(offset) != *needle_ptr.add(offset) {
                                    matches = false;
                                    break;
                                }
                                offset += 1;
                            }

                            if matches {
                                return Some(idx);
                            }
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
