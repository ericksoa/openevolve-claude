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
            return haystack.iter().position(|&b| b == needle[0]);
        }

        let first = needle[0];
        let last = needle[needle_len - 1];
        let end = haystack.len() - needle_len + 1;

        // For short needles, use vectorized first-byte search
        if needle_len <= 8 {
            let first_broadcast = first as u64 * 0x0101010101010101u64;
            let mut i = 0;

            unsafe {
                // Process 8 bytes at a time
                while i + 8 <= end {
                    let chunk = (haystack.as_ptr().add(i) as *const u64).read_unaligned();
                    let xor = chunk ^ first_broadcast;
                    let has_zero = (xor.wrapping_sub(0x0101010101010101u64)) & !xor & 0x8080808080808080u64;

                    if has_zero != 0 {
                        // Check each potential match in this 8-byte block
                        for j in 0..8 {
                            if i + j >= end {
                                break;
                            }
                            if *haystack.get_unchecked(i + j) == first
                                && *haystack.get_unchecked(i + j + needle_len - 1) == last {
                                if haystack.get_unchecked(i + j..i + j + needle_len) == needle {
                                    return Some(i + j);
                                }
                            }
                        }
                    }
                    i += 8;
                }

                // Handle remaining bytes
                while i < end {
                    if *haystack.get_unchecked(i) == first
                        && *haystack.get_unchecked(i + needle_len - 1) == last {
                        if haystack.get_unchecked(i..i + needle_len) == needle {
                            return Some(i);
                        }
                    }
                    i += 1;
                }
            }
        } else {
            // For longer needles, use optimized comparison with early exit
            let mut i = 0;

            unsafe {
                while i < end {
                    if *haystack.get_unchecked(i) == first
                        && *haystack.get_unchecked(i + needle_len - 1) == last {

                        // Compare in chunks of 8 bytes where possible
                        let mut match_found = true;
                        let mut offset = 0;

                        while offset + 8 <= needle_len {
                            let h_chunk = (haystack.as_ptr().add(i + offset) as *const u64).read_unaligned();
                            let n_chunk = (needle.as_ptr().add(offset) as *const u64).read_unaligned();
                            if h_chunk != n_chunk {
                                match_found = false;
                                break;
                            }
                            offset += 8;
                        }

                        if match_found {
                            // Check remaining bytes
                            while offset < needle_len {
                                if *haystack.get_unchecked(i + offset) != *needle.get_unchecked(offset) {
                                    match_found = false;
                                    break;
                                }
                                offset += 1;
                            }

                            if match_found {
                                return Some(i);
                            }
                        }
                    }
                    i += 1;
                }
            }
        }

        None
    }
}
