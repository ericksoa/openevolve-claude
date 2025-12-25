use crate::SubstringSearcher;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { find_avx2(haystack, needle) };
            }
        }

        find_fallback(haystack, needle)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_avx2(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let needle_len = needle.len();
    let first = needle[0];
    let last = needle[needle_len - 1];
    let second = needle[1];

    let mut pos = 0;
    let search_end = haystack.len() - needle_len + 1;

    if needle_len <= 32 {
        let needle_vec = if needle_len == 32 {
            _mm256_loadu_si256(needle.as_ptr() as *const __m256i)
        } else {
            let mut buf = [0u8; 32];
            std::ptr::copy_nonoverlapping(needle.as_ptr(), buf.as_mut_ptr(), needle_len);
            _mm256_loadu_si256(buf.as_ptr() as *const __m256i)
        };

        while pos < search_end {
            if let Some(offset) = memchr::memchr(first, &haystack[pos..search_end]) {
                let idx = pos + offset;

                if *haystack.get_unchecked(idx + needle_len - 1) == last
                    && *haystack.get_unchecked(idx + 1) == second
                {
                    if needle_len <= 32 {
                        let hay_vec = if idx + 32 <= haystack.len() {
                            _mm256_loadu_si256(haystack.as_ptr().add(idx) as *const __m256i)
                        } else {
                            let mut buf = [0u8; 32];
                            let remaining = haystack.len() - idx;
                            std::ptr::copy_nonoverlapping(
                                haystack.as_ptr().add(idx),
                                buf.as_mut_ptr(),
                                remaining,
                            );
                            _mm256_loadu_si256(buf.as_ptr() as *const __m256i)
                        };

                        let cmp = _mm256_cmpeq_epi8(needle_vec, hay_vec);
                        let mask = _mm256_movemask_epi8(cmp) as u32;
                        let expected_mask = (1u32 << needle_len) - 1;

                        if (mask & expected_mask) == expected_mask {
                            return Some(idx);
                        }
                    }
                }

                pos = idx + 1;
            } else {
                return None;
            }
        }
    } else {
        while pos < search_end {
            if let Some(offset) = memchr::memchr(first, &haystack[pos..search_end]) {
                let idx = pos + offset;

                if *haystack.get_unchecked(idx + needle_len - 1) == last
                    && *haystack.get_unchecked(idx + 1) == second
                {
                    let mut matches = true;
                    let mut i = 0;

                    while i + 32 <= needle_len {
                        let needle_chunk =
                            _mm256_loadu_si256(needle.as_ptr().add(i) as *const __m256i);
                        let hay_chunk =
                            _mm256_loadu_si256(haystack.as_ptr().add(idx + i) as *const __m256i);
                        let cmp = _mm256_cmpeq_epi8(needle_chunk, hay_chunk);
                        let mask = _mm256_movemask_epi8(cmp);

                        if mask != -1 {
                            matches = false;
                            break;
                        }
                        i += 32;
                    }

                    if matches && i < needle_len {
                        if haystack.get_unchecked(idx + i..idx + needle_len)
                            == needle.get_unchecked(i..needle_len)
                        {
                            return Some(idx);
                        }
                    } else if matches {
                        return Some(idx);
                    }
                }

                pos = idx + 1;
            } else {
                return None;
            }
        }
    }

    None
}

#[inline(always)]
fn find_fallback(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let needle_len = needle.len();
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
