use crate::SubstringSearcher;

pub struct EvolvedSearcher;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl SubstringSearcher for EvolvedSearcher {
    #[inline(always)]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && needle.len() >= 2 {
                unsafe { return avx2_search(haystack, needle); }
            }
        }

        fallback_search(haystack, needle)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_search(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len == 1 {
        return memchr_avx2(haystack, needle[0]);
    }

    let first = needle[0];
    let second = needle[1];
    let first_vec = _mm256_set1_epi8(first as i8);
    let second_vec = _mm256_set1_epi8(second as i8);

    let mut i = 0;
    let end = haystack_len.saturating_sub(needle_len) + 1;

    if end == 0 {
        return None;
    }

    // SIMD scan for first two bytes
    while i + 32 < end {
        let chunk = _mm256_loadu_si256(haystack.as_ptr().add(i) as *const __m256i);
        let eq_first = _mm256_cmpeq_epi8(chunk, first_vec);
        let mut mask = _mm256_movemask_epi8(eq_first);

        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            let pos = i + offset;

            if pos + 1 < haystack_len && haystack[pos + 1] == second {
                if needle_len == 2 || verify_match(haystack, needle, pos) {
                    return Some(pos);
                }
            }

            mask &= mask - 1;
        }

        i += 32;
    }

    // Scalar tail
    while i < end {
        if haystack[i] == first && haystack[i + 1] == second {
            if needle_len == 2 || verify_match(haystack, needle, i) {
                return Some(i);
            }
        }
        i += 1;
    }

    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn memchr_avx2(haystack: &[u8], byte: u8) -> Option<usize> {
    let byte_vec = _mm256_set1_epi8(byte as i8);
    let mut i = 0;
    let len = haystack.len();

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(haystack.as_ptr().add(i) as *const __m256i);
        let eq = _mm256_cmpeq_epi8(chunk, byte_vec);
        let mask = _mm256_movemask_epi8(eq);

        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }

        i += 32;
    }

    while i < len {
        if haystack[i] == byte {
            return Some(i);
        }
        i += 1;
    }

    None
}

#[inline(always)]
fn verify_match(haystack: &[u8], needle: &[u8], pos: usize) -> bool {
    let needle_len = needle.len();

    if pos + needle_len > haystack.len() {
        return false;
    }

    // Already verified first two bytes, check rest
    unsafe {
        let hay_ptr = haystack.as_ptr().add(pos + 2);
        let needle_ptr = needle.as_ptr().add(2);
        let remaining = needle_len - 2;

        for i in 0..remaining {
            if *hay_ptr.add(i) != *needle_ptr.add(i) {
                return false;
            }
        }
    }

    true
}

#[inline(always)]
fn fallback_search(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len == 1 {
        return haystack.iter().position(|&b| b == needle[0]);
    }

    let first = needle[0];
    let second = needle[1];
    let last = needle[needle_len - 1];

    let mut i = 0;
    let end = haystack_len.saturating_sub(needle_len) + 1;

    while i < end {
        if haystack[i] == first {
            if haystack[i + 1] == second && haystack[i + needle_len - 1] == last {
                if needle_len == 2 || &haystack[i + 2..i + needle_len] == &needle[2..] {
                    return Some(i);
                }
            }
        }
        i += 1;
    }

    None
}
