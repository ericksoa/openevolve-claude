use crate::SubstringSearcher;

pub struct EvolvedSearcher;

impl SubstringSearcher for EvolvedSearcher {
    #[inline]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }

        let needle_len = needle.len();
        let haystack_len = haystack.len();

        // Single byte: use memchr
        if needle_len == 1 {
            return memchr::memchr(needle[0], haystack);
        }

        // Two bytes: optimized path with memchr
        if needle_len == 2 {
            return find_two_bytes(haystack, needle);
        }

        // Three bytes: memchr + dual check
        if needle_len == 3 {
            return find_three_bytes(haystack, needle);
        }

        // Four bytes: u32 word comparison
        if needle_len == 4 {
            return find_four_bytes(haystack, needle);
        }

        // Eight bytes: u64 word comparison
        if needle_len == 8 {
            return find_eight_bytes(haystack, needle);
        }

        // Fallback: optimized Boyer-Moore with word-based comparison
        find_boyer_moore(haystack, needle)
    }
}

#[inline]
fn find_two_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let first = needle[0];
    let second = needle[1];

    if haystack.len() < 2 {
        return None;
    }

    let mut pos = 0;
    while pos < haystack.len() - 1 {
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
    None
}

#[inline]
fn find_three_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if haystack.len() < 3 {
        return None;
    }

    let first = needle[0];
    let second = needle[1];
    let third = needle[2];

    let mut pos = 0;
    while pos < haystack.len() - 2 {
        if let Some(offset) = memchr::memchr(first, &haystack[pos..]) {
            let idx = pos + offset;
            if idx + 2 < haystack.len()
                && haystack[idx + 1] == second
                && haystack[idx + 2] == third
            {
                return Some(idx);
            }
            pos = idx + 1;
        } else {
            return None;
        }
    }
    None
}

#[inline]
fn find_four_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if haystack.len() < 4 {
        return None;
    }

    let first = needle[0];
    let mut pos = 0;

    while pos <= haystack.len() - 4 {
        if let Some(offset) = memchr::memchr(first, &haystack[pos..]) {
            let idx = pos + offset;
            if idx + 4 <= haystack.len() {
                unsafe {
                    let hay_word = u32::from_le_bytes([
                        *haystack.get_unchecked(idx),
                        *haystack.get_unchecked(idx + 1),
                        *haystack.get_unchecked(idx + 2),
                        *haystack.get_unchecked(idx + 3),
                    ]);
                    let needle_word = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);

                    if hay_word == needle_word {
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

#[inline]
fn find_eight_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if haystack.len() < 8 {
        return None;
    }

    let first = needle[0];
    let mut pos = 0;

    while pos <= haystack.len() - 8 {
        if let Some(offset) = memchr::memchr(first, &haystack[pos..]) {
            let idx = pos + offset;
            if idx + 8 <= haystack.len() {
                unsafe {
                    let hay_word = u64::from_le_bytes([
                        *haystack.get_unchecked(idx),
                        *haystack.get_unchecked(idx + 1),
                        *haystack.get_unchecked(idx + 2),
                        *haystack.get_unchecked(idx + 3),
                        *haystack.get_unchecked(idx + 4),
                        *haystack.get_unchecked(idx + 5),
                        *haystack.get_unchecked(idx + 6),
                        *haystack.get_unchecked(idx + 7),
                    ]);
                    let needle_word = u64::from_le_bytes([
                        needle[0], needle[1], needle[2], needle[3],
                        needle[4], needle[5], needle[6], needle[7],
                    ]);

                    if hay_word == needle_word {
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

#[inline]
fn find_boyer_moore(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let needle_len = needle.len();
    let haystack_len = haystack.len();

    if needle_len > haystack_len {
        return None;
    }

    // Build skip table with caching
    let mut skip = [needle_len; 256];
    for (i, &b) in needle[..needle_len - 1].iter().enumerate() {
        skip[b as usize] = needle_len - 1 - i;
    }

    let last_byte = needle[needle_len - 1];
    let first_byte = needle[0];
    let mid_idx = needle_len / 2;
    let mid_byte = needle[mid_idx];

    let mut i = needle_len - 1;

    while i < haystack_len {
        // Check last byte
        if unsafe { *haystack.get_unchecked(i) } == last_byte {
            let start = i + 1 - needle_len;

            // Pre-checks for quick rejection
            if unsafe { *haystack.get_unchecked(start) } == first_byte
                && unsafe { *haystack.get_unchecked(start + mid_idx) } == mid_byte
            {
                // Full word-based comparison
                if word_compare(haystack, start, needle) {
                    return Some(start);
                }
            }
        }

        i += skip[unsafe { *haystack.get_unchecked(i) } as usize];
    }

    None
}

#[inline]
fn word_compare(haystack: &[u8], pos: usize, needle: &[u8]) -> bool {
    let needle_len = needle.len();

    if pos + needle_len > haystack.len() {
        return false;
    }

    // Compare in 8-byte chunks for speed
    let mut i = 0;
    while i + 8 <= needle_len {
        unsafe {
            let hay_word = u64::from_le_bytes([
                *haystack.get_unchecked(pos + i),
                *haystack.get_unchecked(pos + i + 1),
                *haystack.get_unchecked(pos + i + 2),
                *haystack.get_unchecked(pos + i + 3),
                *haystack.get_unchecked(pos + i + 4),
                *haystack.get_unchecked(pos + i + 5),
                *haystack.get_unchecked(pos + i + 6),
                *haystack.get_unchecked(pos + i + 7),
            ]);
            let needle_word = u64::from_le_bytes([
                needle[i],
                needle[i + 1],
                needle[i + 2],
                needle[i + 3],
                needle[i + 4],
                needle[i + 5],
                needle[i + 6],
                needle[i + 7],
            ]);

            if hay_word != needle_word {
                return false;
            }
        }
        i += 8;
    }

    // Remainder
    while i < needle_len {
        if unsafe { *haystack.get_unchecked(pos + i) } != needle[i] {
            return false;
        }
        i += 1;
    }

    true
}
