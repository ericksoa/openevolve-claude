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

        match needle.len() {
            1 => find_byte(haystack, needle[0]),
            2 => find_two_bytes(haystack, needle),
            3 => find_three_bytes(haystack, needle),
            4 => find_four_bytes(haystack, needle),
            _ => find_long(haystack, needle),
        }
    }
}

#[inline(always)]
fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

#[inline(always)]
fn find_two_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let n0 = needle[0];
    let n1 = needle[1];

    let len = haystack.len();
    if len < 2 {
        return None;
    }

    unsafe {
        let mut i = 0;
        let end = len - 1;

        while i < end {
            if *haystack.get_unchecked(i) == n0 && *haystack.get_unchecked(i + 1) == n1 {
                return Some(i);
            }
            i += 1;
        }
    }

    None
}

#[inline(always)]
fn find_three_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let n0 = needle[0];
    let n1 = needle[1];
    let n2 = needle[2];

    let len = haystack.len();
    if len < 3 {
        return None;
    }

    unsafe {
        let mut i = 0;
        let end = len - 2;

        while i < end {
            if *haystack.get_unchecked(i) == n0
                && *haystack.get_unchecked(i + 2) == n2
                && *haystack.get_unchecked(i + 1) == n1 {
                return Some(i);
            }
            i += 1;
        }
    }

    None
}

#[inline(always)]
fn find_four_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let n0 = needle[0];
    let n1 = needle[1];
    let n2 = needle[2];
    let n3 = needle[3];

    let len = haystack.len();
    if len < 4 {
        return None;
    }

    unsafe {
        let mut i = 0;
        let end = len - 3;

        while i < end {
            if *haystack.get_unchecked(i) == n0
                && *haystack.get_unchecked(i + 3) == n3
                && *haystack.get_unchecked(i + 1) == n1
                && *haystack.get_unchecked(i + 2) == n2 {
                return Some(i);
            }
            i += 1;
        }
    }

    None
}

#[inline(always)]
fn find_long(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let first = needle[0];
    let last = needle[needle.len() - 1];
    let needle_len = needle.len();

    let len = haystack.len();
    if len < needle_len {
        return None;
    }

    unsafe {
        let mut i = 0;
        let end = len - needle_len + 1;

        while i < end {
            if *haystack.get_unchecked(i) == first
                && *haystack.get_unchecked(i + needle_len - 1) == last {
                let mut match_found = true;
                for j in 1..needle_len - 1 {
                    if *haystack.get_unchecked(i + j) != needle[j] {
                        match_found = false;
                        break;
                    }
                }
                if match_found {
                    return Some(i);
                }
            }
            i += 1;
        }
    }

    None
}
