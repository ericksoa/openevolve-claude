use crate::SubstringSearcher;

/// Naive O(n*m) search
pub struct NaiveSearcher;

impl SubstringSearcher for NaiveSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() { return Some(0); }
        if needle.len() > haystack.len() { return None; }

        for i in 0..=(haystack.len() - needle.len()) {
            if &haystack[i..i + needle.len()] == needle {
                return Some(i);
            }
        }
        None
    }
}

/// Standard library search
pub struct StdSearcher;

impl SubstringSearcher for StdSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() { return Some(0); }
        haystack.windows(needle.len()).position(|w| w == needle)
    }
}

/// memchr crate (SIMD-optimized)
pub struct MemchrSearcher;

impl SubstringSearcher for MemchrSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        memchr::memmem::find(haystack, needle)
    }
}

/// Boyer-Moore-Horspool
pub struct BoyerMooreSearcher;

impl SubstringSearcher for BoyerMooreSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() { return Some(0); }
        if needle.len() > haystack.len() { return None; }

        // Build bad character table
        let mut skip = [needle.len(); 256];
        for (i, &b) in needle[..needle.len()-1].iter().enumerate() {
            skip[b as usize] = needle.len() - 1 - i;
        }

        let mut i = needle.len() - 1;
        while i < haystack.len() {
            let mut j = needle.len() - 1;
            let mut k = i;

            while needle[j] == haystack[k] {
                if j == 0 { return Some(k); }
                j -= 1;
                k -= 1;
            }

            i += skip[haystack[i] as usize];
        }
        None
    }
}
