use crate::SubstringSearcher;

/// Naive O(n*m) substring search
pub struct NaiveSearcher;

impl SubstringSearcher for NaiveSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }

        let end = haystack.len() - needle.len() + 1;
        for i in 0..end {
            if &haystack[i..i + needle.len()] == needle {
                return Some(i);
            }
        }
        None
    }
}

/// Standard library string search (via str::find)
pub struct StdSearcher;

impl SubstringSearcher for StdSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        // Use windows to avoid UTF-8 conversion
        haystack.windows(needle.len()).position(|w| w == needle)
    }
}

/// memchr crate's memmem searcher (the target to beat)
pub struct MemchrSearcher;

impl SubstringSearcher for MemchrSearcher {
    #[inline]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        memchr::memmem::find(haystack, needle)
    }
}

/// Boyer-Moore-Horspool variant
pub struct BoyerMooreSearcher;

impl SubstringSearcher for BoyerMooreSearcher {
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if needle.len() > haystack.len() {
            return None;
        }

        // Build bad character table
        let mut skip = [needle.len(); 256];
        for (i, &b) in needle[..needle.len() - 1].iter().enumerate() {
            skip[b as usize] = needle.len() - 1 - i;
        }

        let mut i = needle.len() - 1;
        while i < haystack.len() {
            let mut j = needle.len() - 1;
            let mut k = i;

            while needle[j] == haystack[k] {
                if j == 0 {
                    return Some(k);
                }
                j -= 1;
                k -= 1;
            }

            i += skip[haystack[i] as usize];
        }

        None
    }
}
