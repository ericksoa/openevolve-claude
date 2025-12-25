//! String Search Benchmark
//!
//! Evolving algorithms to beat memchr for substring search.

pub mod baselines;
pub mod evolved;

/// Trait for substring search implementations
pub trait SubstringSearcher {
    /// Find the first occurrence of needle in haystack.
    /// Returns the byte offset of the first match, or None if not found.
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use baselines::*;
    use evolved::EvolvedSearcher;

    fn test_searcher<S: SubstringSearcher>(searcher: &S) {
        // Empty needle should find at position 0
        assert_eq!(searcher.find(b"hello", b""), Some(0));

        // Empty haystack with non-empty needle
        assert_eq!(searcher.find(b"", b"a"), None);

        // Both empty
        assert_eq!(searcher.find(b"", b""), Some(0));

        // Needle at start
        assert_eq!(searcher.find(b"hello world", b"hello"), Some(0));

        // Needle at end
        assert_eq!(searcher.find(b"hello world", b"world"), Some(6));

        // Needle in middle
        assert_eq!(searcher.find(b"foo bar baz", b"bar"), Some(4));

        // Single byte needle
        assert_eq!(searcher.find(b"hello", b"e"), Some(1));
        assert_eq!(searcher.find(b"hello", b"l"), Some(2));

        // Needle not found
        assert_eq!(searcher.find(b"hello", b"xyz"), None);

        // Needle longer than haystack
        assert_eq!(searcher.find(b"hi", b"hello"), None);

        // Overlapping pattern (should find first)
        assert_eq!(searcher.find(b"aaaa", b"aa"), Some(0));

        // Case sensitive
        assert_eq!(searcher.find(b"Hello", b"hello"), None);

        // Binary data
        assert_eq!(searcher.find(b"\x00\x01\x02\x03", b"\x01\x02"), Some(1));

        // Repeated pattern
        assert_eq!(searcher.find(b"abcabcabc", b"abc"), Some(0));
        assert_eq!(searcher.find(b"xyzabcabc", b"abc"), Some(3));
    }

    #[test]
    fn test_naive() {
        test_searcher(&NaiveSearcher);
    }

    #[test]
    fn test_std_search() {
        test_searcher(&StdSearcher);
    }

    #[test]
    fn test_memchr_search() {
        test_searcher(&MemchrSearcher);
    }

    #[test]
    fn test_evolved() {
        test_searcher(&EvolvedSearcher);
    }
}
