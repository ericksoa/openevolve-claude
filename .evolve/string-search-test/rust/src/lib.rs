//! String Search Benchmark
pub mod baselines;
pub mod evolved;

/// Trait for substring search implementations
pub trait SubstringSearcher {
    /// Find the first occurrence of needle in haystack
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_searcher<S: SubstringSearcher>(s: &S) {
        // Empty needle
        assert_eq!(s.find(b"hello", b""), Some(0));

        // Basic matches
        assert_eq!(s.find(b"hello world", b"world"), Some(6));
        assert_eq!(s.find(b"hello", b"hello"), Some(0));
        assert_eq!(s.find(b"hello", b"llo"), Some(2));

        // No match
        assert_eq!(s.find(b"hello", b"xyz"), None);

        // Needle longer than haystack
        assert_eq!(s.find(b"hi", b"hello"), None);

        // Single char
        assert_eq!(s.find(b"hello", b"e"), Some(1));
    }

    #[test]
    fn test_naive() { test_searcher(&baselines::NaiveSearcher); }

    #[test]
    fn test_std() { test_searcher(&baselines::StdSearcher); }

    #[test]
    fn test_memchr() { test_searcher(&baselines::MemchrSearcher); }

    #[test]
    fn test_evolved() { test_searcher(&evolved::EvolvedSearcher); }
}
