//! String Search Algorithms
//!
//! This module contains various string search implementations.
//! The `evolved` module contains the algorithm being evolved.

pub mod baselines;
pub mod evolved;

/// Find all occurrences of pattern in text, returning starting indices
pub trait StringSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize>;
}

/// Verify that a search result is correct
pub fn verify_search(text: &[u8], pattern: &[u8], indices: &[usize]) -> bool {
    for &idx in indices {
        if idx + pattern.len() > text.len() {
            return false;
        }
        if &text[idx..idx + pattern.len()] != pattern {
            return false;
        }
    }

    // Also verify we didn't miss any occurrences
    let expected = naive_search(text, pattern);
    indices.len() == expected.len() && indices.iter().all(|i| expected.contains(i))
}

/// Reference implementation for verification
fn naive_search(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return Vec::new();
    }

    let mut results = Vec::new();
    for i in 0..=text.len() - pattern.len() {
        if &text[i..i + pattern.len()] == pattern {
            results.push(i);
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolved::EvolvedSearch;
    use crate::baselines::*;

    #[test]
    fn test_basic_search() {
        let text = b"hello world hello";
        let pattern = b"hello";

        let searcher = EvolvedSearch::new();
        let results = searcher.search(text, pattern);

        assert!(verify_search(text, pattern, &results));
        assert_eq!(results, vec![0, 12]);
    }

    #[test]
    fn test_no_match() {
        let text = b"hello world";
        let pattern = b"xyz";

        let searcher = EvolvedSearch::new();
        let results = searcher.search(text, pattern);

        assert!(results.is_empty());
    }

    #[test]
    fn test_overlapping() {
        let text = b"aaaa";
        let pattern = b"aa";

        let searcher = EvolvedSearch::new();
        let results = searcher.search(text, pattern);

        assert!(verify_search(text, pattern, &results));
        assert_eq!(results, vec![0, 1, 2]);
    }

    #[test]
    fn test_all_baselines() {
        let text = b"the quick brown fox jumps over the lazy dog";
        let pattern = b"the";

        let naive = NaiveSearch::new();
        let kmp = KMPSearch::new();
        let bm = BoyerMooreSearch::new();
        let horspool = HorspoolSearch::new();
        let evolved = EvolvedSearch::new();

        let expected = vec![0, 31];

        assert_eq!(naive.search(text, pattern), expected);
        assert_eq!(kmp.search(text, pattern), expected);
        assert_eq!(bm.search(text, pattern), expected);
        assert_eq!(horspool.search(text, pattern), expected);
        assert_eq!(evolved.search(text, pattern), expected);
    }
}
