//! Evolved String Search Algorithm
//!
//! This module contains the algorithm being evolved.
//! The initial implementation is a simple Horspool variant.
//! Evolution will mutate this code to discover better algorithms.

use crate::StringSearch;

/// The evolved string search algorithm
///
/// This implementation will be mutated to discover
/// novel, high-performance string search algorithms.
pub struct EvolvedSearch {
    // Preprocessing data structures can be added here
    shift_table: Option<[usize; 256]>,
}

impl EvolvedSearch {
    pub fn new() -> Self {
        Self { shift_table: None }
    }

    /// Preprocess the pattern to build lookup tables
    fn preprocess(&mut self, pattern: &[u8]) {
        let m = pattern.len();
        let mut table = [m; 256];

        // Build bad character shift table
        for (i, &byte) in pattern.iter().enumerate().take(m.saturating_sub(1)) {
            table[byte as usize] = m - 1 - i;
        }

        self.shift_table = Some(table);
    }
}

impl Default for EvolvedSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl StringSearch for EvolvedSearch {
    fn search(&self, text: &[u8], pattern: &[u8]) -> Vec<usize> {
        let n = text.len();
        let m = pattern.len();

        // Handle edge cases
        if m == 0 {
            return Vec::new();
        }
        if m > n {
            return Vec::new();
        }
        if m == 1 {
            // Special case: single byte search
            let target = pattern[0];
            return text
                .iter()
                .enumerate()
                .filter(|(_, &b)| b == target)
                .map(|(i, _)| i)
                .collect();
        }

        // Build shift table inline for now
        // (evolution may discover better preprocessing)
        let mut shift = [m; 256];
        for (i, &byte) in pattern.iter().enumerate().take(m - 1) {
            shift[byte as usize] = m - 1 - i;
        }

        let mut results = Vec::new();
        let mut i = 0;

        // Cache pattern bytes for faster access
        let last_pattern_byte = pattern[m - 1];
        let first_pattern_byte = pattern[0];

        while i <= n - m {
            // Quick check: compare last byte first (most likely to differ)
            let last_text_byte = text[i + m - 1];

            if last_text_byte == last_pattern_byte {
                // Check first byte
                if text[i] == first_pattern_byte {
                    // Full comparison (skip first and last, already checked)
                    let mut matched = true;
                    for j in 1..(m - 1) {
                        if text[i + j] != pattern[j] {
                            matched = false;
                            break;
                        }
                    }

                    if matched {
                        results.push(i);
                    }
                }
            }

            // Shift based on the character at the end of current window
            i += shift[last_text_byte as usize].max(1);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolved_basic() {
        let searcher = EvolvedSearch::new();
        let text = b"abracadabra";
        let pattern = b"abra";

        let results = searcher.search(text, pattern);
        assert_eq!(results, vec![0, 7]);
    }

    #[test]
    fn test_evolved_single_char() {
        let searcher = EvolvedSearch::new();
        let text = b"aaabbbccc";
        let pattern = b"b";

        let results = searcher.search(text, pattern);
        assert_eq!(results, vec![3, 4, 5]);
    }

    #[test]
    fn test_evolved_no_match() {
        let searcher = EvolvedSearch::new();
        let text = b"hello world";
        let pattern = b"xyz";

        let results = searcher.search(text, pattern);
        assert!(results.is_empty());
    }

    #[test]
    fn test_evolved_full_match() {
        let searcher = EvolvedSearch::new();
        let text = b"exact";
        let pattern = b"exact";

        let results = searcher.search(text, pattern);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn test_evolved_overlapping() {
        let searcher = EvolvedSearch::new();
        let text = b"aaaaa";
        let pattern = b"aa";

        let results = searcher.search(text, pattern);
        assert_eq!(results, vec![0, 1, 2, 3]);
    }
}
