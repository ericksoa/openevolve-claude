//! Integer Parsing Benchmark
//!
//! Goal: Evolve a u64 parser that beats Rust's standard library

pub mod baselines;
pub mod evolved;

/// Trait for integer parsing implementations
pub trait IntParse {
    /// Parse a byte slice as a u64
    /// Returns None if the input is invalid (empty, non-digit, overflow)
    fn parse_u64(&self, input: &[u8]) -> Option<u64>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::*;
    use crate::evolved::EvolvedParser;

    fn test_parser<P: IntParse>(parser: &P) {
        // Basic cases
        assert_eq!(parser.parse_u64(b"0"), Some(0));
        assert_eq!(parser.parse_u64(b"1"), Some(1));
        assert_eq!(parser.parse_u64(b"123"), Some(123));
        assert_eq!(parser.parse_u64(b"999999999"), Some(999999999));

        // Edge cases
        assert_eq!(parser.parse_u64(b""), None); // empty
        assert_eq!(parser.parse_u64(b"abc"), None); // non-digits
        assert_eq!(parser.parse_u64(b"12a34"), None); // mixed
        assert_eq!(parser.parse_u64(b"-1"), None); // negative (we only do unsigned)
        assert_eq!(parser.parse_u64(b" 123"), None); // leading space
        assert_eq!(parser.parse_u64(b"123 "), None); // trailing space

        // Max u64
        assert_eq!(parser.parse_u64(b"18446744073709551615"), Some(u64::MAX));

        // Overflow
        assert_eq!(parser.parse_u64(b"18446744073709551616"), None); // MAX + 1
        assert_eq!(parser.parse_u64(b"99999999999999999999"), None); // way over

        // Leading zeros (valid)
        assert_eq!(parser.parse_u64(b"007"), Some(7));
        assert_eq!(parser.parse_u64(b"00000"), Some(0));
    }

    #[test]
    fn test_std_parser() {
        test_parser(&StdParser);
    }

    #[test]
    fn test_naive_parser() {
        test_parser(&NaiveParser);
    }

    #[test]
    fn test_unrolled_parser() {
        test_parser(&UnrolledParser);
    }

    #[test]
    fn test_evolved_parser() {
        test_parser(&EvolvedParser::new());
    }
}
