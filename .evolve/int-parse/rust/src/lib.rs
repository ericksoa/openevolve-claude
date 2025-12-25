//! Integer Parsing Benchmark
//!
//! Evolve a faster integer parsing algorithm to beat std::str::parse

pub mod baselines;
pub mod evolved;

/// Trait for integer parsing implementations
///
/// Implementations must parse a byte slice representing a decimal integer
/// and return the parsed i64 value or an error.
pub trait IntParser {
    /// Parse a byte slice as a decimal integer
    /// Returns Ok(value) on success, Err(()) on invalid input
    fn parse(&self, input: &[u8]) -> Result<i64, ()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use baselines::*;
    use evolved::EvolvedParser;

    fn test_parser<P: IntParser>(parser: &P) {
        // Basic positive numbers
        assert_eq!(parser.parse(b"0").unwrap(), 0);
        assert_eq!(parser.parse(b"1").unwrap(), 1);
        assert_eq!(parser.parse(b"123").unwrap(), 123);
        assert_eq!(parser.parse(b"999999999").unwrap(), 999999999);

        // Negative numbers
        assert_eq!(parser.parse(b"-1").unwrap(), -1);
        assert_eq!(parser.parse(b"-123").unwrap(), -123);
        assert_eq!(parser.parse(b"-999999999").unwrap(), -999999999);

        // Edge cases
        assert_eq!(parser.parse(b"9223372036854775807").unwrap(), i64::MAX);
        assert_eq!(parser.parse(b"-9223372036854775808").unwrap(), i64::MIN);

        // Leading zeros
        assert_eq!(parser.parse(b"007").unwrap(), 7);
        assert_eq!(parser.parse(b"-007").unwrap(), -7);

        // Invalid inputs should error
        assert!(parser.parse(b"").is_err());
        assert!(parser.parse(b"-").is_err());
        assert!(parser.parse(b"abc").is_err());
        assert!(parser.parse(b"12a3").is_err());
        assert!(parser.parse(b" 123").is_err());
        assert!(parser.parse(b"123 ").is_err());
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
        test_parser(&EvolvedParser);
    }
}
