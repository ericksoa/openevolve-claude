//! Baseline integer parsing implementations

use crate::IntParse;

/// Standard library parser - the one to beat
pub struct StdParser;

impl IntParse for StdParser {
    #[inline]
    fn parse_u64(&self, input: &[u8]) -> Option<u64> {
        std::str::from_utf8(input).ok()?.parse().ok()
    }
}

/// Naive byte-by-byte parser
pub struct NaiveParser;

impl IntParse for NaiveParser {
    fn parse_u64(&self, input: &[u8]) -> Option<u64> {
        if input.is_empty() {
            return None;
        }

        let mut result: u64 = 0;
        for &byte in input {
            if byte < b'0' || byte > b'9' {
                return None;
            }
            let digit = (byte - b'0') as u64;
            result = result.checked_mul(10)?.checked_add(digit)?;
        }
        Some(result)
    }
}

/// Unrolled parser with some optimizations
pub struct UnrolledParser;

impl IntParse for UnrolledParser {
    #[inline]
    fn parse_u64(&self, input: &[u8]) -> Option<u64> {
        let len = input.len();
        if len == 0 || len > 20 {
            return None;
        }

        // Fast path for single digit
        if len == 1 {
            let d = input[0].wrapping_sub(b'0');
            return if d <= 9 { Some(d as u64) } else { None };
        }

        let mut result: u64 = 0;
        let mut i = 0;

        // Process 4 digits at a time
        while i + 4 <= len {
            let d0 = input[i].wrapping_sub(b'0');
            let d1 = input[i + 1].wrapping_sub(b'0');
            let d2 = input[i + 2].wrapping_sub(b'0');
            let d3 = input[i + 3].wrapping_sub(b'0');

            if d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9 {
                return None;
            }

            result = result.checked_mul(10000)?;
            result = result.checked_add(
                (d0 as u64) * 1000 + (d1 as u64) * 100 + (d2 as u64) * 10 + (d3 as u64)
            )?;
            i += 4;
        }

        // Handle remaining digits
        while i < len {
            let d = input[i].wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            result = result.checked_mul(10)?.checked_add(d as u64)?;
            i += 1;
        }

        Some(result)
    }
}
