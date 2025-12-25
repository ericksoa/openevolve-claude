//! Baseline implementations for integer parsing

use crate::IntParser;

/// Standard library parser using str::parse
pub struct StdParser;

impl IntParser for StdParser {
    #[inline]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        std::str::from_utf8(input)
            .map_err(|_| ())?
            .parse::<i64>()
            .map_err(|_| ())
    }
}

/// Naive byte-by-byte parser
/// Uses negative accumulation to handle i64::MIN correctly
pub struct NaiveParser;

impl IntParser for NaiveParser {
    #[inline]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        if input.is_empty() {
            return Err(());
        }

        let (negative, start) = if input[0] == b'-' {
            if input.len() == 1 {
                return Err(());
            }
            (true, 1)
        } else {
            (false, 0)
        };

        // Accumulate as negative to handle i64::MIN correctly
        let mut result: i64 = 0;
        for &byte in &input[start..] {
            if byte < b'0' || byte > b'9' {
                return Err(());
            }
            let digit = (byte - b'0') as i64;
            result = result.checked_mul(10).ok_or(())?;
            result = result.checked_sub(digit).ok_or(())?;
        }

        if negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}

/// Optimized parser with loop unrolling and fast paths
pub struct UnrolledParser;

impl IntParser for UnrolledParser {
    #[inline]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        if input.is_empty() {
            return Err(());
        }

        let (negative, digits) = if input[0] == b'-' {
            if input.len() == 1 {
                return Err(());
            }
            (true, &input[1..])
        } else {
            (false, input)
        };

        if digits.is_empty() {
            return Err(());
        }

        // Fast path for small numbers (1-4 digits) - no overflow possible
        let len = digits.len();
        if len <= 4 {
            let mut result: i64 = 0;
            for &byte in digits {
                if byte < b'0' || byte > b'9' {
                    return Err(());
                }
                result = result * 10 + (byte - b'0') as i64;
            }
            return Ok(if negative { -result } else { result });
        }

        // General case: accumulate as negative to handle i64::MIN
        let mut result: i64 = 0;
        let mut i = 0;

        // Process 4 digits at a time when possible
        while i + 4 <= len {
            let d0 = digits[i];
            let d1 = digits[i + 1];
            let d2 = digits[i + 2];
            let d3 = digits[i + 3];

            if d0 < b'0' || d0 > b'9' || d1 < b'0' || d1 > b'9'
               || d2 < b'0' || d2 > b'9' || d3 < b'0' || d3 > b'9' {
                return Err(());
            }

            let chunk = ((d0 - b'0') as i64) * 1000
                + ((d1 - b'0') as i64) * 100
                + ((d2 - b'0') as i64) * 10
                + ((d3 - b'0') as i64);

            result = result.checked_mul(10000).ok_or(())?;
            result = result.checked_sub(chunk).ok_or(())?;
            i += 4;
        }

        // Handle remaining digits
        while i < len {
            let byte = digits[i];
            if byte < b'0' || byte > b'9' {
                return Err(());
            }
            result = result.checked_mul(10).ok_or(())?;
            result = result.checked_sub((byte - b'0') as i64).ok_or(())?;
            i += 1;
        }

        if negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}

/// SWAR (SIMD Within A Register) parser - processes 8 bytes at once
pub struct SwarParser;

impl IntParser for SwarParser {
    #[inline]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        if input.is_empty() {
            return Err(());
        }

        let (negative, digits) = if input[0] == b'-' {
            if input.len() == 1 {
                return Err(());
            }
            (true, &input[1..])
        } else {
            (false, input)
        };

        if digits.is_empty() {
            return Err(());
        }

        let len = digits.len();

        // For very short numbers, use simple path (no overflow)
        if len <= 4 {
            let mut result: i64 = 0;
            for &byte in digits {
                if byte < b'0' || byte > b'9' {
                    return Err(());
                }
                result = result * 10 + (byte - b'0') as i64;
            }
            return Ok(if negative { -result } else { result });
        }

        // Accumulate as negative for i64::MIN handling
        let mut result: i64 = 0;
        let mut i = 0;

        while i + 8 <= len {
            // Read 8 bytes
            let chunk = unsafe {
                (digits.as_ptr().add(i) as *const u64).read_unaligned()
            };

            // Check all bytes are digits: subtract '0' and check < 10
            let zeros = 0x3030303030303030u64;
            let adjusted = chunk.wrapping_sub(zeros);

            // Each byte should be 0-9 after subtracting '0'
            // Check if any byte has value >= 10
            let tens = 0x0A0A0A0A0A0A0A0Au64;
            let overflow = adjusted.wrapping_add(0x7676767676767676u64); // 0x80 - 10 = 0x76
            if (overflow & 0x8080808080808080u64) != 0x8080808080808080u64 {
                // Some byte was >= 10, fallback to byte-by-byte
                for j in 0..8 {
                    let byte = digits[i + j];
                    if byte < b'0' || byte > b'9' {
                        return Err(());
                    }
                    result = result.checked_mul(10).ok_or(())?;
                    result = result.checked_sub((byte - b'0') as i64).ok_or(())?;
                }
            } else {
                // Convert 8 ASCII digits to number
                let d = adjusted.to_le_bytes();
                let v = (d[0] as i64) * 10000000
                    + (d[1] as i64) * 1000000
                    + (d[2] as i64) * 100000
                    + (d[3] as i64) * 10000
                    + (d[4] as i64) * 1000
                    + (d[5] as i64) * 100
                    + (d[6] as i64) * 10
                    + (d[7] as i64);

                result = result.checked_mul(100000000).ok_or(())?;
                result = result.checked_sub(v).ok_or(())?;
            }
            i += 8;
        }

        // Handle remaining digits
        while i < len {
            let byte = digits[i];
            if byte < b'0' || byte > b'9' {
                return Err(());
            }
            result = result.checked_mul(10).ok_or(())?;
            result = result.checked_sub((byte - b'0') as i64).ok_or(())?;
            i += 1;
        }

        if negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}
