//! Evolved integer parser - this file gets mutated during evolution

use crate::IntParser;

/// The evolved parser implementation
pub struct EvolvedParser;

impl IntParser for EvolvedParser {
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
        let mut result: i64 = 0;
        let mut i = 0;

        // Process 8 digits at a time
        while i + 8 <= len {
            unsafe {
                let d0 = *digits.get_unchecked(i);
                let d1 = *digits.get_unchecked(i + 1);
                let d2 = *digits.get_unchecked(i + 2);
                let d3 = *digits.get_unchecked(i + 3);
                let d4 = *digits.get_unchecked(i + 4);
                let d5 = *digits.get_unchecked(i + 5);
                let d6 = *digits.get_unchecked(i + 6);
                let d7 = *digits.get_unchecked(i + 7);

                if d0 < b'0' || d0 > b'9' || d1 < b'0' || d1 > b'9' ||
                   d2 < b'0' || d2 > b'9' || d3 < b'0' || d3 > b'9' ||
                   d4 < b'0' || d4 > b'9' || d5 < b'0' || d5 > b'9' ||
                   d6 < b'0' || d6 > b'9' || d7 < b'0' || d7 > b'9' {
                    return Err(());
                }

                let chunk = ((d0 - b'0') as i64) * 10_000_000
                          + ((d1 - b'0') as i64) * 1_000_000
                          + ((d2 - b'0') as i64) * 100_000
                          + ((d3 - b'0') as i64) * 10_000
                          + ((d4 - b'0') as i64) * 1_000
                          + ((d5 - b'0') as i64) * 100
                          + ((d6 - b'0') as i64) * 10
                          + ((d7 - b'0') as i64);

                result = result.checked_mul(100_000_000).ok_or(())?;
                result = result.checked_sub(chunk).ok_or(())?;
            }
            i += 8;
        }

        // Process 4 digits at a time
        if i + 4 <= len {
            unsafe {
                let d0 = *digits.get_unchecked(i);
                let d1 = *digits.get_unchecked(i + 1);
                let d2 = *digits.get_unchecked(i + 2);
                let d3 = *digits.get_unchecked(i + 3);

                if d0 < b'0' || d0 > b'9' || d1 < b'0' || d1 > b'9' ||
                   d2 < b'0' || d2 > b'9' || d3 < b'0' || d3 > b'9' {
                    return Err(());
                }

                let chunk = ((d0 - b'0') as i64) * 1000
                          + ((d1 - b'0') as i64) * 100
                          + ((d2 - b'0') as i64) * 10
                          + ((d3 - b'0') as i64);

                result = result.checked_mul(10_000).ok_or(())?;
                result = result.checked_sub(chunk).ok_or(())?;
            }
            i += 4;
        }

        // Process 2 digits at a time
        if i + 2 <= len {
            unsafe {
                let d0 = *digits.get_unchecked(i);
                let d1 = *digits.get_unchecked(i + 1);

                if d0 < b'0' || d0 > b'9' || d1 < b'0' || d1 > b'9' {
                    return Err(());
                }

                let chunk = ((d0 - b'0') as i64) * 10 + ((d1 - b'0') as i64);

                result = result.checked_mul(100).ok_or(())?;
                result = result.checked_sub(chunk).ok_or(())?;
            }
            i += 2;
        }

        // Process remaining single digit
        if i < len {
            unsafe {
                let byte = *digits.get_unchecked(i);
                if byte < b'0' || byte > b'9' {
                    return Err(());
                }
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub((byte - b'0') as i64).ok_or(())?;
            }
        }

        if negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}
