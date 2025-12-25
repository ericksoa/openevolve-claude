//! Evolved integer parser - this file gets mutated during evolution

use crate::IntParser;

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

        // Fast path for 1-4 digits - no overflow possible
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

        // Fast path for 5-8 digits - use SWAR
        if len <= 8 {
            let mut result: i64 = 0;
            unsafe {
                let ptr = digits.as_ptr();

                if len == 8 {
                    let chunk = (ptr as *const u64).read_unaligned();
                    let bytes = chunk.to_le_bytes();

                    for &byte in &bytes {
                        if byte < b'0' || byte > b'9' {
                            return Err(());
                        }
                        result = result * 10 + (byte - b'0') as i64;
                    }
                } else {
                    for i in 0..len {
                        let byte = *ptr.add(i);
                        if byte < b'0' || byte > b'9' {
                            return Err(());
                        }
                        result = result * 10 + (byte - b'0') as i64;
                    }
                }
            }
            return Ok(if negative { -result } else { result });
        }

        // General case: accumulate as negative to handle i64::MIN
        let mut result: i64 = 0;
        let mut i = 0;

        // Process 8 digits at a time using SWAR when possible
        while i + 8 <= len {
            unsafe {
                let ptr = digits.as_ptr().add(i);
                let chunk = (ptr as *const u64).read_unaligned();
                let bytes = chunk.to_le_bytes();

                let d0 = bytes[0];
                let d1 = bytes[1];
                let d2 = bytes[2];
                let d3 = bytes[3];
                let d4 = bytes[4];
                let d5 = bytes[5];
                let d6 = bytes[6];
                let d7 = bytes[7];

                if d0 < b'0' || d0 > b'9' || d1 < b'0' || d1 > b'9'
                   || d2 < b'0' || d2 > b'9' || d3 < b'0' || d3 > b'9'
                   || d4 < b'0' || d4 > b'9' || d5 < b'0' || d5 > b'9'
                   || d6 < b'0' || d6 > b'9' || d7 < b'0' || d7 > b'9' {
                    return Err(());
                }

                let chunk_val = ((d0 - b'0') as i64) * 10000000
                    + ((d1 - b'0') as i64) * 1000000
                    + ((d2 - b'0') as i64) * 100000
                    + ((d3 - b'0') as i64) * 10000
                    + ((d4 - b'0') as i64) * 1000
                    + ((d5 - b'0') as i64) * 100
                    + ((d6 - b'0') as i64) * 10
                    + ((d7 - b'0') as i64);

                result = result.checked_mul(100000000).ok_or(())?;
                result = result.checked_sub(chunk_val).ok_or(())?;
                i += 8;
            }
        }

        // Process 4 digits at a time for remaining
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
