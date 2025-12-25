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

        // Fast paths for common small cases (1-4 digits)
        match digits.len() {
            1 => {
                let byte = digits[0];
                if byte < b'0' || byte > b'9' {
                    return Err(());
                }
                let val = (byte - b'0') as i64;
                Ok(if negative { -val } else { val })
            }
            2 => {
                let b0 = digits[0];
                let b1 = digits[1];
                if b0 < b'0' || b0 > b'9' || b1 < b'0' || b1 > b'9' {
                    return Err(());
                }
                let val = ((b0 - b'0') as i64) * 10 + ((b1 - b'0') as i64);
                Ok(if negative { -val } else { val })
            }
            3 => {
                let b0 = digits[0];
                let b1 = digits[1];
                let b2 = digits[2];
                if b0 < b'0' || b0 > b'9' || b1 < b'0' || b1 > b'9' || b2 < b'0' || b2 > b'9' {
                    return Err(());
                }
                let val = ((b0 - b'0') as i64) * 100
                        + ((b1 - b'0') as i64) * 10
                        + ((b2 - b'0') as i64);
                Ok(if negative { -val } else { val })
            }
            4 => {
                let b0 = digits[0];
                let b1 = digits[1];
                let b2 = digits[2];
                let b3 = digits[3];
                if b0 < b'0' || b0 > b'9' || b1 < b'0' || b1 > b'9'
                    || b2 < b'0' || b2 > b'9' || b3 < b'0' || b3 > b'9' {
                    return Err(());
                }
                let val = ((b0 - b'0') as i64) * 1000
                        + ((b1 - b'0') as i64) * 100
                        + ((b2 - b'0') as i64) * 10
                        + ((b3 - b'0') as i64);
                Ok(if negative { -val } else { val })
            }
            _ => {
                // For longer numbers, accumulate as negative to handle i64::MIN
                let mut result: i64 = 0;
                for &byte in digits {
                    if byte < b'0' || byte > b'9' {
                        return Err(());
                    }
                    result = result.checked_mul(10).ok_or(())?;
                    result = result.checked_sub((byte - b'0') as i64).ok_or(())?;
                }

                if negative {
                    Ok(result)
                } else {
                    result.checked_neg().ok_or(())
                }
            }
        }
    }
}
