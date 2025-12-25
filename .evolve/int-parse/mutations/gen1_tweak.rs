//! Evolved integer parser - this file gets mutated during evolution

use crate::IntParser;

/// The evolved parser implementation
pub struct EvolvedParser;

impl IntParser for EvolvedParser {
    #[inline]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        let len = input.len();
        if len == 0 {
            return Err(());
        }

        unsafe {
            let ptr = input.as_ptr();
            let first = *ptr;
            let (negative, mut idx) = if first == b'-' {
                if len == 1 {
                    return Err(());
                }
                (true, 1)
            } else {
                (false, 0)
            };

            let mut digit = *ptr.add(idx);
            if digit < b'0' || digit > b'9' {
                return Err(());
            }

            let mut result: i64 = -((digit - b'0') as i64);
            idx += 1;

            while idx < len {
                digit = *ptr.add(idx);
                if digit < b'0' || digit > b'9' {
                    return Err(());
                }

                let new_result = result.wrapping_mul(10).wrapping_sub((digit - b'0') as i64);
                if result < -922337203685477580 || (result == -922337203685477580 && (digit - b'0') > 8) {
                    return Err(());
                }
                result = new_result;
                idx += 1;
            }

            if negative {
                Ok(result)
            } else {
                if result == i64::MIN {
                    return Err(());
                }
                Ok(-result)
            }
        }
    }
}
