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

        unsafe {
            let mut ptr = input.as_ptr();
            let end = ptr.add(input.len());

            let negative = *ptr == b'-';
            ptr = ptr.add(negative as usize);

            if ptr >= end {
                return Err(());
            }

            let mut result: i64 = 0;

            while ptr < end {
                let digit = (*ptr).wrapping_sub(b'0');
                if digit > 9 {
                    return Err(());
                }

                let new_result = result.wrapping_mul(10).wrapping_sub(digit as i64);

                if result < new_result {
                    return Err(());
                }

                result = new_result;
                ptr = ptr.add(1);
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
