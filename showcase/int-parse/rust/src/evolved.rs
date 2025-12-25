//! Evolved Integer Parser
//!
//! Starting point - simple implementation to be evolved

use crate::IntParse;

pub struct EvolvedParser;

impl EvolvedParser {
    pub fn new() -> Self {
        Self
    }
}

impl IntParse for EvolvedParser {
    #[inline]
    fn parse_u64(&self, input: &[u8]) -> Option<u64> {
        let len = input.len();
        if len == 0 || len > 20 {
            return None;
        }

        let mut result: u64 = 0;
        for &byte in input {
            let digit = byte.wrapping_sub(b'0');
            if digit > 9 {
                return None;
            }
            result = result.checked_mul(10)?.checked_add(digit as u64)?;
        }
        Some(result)
    }
}
