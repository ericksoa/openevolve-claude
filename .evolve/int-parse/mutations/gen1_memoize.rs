//! Evolved integer parser - this file gets mutated during evolution

use crate::IntParser;

/// The evolved parser implementation
pub struct EvolvedParser;

// Lookup table for digit validation: valid[b] = true if b is '0'-'9'
static DIGIT_VALID: [bool; 256] = {
    let mut table = [false; 256];
    let mut i = b'0';
    while i <= b'9' {
        table[i as usize] = true;
        i += 1;
    }
    table
};

// Lookup table for digit values: value[b] = b - '0' for valid digits
static DIGIT_VALUE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0u8;
    while i <= 9 {
        table[(b'0' + i) as usize] = i;
        i += 1;
    }
    table
};

// Lookup table for two-digit parsing: two_digit[tens][ones] = -(tens*10 + ones)
static TWO_DIGIT: [[i64; 10]; 10] = {
    let mut table = [[0i64; 10]; 10];
    let mut tens = 0;
    while tens < 10 {
        let mut ones = 0;
        while ones < 10 {
            table[tens][ones] = -((tens * 10 + ones) as i64);
            ones += 1;
        }
        tens += 1;
    }
    table
};

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

        let len = digits.len();
        if len == 0 {
            return Err(());
        }

        // Validate all digits first using lookup table
        for &byte in digits {
            if !DIGIT_VALID[byte as usize] {
                return Err(());
            }
        }

        // Accumulate as negative to handle i64::MIN correctly
        let mut result: i64 = 0;
        let mut i = 0;

        // Process pairs of digits using lookup table
        while i + 1 < len {
            let tens = DIGIT_VALUE[digits[i] as usize] as usize;
            let ones = DIGIT_VALUE[digits[i + 1] as usize] as usize;
            let pair_value = TWO_DIGIT[tens][ones];

            result = result.checked_mul(100).ok_or(())?;
            result = result.checked_add(pair_value).ok_or(())?;
            i += 2;
        }

        // Process remaining single digit if any
        if i < len {
            let digit_val = DIGIT_VALUE[digits[i] as usize] as i64;
            result = result.checked_mul(10).ok_or(())?;
            result = result.checked_sub(digit_val).ok_or(())?;
        }

        if negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}
