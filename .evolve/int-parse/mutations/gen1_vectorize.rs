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

        // For very short inputs, use simple loop
        if len <= 4 {
            let mut result: i64 = 0;
            for &byte in digits {
                if byte < b'0' || byte > b'9' {
                    return Err(());
                }
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub((byte - b'0') as i64).ok_or(())?;
            }
            return if negative {
                Ok(result)
            } else {
                result.checked_neg().ok_or(())
            };
        }

        // SWAR approach for longer inputs
        unsafe {
            let mut result: i64 = 0;
            let ptr = digits.as_ptr();
            let mut pos = 0;

            // Process 8 bytes at a time when possible
            while pos + 8 <= len {
                let chunk = (ptr.add(pos) as *const u64).read_unaligned();

                // Parallel digit validation using SWAR
                // Check all bytes are in range [0x30, 0x39] ('0' to '9')
                let ge_0 = chunk.wrapping_sub(0x3030303030303030u64);
                let le_9 = !chunk.wrapping_add(0xC6C6C6C6C6C6C6C6u64);

                // If any byte is out of range, at least one high bit will be set incorrectly
                if (ge_0 | le_9) & 0x8080808080808080u64 != 0 {
                    return Err(());
                }

                // Extract digits in parallel
                let digits_u64 = chunk & 0x0F0F0F0F0F0F0F0Fu64;

                // Convert to individual bytes (little-endian)
                let b0 = (digits_u64 & 0xFF) as i64;
                let b1 = ((digits_u64 >> 8) & 0xFF) as i64;
                let b2 = ((digits_u64 >> 16) & 0xFF) as i64;
                let b3 = ((digits_u64 >> 24) & 0xFF) as i64;
                let b4 = ((digits_u64 >> 32) & 0xFF) as i64;
                let b5 = ((digits_u64 >> 40) & 0xFF) as i64;
                let b6 = ((digits_u64 >> 48) & 0xFF) as i64;
                let b7 = ((digits_u64 >> 56) & 0xFF) as i64;

                // Accumulate (unrolled multiply-add chain)
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b0).ok_or(())?;
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b1).ok_or(())?;
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b2).ok_or(())?;
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b3).ok_or(())?;
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b4).ok_or(())?;
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b5).ok_or(())?;
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b6).ok_or(())?;
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(b7).ok_or(())?;

                pos += 8;
            }

            // Handle remaining bytes
            while pos < len {
                let byte = *ptr.add(pos);
                if byte < b'0' || byte > b'9' {
                    return Err(());
                }
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub((byte - b'0') as i64).ok_or(())?;
                pos += 1;
            }

            if negative {
                Ok(result)
            } else {
                result.checked_neg().ok_or(())
            }
        }
    }
}
