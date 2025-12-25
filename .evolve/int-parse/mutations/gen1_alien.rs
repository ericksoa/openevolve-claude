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

        // Ultra-fast path for 1-8 digit numbers - branchless SWAR approach
        if len <= 8 {
            return parse_small_swar(digits, negative);
        }

        // For longer numbers, use hybrid approach
        parse_long_hybrid(digits, negative)
    }
}

#[inline(always)]
fn parse_small_swar(digits: &[u8], negative: bool) -> Result<i64, ()> {
    let len = digits.len();

    // Read up to 8 bytes as u64 (unaligned read is safe for small slices)
    let mut buf = [b'0'; 8];
    unsafe {
        std::ptr::copy_nonoverlapping(digits.as_ptr(), buf.as_mut_ptr(), len);
    }

    let chunk = u64::from_le_bytes(buf);

    // Parallel digit validation using SWAR
    // Subtract '0' from all bytes
    let zeros = 0x3030303030303030u64;
    let adjusted = chunk.wrapping_sub(zeros);

    // Check if all relevant bytes are valid digits (0-9)
    // Create a mask for the bytes we care about
    let mut mask = 0xFFFFFFFFFFFFFFFFu64;
    if len < 8 {
        mask = (1u64 << (len * 8)) - 1;
    }

    // Each byte should be 0-9 after subtracting '0'
    // Check by adding 0x76 (0x80 - 10) and seeing if high bit is set
    let check = adjusted & mask;
    let overflow = check.wrapping_add(0x7676767676767676u64 & mask);
    let valid_mask = (1u64 << (len * 8)) - 1;
    let expected_high_bits = 0x8080808080808080u64 & valid_mask;

    if (overflow & expected_high_bits) != expected_high_bits {
        // Fallback to byte-by-byte validation
        for &byte in digits {
            if byte < b'0' || byte > b'9' {
                return Err(());
            }
        }
    }

    // Convert bytes to digits using SWAR multiplication
    let bytes = adjusted.to_le_bytes();

    // Branchless calculation based on length
    // Use lookup table approach for speed
    let result = match len {
        1 => bytes[0] as i64,
        2 => bytes[0] as i64 * 10 + bytes[1] as i64,
        3 => bytes[0] as i64 * 100 + bytes[1] as i64 * 10 + bytes[2] as i64,
        4 => bytes[0] as i64 * 1000 + bytes[1] as i64 * 100 + bytes[2] as i64 * 10 + bytes[3] as i64,
        5 => bytes[0] as i64 * 10000 + bytes[1] as i64 * 1000 + bytes[2] as i64 * 100 + bytes[3] as i64 * 10 + bytes[4] as i64,
        6 => bytes[0] as i64 * 100000 + bytes[1] as i64 * 10000 + bytes[2] as i64 * 1000 + bytes[3] as i64 * 100 + bytes[4] as i64 * 10 + bytes[5] as i64,
        7 => bytes[0] as i64 * 1000000 + bytes[1] as i64 * 100000 + bytes[2] as i64 * 10000 + bytes[3] as i64 * 1000 + bytes[4] as i64 * 100 + bytes[5] as i64 * 10 + bytes[6] as i64,
        8 => bytes[0] as i64 * 10000000 + bytes[1] as i64 * 1000000 + bytes[2] as i64 * 100000 + bytes[3] as i64 * 10000 + bytes[4] as i64 * 1000 + bytes[5] as i64 * 100 + bytes[6] as i64 * 10 + bytes[7] as i64,
        _ => unreachable!(),
    };

    Ok(if negative { -result } else { result })
}

#[inline(always)]
fn parse_long_hybrid(digits: &[u8], negative: bool) -> Result<i64, ()> {
    let len = digits.len();
    let mut result: i64 = 0;
    let mut i = 0;

    // Process 8 digits at a time using SWAR
    while i + 8 <= len {
        let chunk = unsafe {
            (digits.as_ptr().add(i) as *const u64).read_unaligned()
        };

        // Validate and convert 8 digits
        let zeros = 0x3030303030303030u64;
        let adjusted = chunk.wrapping_sub(zeros);

        // Parallel validation
        let overflow = adjusted.wrapping_add(0x7676767676767676u64);
        if (overflow & 0x8080808080808080u64) != 0x8080808080808080u64 {
            // Invalid digit found
            for j in 0..8 {
                if digits[i + j] < b'0' || digits[i + j] > b'9' {
                    return Err(());
                }
            }
            return Err(());
        }

        // Convert 8 ASCII digits using parallel extraction
        let bytes = adjusted.to_le_bytes();
        let chunk_val = (bytes[0] as i64) * 10000000
            + (bytes[1] as i64) * 1000000
            + (bytes[2] as i64) * 100000
            + (bytes[3] as i64) * 10000
            + (bytes[4] as i64) * 1000
            + (bytes[5] as i64) * 100
            + (bytes[6] as i64) * 10
            + (bytes[7] as i64);

        // Accumulate as negative
        result = result.checked_mul(100000000).ok_or(())?;
        result = result.checked_sub(chunk_val).ok_or(())?;
        i += 8;
    }

    // Handle remaining 1-7 digits with optimized unrolling
    let remaining = len - i;
    if remaining > 0 {
        let mut tail_buf = [b'0'; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(digits.as_ptr().add(i), tail_buf.as_mut_ptr(), remaining);
        }

        let chunk = u64::from_le_bytes(tail_buf);
        let zeros = 0x3030303030303030u64;
        let adjusted = chunk.wrapping_sub(zeros);

        // Validate only the bytes we care about
        let mask = (1u64 << (remaining * 8)) - 1;
        let check = adjusted & mask;
        let overflow = check.wrapping_add(0x7676767676767676u64 & mask);
        let expected = 0x8080808080808080u64 & mask;

        if (overflow & expected) != expected {
            for j in 0..remaining {
                if digits[i + j] < b'0' || digits[i + j] > b'9' {
                    return Err(());
                }
            }
        }

        let bytes = adjusted.to_le_bytes();

        // Branchless tail conversion
        let multipliers = [
            [1, 0, 0, 0, 0, 0, 0],
            [10, 1, 0, 0, 0, 0, 0],
            [100, 10, 1, 0, 0, 0, 0],
            [1000, 100, 10, 1, 0, 0, 0],
            [10000, 1000, 100, 10, 1, 0, 0],
            [100000, 10000, 1000, 100, 10, 1, 0],
            [1000000, 100000, 10000, 1000, 100, 10, 1],
        ];

        let muls = &multipliers[remaining - 1];
        let tail_val = (bytes[0] as i64) * muls[0]
            + (bytes[1] as i64) * muls[1]
            + (bytes[2] as i64) * muls[2]
            + (bytes[3] as i64) * muls[3]
            + (bytes[4] as i64) * muls[4]
            + (bytes[5] as i64) * muls[5]
            + (bytes[6] as i64) * muls[6];

        let power_of_10 = [10, 100, 1000, 10000, 100000, 1000000, 10000000];
        result = result.checked_mul(power_of_10[remaining - 1]).ok_or(())?;
        result = result.checked_sub(tail_val).ok_or(())?;
    }

    if negative {
        Ok(result)
    } else {
        result.checked_neg().ok_or(())
    }
}
