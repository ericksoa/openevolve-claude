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

        // Fast path for 1-4 digit numbers (no overflow possible)
        match len {
            1 => {
                let d = digits[0].wrapping_sub(b'0');
                if d > 9 {
                    return Err(());
                }
                let val = d as i64;
                return Ok(if negative { -val } else { val });
            }
            2 => unsafe {
                let d0 = digits.get_unchecked(0).wrapping_sub(b'0');
                let d1 = digits.get_unchecked(1).wrapping_sub(b'0');
                if d0 > 9 || d1 > 9 {
                    return Err(());
                }
                let val = (d0 as i64) * 10 + (d1 as i64);
                return Ok(if negative { -val } else { val });
            },
            3 => unsafe {
                let d0 = digits.get_unchecked(0).wrapping_sub(b'0');
                let d1 = digits.get_unchecked(1).wrapping_sub(b'0');
                let d2 = digits.get_unchecked(2).wrapping_sub(b'0');
                if d0 > 9 || d1 > 9 || d2 > 9 {
                    return Err(());
                }
                let val = (d0 as i64) * 100 + (d1 as i64) * 10 + (d2 as i64);
                return Ok(if negative { -val } else { val });
            },
            4 => unsafe {
                let d0 = digits.get_unchecked(0).wrapping_sub(b'0');
                let d1 = digits.get_unchecked(1).wrapping_sub(b'0');
                let d2 = digits.get_unchecked(2).wrapping_sub(b'0');
                let d3 = digits.get_unchecked(3).wrapping_sub(b'0');
                if d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9 {
                    return Err(());
                }
                let val = (d0 as i64) * 1000 + (d1 as i64) * 100 + (d2 as i64) * 10 + (d3 as i64);
                return Ok(if negative { -val } else { val });
            },
            _ => {}
        }

        // For larger numbers, use negative accumulation
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

                let v0 = d0.wrapping_sub(b'0');
                let v1 = d1.wrapping_sub(b'0');
                let v2 = d2.wrapping_sub(b'0');
                let v3 = d3.wrapping_sub(b'0');
                let v4 = d4.wrapping_sub(b'0');
                let v5 = d5.wrapping_sub(b'0');
                let v6 = d6.wrapping_sub(b'0');
                let v7 = d7.wrapping_sub(b'0');

                if (v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7) > 9 {
                    return Err(());
                }

                let chunk = (v0 as i64) * 10_000_000
                    + (v1 as i64) * 1_000_000
                    + (v2 as i64) * 100_000
                    + (v3 as i64) * 10_000
                    + (v4 as i64) * 1_000
                    + (v5 as i64) * 100
                    + (v6 as i64) * 10
                    + (v7 as i64);

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

                let v0 = d0.wrapping_sub(b'0');
                let v1 = d1.wrapping_sub(b'0');
                let v2 = d2.wrapping_sub(b'0');
                let v3 = d3.wrapping_sub(b'0');

                if (v0 | v1 | v2 | v3) > 9 {
                    return Err(());
                }

                let chunk = (v0 as i64) * 1_000
                    + (v1 as i64) * 100
                    + (v2 as i64) * 10
                    + (v3 as i64);

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

                let v0 = d0.wrapping_sub(b'0');
                let v1 = d1.wrapping_sub(b'0');

                if (v0 | v1) > 9 {
                    return Err(());
                }

                let chunk = (v0 as i64) * 10 + (v1 as i64);

                result = result.checked_mul(100).ok_or(())?;
                result = result.checked_sub(chunk).ok_or(())?;
            }
            i += 2;
        }

        // Process remaining digit
        if i < len {
            unsafe {
                let d = *digits.get_unchecked(i);
                let v = d.wrapping_sub(b'0');
                if v > 9 {
                    return Err(());
                }
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(v as i64).ok_or(())?;
            }
        }

        if negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}
