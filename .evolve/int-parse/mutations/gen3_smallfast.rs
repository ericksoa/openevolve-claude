use crate::IntParser;

pub struct EvolvedParser;

impl IntParser for EvolvedParser {
    #[inline(always)]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        if input.is_empty() {
            return Err(());
        }

        let mut idx = 0;
        let neg = input[0] == b'-';
        if neg {
            idx += 1;
            if idx >= input.len() {
                return Err(());
            }
        }

        let remaining = &input[idx..];
        let len = remaining.len();

        // Fast paths for small numbers (1-4 digits)
        match len {
            1 => {
                let d0 = remaining[0].wrapping_sub(b'0');
                if d0 > 9 {
                    return Err(());
                }
                let val = d0 as i64;
                Ok(if neg { -val } else { val })
            }
            2 => {
                let d0 = remaining[0].wrapping_sub(b'0');
                let d1 = remaining[1].wrapping_sub(b'0');
                if d0 > 9 || d1 > 9 {
                    return Err(());
                }
                let val = (d0 as i64) * 10 + (d1 as i64);
                Ok(if neg { -val } else { val })
            }
            3 => {
                let d0 = remaining[0].wrapping_sub(b'0');
                let d1 = remaining[1].wrapping_sub(b'0');
                let d2 = remaining[2].wrapping_sub(b'0');
                if d0 > 9 || d1 > 9 || d2 > 9 {
                    return Err(());
                }
                let val = (d0 as i64) * 100 + (d1 as i64) * 10 + (d2 as i64);
                Ok(if neg { -val } else { val })
            }
            4 => {
                let d0 = remaining[0].wrapping_sub(b'0');
                let d1 = remaining[1].wrapping_sub(b'0');
                let d2 = remaining[2].wrapping_sub(b'0');
                let d3 = remaining[3].wrapping_sub(b'0');
                if d0 > 9 || d1 > 9 || d2 > 9 || d3 > 9 {
                    return Err(());
                }
                let val = (d0 as i64) * 1000 + (d1 as i64) * 100 + (d2 as i64) * 10 + (d3 as i64);
                Ok(if neg { -val } else { val })
            }
            _ => {
                // 8-4-2-1 unrolling with negative accumulation for larger numbers
                let mut result: i64 = 0;
                let mut i = 0;

                // Process 8 digits at a time
                while i + 8 <= len {
                    let d0 = remaining[i].wrapping_sub(b'0');
                    let d1 = remaining[i + 1].wrapping_sub(b'0');
                    let d2 = remaining[i + 2].wrapping_sub(b'0');
                    let d3 = remaining[i + 3].wrapping_sub(b'0');
                    let d4 = remaining[i + 4].wrapping_sub(b'0');
                    let d5 = remaining[i + 5].wrapping_sub(b'0');
                    let d6 = remaining[i + 6].wrapping_sub(b'0');
                    let d7 = remaining[i + 7].wrapping_sub(b'0');

                    if (d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7) > 9 {
                        return Err(());
                    }

                    let chunk = (d0 as i64) * 10_000_000
                        + (d1 as i64) * 1_000_000
                        + (d2 as i64) * 100_000
                        + (d3 as i64) * 10_000
                        + (d4 as i64) * 1_000
                        + (d5 as i64) * 100
                        + (d6 as i64) * 10
                        + (d7 as i64);

                    result = result.checked_mul(100_000_000).ok_or(())?;
                    result = result.checked_sub(chunk).ok_or(())?;

                    i += 8;
                }

                // Process 4 digits at a time
                if i + 4 <= len {
                    let d0 = remaining[i].wrapping_sub(b'0');
                    let d1 = remaining[i + 1].wrapping_sub(b'0');
                    let d2 = remaining[i + 2].wrapping_sub(b'0');
                    let d3 = remaining[i + 3].wrapping_sub(b'0');

                    if (d0 | d1 | d2 | d3) > 9 {
                        return Err(());
                    }

                    let chunk = (d0 as i64) * 1_000
                        + (d1 as i64) * 100
                        + (d2 as i64) * 10
                        + (d3 as i64);

                    result = result.checked_mul(10_000).ok_or(())?;
                    result = result.checked_sub(chunk).ok_or(())?;

                    i += 4;
                }

                // Process 2 digits at a time
                if i + 2 <= len {
                    let d0 = remaining[i].wrapping_sub(b'0');
                    let d1 = remaining[i + 1].wrapping_sub(b'0');

                    if (d0 | d1) > 9 {
                        return Err(());
                    }

                    let chunk = (d0 as i64) * 10 + (d1 as i64);

                    result = result.checked_mul(100).ok_or(())?;
                    result = result.checked_sub(chunk).ok_or(())?;

                    i += 2;
                }

                // Process last digit if any
                if i < len {
                    let d0 = remaining[i].wrapping_sub(b'0');
                    if d0 > 9 {
                        return Err(());
                    }
                    result = result.checked_mul(10).ok_or(())?;
                    result = result.checked_sub(d0 as i64).ok_or(())?;
                }

                if neg {
                    Ok(result)
                } else {
                    result.checked_neg().ok_or(())
                }
            }
        }
    }
}
