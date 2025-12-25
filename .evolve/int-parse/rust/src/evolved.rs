use crate::IntParser;

pub struct EvolvedParser;

impl IntParser for EvolvedParser {
    #[inline(always)]
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
        let mut result: i64 = 0;
        let mut i = 0;
        let ptr = digits.as_ptr();

        // Process 8 digits at a time
        while i + 8 <= len {
            unsafe {
                let d0 = *ptr.add(i);
                let d1 = *ptr.add(i + 1);
                let d2 = *ptr.add(i + 2);
                let d3 = *ptr.add(i + 3);
                let d4 = *ptr.add(i + 4);
                let d5 = *ptr.add(i + 5);
                let d6 = *ptr.add(i + 6);
                let d7 = *ptr.add(i + 7);

                let v0 = d0.wrapping_sub(b'0');
                let v1 = d1.wrapping_sub(b'0');
                let v2 = d2.wrapping_sub(b'0');
                let v3 = d3.wrapping_sub(b'0');
                let v4 = d4.wrapping_sub(b'0');
                let v5 = d5.wrapping_sub(b'0');
                let v6 = d6.wrapping_sub(b'0');
                let v7 = d7.wrapping_sub(b'0');

                if v0 > 9 || v1 > 9 || v2 > 9 || v3 > 9 || v4 > 9 || v5 > 9 || v6 > 9 || v7 > 9 {
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
                let d0 = *ptr.add(i);
                let d1 = *ptr.add(i + 1);
                let d2 = *ptr.add(i + 2);
                let d3 = *ptr.add(i + 3);

                let v0 = d0.wrapping_sub(b'0');
                let v1 = d1.wrapping_sub(b'0');
                let v2 = d2.wrapping_sub(b'0');
                let v3 = d3.wrapping_sub(b'0');

                if v0 > 9 || v1 > 9 || v2 > 9 || v3 > 9 {
                    return Err(());
                }

                let chunk = (v0 as i64) * 1000
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
                let d0 = *ptr.add(i);
                let d1 = *ptr.add(i + 1);

                let v0 = d0.wrapping_sub(b'0');
                let v1 = d1.wrapping_sub(b'0');

                if v0 > 9 || v1 > 9 {
                    return Err(());
                }

                let chunk = (v0 as i64) * 10 + (v1 as i64);

                result = result.checked_mul(100).ok_or(())?;
                result = result.checked_sub(chunk).ok_or(())?;
            }
            i += 2;
        }

        // Process remaining single digit
        if i < len {
            unsafe {
                let byte = *ptr.add(i);
                let v = byte.wrapping_sub(b'0');
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
