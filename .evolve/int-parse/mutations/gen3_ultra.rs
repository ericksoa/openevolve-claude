use crate::IntParser;

pub struct EvolvedParser;

impl IntParser for EvolvedParser {
    #[inline(always)]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        if input.is_empty() {
            return Err(());
        }

        let mut ptr = input.as_ptr();
        let end = unsafe { ptr.add(input.len()) };

        let negative = unsafe { *ptr == b'-' };
        if negative {
            ptr = unsafe { ptr.add(1) };
            if ptr == end {
                return Err(());
            }
        }

        let len = end as usize - ptr as usize;
        if len == 0 || len > 19 {
            return Err(());
        }

        let mut result: i64 = 0;
        let mut remaining = len;

        // Process 8 digits at a time
        while remaining >= 8 {
            unsafe {
                let d0 = (*ptr).wrapping_sub(b'0');
                let d1 = (*ptr.add(1)).wrapping_sub(b'0');
                let d2 = (*ptr.add(2)).wrapping_sub(b'0');
                let d3 = (*ptr.add(3)).wrapping_sub(b'0');
                let d4 = (*ptr.add(4)).wrapping_sub(b'0');
                let d5 = (*ptr.add(5)).wrapping_sub(b'0');
                let d6 = (*ptr.add(6)).wrapping_sub(b'0');
                let d7 = (*ptr.add(7)).wrapping_sub(b'0');

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
                ptr = ptr.add(8);
                remaining -= 8;
            }
        }

        // Process 4 digits
        if remaining >= 4 {
            unsafe {
                let d0 = (*ptr).wrapping_sub(b'0');
                let d1 = (*ptr.add(1)).wrapping_sub(b'0');
                let d2 = (*ptr.add(2)).wrapping_sub(b'0');
                let d3 = (*ptr.add(3)).wrapping_sub(b'0');

                if (d0 | d1 | d2 | d3) > 9 {
                    return Err(());
                }

                let chunk = (d0 as i64) * 1_000
                    + (d1 as i64) * 100
                    + (d2 as i64) * 10
                    + (d3 as i64);

                result = result.checked_mul(10_000).ok_or(())?;
                result = result.checked_sub(chunk).ok_or(())?;
                ptr = ptr.add(4);
                remaining -= 4;
            }
        }

        // Process 2 digits
        if remaining >= 2 {
            unsafe {
                let d0 = (*ptr).wrapping_sub(b'0');
                let d1 = (*ptr.add(1)).wrapping_sub(b'0');

                if (d0 | d1) > 9 {
                    return Err(());
                }

                let chunk = (d0 as i64) * 10 + (d1 as i64);

                result = result.checked_mul(100).ok_or(())?;
                result = result.checked_sub(chunk).ok_or(())?;
                ptr = ptr.add(2);
                remaining -= 2;
            }
        }

        // Process 1 digit
        if remaining == 1 {
            unsafe {
                let d0 = (*ptr).wrapping_sub(b'0');
                if d0 > 9 {
                    return Err(());
                }
                result = result.checked_mul(10).ok_or(())?;
                result = result.checked_sub(d0 as i64).ok_or(())?;
            }
        }

        if negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}
