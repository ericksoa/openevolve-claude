use crate::IntParser;

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
            let remaining = end.offset_from(ptr) as usize;

            // Process 8 digits at a time
            if remaining >= 8 {
                let chunks = remaining / 8;
                for _ in 0..chunks {
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

                    let chunk = (d0 as i64) * 10000000
                        + (d1 as i64) * 1000000
                        + (d2 as i64) * 100000
                        + (d3 as i64) * 10000
                        + (d4 as i64) * 1000
                        + (d5 as i64) * 100
                        + (d6 as i64) * 10
                        + (d7 as i64);

                    let new_result = result.wrapping_mul(100000000).wrapping_sub(chunk);
                    if result < new_result {
                        return Err(());
                    }

                    result = new_result;
                    ptr = ptr.add(8);
                }
            }

            // Process remaining digits
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
