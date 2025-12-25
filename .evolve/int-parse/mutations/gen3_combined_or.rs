use crate::IntParser;

pub struct EvolvedParser;

impl IntParser for EvolvedParser {
    #[inline(always)]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        if input.is_empty() {
            return Err(());
        }

        let mut idx = 0;
        let len = input.len();
        let is_negative = input[0] == b'-';

        if is_negative {
            idx = 1;
            if idx >= len {
                return Err(());
            }
        }

        let mut result: i64 = 0;

        // Process 8 digits at a time
        while idx + 8 <= len {
            let v0 = input[idx].wrapping_sub(b'0');
            let v1 = input[idx + 1].wrapping_sub(b'0');
            let v2 = input[idx + 2].wrapping_sub(b'0');
            let v3 = input[idx + 3].wrapping_sub(b'0');
            let v4 = input[idx + 4].wrapping_sub(b'0');
            let v5 = input[idx + 5].wrapping_sub(b'0');
            let v6 = input[idx + 6].wrapping_sub(b'0');
            let v7 = input[idx + 7].wrapping_sub(b'0');

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

            idx += 8;
        }

        // Process 4 digits at a time
        if idx + 4 <= len {
            let v0 = input[idx].wrapping_sub(b'0');
            let v1 = input[idx + 1].wrapping_sub(b'0');
            let v2 = input[idx + 2].wrapping_sub(b'0');
            let v3 = input[idx + 3].wrapping_sub(b'0');

            if (v0 | v1 | v2 | v3) > 9 {
                return Err(());
            }

            let chunk = (v0 as i64) * 1_000
                + (v1 as i64) * 100
                + (v2 as i64) * 10
                + (v3 as i64);

            result = result.checked_mul(10_000).ok_or(())?;
            result = result.checked_sub(chunk).ok_or(())?;

            idx += 4;
        }

        // Process 2 digits at a time
        if idx + 2 <= len {
            let v0 = input[idx].wrapping_sub(b'0');
            let v1 = input[idx + 1].wrapping_sub(b'0');

            if (v0 | v1) > 9 {
                return Err(());
            }

            let chunk = (v0 as i64) * 10 + (v1 as i64);

            result = result.checked_mul(100).ok_or(())?;
            result = result.checked_sub(chunk).ok_or(())?;

            idx += 2;
        }

        // Process final digit
        if idx < len {
            let v0 = input[idx].wrapping_sub(b'0');

            if v0 > 9 {
                return Err(());
            }

            result = result.checked_mul(10).ok_or(())?;
            result = result.checked_sub(v0 as i64).ok_or(())?;
        }

        if is_negative {
            Ok(result)
        } else {
            result.checked_neg().ok_or(())
        }
    }
}
