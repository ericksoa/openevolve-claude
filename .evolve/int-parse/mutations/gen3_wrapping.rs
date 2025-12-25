use crate::IntParser;

pub struct EvolvedParser;

impl IntParser for EvolvedParser {
    #[inline(always)]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        if input.is_empty() {
            return Err(());
        }

        let (is_negative, digits) = if input[0] == b'-' {
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

        let mut result: i64 = 0;
        let len = digits.len();

        if len > 19 {
            return Err(());
        }

        for &byte in digits {
            let digit = byte.wrapping_sub(b'0');
            if digit > 9 {
                return Err(());
            }

            result = result.wrapping_mul(10).wrapping_sub(digit as i64);

            if len == 19 && result > 0 {
                return Err(());
            }
        }

        if is_negative {
            Ok(result)
        } else {
            if result == i64::MIN {
                return Err(());
            }
            Ok(-result)
        }
    }
}
