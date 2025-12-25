use crate::IntParser;

pub struct EvolvedParser;

impl IntParser for EvolvedParser {
    #[inline(always)]
    fn parse(&self, input: &[u8]) -> Result<i64, ()> {
        parse_int(input)
    }
}

#[inline(always)]
fn parse_int(input: &[u8]) -> Result<i64, ()> {
    let len = input.len();
    if len == 0 {
        return Err(());
    }

    let (start, negative) = if input[0] == b'-' {
        if len == 1 {
            return Err(());
        }
        (1, true)
    } else {
        (0, false)
    };

    let mut acc: i64 = 0;
    let mut i = start;
    let remaining = len - start;

    // Process 8 digits at a time
    let end8 = start + (remaining / 8) * 8;
    while i < end8 {
        let d0 = input[i].wrapping_sub(b'0');
        let d1 = input[i + 1].wrapping_sub(b'0');
        let d2 = input[i + 2].wrapping_sub(b'0');
        let d3 = input[i + 3].wrapping_sub(b'0');
        let d4 = input[i + 4].wrapping_sub(b'0');
        let d5 = input[i + 5].wrapping_sub(b'0');
        let d6 = input[i + 6].wrapping_sub(b'0');
        let d7 = input[i + 7].wrapping_sub(b'0');

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

        let new_acc = acc.wrapping_mul(100000000).wrapping_sub(chunk);
        if acc < 0 && new_acc > acc {
            return Err(());
        }
        acc = new_acc;
        i += 8;
    }

    // Process 4 digits at a time
    if i + 4 <= len {
        let d0 = input[i].wrapping_sub(b'0');
        let d1 = input[i + 1].wrapping_sub(b'0');
        let d2 = input[i + 2].wrapping_sub(b'0');
        let d3 = input[i + 3].wrapping_sub(b'0');

        if (d0 | d1 | d2 | d3) > 9 {
            return Err(());
        }

        let chunk = (d0 as i64) * 1000 + (d1 as i64) * 100 + (d2 as i64) * 10 + (d3 as i64);
        let new_acc = acc.wrapping_mul(10000).wrapping_sub(chunk);
        if acc < 0 && new_acc > acc {
            return Err(());
        }
        acc = new_acc;
        i += 4;
    }

    // Process 2 digits at a time
    if i + 2 <= len {
        let d0 = input[i].wrapping_sub(b'0');
        let d1 = input[i + 1].wrapping_sub(b'0');

        if (d0 | d1) > 9 {
            return Err(());
        }

        let chunk = (d0 as i64) * 10 + (d1 as i64);
        let new_acc = acc.wrapping_mul(100).wrapping_sub(chunk);
        if acc < 0 && new_acc > acc {
            return Err(());
        }
        acc = new_acc;
        i += 2;
    }

    // Process remaining digit
    if i < len {
        let d = input[i].wrapping_sub(b'0');
        if d > 9 {
            return Err(());
        }
        let new_acc = acc.wrapping_mul(10).wrapping_sub(d as i64);
        if acc < 0 && new_acc > acc {
            return Err(());
        }
        acc = new_acc;
    }

    if negative {
        Ok(acc)
    } else {
        if acc == i64::MIN {
            return Err(());
        }
        Ok(-acc)
    }
}
