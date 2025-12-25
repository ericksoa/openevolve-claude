use crate::SubstringSearcher;

pub struct EvolvedSearcher;

impl SubstringSearcher for EvolvedSearcher {
    #[inline]
    fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        let n = haystack.len();
        let m = needle.len();

        if m == 0 {
            return Some(0);
        }
        if m > n {
            return None;
        }

        unsafe {
            match m {
                1 => {
                    let nb = *needle.get_unchecked(0);
                    for i in 0..n {
                        if *haystack.get_unchecked(i) == nb {
                            return Some(i);
                        }
                    }
                    None
                }
                2 => {
                    let n0 = *needle.get_unchecked(0);
                    let n1 = *needle.get_unchecked(1);
                    for i in 0..n - 1 {
                        if *haystack.get_unchecked(i) == n0 && *haystack.get_unchecked(i + 1) == n1 {
                            return Some(i);
                        }
                    }
                    None
                }
                3 => {
                    let n0 = *needle.get_unchecked(0);
                    let n1 = *needle.get_unchecked(1);
                    let n2 = *needle.get_unchecked(2);
                    for i in 0..n - 2 {
                        if *haystack.get_unchecked(i) == n0
                            && *haystack.get_unchecked(i + 1) == n1
                            && *haystack.get_unchecked(i + 2) == n2
                        {
                            return Some(i);
                        }
                    }
                    None
                }
                4 => {
                    let n0 = *needle.get_unchecked(0);
                    let n1 = *needle.get_unchecked(1);
                    let n2 = *needle.get_unchecked(2);
                    let n3 = *needle.get_unchecked(3);
                    for i in 0..n - 3 {
                        if *haystack.get_unchecked(i) == n0
                            && *haystack.get_unchecked(i + 1) == n1
                            && *haystack.get_unchecked(i + 2) == n2
                            && *haystack.get_unchecked(i + 3) == n3
                        {
                            return Some(i);
                        }
                    }
                    None
                }
                _ => Self::search_long(haystack, needle, n, m),
            }
        }
    }
}

impl EvolvedSearcher {
    #[inline(always)]
    unsafe fn search_long(
        haystack: &[u8],
        needle: &[u8],
        n: usize,
        m: usize,
    ) -> Option<usize> {
        let mut skip = [m; 256];
        for i in 0..m - 1 {
            skip[*needle.get_unchecked(i) as usize] = m - 1 - i;
        }

        let last = *needle.get_unchecked(m - 1);
        let first = *needle.get_unchecked(0);

        let mut i = m - 1;
        while i < n {
            let h = *haystack.get_unchecked(i);

            if h == last && *haystack.get_unchecked(i + 1 - m) == first {
                let mut j = 1;
                let mut match_found = true;
                while j < m {
                    if *haystack.get_unchecked(i + 1 - m + j) != *needle.get_unchecked(j) {
                        match_found = false;
                        break;
                    }
                    j += 1;
                }

                if match_found {
                    return Some(i + 1 - m);
                }
            }

            i += skip[h as usize];
        }

        None
    }
}
