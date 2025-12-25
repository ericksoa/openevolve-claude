use crate::Sorter;

pub struct EvolvedSorter;

impl Sorter for EvolvedSorter {
    fn sort(&self, data: &mut [u64]) {
        pdqsort(data);
    }
}

const INSERTION_THRESHOLD: usize = 32;
const MAX_DEPTH: usize = 32;

#[inline]
fn pdqsort(data: &mut [u64]) {
    if data.len() <= 1 {
        return;
    }
    pdqsort_recursive(data, 0);
}

#[inline]
fn pdqsort_recursive(data: &mut [u64], depth: usize) {
    let mut len = data.len();

    loop {
        if len <= INSERTION_THRESHOLD {
            insertion_sort(data);
            return;
        }

        if depth >= MAX_DEPTH {
            heapsort(data);
            return;
        }

        let pivot_idx = partition_lomuto(data);
        let (left, right) = data.split_at_mut(pivot_idx);

        if left.len() < right.len() {
            pdqsort_recursive(left, depth + 1);
            data = right;
            len = right.len();
        } else {
            pdqsort_recursive(&mut right[1..], depth + 1);
            data = left;
            len = left.len();
        }
    }
}

#[inline]
fn partition_lomuto(data: &mut [u64]) -> usize {
    let len = data.len();
    if len == 0 {
        return 0;
    }

    let mid = len / 2;

    // Median-of-three pivot selection
    if data[0] > data[mid] {
        data.swap(0, mid);
    }
    if data[mid] > data[len - 1] {
        data.swap(mid, len - 1);
    }
    if data[0] > data[mid] {
        data.swap(0, mid);
    }

    let pivot = data[mid];
    data.swap(mid, len - 1);

    unsafe {
        let ptr = data.as_mut_ptr();
        let mut i = 0;

        for j in 0..len - 1 {
            if *ptr.add(j) <= pivot {
                core::ptr::swap(ptr.add(i), ptr.add(j));
                i += 1;
            }
        }

        core::ptr::swap(ptr.add(i), ptr.add(len - 1));
        i
    }
}

#[inline]
fn insertion_sort(data: &mut [u64]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;

        unsafe {
            let ptr = data.as_mut_ptr();
            while j > 0 && *ptr.add(j - 1) > key {
                *ptr.add(j) = *ptr.add(j - 1);
                j -= 1;
            }
            *ptr.add(j) = key;
        }
    }
}

#[inline]
fn heapsort(data: &mut [u64]) {
    let len = data.len();
    if len <= 1 {
        return;
    }

    for i in (0..len / 2).rev() {
        sift_down(data, i, len);
    }

    for i in (1..len).rev() {
        data.swap(0, i);
        sift_down(data, 0, i);
    }
}

#[inline]
fn sift_down(data: &mut [u64], mut i: usize, n: usize) {
    unsafe {
        let ptr = data.as_mut_ptr();

        loop {
            let mut largest = i;
            let left = 2 * i + 1;
            let right = 2 * i + 2;

            if left < n && *ptr.add(left) > *ptr.add(largest) {
                largest = left;
            }
            if right < n && *ptr.add(right) > *ptr.add(largest) {
                largest = right;
            }

            if largest != i {
                core::ptr::swap(ptr.add(i), ptr.add(largest));
                i = largest;
            } else {
                break;
            }
        }
    }
}
