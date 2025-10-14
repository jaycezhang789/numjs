use std::sync::atomic::{AtomicUsize, Ordering};

static COPY_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn record_copy_bytes(bytes: usize) {
    COPY_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn copy_bytes_total() -> u64 {
    COPY_BYTES.load(Ordering::Relaxed) as u64
}

pub fn take_copy_bytes() -> u64 {
    COPY_BYTES.swap(0, Ordering::Relaxed) as u64
}

pub fn reset_copy_bytes() {
    COPY_BYTES.store(0, Ordering::Relaxed);
}
