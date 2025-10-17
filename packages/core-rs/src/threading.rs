use std::env;
use std::sync::OnceLock;

#[allow(dead_code)]
const THREAD_ENV: &str = "NUMJS_CPU_THREADS";
const DISABLE_ENV: &str = "NUMJS_DISABLE_PARALLEL";
const MIN_ELEMS_ENV: &str = "NUMJS_PARALLEL_MIN_ELEMS";
const MIN_FLOPS_ENV: &str = "NUMJS_PARALLEL_MIN_FLOPS";

const DEFAULT_MIN_ELEMS: usize = 16 * 16;
const DEFAULT_MIN_FLOPS: usize = 128 * 128 * 32;

#[allow(dead_code)]
static THREAD_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
static DISABLE_PARALLEL: OnceLock<bool> = OnceLock::new();
static MIN_ELEMENTS: OnceLock<usize> = OnceLock::new();
static MIN_FLOPS: OnceLock<usize> = OnceLock::new();

#[allow(dead_code)]
pub fn thread_override() -> Option<usize> {
    *THREAD_OVERRIDE.get_or_init(|| match env::var(THREAD_ENV) {
        Ok(value) => match value.trim() {
            "" => None,
            raw => match raw.parse::<usize>() {
                Ok(0) => None,
                Ok(n) => Some(n.max(1)),
                Err(_) => None,
            },
        },
        Err(_) => None,
    })
}

pub fn parallel_disabled() -> bool {
    *DISABLE_PARALLEL.get_or_init(|| {
        matches!(
            env::var(DISABLE_ENV)
                .ok()
                .map(|raw| raw.trim().to_ascii_lowercase()),
            Some(ref value) if value == "1" || value == "true" || value == "yes"
        )
    })
}

fn min_elements_threshold() -> usize {
    *MIN_ELEMENTS.get_or_init(|| {
        env::var(MIN_ELEMS_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_MIN_ELEMS)
    })
}

fn min_flops_threshold() -> usize {
    *MIN_FLOPS.get_or_init(|| {
        env::var(MIN_FLOPS_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_MIN_FLOPS)
    })
}

pub fn should_parallelize(m: usize, n: usize, k: usize) -> bool {
    if parallel_disabled() {
        return false;
    }
    let elems = m * n;
    let flops = elems.saturating_mul(k);
    elems >= min_elements_threshold() && flops >= min_flops_threshold()
}

#[cfg(feature = "parallel")]
use std::sync::Once;

#[cfg(feature = "parallel")]
static INIT_RAYON: Once = Once::new();

#[cfg(feature = "parallel")]
pub fn ensure_rayon_pool() {
    use rayon::ThreadPoolBuilder;
    INIT_RAYON.call_once(|| {
        if let Some(threads) = thread_override() {
            let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();
        } else {
            let _ = ThreadPoolBuilder::new().build_global();
        }
    });
}

#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn ensure_rayon_pool() {}
