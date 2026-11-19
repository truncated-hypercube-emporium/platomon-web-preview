//! The collector thread (native builds only).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::{RawMetrics, gpu, system};

type Shared = Arc<Mutex<Option<RawMetrics>>>;

pub struct Collector {
    shared: Shared,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    gpu_backend: Arc<Mutex<&'static str>>,
}

impl Collector {
    pub fn spawn(poll_hz: f32) -> Self {
        let shared: Shared = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let gpu_backend = Arc::new(Mutex::new("probing"));
        let interval = Duration::from_secs_f32(1.0 / poll_hz.clamp(0.5, 20.0))
            .max(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);

        let (s, st, gb) = (shared.clone(), stop.clone(), gpu_backend.clone());
        let handle = thread::Builder::new()
            .name("platomon-collector".into())
            .spawn(move || run(s, st, gb, interval))
            .ok();

        Self { shared, stop, handle, gpu_backend }
    }

    /// Latest snapshot, if any has been published yet.
    pub fn latest(&self) -> Option<RawMetrics> {
        self.shared.lock().ok().and_then(|g| g.clone())
    }

    pub fn gpu_backend(&self) -> &'static str {
        self.gpu_backend.lock().map(|g| *g).unwrap_or("?")
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

fn run(shared: Shared, stop: Arc<AtomicBool>, gpu_backend: Arc<Mutex<&'static str>>, interval: Duration) {
    // Construction touches OS APIs; a panic must never take the app down.
    let Ok(mut sys) = catch_unwind(system::SystemSampler::new) else { return };
    let mut gpu = catch_unwind(gpu::GpuSampler::new).ok();
    if let Ok(mut g) = gpu_backend.lock() {
        *g = gpu.as_ref().map_or("none", |g| g.backend_name());
    }

    let mut seq = 0u64;
    let mut last = Instant::now();
    thread::park_timeout(interval);

    while !stop.load(Ordering::Relaxed) {
        let start = Instant::now();
        let elapsed = start.duration_since(last).as_secs_f64();
        last = start;

        let sampled = catch_unwind(AssertUnwindSafe(|| {
            let mut m = sys.sample(elapsed);
            if let Some(g) = gpu.as_mut().and_then(|g| g.sample()) {
                m.gpu_util = Some(g.util);
                m.gpu_mem_util = g.mem_util;
            }
            m
        }));
        let mut m = match sampled {
            Ok(m) => m,
            Err(_) => {
                gpu = None; // a backend that panicked once is not trusted again
                RawMetrics { elapsed, ..RawMetrics::default() }
            }
        };

        seq += 1;
        m.seq = seq;
        if let Ok(mut g) = shared.lock() {
            *g = Some(m);
        }

        let spent = start.elapsed();
        if spent < interval {
            thread::park_timeout(interval - spent);
        }
    }
}
