//! Background collector thread. Polls the system at `poll_hz` and publishes
//! the latest immutable [`RawMetrics`] snapshot.
//!
//! The sample types are platform-neutral; the thread and the OS samplers
//! exist only in native builds (the browser build feeds [`crate::sim`]).

#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
#[cfg(not(target_arch = "wasm32"))]
pub mod system;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::Collector;

/// Throughput of one disk or network adapter, bytes/s.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IoSample {
    /// Adapter name, or disk name.
    pub name: String,
    /// Alternative name accepted for selection (a disk's mount point).
    pub alt: Option<String>,
    /// read / rx
    pub a_bps: f64,
    /// write / tx
    pub b_bps: f64,
}

impl IoSample {
    /// Case-insensitive match on name or alt, ignoring trailing path separators.
    pub fn matches(&self, want: &str) -> bool {
        let norm = |s: &str| s.trim().trim_end_matches(['\\', '/']).to_ascii_lowercase();
        let w = norm(want);
        !w.is_empty() && (norm(&self.name) == w || self.alt.as_deref().is_some_and(|a| norm(a) == w))
    }
}

/// One raw sample. `None` means "this platform / machine can't provide it".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawMetrics {
    /// Increments with every published sample.
    pub seq: u64,
    /// Seconds since the previous sample.
    pub elapsed: f64,
    /// System-wide CPU utilisation, 0..=100.
    pub cpu_total: Option<f32>,
    /// This process's CPU utilisation on the same system-wide 0..=100 basis.
    pub cpu_self: Option<f32>,
    pub mem_used: Option<u64>,
    pub mem_total: Option<u64>,
    pub disk_read_bps: Option<f64>,
    pub disk_write_bps: Option<f64>,
    pub net_rx_bps: Option<f64>,
    pub net_tx_bps: Option<f64>,
    /// 0..=100
    pub gpu_util: Option<f32>,
    /// 0..=100
    pub gpu_mem_util: Option<f32>,
    /// Per disk (empty when unavailable).
    pub disks: Vec<IoSample>,
    /// Per network adapter, loopback excluded (empty when unavailable).
    pub nics: Vec<IoSample>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_sample_matching() {
        let d = IoSample { name: "Windows".into(), alt: Some(r"C:\".into()), ..Default::default() };
        assert!(d.matches("windows"));
        assert!(d.matches("C:"));
        assert!(d.matches(r"c:\"));
        assert!(!d.matches("D:"));
        assert!(!d.matches(""));
        let n = IoSample { name: "eth0".into(), ..Default::default() };
        assert!(n.matches("ETH0") && !n.matches("eth1"));
    }
}
