//! Simulated sensors: a deterministic, time-varying load sequence that
//! produces the same [`RawMetrics`] the OS collector does, so everything
//! downstream (smoothing, rolling peaks, activity score, fps) behaves as on a
//! real machine. Used by the browser build.

use std::f64::consts::TAU;
use std::str::FromStr;

use crate::collector::{IoSample, RawMetrics};

/// Shape of the overall load over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// Smooth idle → busy → idle waves (40 s period) with random bursts.
    #[default]
    Cycle,
    /// Mostly idle with short random bursts to full load.
    Bursts,
    /// Linear ramp 0 → 100 % → 0 over 40 s.
    Ramp,
    /// 0, 25, 50, 75, 100 % held 8 s each.
    Steps,
    /// Near-idle machine.
    Idle,
    /// Sustained heavy load.
    Stress,
    /// Fixed level from [`Sim::manual`].
    Manual,
}

impl Profile {
    pub const ALL: [Profile; 7] =
        [Profile::Cycle, Profile::Bursts, Profile::Ramp, Profile::Steps, Profile::Idle, Profile::Stress, Profile::Manual];

    pub fn id(self) -> &'static str {
        match self {
            Profile::Cycle => "cycle",
            Profile::Bursts => "bursts",
            Profile::Ramp => "ramp",
            Profile::Steps => "steps",
            Profile::Idle => "idle",
            Profile::Stress => "stress",
            Profile::Manual => "manual",
        }
    }
}

impl FromStr for Profile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|p| p.id() == s).ok_or_else(|| format!("unknown load profile '{s}'"))
    }
}

/// Simulated disks (name, mount) and adapters.
pub const DISKS: [(&str, &str); 3] = [("nvme0", r"C:\"), ("sata0", r"D:\"), ("usb0", r"E:\")];
pub const NICS: [&str; 3] = ["Ethernet", "Wi-Fi", "vEthernet"];

const MB: f64 = 1e6;
const MEM_TOTAL: u64 = 32 << 30;

#[derive(Debug, Clone)]
pub struct Sim {
    pub profile: Profile,
    /// Simulated seconds per real second.
    pub speed: f64,
    /// 0..=1, used by [`Profile::Manual`].
    pub manual: f64,
    seed: u64,
    t: f64,
    seq: u64,
}

/// Hash → 0..1.
fn hash01(seed: u64, a: u64, b: u64) -> f64 {
    let mut x = seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Smooth 1-D value noise in 0..1 (one knot per second).
fn noise(seed: u64, ch: u64, t: f64) -> f64 {
    let i = t.floor();
    let f = t - i;
    let s = f * f * (3.0 - 2.0 * f);
    let (a, b) = (hash01(seed, ch, i as u64), hash01(seed, ch, i as u64 + 1));
    a + (b - a) * s
}

/// 1 while a random burst is active: windows of `every` seconds, each with a
/// `chance` of a burst lasting 2–5 s at a random offset.
fn burst(seed: u64, ch: u64, t: f64, every: f64, chance: f64) -> f64 {
    let k = (t / every).floor();
    let w = k as u64;
    if hash01(seed, ch + 100, w) > chance {
        return 0.0;
    }
    let len = 2.0 + 3.0 * hash01(seed, ch + 200, w);
    let start = k * every + (every - len) * hash01(seed, ch + 300, w);
    let x = t - start;
    if (0.0..len).contains(&x) {
        // soft edges
        (x / 0.4).min(1.0).min((len - x) / 0.4)
    } else {
        0.0
    }
}

impl Default for Sim {
    fn default() -> Self {
        Self::new(Profile::Cycle, 1.0, 1)
    }
}

impl Sim {
    pub fn new(profile: Profile, speed: f64, seed: u64) -> Self {
        Self { profile, speed, manual: 0.5, seed, t: 0.0, seq: 0 }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Simulated time, seconds.
    pub fn time(&self) -> f64 {
        self.t
    }

    /// Overall load 0..=1 at simulated time `t`.
    pub fn envelope(&self, t: f64) -> f64 {
        let s = self.seed;
        let e = match self.profile {
            Profile::Cycle => {
                let wave = 0.5 - 0.5 * (TAU * t / 40.0).cos();
                (0.05 + 0.8 * wave + 0.15 * noise(s, 1, t * 0.5)).max(burst(s, 1, t, 15.0, 0.5))
            }
            Profile::Bursts => (0.04 + 0.06 * noise(s, 2, t)).max(burst(s, 2, t, 8.0, 0.6)),
            Profile::Ramp => {
                let p = (t / 40.0).fract();
                if p < 0.5 { 2.0 * p } else { 2.0 - 2.0 * p }
            }
            Profile::Steps => ((t / 8.0).floor() as u64 % 5) as f64 * 0.25,
            Profile::Idle => 0.02 + 0.04 * noise(s, 3, t),
            Profile::Stress => 0.88 + 0.12 * noise(s, 4, t * 2.0),
            Profile::Manual => self.manual,
        };
        e.clamp(0.0, 1.0)
    }

    /// One component's share of the envelope: phase-shifted and noisy so the
    /// solids don't move in lockstep.
    fn comp(&self, ch: u64, t: f64, e: f64) -> f64 {
        let phase = hash01(self.seed, ch, 999) * TAU;
        let wobble = 0.75 + 0.25 * (t * (0.11 + 0.05 * ch as f64) + phase).sin();
        let n = noise(self.seed, ch + 10, t * 1.3) - 0.5;
        let flat = matches!(self.profile, Profile::Ramp | Profile::Steps | Profile::Manual);
        if flat { (e + 0.04 * n).clamp(0.0, 1.0) } else { (e * wobble + 0.12 * n).clamp(0.0, 1.0) }
    }

    /// Advance by `dt` real seconds and return a sample, as the collector would.
    pub fn sample(&mut self, dt: f64) -> RawMetrics {
        let dt = if dt.is_finite() { dt.clamp(0.0, 5.0) } else { 0.0 };
        self.t += dt * self.speed.clamp(0.0, 100.0);
        self.seq += 1;
        let t = self.t;
        let e = self.envelope(t);
        let c = |ch| self.comp(ch, t, e);

        let cpu = 2.0 + 96.0 * c(1);
        let ram = 0.30 + 0.55 * (0.6 * c(2) + 0.4 * noise(self.seed, 20, t * 0.05));
        let disk = c(3);
        let net = c(4);
        let gpu = c(5);

        let disks: Vec<IoSample> = DISKS
            .iter()
            .enumerate()
            .map(|(i, (name, mount))| {
                let share = [0.7, 0.25, 0.05][i] * (0.5 + noise(self.seed, 30 + i as u64, t * 0.4));
                IoSample {
                    name: (*name).into(),
                    alt: Some((*mount).into()),
                    a_bps: (0.2 * MB + 900.0 * MB * disk * share) * (0.4 + 0.6 * noise(self.seed, 40 + i as u64, t)),
                    b_bps: (0.1 * MB + 500.0 * MB * disk * share) * (0.4 + 0.6 * noise(self.seed, 50 + i as u64, t)),
                }
            })
            .collect();
        let nics: Vec<IoSample> = NICS
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let share = [0.75, 0.2, 0.05][i] * (0.5 + noise(self.seed, 60 + i as u64, t * 0.3));
                IoSample {
                    name: (*name).into(),
                    alt: None,
                    a_bps: 0.05 * MB + 110.0 * MB * net * share,
                    b_bps: 0.02 * MB + 25.0 * MB * net * share * (0.5 + noise(self.seed, 70 + i as u64, t)),
                }
            })
            .collect();
        let sum = |v: &[IoSample], f: fn(&IoSample) -> f64| v.iter().map(f).sum::<f64>();

        RawMetrics {
            seq: self.seq,
            elapsed: dt,
            cpu_total: Some(cpu as f32),
            cpu_self: Some(0.0),
            mem_used: Some((MEM_TOTAL as f64 * ram) as u64),
            mem_total: Some(MEM_TOTAL),
            disk_read_bps: Some(sum(&disks, |d| d.a_bps)),
            disk_write_bps: Some(sum(&disks, |d| d.b_bps)),
            net_rx_bps: Some(sum(&nics, |n| n.a_bps)),
            net_tx_bps: Some(sum(&nics, |n| n.b_bps)),
            gpu_util: Some((1.0 + 98.0 * gpu) as f32),
            gpu_mem_util: Some((10.0 + 60.0 * gpu) as f32),
            disks,
            nics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        for p in Profile::ALL {
            let (mut a, mut b) = (Sim::new(p, 2.0, 7), Sim::new(p, 2.0, 7));
            for _ in 0..400 {
                let (x, y) = (a.sample(0.25), b.sample(0.25));
                assert_eq!(x, y, "{p:?} is deterministic");
                let cpu = x.cpu_total.unwrap();
                assert!((0.0..=100.0).contains(&cpu), "{p:?} cpu {cpu}");
                assert!(x.mem_used.unwrap() <= x.mem_total.unwrap());
                assert!(x.disks.len() == 3 && x.nics.len() == 3);
                assert!(x.disk_read_bps.unwrap() >= 0.0 && x.net_rx_bps.unwrap() >= 0.0);
            }
        }
    }

    #[test]
    fn profiles_span_their_range() {
        let range = |p: Profile| {
            let s = Sim::new(p, 1.0, 3);
            let v: Vec<f64> = (0..800).map(|i| s.envelope(i as f64 * 0.1)).collect();
            (v.iter().copied().fold(1.0, f64::min), v.iter().copied().fold(0.0, f64::max))
        };
        let (lo, hi) = range(Profile::Cycle);
        assert!(lo < 0.2 && hi > 0.8, "cycle {lo}..{hi}");
        let (lo, hi) = range(Profile::Ramp);
        assert!(lo < 0.01 && hi > 0.99);
        assert!(range(Profile::Idle).1 < 0.1);
        assert!(range(Profile::Stress).0 > 0.85);
        let mut m = Sim::new(Profile::Manual, 1.0, 1);
        m.manual = 0.3;
        assert_eq!(m.envelope(12.0), 0.3);
        assert_eq!("BURSTS".parse::<Profile>().unwrap(), Profile::Bursts);
    }
}
