//! The orbit layout: one reference solid at the centre and the others on
//! Kepler orbits of different sizes around it.
//!
//! Each orbit is an ellipse with its own eccentricity, inclination, node and
//! periapsis direction; solids move along it by Kepler's equation, so they
//! speed up near periapsis and slow down far out, smoothly. When the
//! reference changes (it follows the largest or fastest-spinning solid), the
//! new one glides in to the centre and the old one glides back out onto its
//! orbit on a critically damped spring, so nothing ever jumps.

use glam::{Mat3A, Vec3A};
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

use crate::metrics::Channel;

/// Which solid the others orbit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OrbitCenter {
    /// The CPU's solid.
    #[default]
    Cpu,
    Ram,
    /// The network's solid (or a network adapter's).
    Net,
    /// The disk's solid (or a disk device's).
    Disk,
    /// Whichever solid is largest (busiest) at the moment.
    Size,
    /// Whichever solid spins fastest at the moment.
    Spin,
}

impl OrbitCenter {
    pub const ALL: [OrbitCenter; 6] =
        [OrbitCenter::Cpu, OrbitCenter::Ram, OrbitCenter::Net, OrbitCenter::Disk, OrbitCenter::Size, OrbitCenter::Spin];

    pub fn id(self) -> &'static str {
        match self {
            OrbitCenter::Cpu => "cpu",
            OrbitCenter::Ram => "ram",
            OrbitCenter::Net => "net",
            OrbitCenter::Disk => "disk",
            OrbitCenter::Size => "size",
            OrbitCenter::Spin => "spin",
        }
    }

    /// Shown name.
    pub fn label(self) -> &'static str {
        match self {
            OrbitCenter::Cpu => "CPU",
            OrbitCenter::Ram => "RAM",
            OrbitCenter::Net => "Network",
            OrbitCenter::Disk => "Disk",
            OrbitCenter::Size => "Size (largest solid)",
            OrbitCenter::Spin => "Spin (fastest solid)",
        }
    }

    /// Follows the solids' activity instead of one metric.
    pub fn dynamic(self) -> bool {
        matches!(self, OrbitCenter::Size | OrbitCenter::Spin)
    }
}

impl std::str::FromStr for OrbitCenter {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        let s = match s.as_str() {
            "network" | "nic" => "net",
            "disk-dev" => "disk",
            "memory" => "ram",
            other => other,
        };
        Self::ALL.into_iter().find(|c| c.id() == s).ok_or_else(|| format!("unknown orbit centre '{s}' (cpu|ram|net|disk|size|spin)"))
    }
}

/// The orbit layout's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitParams {
    pub center: OrbitCenter,
    /// Angular speed of the innermost orbit, radians/s (outer orbits are
    /// slower, as the square root of their size).
    pub velocity: f32,
    /// Tilt of the orbits from the horizontal, degrees.
    pub inclination: f32,
    /// 0 = circles, toward 1 = long ellipses.
    pub eccentricity: f32,
    /// 0 = every orbit alike (one plane, one shape), 1 = each orbit its own
    /// size step, speed, tilt, shape and direction.
    pub variation: f32,
}

impl Default for OrbitParams {
    fn default() -> Self {
        Self { center: OrbitCenter::Cpu, velocity: 0.35, inclination: 15.0, eccentricity: 0.15, variation: 0.3 }
    }
}

pub const VELOCITY_RANGE: (f32, f32) = (0.05, 2.0);
pub const INCLINATION_MAX: f32 = 90.0;
pub const ECCENTRICITY_MAX: f32 = 0.8;
/// No orbit goes past this eccentricity, variation or not.
const ECCENTRICITY_CAP: f32 = 0.85;

impl OrbitParams {
    /// Every value finite and in range.
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        let fix = |v: f32, d: f32, lo: f32, hi: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        Self {
            center: self.center,
            velocity: fix(self.velocity, d.velocity, VELOCITY_RANGE.0, VELOCITY_RANGE.1),
            inclination: fix(self.inclination, d.inclination, 0.0, INCLINATION_MAX),
            eccentricity: fix(self.eccentricity, d.eccentricity, 0.0, ECCENTRICITY_MAX),
            variation: fix(self.variation, d.variation, 0.0, 1.0),
        }
    }
}

/// One Kepler orbit around the centre.
#[derive(Debug, Clone, Copy)]
struct Orbit {
    /// Semi-major axis.
    a: f32,
    e: f32,
    /// Mean motion, radians/s.
    n: f32,
    /// Mean anomaly, radians in 0..TAU.
    m: f32,
    /// Unit vectors toward periapsis and 90 degrees on along the motion.
    p: Vec3A,
    q: Vec3A,
}

impl Orbit {
    /// Offset from the centre now.
    fn offset(&self) -> Vec3A {
        // Kepler's equation M = E - e sin E, by Newton's method from a
        // starting guess that converges for every e < 1.
        let (m, e) = (self.m, self.e);
        let mut ea = if e > 0.8 { PI } else { m + e * m.sin() };
        for _ in 0..8 {
            let f = ea - e * ea.sin() - m;
            ea -= f / (1.0 - e * ea.cos());
        }
        let b = self.a * (1.0 - e * e).sqrt();
        self.p * (self.a * (ea.cos() - e)) + self.q * (b * ea.sin())
    }

    /// Furthest distance from the centre (apoapsis).
    fn apoapsis(&self) -> f32 {
        self.a * (1.0 + self.e)
    }
}

/// How much larger each orbit is than the one inside it at the standard
/// spacing: 10% of its radius.
pub const ORBIT_STEP: f32 = 0.1;

/// A solid must lead the reference by this much...
const CENTER_MARGIN: f32 = 0.08;
/// ...for this long (seconds) before it takes the centre.
const CENTER_HOLD: f32 = 1.5;
/// Natural frequency of the spring that glides solids to and from the
/// centre, radians/s (about two seconds to settle).
const GLIDE_RATE: f32 = 2.2;

/// The orbits of every solid and which one sits in the centre.
#[derive(Debug, Clone)]
pub struct Orbits {
    pub centre: Vec3A,
    /// Per slot; `None` for a fixed reference (it never leaves the centre).
    orbits: Vec<Option<Orbit>>,
    /// Per slot: (pull toward the centre, 1 = at it; its rate).
    pull: Vec<(f32, f32)>,
    center: OrbitCenter,
    reference: usize,
    challenger: Option<(usize, f32)>,
    /// Orbit number per slot (0 = innermost; the reference's is 0 too).
    pub index: Vec<usize>,
    /// Furthest any solid's centre gets from the centre.
    pub reach: f32,
    /// Greatest inclination of any orbit, radians.
    pub max_tilt: f32,
}

fn signed(rng: &mut fastrand::Rng) -> f32 {
    rng.f32() * 2.0 - 1.0
}

impl Orbits {
    /// Orbits for `order` (slots in placement order: the first is the
    /// reference unless `center_slot` names one), the innermost at least
    /// `min_r` out and every periapsis `gap` clear of the centre; each orbit
    /// is `step` (a share of its radius, 0.1 = 10%) larger than the one
    /// inside it. The centre is raised so the lowest any solid can go stays
    /// `clearance` above the ground.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        order: &[usize],
        p: &OrbitParams,
        center_slot: Option<usize>,
        gap: f32,
        min_r: f32,
        step: f32,
        clearance: f32,
        seed: u64,
    ) -> Self {
        let p = p.sanitized();
        let n = order.len();
        let reference = if p.center.dynamic() { order.first().copied().unwrap_or(0) } else { center_slot.or(order.first().copied()).unwrap_or(0) };
        // A fixed reference has no orbit; a moving one keeps one to go back to.
        let orbiting: Vec<usize> = order.iter().copied().filter(|&i| p.center.dynamic() || i != reference).collect();
        let mut rng = fastrand::Rng::with_seed(seed ^ 0x6f72_6269_7473);
        let var = p.variation;
        let mut orbits: Vec<Option<Orbit>> = vec![None; n];
        let mut index = vec![0; n];
        let (mut prev, mut a0, mut reach, mut height, mut max_tilt) = (None::<f32>, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (k, &slot) in orbiting.iter().enumerate() {
            let e = (p.eccentricity + 0.3 * var * signed(&mut rng)).clamp(0.0, ECCENTRICITY_CAP);
            let tilt = (p.inclination + 30.0 * var * signed(&mut rng)).clamp(0.0, INCLINATION_MAX).to_radians();
            let node = PI * var * signed(&mut rng);
            let peri = PI * var * signed(&mut rng);
            // Each orbit is `step` of its radius beyond the last (10% at the
            // standard spacing), varied per orbit.
            let grow = 1.0 + step.max(0.0) * (1.0 + 0.75 * var * signed(&mut rng)).max(0.5);
            let speed = (1.0 + 0.6 * var * signed(&mut rng)).max(0.2);
            let start = signed(&mut rng);
            // the periapsis stays a full gap clear of the centre solid
            let a = prev.map_or(min_r, |r| r * grow).max(gap / (1.0 - e));
            if k == 0 {
                a0 = a;
            }
            prev = Some(a);
            let r = Mat3A::from_rotation_z(node) * Mat3A::from_rotation_x(tilt) * Mat3A::from_rotation_z(peri);
            let o = Orbit {
                a,
                e,
                n: p.velocity * (a0 / a).sqrt() * speed,
                // spread the starting points (golden angle) so solids don't line up
                m: (k as f32 * 2.399_963 + PI * var * start).rem_euclid(TAU),
                p: r * Vec3A::X,
                q: r * Vec3A::Y,
            };
            reach = reach.max(o.apoapsis());
            // any point of an orbit tilted by `tilt` is at most r·sin(tilt) below the centre
            height = height.max(o.apoapsis() * tilt.sin());
            max_tilt = max_tilt.max(tilt);
            orbits[slot] = Some(o);
            index[slot] = k;
        }
        let pull = (0..n).map(|i| (if i == reference { 1.0 } else { 0.0 }, 0.0)).collect();
        Self {
            centre: Vec3A::new(0.0, 0.0, height + clearance),
            orbits,
            pull,
            center: p.center,
            reference,
            challenger: None,
            index,
            reach,
            max_tilt,
        }
    }

    /// The slot in the centre (or gliding to it).
    pub fn reference(&self) -> usize {
        self.reference
    }

    /// Solid `i`'s centre now.
    pub fn position(&self, i: usize) -> Vec3A {
        let w = self.pull.get(i).map_or(0.0, |p| p.0).clamp(0.0, 1.0);
        match self.orbits.get(i).copied().flatten() {
            Some(o) => self.centre + o.offset() * (1.0 - w),
            None => self.centre,
        }
    }

    /// Move every solid along its orbit by `dt` seconds and, for a moving
    /// reference, hand the centre to the largest or fastest solid.
    pub fn advance(&mut self, channels: &[Channel], dt: f32) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        for o in self.orbits.iter_mut().flatten() {
            o.m = (o.m + o.n * dt).rem_euclid(TAU);
        }
        if self.center.dynamic() {
            self.pick_reference(channels, dt);
        }
        // Critically damped spring toward 1 (the reference) or 0, solved
        // exactly so a long frame can't overshoot or wobble.
        let decay = (-GLIDE_RATE * dt).exp();
        for (i, (w, v)) in self.pull.iter_mut().enumerate() {
            let target = if i == self.reference { 1.0 } else { 0.0 };
            let x0 = *w - target;
            let c = *v + GLIDE_RATE * x0;
            *w = target + (x0 + c * dt) * decay;
            *v = (*v - GLIDE_RATE * c * dt) * decay;
        }
    }

    fn pick_reference(&mut self, channels: &[Channel], dt: f32) {
        let score = |i: usize| -> f32 {
            match channels.get(i).filter(|c| c.available) {
                None => 0.0,
                Some(c) if self.center == OrbitCenter::Spin => 0.5 * (c.spin_y + c.spin_x),
                Some(c) => c.level,
            }
        };
        let n = self.pull.len();
        let Some(best) = (0..n).max_by(|&a, &b| score(a).total_cmp(&score(b))) else { return };
        if best == self.reference || score(best) <= score(self.reference) + CENTER_MARGIN {
            self.challenger = None;
            return;
        }
        let held = match self.challenger {
            Some((c, t)) if c == best => t + dt,
            _ => dt,
        };
        if held >= CENTER_HOLD {
            self.reference = best;
            self.challenger = None;
        } else {
            self.challenger = Some((best, held));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(levels: &[f32]) -> Vec<Channel> {
        levels.iter().map(|&l| Channel { available: true, level: l, spin_y: l, spin_x: l, activity: l }).collect()
    }

    #[test]
    fn kepler_orbits_stay_on_their_ellipse_and_move_smoothly() {
        let p = OrbitParams { eccentricity: 0.6, inclination: 40.0, variation: 0.8, ..OrbitParams::default() };
        let order: Vec<usize> = (0..6).collect();
        let mut o = Orbits::new(&order, &p, Some(0), 2.6, 2.7, ORBIT_STEP, 2.0, 5);
        let mut last: Vec<Vec3A> = (0..6).map(|i| o.position(i)).collect();
        for _ in 0..2000 {
            o.advance(&ch(&[0.5; 6]), 1.0 / 60.0);
            for (i, prev) in last.iter_mut().enumerate() {
                let now = o.position(i);
                let r = (now - o.centre).length();
                if i == 0 {
                    assert!(r < 1e-5, "the fixed reference stays put");
                } else {
                    let orb = o.orbits[i].unwrap();
                    assert!(r >= orb.a * (1.0 - orb.e) - 1e-3 && r <= orb.apoapsis() + 1e-3, "slot {i}: r {r}");
                    assert!(r >= 2.6 - 1e-3, "clear of the centre solid");
                }
                assert!(now.z >= 2.0 - 1e-3, "never below the clearance: {}", now.z);
                assert!((now - *prev).length() < 1.0, "no jumps");
                *prev = now;
            }
        }
    }

    #[test]
    fn variation_spreads_the_orbits() {
        let order: Vec<usize> = (0..5).collect();
        let tilts = |var: f32| {
            let p = OrbitParams { inclination: 30.0, variation: var, ..OrbitParams::default() };
            let o = Orbits::new(&order, &p, Some(0), 2.6, 2.7, ORBIT_STEP, 0.0, 9);
            o.orbits.iter().flatten().map(|o| o.q.z.asin().to_degrees()).collect::<Vec<f32>>()
        };
        let same = tilts(0.0);
        assert!(same.iter().all(|t| (t - 30.0).abs() < 1e-3), "{same:?}");
        let spread = tilts(1.0);
        assert!(spread.iter().any(|t| (t - 30.0).abs() > 3.0), "{spread:?}");
        // without variation each orbit is the step (10%) larger than the last
        let flat = OrbitParams { eccentricity: 0.0, variation: 0.0, ..OrbitParams::default() };
        for step in [ORBIT_STEP, 0.4] {
            let o = Orbits::new(&order, &flat, Some(0), 2.6, 2.7, step, 0.0, 9);
            let a: Vec<f32> = order[1..].iter().map(|&i| o.orbits[i].unwrap().a).collect();
            assert!((a[0] - 2.7).abs() < 1e-4, "{a:?}");
            assert!(a.windows(2).all(|w| (w[1] / w[0] - (1.0 + step)).abs() < 1e-4), "step {step}: {a:?}");
        }
    }

    #[test]
    fn a_moving_reference_glides_to_the_busiest_solid() {
        let p = OrbitParams { center: OrbitCenter::Size, ..OrbitParams::default() };
        let order: Vec<usize> = (0..4).collect();
        let mut o = Orbits::new(&order, &p, None, 2.6, 2.7, ORBIT_STEP, 1.0, 1);
        assert_eq!(o.reference(), 0);
        let busy = ch(&[0.1, 0.2, 0.9, 0.3]);
        let mut prev = o.position(2);
        for step in 0..400 {
            o.advance(&busy, 1.0 / 60.0);
            let now = o.position(2);
            assert!((now - prev).length() < 0.2, "step {step}: smooth glide");
            prev = now;
        }
        assert_eq!(o.reference(), 2);
        assert!((o.position(2) - o.centre).length() < 0.05, "arrived at the centre");
        assert!((o.position(0) - o.centre).length() > 2.0, "the old one went back out");
        assert!("Network".parse::<OrbitCenter>().unwrap() == OrbitCenter::Net && "x".parse::<OrbitCenter>().is_err());
    }
}
