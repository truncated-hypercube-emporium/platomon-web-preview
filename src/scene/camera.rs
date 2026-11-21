//! Camera, and the rig that places it: static, orbiting the layout's
//! centroid, gliding toward the busiest solid, orbiting while tilting
//! between 20° and 80°, or orbiting on a three-wave resonance.

use glam::Vec3A;
use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CameraMode {
    #[default]
    Static,
    Orbit,
    /// Glide toward the busiest solid.
    Focus,
    /// Orbit while lifting from 20° to 80° elevation over the orbit point in
    /// one orbital period, holding at each end for half a period, then back.
    Tilt,
    /// Orbit on a third-order (3:1) disc resonance: the elevation rises and
    /// falls sinusoidally over 20° three times per turn, with the distance
    /// breathing in step, tracing a closed three-lobed rosette.
    Resonance,
}

impl CameraMode {
    pub const ALL: [CameraMode; 5] =
        [CameraMode::Static, CameraMode::Orbit, CameraMode::Focus, CameraMode::Tilt, CameraMode::Resonance];

    pub fn id(self) -> &'static str {
        match self {
            CameraMode::Static => "static",
            CameraMode::Orbit => "orbit",
            CameraMode::Focus => "focus",
            CameraMode::Tilt => "tilt",
            CameraMode::Resonance => "resonance",
        }
    }
}

impl FromStr for CameraMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "static" => Ok(Self::Static),
            "orbit" => Ok(Self::Orbit),
            "focus" => Ok(Self::Focus),
            "tilt" | "lift" | "arc" => Ok(Self::Tilt),
            "resonance" | "sine" | "wave" => Ok(Self::Resonance),
            _ => Err(format!("unknown camera '{s}' (static|orbit|focus|tilt|resonance)")),
        }
    }
}

/// Orbit direction, seen from above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OrbitDir {
    #[default]
    Random,
    Cw,
    Ccw,
}

impl FromStr for OrbitDir {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "random" => Ok(Self::Random),
            "cw" => Ok(Self::Cw),
            "ccw" => Ok(Self::Ccw),
            _ => Err(format!("unknown orbit direction '{s}' (random|cw|ccw)")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: Vec3A,
    pub forward: Vec3A,
    pub right: Vec3A,
    pub up: Vec3A,
    /// Minimum horizontal / vertical field of view, radians (fit whichever is tighter).
    pub hfov: f32,
    pub vfov: f32,
}

impl Camera {
    pub fn look_at(pos: Vec3A, target: Vec3A, hfov_deg: f32, vfov_deg: f32) -> Self {
        let forward = (target - pos).normalize();
        let right = forward.cross(Vec3A::Y).normalize();
        let up = right.cross(forward);
        Self { pos, forward, right, up, hfov: hfov_deg.to_radians(), vfov: vfov_deg.to_radians() }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraParams {
    pub mode: CameraMode,
    pub period: f32,
    pub dir: OrbitDir,
    pub radius: Option<f32>,
    pub height: Option<f32>,
    pub bob: f32,
    pub seed: Option<u64>,
    /// Focus mode: 0 = only turn toward the busiest solid, 1 = close-up.
    pub focus_zoom: f32,
    /// Focus mode: glide time constant, seconds.
    pub focus_tau: f32,
    /// Move in and out so every solid stays in frame (all modes but focus).
    pub auto_distance: bool,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self {
            mode: CameraMode::Static,
            period: 60.0,
            dir: OrbitDir::Random,
            radius: None,
            height: None,
            bob: 0.3,
            seed: None,
            focus_zoom: 0.6,
            focus_tau: 1.5,
            auto_distance: false,
        }
    }
}

/// Focus mode close-up distance from the solid's centre, world units
/// (a full-size solid then fills most of the 30° vertical view).
pub const FOCUS_CLOSE: f32 = 5.0;

/// Distance and height per unit of layout extent (tuned on the 5-solid ring).
const RADIUS_PER_EXTENT: f32 = 2.32;
const HEIGHT_PER_EXTENT: f32 = 1.4;
const TARGET_Y: f32 = 0.55;
const HFOV_DEG: f32 = 50.0;
const VFOV_DEG: f32 = 30.0;
/// Tilt mode: lowest and highest camera elevation above the look-at
/// point, radians (20° and 80°: never straight down at the floor).
pub const TILT_LOW: f32 = 20.0 * PI / 180.0;
pub const TILT_HIGH: f32 = 80.0 * PI / 180.0;
/// Resonance mode: peak-to-peak elevation swing (20°), vertical waves per
/// orbit (a third-order, 3:1 resonance) and relative radial breathing, in
/// phase with the waves so the closed orbit is a three-lobed rosette.
pub const RESONANCE_SWING: f32 = 20.0 * PI / 180.0;
pub const RESONANCE_ORDER: f32 = 3.0;
pub const RESONANCE_BREATH: f32 = 0.08;

/// Dynamic distance: the camera's distance as a multiple of its normal one.
pub const ZOOM_MIN: f32 = 0.35;
pub const ZOOM_MAX: f32 = 3.0;
/// Dynamic distance: share of the half field of view a solid may reach
/// (a little room around the edges).
pub const FIT_MARGIN: f32 = 0.9;
/// Dynamic distance: smoothing time constant when coming in, seconds
/// (backing out follows the fit at once, so no solid is ever clipped; the
/// fit itself moves smoothly with the solids and the camera).
const ZOOM_IN_TAU: f32 = 1.5;

/// Whether a sphere (`c`, `r`) lies inside `cam`'s guaranteed view (both
/// half-angles, scaled by [`FIT_MARGIN`]).
fn sphere_in_view(cam: &Camera, c: Vec3A, r: f32) -> bool {
    let v = c - cam.pos;
    let d = v.length();
    let z = v.dot(cam.forward);
    if d <= r || z <= 0.0 {
        return false;
    }
    let spread = (r / d).clamp(0.0, 1.0).asin();
    let (ah, av) = (0.5 * cam.hfov * FIT_MARGIN, 0.5 * cam.vfov * FIT_MARGIN);
    v.dot(cam.right).abs().atan2(z) + spread <= ah && v.dot(cam.up).abs().atan2(z) + spread <= av
}

/// 0 → 1 easing that starts and stops at rest.
fn ease(u: f32) -> f32 {
    0.5 - 0.5 * (PI * u.clamp(0.0, 1.0)).cos()
}

#[derive(Debug, Clone)]
pub struct CameraRig {
    mode: CameraMode,
    /// +1 = counter-clockwise from above, −1 = clockwise.
    dir: f32,
    omega: f32,
    radius: f32,
    height: f32,
    bob: f32,
    angle: f32,
    centroid: Vec3A,
    focus_zoom: f32,
    focus_tau: f32,
    /// Focus mode: smoothed eye position and look-at point.
    eye: Vec3A,
    look: Vec3A,
    /// Tilt mode: seconds into the arc cycle (arc, hold, arc back, hold).
    arc_t: f32,
    /// Tilt mode: seconds per arc (one orbital period).
    arc_secs: f32,
    /// Dynamic distance: on, the smoothed distance factor, and whether it
    /// has been fitted yet (the first fit snaps).
    auto_distance: bool,
    zoom: f32,
    fitted: bool,
}

impl CameraRig {
    /// `centroid`: mean solid position; `extent`: layout radius around it.
    pub fn new(p: &CameraParams, centroid: Vec3A, extent: f32) -> Self {
        let dir = match p.dir {
            OrbitDir::Ccw => 1.0,
            OrbitDir::Cw => -1.0,
            OrbitDir::Random => {
                let mut rng = p.seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
                if rng.bool() { 1.0 } else { -1.0 }
            }
        };
        let mut rig = Self {
            mode: p.mode,
            dir,
            omega: TAU / p.period.max(0.1),
            radius: p.radius.unwrap_or(RADIUS_PER_EXTENT * extent),
            height: p.height.unwrap_or(HEIGHT_PER_EXTENT * extent),
            bob: p.bob,
            angle: 0.0,
            centroid,
            focus_zoom: if p.focus_zoom.is_finite() { p.focus_zoom.clamp(0.0, 1.0) } else { 0.6 },
            focus_tau: if p.focus_tau.is_finite() { p.focus_tau.max(0.05) } else { 1.5 },
            eye: Vec3A::ZERO,
            look: Vec3A::ZERO,
            arc_t: 0.0,
            arc_secs: p.period.max(0.1),
            auto_distance: p.auto_distance && p.mode != CameraMode::Focus,
            zoom: 1.0,
            fitted: false,
        };
        (rig.eye, rig.look) = rig.overview();
        // Tilt starts on its way up, at the overview's elevation (clamped to
        // the arc).
        let e0 = rig.overview_elevation().clamp(TILT_LOW, TILT_HIGH);
        let u = (1.0 - 2.0 * (e0 - TILT_LOW) / (TILT_HIGH - TILT_LOW)).clamp(-1.0, 1.0).acos() / PI;
        rig.arc_t = u * rig.arc_secs;
        rig
    }

    /// Elevation of the overview eye above the look-at point, radians.
    fn overview_elevation(&self) -> f32 {
        (self.height - TARGET_Y).atan2(self.radius).clamp(0.0, FRAC_PI_2)
    }

    /// Distance of the overview eye from the look-at point.
    fn overview_distance(&self) -> f32 {
        self.radius.hypot(self.height - TARGET_Y).max(1e-3)
    }

    /// The furthest the eye ever gets from the look-at point, in any mode,
    /// with dynamic distance at its widest. Focus mode's eye stays between
    /// the overview eye and a solid, so the overview bounds it too.
    pub fn reach(&self) -> f32 {
        let d = match self.mode {
            CameraMode::Orbit => self.radius.hypot(self.height + self.bob.abs() - TARGET_Y),
            CameraMode::Resonance => self.overview_distance() * (1.0 + RESONANCE_BREATH),
            _ => self.overview_distance(),
        };
        if self.auto_distance { d * ZOOM_MAX } else { d }
    }

    /// The point the overview, tilt and resonance cameras look at.
    pub fn target(&self) -> Vec3A {
        self.centroid + Vec3A::new(0.0, TARGET_Y, 0.0)
    }

    /// Tilt mode: one cycle of arc up, hold, arc down, hold.
    fn arc_cycle(&self) -> f32 {
        3.0 * self.arc_secs
    }

    /// The camera's elevation above the look-at point, radians. Tilt: from
    /// [`TILT_LOW`] up to [`TILT_HIGH`] in one orbital period (easing in and
    /// out), half a period held, back down, half a period held. Resonance:
    /// the overview elevation ± half of [`RESONANCE_SWING`], three waves per
    /// orbit. Other modes: the overview's.
    pub fn elevation(&self) -> f32 {
        match self.mode {
            CameraMode::Tilt => {
                let (a, h) = (self.arc_secs, 0.5 * self.arc_secs);
                let t = self.arc_t;
                let k = if t < a {
                    ease(t / a)
                } else if t < a + h {
                    1.0
                } else if t < 2.0 * a + h {
                    1.0 - ease((t - a - h) / a)
                } else {
                    0.0
                };
                TILT_LOW + (TILT_HIGH - TILT_LOW) * k
            }
            CameraMode::Resonance => {
                let mid = self.overview_elevation().clamp(TILT_LOW + 0.5 * RESONANCE_SWING, TILT_HIGH - 0.5 * RESONANCE_SWING);
                mid + 0.5 * RESONANCE_SWING * (RESONANCE_ORDER * self.angle).sin()
            }
            _ => self.overview_elevation(),
        }
    }

    /// Tilt and resonance cameras: at `elevation()` above the look-at point,
    /// in the vertical plane through the orbit position, at the overview's
    /// distance (resonance: breathing ±[`RESONANCE_BREATH`] with the waves).
    fn arc_camera(&self) -> Camera {
        let e = self.elevation();
        let (s, c) = self.angle.sin_cos();
        let h = Vec3A::new(s, 0.0, c);
        let target = self.centroid + Vec3A::new(0.0, TARGET_Y, 0.0);
        let breath = if self.mode == CameraMode::Resonance {
            1.0 + RESONANCE_BREATH * (RESONANCE_ORDER * self.angle).cos()
        } else {
            1.0
        };
        let pos = target + (h * e.cos() + Vec3A::Y * e.sin()) * (self.overview_distance() * breath);
        Camera::look_at(pos, target, HFOV_DEG, VFOV_DEG)
    }

    pub fn direction(&self) -> f32 {
        self.dir
    }

    pub fn angle(&self) -> f32 {
        self.angle
    }

    /// Eye and look-at point of the overview (static / orbit) shot.
    fn overview(&self) -> (Vec3A, Vec3A) {
        let (s, c) = self.angle.sin_cos();
        let bob = if self.mode == CameraMode::Orbit { self.bob * (2.0 * self.angle).sin() } else { 0.0 };
        let pos = self.centroid + Vec3A::new(self.radius * s, self.height + bob, self.radius * c);
        (pos, self.centroid + Vec3A::new(0.0, TARGET_Y, 0.0))
    }

    /// Where focus mode wants to be for a solid centred at `p`: on the line
    /// from the solid toward the overview eye, `focus_zoom` of the way from
    /// the overview distance to a close-up.
    pub fn focus_shot(&self, p: Vec3A) -> (Vec3A, Vec3A) {
        let (base, _) = self.overview();
        let to_base = base - p;
        let far = to_base.length().max(1e-3);
        let dist = far + (FOCUS_CLOSE.min(far) - far) * self.focus_zoom;
        (p + to_base / far * dist, p)
    }

    /// Advance by wall-clock `dt`. `focus`: centre of the solid focus mode
    /// should move toward (`None` = overview).
    pub fn advance(&mut self, dt: f32, focus: Option<Vec3A>) {
        if matches!(self.mode, CameraMode::Orbit | CameraMode::Tilt | CameraMode::Resonance) {
            self.angle = (self.angle + self.dir * self.omega * dt).rem_euclid(TAU);
        }
        if self.mode == CameraMode::Tilt {
            self.arc_t = (self.arc_t + dt.max(0.0)).rem_euclid(self.arc_cycle());
        }
        if self.mode == CameraMode::Focus {
            let (eye, look) = focus.map_or_else(|| self.overview(), |p| self.focus_shot(p));
            let a = 1.0 - (-dt.max(0.0) / self.focus_tau).exp();
            self.eye += (eye - self.eye) * a;
            self.look += (look - self.look) * a;
        }
    }

    /// The current distance factor (1 = the mode's own distance).
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn auto_distance(&self) -> bool {
        self.auto_distance
    }

    /// `base` moved along its line to the look-at point to `k` times its
    /// distance.
    fn scaled(&self, base: &Camera, k: f32) -> Camera {
        let target = self.centroid + Vec3A::new(0.0, TARGET_Y, 0.0);
        Camera { pos: target + (base.pos - target) * k, ..*base }
    }

    /// Dynamic distance: the smallest distance factor at which every sphere
    /// (centre, radius) fits in view, found by bisection.
    pub fn fit_factor(&self, spheres: &[(Vec3A, f32)]) -> f32 {
        let base = self.base_camera();
        let fits = |k: f32| {
            let cam = self.scaled(&base, k);
            spheres.iter().all(|&(c, r)| sphere_in_view(&cam, c, r))
        };
        if fits(ZOOM_MIN) {
            return ZOOM_MIN;
        }
        if !fits(ZOOM_MAX) {
            return ZOOM_MAX;
        }
        let (mut lo, mut hi) = (ZOOM_MIN, ZOOM_MAX);
        for _ in 0..20 {
            let m = 0.5 * (lo + hi);
            if fits(m) { hi = m } else { lo = m }
        }
        hi
    }

    /// Dynamic distance: glide toward the distance that keeps every sphere
    /// in frame (at once outward, gently inward; the first call snaps).
    pub fn fit(&mut self, spheres: &[(Vec3A, f32)], dt: f32) {
        if !self.auto_distance || spheres.is_empty() {
            return;
        }
        let want = self.fit_factor(spheres);
        if !self.fitted {
            self.zoom = want;
            self.fitted = true;
            return;
        }
        if want >= self.zoom {
            self.zoom = want;
        } else {
            self.zoom += (want - self.zoom) * (1.0 - (-dt.max(0.0) / ZOOM_IN_TAU).exp());
        }
    }

    pub fn camera(&self) -> Camera {
        let base = self.base_camera();
        if self.auto_distance { self.scaled(&base, self.zoom) } else { base }
    }

    /// The mode's camera before any dynamic distance.
    fn base_camera(&self) -> Camera {
        if matches!(self.mode, CameraMode::Tilt | CameraMode::Resonance) {
            return self.arc_camera();
        }
        let (pos, target) = if self.mode == CameraMode::Focus { (self.eye, self.look) } else { self.overview() };
        Camera::look_at(pos, target, HFOV_DEG, VFOV_DEG)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(mode: CameraMode, dir: OrbitDir) -> CameraParams {
        CameraParams { mode, dir, period: 10.0, ..CameraParams::default() }
    }

    #[test]
    fn static_never_moves() {
        let mut r = CameraRig::new(&params(CameraMode::Static, OrbitDir::Ccw), Vec3A::ZERO, 3.7);
        let before = r.camera().pos;
        r.advance(5.0, Some(Vec3A::X));
        assert_eq!(r.camera().pos, before);
        assert!(before.z > 0.0 && before.x.abs() < 1e-6, "in front, on +z");
    }

    #[test]
    fn orbit_circles_centroid_at_period() {
        let c = Vec3A::new(1.0, 0.0, -2.0);
        let mut r = CameraRig::new(&CameraParams { bob: 0.0, ..params(CameraMode::Orbit, OrbitDir::Ccw) }, c, 3.7);
        let d0 = (r.camera().pos - c) * Vec3A::new(1.0, 0.0, 1.0);
        r.advance(2.5, None); // quarter turn
        let p = r.camera().pos - c;
        assert!(((p * Vec3A::new(1.0, 0.0, 1.0)).length() - d0.length()).abs() < 1e-4, "constant radius");
        assert!(p.x > 0.0 && p.z.abs() < 1e-3, "ccw from +z goes to +x: {p:?}");
        r.advance(7.5, None);
        assert!(r.angle() < 1e-4 || (TAU - r.angle()) < 1e-4, "full turn after one period");
        // looks at the centroid
        let cam = r.camera();
        let to = (c + Vec3A::new(0.0, TARGET_Y, 0.0) - cam.pos).normalize();
        assert!(cam.forward.dot(to) > 0.9999);
    }

    #[test]
    fn directions() {
        let cw = CameraRig::new(&params(CameraMode::Orbit, OrbitDir::Cw), Vec3A::ZERO, 1.0);
        assert_eq!(cw.direction(), -1.0);
        let a = CameraRig::new(&CameraParams { seed: Some(42), ..params(CameraMode::Orbit, OrbitDir::Random) }, Vec3A::ZERO, 1.0);
        let b = CameraRig::new(&CameraParams { seed: Some(42), ..params(CameraMode::Orbit, OrbitDir::Random) }, Vec3A::ZERO, 1.0);
        assert_eq!(a.direction(), b.direction(), "seeded random is repeatable");
        let seen: std::collections::HashSet<i32> = (0..64)
            .map(|s| CameraRig::new(&CameraParams { seed: Some(s), ..params(CameraMode::Orbit, OrbitDir::Random) }, Vec3A::ZERO, 1.0).direction() as i32)
            .collect();
        assert_eq!(seen.len(), 2, "random yields both directions");
    }

    #[test]
    fn focus_glides_toward_the_solid_and_back() {
        let p = CameraParams { focus_zoom: 1.0, focus_tau: 0.5, ..params(CameraMode::Focus, OrbitDir::Cw) };
        let mut r = CameraRig::new(&p, Vec3A::ZERO, 3.7);
        let start = r.camera().pos;
        let solid = Vec3A::new(2.7, 1.0, 0.0);
        let (want, _) = r.focus_shot(solid);
        assert!(((want - solid).length() - FOCUS_CLOSE).abs() < 1e-3, "zoom 1 = close-up distance");
        r.advance(0.5, Some(solid));
        let mid = r.camera().pos;
        assert!((mid - want).length() < (start - want).length(), "moving toward it");
        assert!((mid - want).length() > 0.1, "but smoothly, not a jump");
        for _ in 0..100 {
            r.advance(0.1, Some(solid));
        }
        let cam = r.camera();
        assert!((cam.pos - want).length() < 1e-2);
        assert!(cam.forward.dot((solid - cam.pos).normalize()) > 0.9999, "looks at the solid");
        for _ in 0..100 {
            r.advance(0.1, None);
        }
        assert!((r.camera().pos - start).length() < 1e-2, "back to the overview");
        // zoom 0: same distance as the overview, only turns
        let r0 = CameraRig::new(&CameraParams { focus_zoom: 0.0, ..p }, Vec3A::ZERO, 3.7);
        let (eye0, _) = r0.focus_shot(solid);
        assert!(((eye0 - solid).length() - (start - solid).length()).abs() < 1e-3);
    }

    #[test]
    fn tilt_orbits_between_20_and_80_degrees_and_holds_at_each_end() {
        let c = Vec3A::new(1.0, 0.5, -2.0);
        let period = 10.0;
        let p = CameraParams { period, ..params(CameraMode::Tilt, OrbitDir::Ccw) };
        let mut r = CameraRig::new(&p, c, 3.7);
        let target = c + Vec3A::new(0.0, TARGET_Y, 0.0);
        let ov = CameraRig::new(&CameraParams { bob: 0.0, ..params(CameraMode::Static, OrbitDir::Ccw) }, c, 3.7).camera();
        let dist = (ov.pos - target).length();
        assert!((r.camera().pos - ov.pos).length() < 1e-3, "starts at the overview shot");
        let dt = 0.01f32;
        let steps = (6.0 * period / dt) as usize; // two cycles
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let (mut held_hi, mut held_lo) = (0.0f32, 0.0f32);
        let mut prev = r.camera();
        for _ in 0..steps {
            r.advance(dt, None);
            let cam = r.camera();
            let e = r.elevation();
            let rel = cam.pos - target;
            assert!((rel.length() - dist).abs() < 1e-3, "on the arc");
            assert!((rel.normalize().y - e.sin()).abs() < 1e-4, "at its elevation");
            assert!((TILT_LOW - 1e-5..=TILT_HIGH + 1e-5).contains(&e), "between 20 and 80 degrees: {}", e.to_degrees());
            assert!(cam.forward.dot(-rel.normalize()) > 0.9999 && cam.up.y > 0.0, "looks at the orbit point, upright");
            assert!(cam.right.dot(prev.right) > 0.99, "smooth");
            (lo, hi) = (lo.min(e), hi.max(e));
            if e == TILT_HIGH {
                held_hi += dt;
            }
            if e == TILT_LOW {
                held_lo += dt;
            }
            prev = cam;
        }
        assert!((lo - TILT_LOW).abs() < 1e-5 && (hi - TILT_HIGH).abs() < 1e-5, "{} .. {}", lo.to_degrees(), hi.to_degrees());
        // two cycles: two holds at each end, half a period each
        assert!((held_hi - period).abs() < 0.05 && (held_lo - period).abs() < 0.05, "holds {held_hi} / {held_lo}");
        let mut r2 = CameraRig::new(&p, c, 3.7);
        let a0 = r2.angle();
        r2.advance(2.5, None);
        assert!(((r2.angle() - a0) - TAU / 4.0).abs() < 1e-4, "orbit keeps turning");
    }

    #[test]
    fn resonance_rises_and_falls_three_times_per_orbit_over_20_degrees() {
        let c = Vec3A::new(0.0, 0.0, 0.0);
        let period = 12.0;
        let mut r = CameraRig::new(&CameraParams { period, ..params(CameraMode::Resonance, OrbitDir::Cw) }, c, 3.7);
        let target = c + Vec3A::new(0.0, TARGET_Y, 0.0);
        let dt = 0.005f32;
        let mut es = Vec::new();
        let start = r.camera().pos;
        let mut prev = r.camera();
        for _ in 0..(period / dt).round() as usize {
            r.advance(dt, None);
            let cam = r.camera();
            assert!(cam.right.dot(prev.right) > 0.999 && cam.up.y > 0.0, "smooth and upright");
            assert!(cam.forward.dot((target - cam.pos).normalize()) > 0.9999);
            es.push(r.elevation());
            prev = cam;
        }
        let (lo, hi) = es.iter().fold((f32::MAX, f32::MIN), |(a, b), &e| (a.min(e), b.max(e)));
        assert!(((hi - lo) - RESONANCE_SWING).abs() < 1e-3, "20 degree swing: {}", (hi - lo).to_degrees());
        assert!(lo >= TILT_LOW - 1e-5 && hi <= TILT_HIGH + 1e-5);
        let peaks = es.windows(3).filter(|w| w[1] > w[0] && w[1] >= w[2]).count();
        assert_eq!(peaks, 3, "three waves per orbit");
        assert!((r.camera().pos - start).length() < 1e-2, "closed orbit after one turn");
    }
    #[test]
    fn dynamic_distance_keeps_every_solid_in_view() {
        let c = Vec3A::new(0.0, 0.0, 0.0);
        let spheres = |spread: f32| -> Vec<(Vec3A, f32)> {
            (0..6).map(|i| {
                let a = i as f32 * TAU / 6.0;
                (Vec3A::new(spread * a.cos(), 1.0, spread * a.sin()), 1.0)
            })
            .collect()
        };
        for mode in [CameraMode::Static, CameraMode::Orbit, CameraMode::Tilt, CameraMode::Resonance] {
            let p = CameraParams { auto_distance: true, bob: 0.0, ..params(mode, OrbitDir::Ccw) };
            let mut r = CameraRig::new(&p, c, 3.7);
            // wide layout: backs out until everything fits
            let wide = spheres(9.0);
            r.fit(&wide, 0.016);
            assert!(r.zoom() > 1.0, "{mode:?}: wide layout moves out ({})", r.zoom());
            let cam = r.camera();
            assert!(wide.iter().all(|&(s, rad)| sphere_in_view(&cam, s, rad)), "{mode:?}: all in view");
            // tight layout: glides back in, gently, and still fits all the way
            let tight = spheres(1.5);
            let mut prev = r.zoom();
            for _ in 0..400 {
                r.advance(0.016, None);
                r.fit(&tight, 0.016);
                assert!(r.zoom() <= prev + 1e-4 || r.zoom() == r.fit_factor(&tight), "{mode:?}: moves in, or out only as far as the fit");
                prev = r.zoom();
                let cam = r.camera();
                assert!(tight.iter().all(|&(s, rad)| sphere_in_view(&cam, s, rad)), "{mode:?}: in view while coming in");
            }
            if matches!(mode, CameraMode::Static | CameraMode::Orbit) {
                assert!(r.zoom() < 1.0, "{mode:?}: tight layout comes in ({})", r.zoom());
            }
            assert!(r.zoom() >= r.fit_factor(&tight) - 1e-4, "{mode:?}: never closer than the fit");
            if mode == CameraMode::Static {
                assert!((r.zoom() - r.fit_factor(&tight)).abs() < 0.05, "settles on the fit");
            }
            // looks at the same point throughout
            let target = c + Vec3A::new(0.0, TARGET_Y, 0.0);
            assert!(r.camera().forward.dot((target - r.camera().pos).normalize()) > 0.9999);
        }
        // off, or the focus camera: distance untouched
        for p in [CameraParams { auto_distance: false, ..params(CameraMode::Orbit, OrbitDir::Ccw) }, CameraParams { auto_distance: true, ..params(CameraMode::Focus, OrbitDir::Ccw) }] {
            let mut r = CameraRig::new(&p, c, 3.7);
            let before = r.camera().pos;
            r.fit(&spheres(9.0), 0.016);
            assert_eq!((r.zoom(), r.camera().pos), (1.0, before));
        }
    }

    #[test]
    fn parse_enums() {
        for m in CameraMode::ALL {
            assert_eq!(m.id().parse::<CameraMode>().unwrap(), m);
        }
        assert_eq!("focus".parse::<CameraMode>().unwrap(), CameraMode::Focus);
        assert_eq!("ORBIT".parse::<CameraMode>().unwrap(), CameraMode::Orbit);
        assert_eq!("cw".parse::<OrbitDir>().unwrap(), OrbitDir::Cw);
        assert!("sideways".parse::<OrbitDir>().is_err());
    }
}
