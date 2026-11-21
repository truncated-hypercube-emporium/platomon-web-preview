//! The background (ground plane at y = 0): luminosity, checkerboard-family
//! patterns, translation (scroll + circular sway) and rotation about the
//! camera's focal point.

use glam::{Vec2, Vec3A};
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;
use std::str::FromStr;

/// `--bg-lum` value that reproduces the original ground (albedo 0.15).
pub const LUM_REF: f32 = 5.0;
pub const ALBEDO_AT_REF: f32 = 0.15;
pub const LUM_MAX: f32 = 10.0;
/// Approximate angle one terminal cell subtends (≈ 50° over 120 columns), for
/// fading patterns to their mean once tiles shrink below a cell.
const PIXEL_ANGLE: f32 = 0.0075;

/// Mean albedo for a luminosity value (linear: 5 → 0.15, 0 → off).
pub fn albedo_for_lum(lum: f32) -> f32 {
    ALBEDO_AT_REF * lum.clamp(0.0, LUM_MAX) / LUM_REF
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Pattern {
    /// Uniform, the original look.
    #[default]
    Solid,
    /// Square checkerboard.
    Checker,
    /// Checkerboard rotated 45°.
    Diamond,
    /// Checkerboard with every other row shifted half a tile.
    Brick,
    /// Each square split along its diagonal, alternating.
    Triangles,
    /// Hexagonal tiles in three tones.
    Hex,
    /// Polar checkerboard: rings × 12 sectors around the focal point.
    Radial,
    /// Quasicrystal: N plane waves at evenly spaced angles, summed and folded
    /// into bands; a quasiperiodic (never repeating) tiling that flows as the
    /// waves' phase advances.
    Quasicrystal,
}

impl Pattern {
    pub const ALL: [Pattern; 8] = [
        Pattern::Solid,
        Pattern::Checker,
        Pattern::Diamond,
        Pattern::Brick,
        Pattern::Triangles,
        Pattern::Hex,
        Pattern::Radial,
        Pattern::Quasicrystal,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Pattern::Solid => "solid",
            Pattern::Checker => "checker",
            Pattern::Diamond => "diamond",
            Pattern::Brick => "brick",
            Pattern::Triangles => "triangles",
            Pattern::Hex => "hex",
            Pattern::Radial => "radial",
            Pattern::Quasicrystal => "quasicrystal",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Pattern::Solid => "uniform (original look)",
            Pattern::Checker => "square checkerboard",
            Pattern::Diamond => "checkerboard rotated 45 degrees",
            Pattern::Brick => "checkerboard, alternate rows offset half a tile",
            Pattern::Triangles => "squares split on the diagonal, alternating",
            Pattern::Hex => "hexagonal tiles, three tones",
            Pattern::Radial => "polar checkerboard: rings x 12 sectors around the focal point",
            Pattern::Quasicrystal => "quasicrystal: interfering plane waves, animated (--bg-waves, --bg-wave-speed)",
        }
    }
}

impl FromStr for Pattern {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        Pattern::ALL
            .into_iter()
            .find(|p| p.id() == s)
            .ok_or_else(|| format!("unknown background pattern '{s}' (see --list patterns)"))
    }
}

/// Rotation direction, seen from above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Spin {
    #[default]
    Off,
    Cw,
    Ccw,
}

impl FromStr for Spin {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Ok(Spin::Off),
            "cw" => Ok(Spin::Cw),
            "ccw" => Ok(Spin::Ccw),
            _ => Err(format!("unknown rotation '{s}' (off|cw|ccw)")),
        }
    }
}

/// Parse `DX:DZ` (also `DX,DZ`) into world units per second.
pub fn parse_vec2(s: &str) -> Result<[f32; 2], String> {
    let (a, b) = s.split_once([':', ',']).ok_or_else(|| format!("expected DX:DZ, got '{s}'"))?;
    match (a.trim().parse::<f32>(), b.trim().parse::<f32>()) {
        (Ok(x), Ok(z)) if x.is_finite() && z.is_finite() => Ok([x, z]),
        _ => Err(format!("expected two numbers DX:DZ, got '{s}'")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundParams {
    /// 0 = no ground at all, 5 = original, 10 = brightest.
    pub lum: f32,
    pub pattern: Pattern,
    /// Tile tone spread around the mean, 0..=1.
    pub contrast: f32,
    /// Tile size, world units (a solid at full scale is 2 across).
    pub scale: f32,
    /// Constant translation, world units/s (x, z).
    pub scroll: [f32; 2],
    /// Radius of a circular back-and-forth translation; 0 = none.
    pub sway: f32,
    pub sway_period: f32,
    pub rotate: Spin,
    /// Seconds per revolution about the focal point.
    pub rotate_period: f32,
    /// Quasicrystal: number of plane waves (the pattern's rotational
    /// symmetry; 5 or 7 look classic), [`WAVES_MIN`]..=[`WAVES_MAX`].
    pub waves: u32,
    /// Quasicrystal: phase speed, radians per second (0 = still).
    pub wave_speed: f32,
}

/// Quasicrystal wave count range and defaults.
pub const WAVES_MIN: u32 = 3;
pub const WAVES_MAX: u32 = 24;
pub const WAVES_DEFAULT: u32 = 7;
pub const WAVE_SPEED_DEFAULT: f32 = 0.6;

impl Default for GroundParams {
    fn default() -> Self {
        Self {
            lum: LUM_REF,
            pattern: Pattern::Solid,
            contrast: 0.5,
            scale: 1.0,
            scroll: [0.0, 0.0],
            sway: 0.0,
            sway_period: 10.0,
            rotate: Spin::Off,
            rotate_period: 60.0,
            waves: WAVES_DEFAULT,
            wave_speed: WAVE_SPEED_DEFAULT,
        }
    }
}

/// Rotate (x, z) about the y axis, same convention as the camera orbit:
/// positive angles turn +z toward +x (counter-clockwise seen from above).
#[inline(always)]
pub fn rotate_xz(v: Vec2, angle: f32) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(v.x * c + v.y * s, v.y * c - v.x * s)
}

#[derive(Debug, Clone)]
pub struct Ground {
    pub enabled: bool,
    base: f32,
    contrast: f32,
    inv_scale: f32,
    pattern: Pattern,
    /// Focal point on the ground (x, z): rotation centre.
    pivot: Vec2,
    scroll: Vec2,
    sway: f32,
    sway_omega: f32,
    /// Signed rad/s (+ = ccw).
    rot_rate: f32,
    time: f32,
    /// Accumulated scroll, world units.
    travelled: Vec2,
    angle: f32,
    /// Per-frame constants for [`Ground::pattern_coords`], refreshed by
    /// [`Ground::advance`] instead of recomputed for every pixel: sine and
    /// cosine of `-angle`, and the current [`Ground::offset`].
    frame_sin: f32,
    frame_cos: f32,
    frame_offset: Vec2,
    /// Quasicrystal lookup table: the waves' unit directions (first
    /// `n_waves`), their phase speed and this frame's phase.
    wave_dirs: [Vec2; WAVES_MAX as usize],
    n_waves: usize,
    wave_speed: f32,
    frame_phase: f32,
}

impl Ground {
    pub fn new(p: &GroundParams, pivot: Vec2) -> Self {
        let rot = match p.rotate {
            Spin::Off => 0.0,
            Spin::Ccw => TAU / p.rotate_period.max(0.1),
            Spin::Cw => -TAU / p.rotate_period.max(0.1),
        };
        let mut g = Self {
            enabled: p.lum > 0.0,
            base: albedo_for_lum(p.lum),
            contrast: p.contrast.clamp(0.0, 1.0),
            inv_scale: 1.0 / p.scale.max(1e-3),
            pattern: p.pattern,
            pivot,
            scroll: Vec2::from(p.scroll),
            sway: p.sway.max(0.0),
            sway_omega: TAU / p.sway_period.max(0.1),
            rot_rate: rot,
            time: 0.0,
            travelled: Vec2::ZERO,
            angle: 0.0,
            frame_sin: 0.0,
            frame_cos: 1.0,
            frame_offset: Vec2::ZERO,
            wave_dirs: [Vec2::ZERO; WAVES_MAX as usize],
            n_waves: 0,
            wave_speed: if p.wave_speed.is_finite() { p.wave_speed } else { WAVE_SPEED_DEFAULT },
            frame_phase: 0.0,
        };
        g.n_waves = wave_dirs(p.waves, &mut g.wave_dirs);
        g.refresh_frame();
        g
    }

    fn refresh_frame(&mut self) {
        (self.frame_sin, self.frame_cos) = (-self.angle).sin_cos();
        self.frame_offset = self.offset();
        self.frame_phase = (self.wave_speed * self.time).rem_euclid(TAU);
    }

    pub fn mean_albedo(&self) -> f32 {
        self.base
    }

    pub fn angle(&self) -> f32 {
        self.angle
    }

    /// Advance motion by wall-clock `dt`.
    pub fn advance(&mut self, dt: f32) {
        self.time += dt;
        self.travelled += self.scroll * dt;
        self.angle = (self.angle + self.rot_rate * dt).rem_euclid(TAU);
        self.refresh_frame();
    }

    /// Current translation of the pattern (scroll + sway), world units.
    pub fn offset(&self) -> Vec2 {
        let ph = self.sway_omega * self.time;
        // Circle starting at the origin so enabling sway causes no jump.
        self.travelled + Vec2::new(ph.sin(), ph.cos() - 1.0) * self.sway
    }

    /// Pattern-space coordinates (tile units) of a world point on the ground.
    /// The pattern is rotated by `angle` about the pivot, then translated.
    #[inline]
    pub fn pattern_coords(&self, x: f32, z: f32) -> Vec2 {
        let v = Vec2::new(x, z) - self.pivot;
        let (s, c) = (self.frame_sin, self.frame_cos);
        let d = Vec2::new(v.x * c + v.y * s, v.y * c - v.x * s);
        (d - self.frame_offset) * self.inv_scale
    }

    /// Albedo at ground point `p`, hit at distance `t` (for anti-alias fading).
    #[inline]
    pub fn albedo(&self, p: Vec3A, t: f32) -> f32 {
        if self.pattern == Pattern::Solid || self.contrast <= 0.0 {
            return self.base;
        }
        let fade = (1.0 - 1.5 * t * PIXEL_ANGLE * self.inv_scale).clamp(0.0, 1.0);
        if fade <= 0.0 {
            return self.base;
        }
        let uv = self.pattern_coords(p.x, p.z);
        let k = if self.pattern == Pattern::Quasicrystal {
            quasicrystal(uv, &self.wave_dirs[..self.n_waves], self.frame_phase)
        } else {
            tone(self.pattern, uv)
        };
        (self.base * (1.0 + self.contrast * fade * k)).clamp(0.0, 0.95)
    }
}

/// Fill `out` with the unit directions of `n` plane waves at angles
/// `k·π/n` (clamped to the wave range); returns how many.
fn wave_dirs(n: u32, out: &mut [Vec2; WAVES_MAX as usize]) -> usize {
    let n = n.clamp(WAVES_MIN, WAVES_MAX) as usize;
    for (k, d) in out.iter_mut().take(n).enumerate() {
        let a = std::f32::consts::PI * k as f32 / n as f32;
        *d = Vec2::new(a.cos(), a.sin());
    }
    n
}

/// Entries in [`cos_turns`]'s table (one full turn).
const COS_LUT: usize = 1024;

/// `cos(2π·x)` from a lookup table with linear interpolation (error below
/// 1e-5), several times cheaper than `f32::cos` in the quasicrystal's
/// per-pixel wave sum.
#[inline(always)]
fn cos_turns(x: f32) -> f32 {
    static T: std::sync::OnceLock<Box<[f32]>> = std::sync::OnceLock::new();
    let t = T.get_or_init(|| (0..=COS_LUT).map(|k| (TAU * k as f32 / COS_LUT as f32).cos()).collect());
    let f = (x - x.floor()) * COS_LUT as f32;
    let i = (f as usize).min(COS_LUT - 1);
    let w = f - i as f32;
    t[i] + (t[i + 1] - t[i]) * w
}

/// Quasicrystal tone in −1..=1 at `uv` (tile units): the plane waves `dirs`
/// (one wavelength per tile) at `phase`, each mapped to 0..=1 and summed,
/// then folded back and forth into 0..=1 bands so the interference shows as
/// sharp, many-fold symmetric rosettes.
#[inline]
pub fn quasicrystal(uv: Vec2, dirs: &[Vec2], phase: f32) -> f32 {
    let s: f32 = dirs.iter().map(|d| 0.5 + 0.5 * cos_turns(uv.dot(*d) + phase / TAU)).sum();
    let k = s.floor();
    let f = s - k;
    let v = if (k as i64) & 1 == 1 { 1.0 - f } else { f };
    2.0 * v - 1.0
}

#[inline(always)]
fn parity(n: f32) -> f32 {
    if (n as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 }
}

/// Tile tone in −1..=1 for pattern-space `uv` (tile units).
pub fn tone(pattern: Pattern, uv: Vec2) -> f32 {
    let (u, v) = (uv.x, uv.y);
    match pattern {
        Pattern::Solid => 0.0,
        Pattern::Checker => parity(u.floor() + v.floor()),
        Pattern::Diamond => {
            let k = std::f32::consts::FRAC_1_SQRT_2;
            parity(((u + v) * k).floor() + ((v - u) * k).floor())
        }
        Pattern::Brick => {
            let row = v.floor();
            let shift = if (row as i64).rem_euclid(2) == 1 { 0.5 } else { 0.0 };
            parity((u + shift).floor() + row)
        }
        Pattern::Triangles => {
            let (fu, fv) = (u - u.floor(), v - v.floor());
            parity(u.floor() + v.floor() + if fu > fv { 1.0 } else { 0.0 })
        }
        Pattern::Hex => {
            // Pointy-top hexes of circumradius 0.5 tile; cube-round to a cell,
            // then 3-colour by (q − r) mod 3.
            let size = 0.5;
            let q = (3f32.sqrt() / 3.0 * u - v / 3.0) / size;
            let r = (2.0 / 3.0 * v) / size;
            let (x, z) = (q, r);
            let y = -x - z;
            let (mut rx, ry, mut rz) = (x.round(), y.round(), z.round());
            let (dx, dy, dz) = ((rx - x).abs(), (ry - y).abs(), (rz - z).abs());
            if dx > dy && dx > dz {
                rx = -ry - rz;
            } else if dy <= dz {
                rz = -rx - ry;
            }
            match ((rx - rz) as i64).rem_euclid(3) {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            }
        }
        Pattern::Radial => {
            let r = uv.length();
            let sector = (v.atan2(u) / TAU * 12.0).floor();
            parity(r.floor() + sector)
        }
        Pattern::Quasicrystal => {
            let mut dirs = [Vec2::ZERO; WAVES_MAX as usize];
            let n = wave_dirs(WAVES_DEFAULT, &mut dirs);
            quasicrystal(uv, &dirs[..n], 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ground(p: GroundParams) -> Ground {
        Ground::new(&p, Vec2::ZERO)
    }

    #[test]
    fn luminosity_scale() {
        assert_eq!(albedo_for_lum(5.0), 0.15);
        assert_eq!(albedo_for_lum(0.0), 0.0);
        assert!((albedo_for_lum(10.0) - 0.30).abs() < 1e-6);
        assert_eq!(albedo_for_lum(99.0), albedo_for_lum(LUM_MAX));
        assert!(!ground(GroundParams { lum: 0.0, ..Default::default() }).enabled);
        let g = ground(GroundParams::default());
        assert!(g.enabled);
        assert_eq!(g.albedo(Vec3A::new(3.3, 0.0, -7.1), 4.0), 0.15, "solid = original look");
    }

    #[test]
    fn checker_alternates_and_keeps_mean() {
        let g = ground(GroundParams { pattern: Pattern::Checker, contrast: 0.5, ..Default::default() });
        let a = g.albedo(Vec3A::new(0.5, 0.0, 0.5), 1.0);
        let b = g.albedo(Vec3A::new(1.5, 0.0, 0.5), 1.0);
        let c = g.albedo(Vec3A::new(1.5, 0.0, 1.5), 1.0);
        assert!(a > b && (a - c).abs() < 1e-6, "{a} {b} {c}");
        assert!(((a + b) / 2.0 - 0.15).abs() < 1e-3, "mean equals luminosity");
        // contrast 0 = uniform; far away fades to the mean
        let flat = ground(GroundParams { pattern: Pattern::Checker, contrast: 0.0, ..Default::default() });
        assert_eq!(flat.albedo(Vec3A::new(0.5, 0.0, 0.5), 1.0), 0.15);
        assert_eq!(g.albedo(Vec3A::new(0.5, 0.0, 0.5), 1000.0), 0.15);
    }

    #[test]
    fn every_pattern_has_both_tones() {
        for p in Pattern::ALL.into_iter().filter(|p| *p != Pattern::Solid) {
            let mut seen = std::collections::BTreeSet::new();
            for i in 0..40 {
                for j in 0..40 {
                    let uv = Vec2::new(i as f32 * 0.37 - 7.0, j as f32 * 0.41 - 8.0);
                    // tiled patterns give exact tones; the quasicrystal shades smoothly
                    seen.insert(tone(p, uv).round() as i32);
                }
            }
            assert!(seen.contains(&-1) && seen.contains(&1), "{p:?}: {seen:?}");
            if p == Pattern::Hex {
                assert_eq!(seen.len(), 3, "hex uses three tones");
            }
        }
    }

    #[test]
    fn scroll_and_sway_translate() {
        let mut g = ground(GroundParams { pattern: Pattern::Checker, scroll: [1.0, 0.0], ..Default::default() });
        let before = g.pattern_coords(0.25, 0.25);
        g.advance(2.0);
        let after = g.pattern_coords(0.25, 0.25);
        assert!((before - after - Vec2::new(2.0, 0.0)).length() < 1e-5, "pattern moved +2 in x");

        let mut s = ground(GroundParams { sway: 1.5, sway_period: 4.0, ..Default::default() });
        assert_eq!(s.offset(), Vec2::ZERO, "no jump at start");
        s.advance(2.0); // half period: opposite side of the circle
        assert!((s.offset() - Vec2::new(0.0, -3.0)).length() < 1e-4, "{:?}", s.offset());
        s.advance(2.0);
        assert!(s.offset().length() < 1e-4, "back after one period");
    }

    #[test]
    fn rotation_about_pivot_in_both_directions() {
        let pivot = Vec2::new(2.0, -1.0);
        let p = |spin| GroundParams { rotate: spin, rotate_period: 8.0, ..Default::default() };
        let mut ccw = Ground::new(&p(Spin::Ccw), pivot);
        let mut cw = Ground::new(&p(Spin::Cw), pivot);
        // a pattern feature at pivot + (0, 1) (toward +z)
        let feature = ccw.pattern_coords(pivot.x, pivot.y + 1.0);
        ccw.advance(2.0); // quarter turn
        cw.advance(2.0);
        // ccw carries it to +x of the pivot, cw to −x (same convention as the orbit camera)
        assert!((ccw.pattern_coords(pivot.x + 1.0, pivot.y) - feature).length() < 1e-4);
        assert!((cw.pattern_coords(pivot.x - 1.0, pivot.y) - feature).length() < 1e-4);
        // the pivot itself never moves
        assert!((ccw.pattern_coords(pivot.x, pivot.y)).length() < 1e-5);
        let off = Ground::new(&p(Spin::Off), pivot);
        assert_eq!(off.angle(), 0.0);
    }

    #[test]
    fn quasicrystal_is_symmetric_bounded_and_flows() {
        for n in [5u32, 7, 12] {
            let mut dirs = [Vec2::ZERO; WAVES_MAX as usize];
            let k = wave_dirs(n, &mut dirs);
            assert_eq!(k, n as usize);
            let d = &dirs[..k];
            let mut seen = (f32::MAX, f32::MIN);
            for i in 0..400 {
                let uv = Vec2::new((i % 20) as f32 * 0.137 - 1.3, (i / 20) as f32 * 0.171 - 1.7);
                let v = quasicrystal(uv, d, 0.0);
                assert!((-1.0..=1.0).contains(&v), "{n}: {v}");
                seen = (seen.0.min(v), seen.1.max(v));
                // n-fold symmetric: turning by π/n maps the waves onto themselves
                let turned = rotate_xz(uv, std::f32::consts::PI / n as f32);
                assert!((quasicrystal(turned, d, 0.0) - v).abs() < 2e-3, "{n}-fold symmetry at {uv:?}");
            }
            assert!(seen.0 < -0.9 && seen.1 > 0.9, "{n}: full tone range {seen:?}");
        }
        // wave counts are clamped
        let mut dirs = [Vec2::ZERO; WAVES_MAX as usize];
        assert_eq!(wave_dirs(1, &mut dirs), WAVES_MIN as usize);
        assert_eq!(wave_dirs(99, &mut dirs), WAVES_MAX as usize);
        // it flows over time; the tiled patterns stay put
        let p = |pattern| GroundParams { pattern, contrast: 1.0, scale: 1.0, ..Default::default() };
        let pt = Vec3A::new(0.37, 0.0, -0.61);
        let mut q = ground(p(Pattern::Quasicrystal));
        let mut c = ground(p(Pattern::Checker));
        let (q0, c0) = (q.albedo(pt, 1.0), c.albedo(pt, 1.0));
        q.advance(1.3);
        c.advance(1.3);
        assert_ne!(q.albedo(pt, 1.0), q0, "quasicrystal moves");
        assert_eq!(c.albedo(pt, 1.0), c0, "checker doesn't");
        let still = GroundParams { wave_speed: 0.0, ..p(Pattern::Quasicrystal) };
        let mut s = ground(still);
        let s0 = s.albedo(pt, 1.0);
        s.advance(5.0);
        assert_eq!(s.albedo(pt, 1.0), s0, "speed 0 holds still");
        assert_eq!("Quasicrystal".parse::<Pattern>().unwrap(), Pattern::Quasicrystal);
    }

    #[test]
    fn parsing() {
        assert_eq!("Checker".parse::<Pattern>().unwrap(), Pattern::Checker);
        assert!("plaid".parse::<Pattern>().is_err());
        assert_eq!("ccw".parse::<Spin>().unwrap(), Spin::Ccw);
        assert_eq!(parse_vec2("0.5:-1").unwrap(), [0.5, -1.0]);
        assert_eq!(parse_vec2("2,3").unwrap(), [2.0, 3.0]);
        assert!(parse_vec2("1").is_err() && parse_vec2("a:b").is_err());
    }
}
