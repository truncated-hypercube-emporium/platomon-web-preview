//! The background (ground plane at z = 0): luminosity, checkerboard-family
//! patterns, translation (scroll + circular sway) and rotation about the
//! camera's focal point.
//!
//! The patterns are laid out in their own floor frame, (x, −y): across, and
//! toward the camera's starting view, which is the frame they were designed
//! in. [`floor_frame`] maps world points into it.

use glam::{Vec2, Vec3A};
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;
use std::str::FromStr;

/// `--bg-lum` value that reproduces the original ground (albedo 0.15).
pub const LUM_REF: f32 = 5.0;
pub const ALBEDO_AT_REF: f32 = 0.15;
pub const LUM_MAX: f32 = 10.0;


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
    /// Raindrops: drops landing at random, each sending out a ring of
    /// ripples that spreads, fades and overlaps the others.
    Rain,
    /// Ocean: a sum of wind-driven waves, long ones faster and taller
    /// (deep-water dispersion), with sharpened crests.
    Ocean,
    /// Julia set (fractal) whose constant drifts round a circle.
    Julia,
    /// Sierpinski carpet (fractal): squares cut out at five scales.
    Carpet,
    /// Clouds (fractal noise): five octaves of value noise, drifting.
    Clouds,
    /// Polka dots on a triangular lattice, their sizes and places varied at will;
    /// optionally inverted on one side of a line through the centre.
    Polka,
}

impl Pattern {
    pub const ALL: [Pattern; 14] = [
        Pattern::Solid,
        Pattern::Checker,
        Pattern::Diamond,
        Pattern::Brick,
        Pattern::Triangles,
        Pattern::Hex,
        Pattern::Radial,
        Pattern::Quasicrystal,
        Pattern::Rain,
        Pattern::Ocean,
        Pattern::Julia,
        Pattern::Carpet,
        Pattern::Clouds,
        Pattern::Polka,
    ];

    /// Moves by itself (its speed is the flow, `wave_speed`).
    pub fn animated(self) -> bool {
        matches!(self, Pattern::Quasicrystal | Pattern::Rain | Pattern::Ocean | Pattern::Julia | Pattern::Clouds)
    }

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
            Pattern::Rain => "rain",
            Pattern::Ocean => "ocean",
            Pattern::Julia => "julia",
            Pattern::Carpet => "carpet",
            Pattern::Clouds => "clouds",
            Pattern::Polka => "polka",
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
            Pattern::Rain => "raindrops: random drops sending out spreading rings of ripples (--bg-drops, --bg-wave-speed)",
            Pattern::Ocean => "ocean: wind-driven waves, long ones faster, sharpened crests (--bg-wave-speed)",
            Pattern::Julia => "Julia set fractal, its constant drifting round a circle (--bg-wave-speed)",
            Pattern::Carpet => "Sierpinski carpet fractal: squares cut out at five scales",
            Pattern::Clouds => "fractal noise: five octaves of value noise, drifting (--bg-wave-speed)",
            Pattern::Polka => "polka dots, sizes and places varied at will, one half optionally inverted (--bg-dot-*, --bg-invert-half)",
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

/// Parse `DX:DY` (also `DX,DY`) into world units per second.
pub fn parse_vec2(s: &str) -> Result<[f32; 2], String> {
    let (a, b) = s.split_once([':', ',']).ok_or_else(|| format!("expected DX:DZ, got '{s}'"))?;
    match (a.trim().parse::<f32>(), b.trim().parse::<f32>()) {
        (Ok(x), Ok(y)) if x.is_finite() && y.is_finite() => Ok([x, y]),
        _ => Err(format!("expected two numbers DX:DY, got '{s}'")),
    }
}

/// World floor point (x, y) in the patterns' own frame, (x, −y).
#[inline(always)]
pub fn floor_frame(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, -y)
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
    /// Constant translation across the floor, world units/s (x, y).
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
    /// Flow of the animated patterns: the quasicrystal's phase speed,
    /// radians per second (0 = still); the others move in proportion to it
    /// ([`WAVE_SPEED_DEFAULT`] = their natural pace).
    pub wave_speed: f32,
    /// Raindrops: drops rippling at once, [`DROPS_MIN`]..=[`DROPS_MAX`].
    pub drops: u32,
    /// Seeds the raindrops' places and the ocean's waves.
    pub seed: u64,
    /// The patterns' own settings.
    pub shape: PatternShape,
}

/// Settings of single patterns (each only matters for its own).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatternShape {
    /// Rain: distance between ripples, tiles.
    pub ripple: f32,
    /// Rain: seconds a drop's rings last (at the natural pace).
    pub drop_life: f32,
    /// Ocean: the wind's direction, degrees.
    pub wind: f32,
    /// Ocean: how sharp the crests are, 0 (plain sine waves) up.
    pub chop: f32,
    /// Julia set: iterations (more = finer detail at the set's edge).
    pub julia_iter: u32,
    /// Julia set: radius of the circle its constant drifts round.
    pub julia_shape: f32,
    /// Sierpinski carpet: scales holes are cut at.
    pub carpet_levels: u32,
    /// Clouds: octaves of noise, and how strong each finer one is.
    pub octaves: u32,
    pub roughness: f32,
    /// Radial: sectors round the focal point.
    pub sectors: u32,
    /// Polka dots: each dot's size across and the distance between
    /// neighbouring dots, tiles; how much both vary from dot to dot (0..=1);
    /// and whether one half of the floor (one side of a line through the
    /// centre) is inverted, dark dots on light.
    pub dot_size: f32,
    pub dot_spacing: f32,
    pub dot_variability: f32,
    pub invert_half: bool,
}

impl Default for PatternShape {
    fn default() -> Self {
        Self {
            ripple: 0.9,
            drop_life: 4.5,
            wind: 30.0,
            chop: 0.6,
            julia_iter: 28,
            julia_shape: 0.7885,
            carpet_levels: 5,
            octaves: 5,
            roughness: 0.5,
            sectors: 12,
            dot_size: 0.5,
            dot_spacing: 1.0,
            dot_variability: 0.0,
            invert_half: false,
        }
    }
}

impl PatternShape {
    /// Every value inside its range (`d` for non-finite ones).
    pub fn sanitized(self) -> Self {
        let d = Self::default();
        let f = |v: f32, def: f32, lo: f32, hi: f32| if v.is_finite() { v.clamp(lo, hi) } else { def };
        Self {
            ripple: f(self.ripple, d.ripple, 0.2, 4.0),
            drop_life: f(self.drop_life, d.drop_life, 0.5, 30.0),
            wind: if self.wind.is_finite() { self.wind.rem_euclid(360.0) } else { d.wind },
            chop: f(self.chop, d.chop, 0.0, 2.0),
            julia_iter: self.julia_iter.clamp(4, 128),
            julia_shape: f(self.julia_shape, d.julia_shape, 0.3, 1.2),
            carpet_levels: self.carpet_levels.clamp(1, 8),
            octaves: self.octaves.clamp(1, 10),
            roughness: f(self.roughness, d.roughness, 0.1, 0.9),
            sectors: self.sectors.clamp(2, 64),
            dot_size: f(self.dot_size, d.dot_size, 0.05, 4.0),
            dot_spacing: f(self.dot_spacing, d.dot_spacing, 0.25, 8.0),
            dot_variability: f(self.dot_variability, d.dot_variability, 0.0, 1.0),
            invert_half: self.invert_half,
        }
    }
}

/// Quasicrystal wave count range and defaults.
pub const WAVES_MIN: u32 = 3;
pub const WAVES_MAX: u32 = 24;
pub const WAVES_DEFAULT: u32 = 7;
pub const WAVE_SPEED_DEFAULT: f32 = 0.6;
/// Raindrops rippling at once.
pub const DROPS_MIN: u32 = 1;
pub const DROPS_MAX: u32 = 48;
pub const DROPS_DEFAULT: u32 = 12;

/// Raindrops land in a square this many tiles across, repeated over the
/// floor; rings spread at `RAIN_SPEED` tiles a second with ripples
/// `RAIN_WAVELENGTH` apart and last `RAIN_LIFE` seconds (at the natural pace).
const RAIN_CELL: f32 = 12.0;
const RAIN_SPEED: f32 = 1.4;
/// How many wavelengths deep a ring of ripples is.
const RAIN_TRAIN: f32 = 3.0;
/// Ocean: wave components, the longest wavelength (tiles) and the ratio
/// between neighbouring ones, gravity in tile units, and the wind's spread.
const OCEAN_WAVES: usize = 12;
const OCEAN_LONGEST: f32 = 4.0;
const OCEAN_RATIO: f32 = 0.76;
const OCEAN_G: f32 = 1.6;
const OCEAN_SPREAD: f32 = 1.1;
/// Julia set: tiles per unit of the complex plane, iterations, and the
/// constant's circle (radius, turns per second at the natural pace).
const JULIA_SCALE: f32 = 4.0;
const JULIA_TURN: f32 = 0.006;
/// Clouds: the drift in tiles per second at the natural pace.
const CLOUD_DRIFT: f32 = 0.25;

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
            drops: DROPS_DEFAULT,
            seed: 0,
            shape: PatternShape::default(),
        }
    }
}

/// One raindrop: where it landed (tile units, inside the rain cell), how
/// long ago, and how long its rings last.
#[derive(Debug, Clone, Copy)]
struct Drop {
    pos: Vec2,
    age: f32,
    life: f32,
}

/// One ocean wave: direction × wavenumber (cycles per tile), height, its
/// angular frequency (turns per second) and starting phase (turns).
#[derive(Debug, Clone, Copy, Default)]
struct OceanWave {
    k: Vec2,
    amp: f32,
    omega: f32,
    phase: f32,
}

/// Rotate a point of the floor frame ([`floor_frame`]) about its origin,
/// same convention as the camera orbit: positive angles turn the front (−y
/// in the world) toward +x, counter-clockwise seen from above.
#[inline(always)]
pub fn rotate_floor(v: Vec2, angle: f32) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(v.x * c + v.y * s, v.y * c - v.x * s)
}

#[derive(Debug, Clone)]
pub struct Ground {
    pub enabled: bool,
    base: f32,
    contrast: f32,
    inv_scale: f32,
    /// How fast the pattern fades to its mean with distance: `inv_scale`
    /// over its finest feature's size in tiles (smaller dots fade sooner).
    fade_rate: f32,
    pattern: Pattern,
    /// Focal point on the ground in the floor frame: rotation centre.
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
    /// Animated patterns other than the quasicrystal: seconds of their own
    /// time (the flow relative to its natural pace).
    flow_time: f32,
    /// Raindrops, and per frame each one's (place, ring radius, strength).
    drops: Vec<Drop>,
    rain_rng: fastrand::Rng,
    frame_drops: Vec<(Vec2, f32, f32)>,
    /// Ocean waves, and per frame each one's phase (turns).
    ocean: [OceanWave; OCEAN_WAVES],
    ocean_sum: f32,
    frame_ocean: [f32; OCEAN_WAVES],
    /// This frame's Julia constant and cloud drift.
    frame_julia: Vec2,
    frame_drift: Vec2,
    shape: PatternShape,
}

impl Ground {
    /// `pivot`: the focal point on the floor, world (x, y).
    pub fn new(p: &GroundParams, pivot: Vec2) -> Self {
        let pivot = floor_frame(pivot.x, pivot.y);
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
            fade_rate: 1.0 / p.scale.max(1e-3),
            pattern: p.pattern,
            pivot,
            scroll: floor_frame(p.scroll[0], p.scroll[1]),
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
            flow_time: 0.0,
            drops: Vec::new(),
            rain_rng: fastrand::Rng::with_seed(p.seed ^ 0x7261_696e),
            frame_drops: Vec::new(),
            ocean: [OceanWave::default(); OCEAN_WAVES],
            ocean_sum: 1.0,
            frame_ocean: [0.0; OCEAN_WAVES],
            frame_julia: Vec2::ZERO,
            frame_drift: Vec2::ZERO,
            shape: p.shape.sanitized(),
        };
        if g.pattern == Pattern::Polka {
            g.fade_rate /= g.shape.dot_size.min(g.shape.dot_spacing).min(1.0);
        }
        g.n_waves = wave_dirs(p.waves, &mut g.wave_dirs);
        // raindrops at every stage of their lives, so it starts mid-shower
        let n = p.drops.clamp(DROPS_MIN, DROPS_MAX) as usize;
        for k in 0..n {
            let mut d = g.new_drop();
            d.age = d.life * (k as f32 + g.rain_rng.f32()) / n as f32;
            g.drops.push(d);
        }
        // the ocean: wavelengths from the longest down by a fixed ratio,
        // round a seeded wind, taller the longer (an ocean-like spectrum)
        let mut rng = fastrand::Rng::with_seed(p.seed ^ 0x006f_6365_616e);
        let wind = g.shape.wind.to_radians();
        let mut sum = 0.0;
        for (i, w) in g.ocean.iter_mut().enumerate() {
            let lambda = OCEAN_LONGEST * OCEAN_RATIO.powi(i as i32);
            let dir = wind + OCEAN_SPREAD * (rng.f32() - 0.5) * (1.0 + i as f32 / OCEAN_WAVES as f32);
            let k = 1.0 / lambda;
            *w = OceanWave {
                k: Vec2::new(dir.cos(), dir.sin()) * k,
                amp: lambda.powf(0.8),
                // deep water: ω = √(g·2πk), here in turns per second
                omega: (OCEAN_G * TAU * k).sqrt() / TAU,
                phase: rng.f32(),
            };
            sum += w.amp;
        }
        g.ocean_sum = sum.max(1e-3);
        g.refresh_frame();
        g
    }

    fn new_drop(&mut self) -> Drop {
        let life = self.shape.drop_life;
        let r = &mut self.rain_rng;
        Drop { pos: Vec2::new(r.f32(), r.f32()) * RAIN_CELL, age: 0.0, life: life * (0.7 + 0.6 * r.f32()) }
    }

    fn refresh_frame(&mut self) {
        (self.frame_sin, self.frame_cos) = (-self.angle).sin_cos();
        self.frame_offset = self.offset();
        self.frame_phase = (self.wave_speed * self.time).rem_euclid(TAU);
        let t = self.flow_time;
        match self.pattern {
            Pattern::Rain => {
                self.frame_drops.clear();
                self.frame_drops.extend(self.drops.iter().map(|d| {
                    // strong when fresh, gone by the end of its life
                    let left = (1.0 - d.age / d.life).clamp(0.0, 1.0);
                    (d.pos, RAIN_SPEED * d.age, left * left)
                }));
            }
            Pattern::Ocean => {
                for (p, w) in self.frame_ocean.iter_mut().zip(&self.ocean) {
                    *p = (w.phase - w.omega * t).rem_euclid(1.0);
                }
            }
            Pattern::Julia => {
                let a = TAU * JULIA_TURN * t;
                self.frame_julia = Vec2::new(a.cos(), a.sin()) * self.shape.julia_shape;
            }
            Pattern::Clouds => self.frame_drift = Vec2::new(0.8, 0.6) * CLOUD_DRIFT * t,
            _ => {}
        }
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
        // the animated patterns' own time runs at the flow's pace
        let pace = self.wave_speed / WAVE_SPEED_DEFAULT;
        self.flow_time += dt * pace;
        if self.pattern == Pattern::Rain {
            let step = dt * pace.abs();
            for k in 0..self.drops.len() {
                self.drops[k].age += step;
                if self.drops[k].age >= self.drops[k].life {
                    self.drops[k] = self.new_drop();
                }
            }
        }
        self.refresh_frame();
    }

    /// Current translation of the pattern (scroll + sway), world units.
    pub fn offset(&self) -> Vec2 {
        let ph = self.sway_omega * self.time;
        // Circle starting at the origin so enabling sway causes no jump.
        self.travelled + Vec2::new(ph.sin(), ph.cos() - 1.0) * self.sway
    }

    /// Pattern-space coordinates (tile units) of world floor point (x, y).
    /// The pattern is rotated by `angle` about the pivot, then translated.
    #[inline]
    pub fn pattern_coords(&self, x: f32, y: f32) -> Vec2 {
        let v = floor_frame(x, y) - self.pivot;
        let (s, c) = (self.frame_sin, self.frame_cos);
        let d = Vec2::new(v.x * c + v.y * s, v.y * c - v.x * s);
        (d - self.frame_offset) * self.inv_scale
    }

    /// Albedo at ground point `p`, where a pixel covers a patch `footprint`
    /// across (world units): the pattern fades to its mean tone as its tiles
    /// shrink toward a pixel, instead of aliasing.
    #[inline]
    pub fn albedo(&self, p: Vec3A, footprint: f32) -> f32 {
        if self.pattern == Pattern::Solid || self.contrast <= 0.0 {
            return self.base;
        }
        let fade = (1.0 - 1.5 * footprint * self.fade_rate).clamp(0.0, 1.0);
        if fade <= 0.0 {
            return self.base;
        }
        let mut k = self.tone_at(self.pattern_coords(p.x, p.y));
        if self.pattern == Pattern::Polka && self.shape.invert_half && self.on_inverted_half(p.x, p.y) {
            k = -k;
        }
        (self.base * (1.0 + self.contrast * fade * k)).clamp(0.0, 0.95)
    }

    /// Whether world floor point (x, y) is on the inverted half: past the line
    /// through the focal point that turns with the pattern (it doesn't
    /// scroll or sway with it, so it stays through the centre).
    #[inline]
    pub fn on_inverted_half(&self, x: f32, y: f32) -> bool {
        let v = floor_frame(x, y) - self.pivot;
        v.y * self.frame_cos - v.x * self.frame_sin < 0.0
    }

    /// The pattern's tone in −1..=1 at `uv` (tile units) this frame.
    #[inline]
    pub fn tone_at(&self, uv: Vec2) -> f32 {
        match self.pattern {
            Pattern::Quasicrystal => quasicrystal(uv, &self.wave_dirs[..self.n_waves], self.frame_phase),
            Pattern::Rain => rain(uv, &self.frame_drops, self.shape.ripple),
            Pattern::Ocean => ocean(uv, &self.ocean, &self.frame_ocean, self.ocean_sum, self.shape.chop),
            Pattern::Julia => julia(uv, self.frame_julia, self.shape.julia_iter),
            Pattern::Clouds => clouds(uv + self.frame_drift, self.shape.octaves, self.shape.roughness),
            Pattern::Carpet => carpet(uv, self.shape.carpet_levels),
            Pattern::Radial => radial(uv, self.shape.sectors),
            Pattern::Polka => polka_placed(uv, self.shape.dot_size, self.shape.dot_spacing, self.shape.dot_variability),
            p => tone(p, uv),
        }
    }
}

/// Raindrop ripples in −1..=1: each drop's ring (`drops`: place, radius,
/// strength) is a short train of ripples behind its front, weaker the
/// further it has spread; the drops repeat every [`RAIN_CELL`] tiles.
#[inline]
fn rain(uv: Vec2, drops: &[(Vec2, f32, f32)], wavelength: f32) -> f32 {
    let mut h = 0.0;
    for &(pos, radius, strength) in drops {
        let mut d = uv - pos;
        // nearest copy of the drop
        d -= (d / RAIN_CELL).round() * RAIN_CELL;
        let r = d.length();
        // how far behind the front this point is, in wavelengths
        let behind = (radius - r) / wavelength;
        if !(-0.25..RAIN_TRAIN).contains(&behind) {
            continue;
        }
        let lead = (behind * 4.0 + 1.0).clamp(0.0, 1.0);
        let tail = 1.0 - behind / RAIN_TRAIN;
        h += strength * lead * tail * cos_turns(behind) / (1.0 + 0.15 * r);
    }
    (2.5 * h).clamp(-1.0, 1.0)
}

/// The ocean's height in −1..=1: the waves summed (`phase`: each one's this
/// frame), the crests sharpened as on real water.
#[inline]
fn ocean(uv: Vec2, waves: &[OceanWave; OCEAN_WAVES], phase: &[f32; OCEAN_WAVES], sum: f32, chop: f32) -> f32 {
    let mut h = 0.0;
    for (w, p) in waves.iter().zip(phase) {
        h += w.amp * cos_turns(uv.dot(w.k) + p);
    }
    let h = 2.2 * h / sum;
    // peaked crests, flat troughs
    let s = if h > 0.0 { h + chop * h * h } else { (1.0 - 0.5 * chop).max(0.3) * h };
    s.clamp(-1.0, 1.0)
}

/// The Julia set of `c` in −1..=1: inside dark, outside in bands by how
/// fast points escape (smoothed), the set [`JULIA_SCALE`] tiles per unit.
#[inline]
fn julia(uv: Vec2, c: Vec2, iterations: u32) -> f32 {
    let (mut x, mut y) = (uv.x / JULIA_SCALE, uv.y / JULIA_SCALE);
    for n in 0..iterations {
        let (xx, yy) = (x * x, y * y);
        if xx + yy > 16.0 {
            // smooth escape count
            let nu = n as f32 + 1.0 - ((xx + yy).ln() * 0.5).ln() / std::f32::consts::LN_2;
            return cos_turns(nu / 6.0);
        }
        (x, y) = (xx - yy + c.x, 2.0 * x * y + c.y);
    }
    -1.0
}

/// Polar checkerboard: unit rings × `sectors` sectors round the origin.
#[inline]
fn radial(uv: Vec2, sectors: u32) -> f32 {
    let sector = (uv.y.atan2(uv.x) / TAU * sectors as f32).floor();
    parity(uv.length().floor() + sector)
}

/// Sierpinski carpet: −1 in a square cut out at any of `levels`
/// scales (the smallest a third of a tile), else 1.
#[inline]
fn carpet(uv: Vec2, levels: u32) -> f32 {
    let (mut u, mut v) = ((uv.x * 3.0).floor() as i64, (uv.y * 3.0).floor() as i64);
    for _ in 0..levels {
        if u.rem_euclid(3) == 1 && v.rem_euclid(3) == 1 {
            return -1.0;
        }
        (u, v) = (u.div_euclid(3), v.div_euclid(3));
    }
    1.0
}

/// [`polka`] as laid on the floor: the lattice shifted half a spacing along
/// the floor frame's second axis (world y), so the focal point sits between
/// two dots rather than on one.
#[inline]
fn polka_placed(uv: Vec2, size: f32, spacing: f32, variability: f32) -> f32 {
    polka(uv - Vec2::new(0.0, 0.5 * spacing), size, spacing, variability)
}

/// Polka dots in −1..=1: a dot (1) `size` tiles across on every point of a
/// triangular lattice `spacing` tiles apart (rows `spacing`·√3/2 apart,
/// every other row shifted half a spacing: each dot has six nearest
/// neighbours), on −1 between them. `variability` (0..=1) spreads both:
/// each dot's size varies from 0 up to twice `size`, and it moves off its
/// lattice point by up to half the spacing each way, so its neighbours
/// range from touching to twice as far.
#[inline]
fn polka(uv: Vec2, size: f32, spacing: f32, variability: f32) -> f32 {
    const ROW: f32 = 0.866_025_4; // √3/2
    let g = uv / spacing;
    let r = 0.5 * size / spacing;
    // lattice coordinates: g = i·(1, 0) + j·(½, √3/2)
    let j = g.y / ROW;
    let i = g.x - 0.5 * j;
    let (i0, j0) = (i.floor() as i32, j.floor() as i32);
    // how far a dot can reach: its largest size plus its largest move.
    // Within half a spacing of a dot, that dot is the nearest lattice
    // point, a corner of the cell (i0, j0)..(i0+1, j0+1) the point is in;
    // further, rows are √3/2 apart, so look that many rows round it
    let far = r * (1.0 + variability) + 0.71 * variability;
    let reach = if far <= 0.5 { 0 } else { ((far / ROW).ceil() as i32).min(2) };
    for dj in -reach..=reach + 1 {
        for di in -reach..=reach + 1 {
            let (x, y) = (i0.wrapping_add(di), j0.wrapping_add(dj));
            let mut c = Vec2::new(x as f32 + 0.5 * y as f32, y as f32 * ROW);
            let mut rad = r;
            if variability > 0.0 {
                rad *= 1.0 + variability * (2.0 * hash2(x, y) - 1.0);
                let jitter = Vec2::new(
                    hash2(y.wrapping_add(7919), x.wrapping_sub(3571)),
                    hash2(x.wrapping_mul(3).wrapping_add(11), y.wrapping_mul(5).wrapping_sub(7)),
                );
                c += variability * (jitter - 0.5);
            }
            if (g - c).length_squared() < rad * rad {
                return 1.0;
            }
        }
    }
    -1.0
}
/// A lattice point's random value in 0..=1.
#[inline(always)]
fn hash2(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65535.0
}

/// Smoothly interpolated value noise in 0..=1.
#[inline]
fn value_noise(p: Vec2) -> f32 {
    let (fx, fy) = (p.x.floor(), p.y.floor());
    let (ix, iy) = (fx as i32, fy as i32);
    let (tx, ty) = (p.x - fx, p.y - fy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let a = hash2(ix, iy) + (hash2(ix + 1, iy) - hash2(ix, iy)) * sx;
    let b = hash2(ix, iy + 1) + (hash2(ix + 1, iy + 1) - hash2(ix, iy + 1)) * sx;
    a + (b - a) * sy
}

/// Fractal (Brownian) noise in −1..=1: `octaves` octaves, each twice as
/// fine and `roughness` as strong as the last, the coarsest 3 tiles across.
#[inline]
fn clouds(uv: Vec2, octaves: u32, roughness: f32) -> f32 {
    let (mut p, mut amp, mut sum, mut norm) = (uv / 3.0, 1.0, 0.0, 0.0);
    for o in 0..octaves {
        sum += amp * value_noise(p + Vec2::splat(o as f32 * 17.3));
        norm += amp;
        // turn each octave a little so the lattice doesn't show
        p = Vec2::new(p.x * 1.6 - p.y * 1.2, p.x * 1.2 + p.y * 1.6);
        amp *= roughness;
    }
    ((sum / norm - 0.5) * 3.2).clamp(-1.0, 1.0)
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
        Pattern::Radial => radial(uv, PatternShape::default().sectors),
        Pattern::Quasicrystal => {
            let mut dirs = [Vec2::ZERO; WAVES_MAX as usize];
            let n = wave_dirs(WAVES_DEFAULT, &mut dirs);
            quasicrystal(uv, &dirs[..n], 0.0)
        }
        Pattern::Carpet => carpet(uv, PatternShape::default().carpet_levels),
        Pattern::Polka => {
            let d = PatternShape::default();
            polka_placed(uv, d.dot_size, d.dot_spacing, d.dot_variability)
        }
        // the moving patterns as they start, with the default settings
        Pattern::Rain | Pattern::Ocean | Pattern::Julia | Pattern::Clouds => {
            Ground::new(&GroundParams { pattern, ..GroundParams::default() }, Vec2::ZERO).tone_at(uv)
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
        assert_eq!(g.albedo(Vec3A::new(3.3, 7.1, 0.0), 0.03), 0.15, "solid = original look");
    }

    #[test]
    fn checker_alternates_and_keeps_mean() {
        let g = ground(GroundParams { pattern: Pattern::Checker, contrast: 0.5, ..Default::default() });
        let a = g.albedo(Vec3A::new(0.5, -0.5, 0.0), 0.0075);
        let b = g.albedo(Vec3A::new(1.5, -0.5, 0.0), 0.0075);
        let c = g.albedo(Vec3A::new(1.5, -1.5, 0.0), 0.0075);
        assert!(a > b && (a - c).abs() < 1e-6, "{a} {b} {c}");
        assert!(((a + b) / 2.0 - 0.15).abs() < 1e-3, "mean equals luminosity");
        // contrast 0 = uniform; far away fades to the mean
        let flat = ground(GroundParams { pattern: Pattern::Checker, contrast: 0.0, ..Default::default() });
        assert_eq!(flat.albedo(Vec3A::new(0.5, -0.5, 0.0), 0.0075), 0.15);
        assert_eq!(g.albedo(Vec3A::new(0.5, -0.5, 0.0), 1000.0), 0.15);
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
        let before = g.pattern_coords(0.25, -0.25);
        g.advance(2.0);
        let after = g.pattern_coords(0.25, -0.25);
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
        let pivot = Vec2::new(2.0, 1.0);
        let p = |spin| GroundParams { rotate: spin, rotate_period: 8.0, ..Default::default() };
        let mut ccw = Ground::new(&p(Spin::Ccw), pivot);
        let mut cw = Ground::new(&p(Spin::Cw), pivot);
        // a pattern feature at pivot − (0, 1) (toward −y, the front)
        let feature = ccw.pattern_coords(pivot.x, pivot.y - 1.0);
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
                let turned = rotate_floor(uv, std::f32::consts::PI / n as f32);
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
        let pt = Vec3A::new(0.37, 0.61, 0.0);
        let mut q = ground(p(Pattern::Quasicrystal));
        let mut c = ground(p(Pattern::Checker));
        let (q0, c0) = (q.albedo(pt, 0.0075), c.albedo(pt, 0.0075));
        q.advance(1.3);
        c.advance(1.3);
        assert_ne!(q.albedo(pt, 0.0075), q0, "quasicrystal moves");
        assert_eq!(c.albedo(pt, 0.0075), c0, "checker doesn't");
        let still = GroundParams { wave_speed: 0.0, ..p(Pattern::Quasicrystal) };
        let mut s = ground(still);
        let s0 = s.albedo(pt, 0.0075);
        s.advance(5.0);
        assert_eq!(s.albedo(pt, 0.0075), s0, "speed 0 holds still");
        assert_eq!("Quasicrystal".parse::<Pattern>().unwrap(), Pattern::Quasicrystal);
    }

    #[test]
    fn new_patterns_are_bounded_move_with_the_flow_and_hold_at_zero() {
        let pts: Vec<Vec2> = (0..900).map(|i| Vec2::new((i % 30) as f32 * 0.43 - 6.0, (i / 30) as f32 * 0.39 - 5.5)).collect();
        for p in [Pattern::Rain, Pattern::Ocean, Pattern::Julia, Pattern::Clouds, Pattern::Carpet] {
            let params = GroundParams { pattern: p, contrast: 1.0, ..Default::default() };
            let mut g = ground(params);
            let before: Vec<f32> = pts.iter().map(|&uv| g.tone_at(uv)).collect();
            assert!(before.iter().all(|v| (-1.0..=1.0).contains(v)), "{p:?} bounded");
            let spread = before.iter().copied().fold(f32::MIN, f32::max) - before.iter().copied().fold(f32::MAX, f32::min);
            assert!(spread > 1.0, "{p:?}: tones vary ({spread})");
            g.advance(2.0);
            let moved = pts.iter().zip(&before).filter(|(uv, b)| (g.tone_at(**uv) - **b).abs() > 1e-3).count();
            assert_eq!(moved > 0, p.animated(), "{p:?} moves only when animated ({moved})");
            let mut still = ground(GroundParams { wave_speed: 0.0, ..params });
            let s0: Vec<f32> = pts.iter().map(|&uv| still.tone_at(uv)).collect();
            still.advance(3.0);
            assert!(pts.iter().zip(&s0).all(|(uv, s)| still.tone_at(*uv) == *s), "{p:?}: flow 0 holds still");
        }
        // raindrops keep landing: after many lifetimes every drop has been
        // replaced, and the count follows the setting
        let mut r = ground(GroundParams { pattern: Pattern::Rain, drops: 5, ..Default::default() });
        let first: Vec<Vec2> = r.drops.iter().map(|d| d.pos).collect();
        assert_eq!(first.len(), 5);
        for _ in 0..200 {
            r.advance(0.25);
        }
        assert!(r.drops.iter().zip(&first).all(|(d, f)| d.pos != *f), "every drop replaced");
        assert!(r.drops.iter().all(|d| d.age < d.life && (0.0..=RAIN_CELL).contains(&d.pos.x)));
        // the carpet: the centre of each scale's middle square is cut out
        assert_eq!(carpet(Vec2::new(0.5, 0.5), 5), -1.0, "smallest hole: a third of a tile");
        assert_eq!(carpet(Vec2::new(0.1, 0.1), 5), 1.0);
        assert_eq!(carpet(Vec2::new(1.1, 1.1), 5), -1.0, "the 3x larger hole");
        assert_eq!(carpet(Vec2::new(1.1, 1.1), 1), 1.0, "one level: only the smallest holes");
        // the Julia set: the origin's orbit stays bounded for a small constant
        assert_eq!(julia(Vec2::ZERO, Vec2::new(-0.1, 0.1), 28), -1.0);
        assert!(julia(Vec2::splat(40.0), Vec2::ZERO, 28) > -1.0, "far points escape");
    }

    #[test]
    fn pattern_settings_change_their_pattern() {
        let pts: Vec<Vec2> = (0..400).map(|i| Vec2::new((i % 20) as f32 * 0.31 - 3.0, (i / 20) as f32 * 0.29 - 2.8)).collect();
        let tones = |pattern, shape| {
            let g = ground(GroundParams { pattern, shape, ..Default::default() });
            pts.iter().map(|&uv| g.tone_at(uv)).collect::<Vec<f32>>()
        };
        let d = PatternShape::default();
        for (pattern, changed) in [
            (Pattern::Rain, PatternShape { ripple: 1.6, ..d }),
            (Pattern::Ocean, PatternShape { wind: 120.0, ..d }),
            (Pattern::Ocean, PatternShape { chop: 0.0, ..d }),
            (Pattern::Julia, PatternShape { julia_shape: 0.65, ..d }),
            (Pattern::Julia, PatternShape { julia_iter: 8, ..d }),
            (Pattern::Carpet, PatternShape { carpet_levels: 1, ..d }),
            (Pattern::Clouds, PatternShape { octaves: 2, ..d }),
            (Pattern::Clouds, PatternShape { roughness: 0.8, ..d }),
            (Pattern::Radial, PatternShape { sectors: 6, ..d }),
            (Pattern::Polka, PatternShape { dot_size: 0.8, ..d }),
            (Pattern::Polka, PatternShape { dot_spacing: 1.7, ..d }),
            (Pattern::Polka, PatternShape { dot_variability: 0.6, ..d }),
        ] {
            assert_ne!(tones(pattern, d), tones(pattern, changed), "{pattern:?} {changed:?}");
            // and the default settings are the static tone's
            if !pattern.animated() {
                assert!(pts.iter().zip(tones(pattern, d)).all(|(uv, v)| tone(pattern, *uv) == v), "{pattern:?}");
            }
        }
        let wild = PatternShape { ripple: f32::NAN, wind: 400.0, julia_iter: 0, sectors: 999, ..d }.sanitized();
        assert_eq!((wild.ripple, wild.wind, wild.julia_iter, wild.sectors), (d.ripple, 40.0, 4, 64));
    }

    #[test]
    fn polka_dots_sit_on_a_triangular_lattice_vary_and_invert_one_half() {
        let h = 3f32.sqrt() / 2.0;
        // a dot on every lattice point: rows h apart, every other one
        // shifted half a spacing; gaps at the triangles' centres
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (0.5, h), (-0.5, h), (3.0, -2.0 * h), (2.5, -3.0 * h)] {
            assert_eq!(polka(Vec2::new(x, y), 0.5, 1.0, 0.0), 1.0, "dot at {x},{y}");
        }
        assert_eq!(polka(Vec2::new(0.5, h / 3.0), 0.5, 1.0, 0.0), -1.0, "triangle centre");
        assert_eq!(polka(Vec2::new(0.0, h), 0.5, 1.0, 0.0), -1.0, "not a square grid");
        // on the floor the lattice starts half a spacing along y: no dot at
        // the focal point, one half a spacing from it
        for spacing in [1.0, 2.0] {
            assert_eq!(polka_placed(Vec2::ZERO, 0.5, spacing, 0.0), -1.0, "focal point between dots ({spacing})");
            assert_eq!(polka_placed(Vec2::new(0.0, 0.5 * spacing), 0.5, spacing, 0.0), 1.0, "dot half a spacing along y ({spacing})");
        }
        assert_eq!(polka(Vec2::new(1.0, 2.0 * h), 0.5, 2.0, 0.0), 1.0, "spacing 2");
        assert_eq!(polka(Vec2::new(1.0, 2.0 * h / 3.0), 0.5, 2.0, 0.0), -1.0);
        // six nearest neighbours, all a spacing away
        let near = (-3..=3)
            .flat_map(|a| (-3..=3).map(move |b| Vec2::new(a as f32 + 0.5 * b as f32, b as f32 * h)))
            .filter(|p| (p.length() - 1.0).abs() < 1e-4)
            .filter(|p| polka(*p, 0.2, 1.0, 0.0) > 0.0)
            .count();
        assert_eq!(near, 6);
        // dots wide enough to reach the triangles' centres merge into one field
        assert_eq!(polka(Vec2::new(0.5, h / 3.0), 1.6, 1.0, 0.0), 1.0);
        // variability spreads sizes and places: coverage per cell differs,
        // and dots leave their grid points
        let cover = |var: f32, cell: i32| {
            (0..400).filter(|i| polka(Vec2::new(cell as f32 + (i % 20) as f32 / 20.0 + 0.025, (i / 20) as f32 / 20.0 + 0.025), 0.6, 1.0, var) > 0.0).count()
        };
        let even: Vec<usize> = (0..12).map(|c| cover(0.0, c * 2)).collect();
        assert!(even.windows(2).all(|w| w[0] == w[1]), "all alike: {even:?}");
        let varied: Vec<usize> = (0..12).map(|c| cover(1.0, c * 2)).collect();
        let (lo, hi) = (varied.iter().min().unwrap(), varied.iter().max().unwrap());
        assert!(hi - lo > 40, "sizes spread: {varied:?}");
        let off_centre = (0..40).filter(|c| polka(Vec2::new(*c as f32 * 3.0, 0.0), 0.2, 1.0, 1.0) < 0.0).count();
        assert!(off_centre > 5, "dots leave their grid points ({off_centre})");
        // invert half: one side of the line through the focal point is
        // flipped; turning the pattern turns the line with it
        let shape = PatternShape { invert_half: true, ..PatternShape::default() };
        let params = GroundParams { pattern: Pattern::Polka, contrast: 1.0, shape, ..Default::default() };
        let pivot = Vec2::new(0.25, 0.75);
        let plain = Ground::new(&GroundParams { shape: PatternShape::default(), ..params }, pivot);
        let split = Ground::new(&params, pivot);
        let at = |g: &Ground, x: f32, y: f32| g.albedo(Vec3A::new(x, y, 0.0), 0.0075);
        // the near half (toward −y, the camera's starting side) as it was,
        // the far half inverted
        for (x, y) in [(2.3, -0.4), (-5.1, -3.3), (0.9, -7.2)] {
            assert_eq!(at(&split, x, y), at(&plain, x, y), "this side as it was");
            let (xm, ym) = (x, 2.0 * pivot.y - y);
            assert!((at(&split, xm, ym) - at(&plain, xm, ym)).abs() > 0.05, "the other side inverted at {xm},{ym}");
        }
        let mut turning = Ground::new(&GroundParams { rotate: Spin::Ccw, rotate_period: 8.0, ..params }, pivot);
        assert!(turning.on_inverted_half(pivot.x, pivot.y + 1.0) && !turning.on_inverted_half(pivot.x - 1.0, pivot.y - 0.01));
        turning.advance(4.0); // half a turn
        assert!(!turning.on_inverted_half(pivot.x, pivot.y + 1.0) && turning.on_inverted_half(pivot.x, pivot.y - 1.0));
        // other patterns ignore it
        let checker = Ground::new(&GroundParams { pattern: Pattern::Checker, ..params }, pivot);
        let checker_plain = Ground::new(&GroundParams { pattern: Pattern::Checker, shape: PatternShape::default(), ..params }, pivot);
        assert_eq!(at(&checker, -3.3, 1.2), at(&checker_plain, -3.3, 1.2));
        // small dots fade out sooner than their tiles would
        let tiny = Ground::new(&GroundParams { shape: PatternShape { dot_size: 0.1, ..PatternShape::default() }, ..params }, pivot);
        assert_eq!(tiny.albedo(Vec3A::new(0.5, -0.5, 0.0), 0.1), tiny.mean_albedo());
        let s = PatternShape { dot_size: 0.0, dot_spacing: 99.0, dot_variability: f32::NAN, ..PatternShape::default() }.sanitized();
        assert_eq!((s.dot_size, s.dot_spacing, s.dot_variability), (0.05, 8.0, 0.0));
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
