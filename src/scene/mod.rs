//! The scene: one solid per slot on concentric rings, a ground plane at
//! z = 0, one spherical area light and a dim uniform sky.

pub mod camera;
pub mod ground;
pub mod orbit;
pub mod sdf;

use glam::{Mat3A, Vec2, Vec3A};
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

use crate::metrics::Channel;
pub use camera::{Camera, CameraParams, CameraRig};
pub use ground::{Ground, GroundParams};
pub use orbit::{OrbitCenter, OrbitParams, Orbits};
use sdf::{Polyhedron, SolidKind, inverse_rotation};

/// Circumradius of every solid at scale 1.
pub const BASE_RADIUS: f32 = 1.0;
/// Radius of the innermost ring when it holds more than one solid.
pub const MIN_RING_RADIUS: f32 = 2.7;
/// Centre-to-centre spacing between neighbours and between rings.
pub const SPACING: f32 = 2.0 * BASE_RADIUS + 0.6;
/// Automatic placement fills a ring with this many solids before starting the next.
pub const RING_CAPACITY: usize = 8;
/// Gap between a solid's circumsphere and the ground.
pub const HOVER: f32 = 0.08;
/// Solids float so that even the largest halo (1.8x the radius, the ±60%
/// swing) stays this far above the ground.
pub const HALO_REACH_MAX: f32 = 1.8;
pub const FLOAT_GAP: f32 = 0.1;
/// Layout extent the light and camera were tuned for (default 5-solid ring).
const REF_EXTENT: f32 = MIN_RING_RADIUS + BASE_RADIUS;

/// Slightly different albedo per slot so neighbours separate in monochrome.
const ALBEDO: [f32; 8] = [0.85, 0.70, 0.62, 0.78, 0.66, 0.74, 0.58, 0.81];

/// Default slot list (the five classic resources).
pub const DEFAULT_SOLIDS: [SolidKind; 5] = [
    SolidKind::Icosahedron,
    SolidKind::Cube,
    SolidKind::Octahedron,
    SolidKind::Tetrahedron,
    SolidKind::Dodecahedron,
];

#[derive(Debug, Clone, Copy)]
pub struct Instance {
    pub poly: Polyhedron,
    pub center: Vec3A,
    /// Circumradius in world units.
    pub scale: f32,
    pub inv_scale: f32,
    pub inv_rot: Mat3A,
    pub albedo: f32,
}

impl Instance {
    /// `sdf(R⁻¹ · (p − c) / s) · s`, with `dc = p − c` precomputed.
    #[inline(always)]
    pub fn sdf_local(&self, dc: Vec3A) -> f32 {
        self.poly.eval(self.inv_rot * (dc * self.inv_scale)) * self.scale
    }

    #[inline(always)]
    pub fn sdf(&self, p: Vec3A) -> f32 {
        self.sdf_local(p - self.center)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub center: Vec3A,
    pub radius: f32,
    pub emission: f32,
}

/// Where the light sits relative to the centroid, per unit of layout scale,
/// and its radius (tuned on the default 5-solid ring).
const LIGHT_OFFSET: Vec3A = Vec3A::new(-7.0, -4.0, 11.0);
const LIGHT_RADIUS: f32 = 2.2;
/// The light's near surface stays at least this many times the camera's
/// furthest reach from the centroid.
const LIGHT_CLEARANCE: f32 = 1.2;

/// The scene light, pushed out along its usual direction until no camera
/// position can reach or look through it. Its radius grows by the same
/// factor, so it covers the same angle from the solids and lights them as
/// before (irradiance depends only on that angle, not on the distance).
fn light_beyond(centroid: Vec3A, k: f32, rig: &CameraRig) -> Light {
    let (d, r) = (LIGHT_OFFSET.length() * k, LIGHT_RADIUS * k);
    let reach = rig.reach() + (rig.target() - centroid).length();
    let push = (LIGHT_CLEARANCE * reach / (d - r)).max(1.0);
    Light { center: centroid + LIGHT_OFFSET * (k * push), radius: r * push, emission: 90.0 }
}

/// Animation ranges from the config.
#[derive(Debug, Clone, Copy)]
pub struct AnimParams {
    pub min_scale: f32,
    pub max_scale: f32,
    pub idle_spin: f32,
    pub max_spin: f32,
}

impl Default for AnimParams {
    fn default() -> Self {
        Self { min_scale: 0.45, max_scale: 1.0, idle_spin: 0.15, max_spin: 3.0 }
    }
}

/// One solid to place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotLayout {
    pub solid: SolidKind,
    /// Requested ring; `None` = automatic.
    pub ring: Option<u32>,
}

/// Ground-plane positions (z = 0) and the compacted ring index of every slot.
///
/// Explicit rings are honoured; automatic slots fill rings from the inside,
/// `RING_CAPACITY` per ring. Empty rings are skipped, so ring numbers only
/// order the rings. Within a ring, slots keep their `show` order, starting at
/// the front (−y, toward the camera's starting view); odd rings are rotated half a step so solids interleave.
pub fn ring_layout(rings: &[Option<u32>]) -> (Vec<Vec3A>, Vec<usize>) {
    ring_layout_spaced(rings, 1.0)
}

/// [`ring_layout`] with neighbour spacing and ring radii scaled by `spacing`.
pub fn ring_layout_spaced(rings: &[Option<u32>], spacing: f32) -> (Vec<Vec3A>, Vec<usize>) {
    let (gap, min_r) = (SPACING * spacing, MIN_RING_RADIUS * spacing);
    let (compact, nrings) = assign_rings(rings);

    let mut radius_prev: Option<f32> = None;
    let mut radii = Vec::with_capacity(nrings);
    for k in 0..nrings {
        let n = compact.iter().filter(|&&c| c == k).count();
        let chord = if n > 1 { gap / (2.0 * (PI / n as f32).sin()) } else { 0.0 };
        let r = match radius_prev {
            None if n == 1 => 0.0,
            None => chord.max(min_r),
            Some(prev) => chord.max(prev + gap),
        };
        radii.push(r);
        radius_prev = Some(r);
    }

    let mut pos = vec![Vec3A::ZERO; rings.len()];
    for (k, &r) in radii.iter().enumerate() {
        let members: Vec<usize> = (0..rings.len()).filter(|&i| compact[i] == k).collect();
        let n = members.len() as f32;
        let offset = if k % 2 == 1 { PI / n } else { 0.0 };
        for (j, &i) in members.iter().enumerate() {
            let th = offset + j as f32 * TAU / n;
            pos[i] = Vec3A::new(r * th.sin(), -(r * th.cos()), 0.0);
        }
    }
    (pos, compact)
}

/// `n` points spread evenly over the unit sphere (Fibonacci lattice),
/// turned by `yaw` about the vertical axis.
pub fn fibonacci_sphere(n: usize, yaw: f32) -> Vec<Vec3A> {
    let golden = PI * (3.0 - 5f32.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = yaw + i as f32 * golden;
            Vec3A::new(r * th.sin(), -(r * th.cos()), z)
        })
        .collect()
}

fn min_gap(pos: &[Vec3A]) -> f32 {
    let mut m = f32::MAX;
    for i in 0..pos.len() {
        for j in i + 1..pos.len() {
            m = m.min((pos[i] - pos[j]).length());
        }
    }
    m
}

/// Centres on concentric spherical shells (ring index = shell index, same
/// assignment rules as [`ring_layout`]), around a centre raised so the lowest
/// full-size solid clears the ground. Returns (centres, shell index, centre).
pub fn shell_layout(rings: &[Option<u32>]) -> (Vec<Vec3A>, Vec<usize>, Vec3A) {
    shell_layout_spaced(rings, 1.0)
}

/// [`shell_layout`] with neighbour spacing and shell radii scaled by `spacing`.
pub fn shell_layout_spaced(rings: &[Option<u32>], spacing: f32) -> (Vec<Vec3A>, Vec<usize>, Vec3A) {
    let (gap, min_r) = (SPACING * spacing, MIN_RING_RADIUS * spacing);
    let (compact, nshells) = assign_rings(rings);
    let mut dirs: Vec<Vec<Vec3A>> = Vec::with_capacity(nshells);
    let mut radii: Vec<f32> = Vec::with_capacity(nshells);
    let mut prev: Option<f32> = None;
    for k in 0..nshells {
        let n = compact.iter().filter(|&&c| c == k).count();
        // Alternate shells are turned so solids don't line up radially.
        let d = fibonacci_sphere(n, k as f32 * 0.9);
        let fit = if n > 1 { gap / min_gap(&d) } else { 0.0 };
        let r = match prev {
            None if n == 1 => 0.0,
            None => fit.max(min_r),
            Some(p) => fit.max(p + gap),
        };
        dirs.push(d);
        radii.push(r);
        prev = Some(r);
    }
    let outer = radii.last().copied().unwrap_or(0.0);
    let centre = Vec3A::new(0.0, 0.0, outer + BASE_RADIUS + HOVER);
    let mut pos = vec![centre; rings.len()];
    let mut next = vec![0usize; nshells];
    for (i, &k) in compact.iter().enumerate() {
        pos[i] = centre + dirs[k][next[k]] * radii[k];
        next[k] += 1;
    }
    (pos, compact, centre)
}

/// `n` ground positions evenly spaced `SPACING × spacing` apart along x,
/// centred on the origin.
pub fn line_layout(n: usize, spacing: f32) -> Vec<Vec3A> {
    let gap = SPACING * spacing;
    (0..n).map(|i| Vec3A::new((i as f32 - (n as f32 - 1.0) * 0.5) * gap, 0.0, 0.0)).collect()
}

/// `n` centres one above another on the vertical axis, `SPACING × spacing`
/// apart, the lowest at z = 0 (the scene lifts the stack clear of the floor).
pub fn stack_layout(n: usize, spacing: f32) -> Vec<Vec3A> {
    let gap = SPACING * spacing;
    (0..n).map(|i| Vec3A::new(0.0, 0.0, i as f32 * gap)).collect()
}

/// The order slots take the layout's places in (first place first: the
/// inner ring's front, the inner shell, the line's left end, the stack's
/// bottom).
pub fn placement_order(kinds: &[SolidKind], ordering: Ordering, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..kinds.len()).collect();
    match ordering {
        Ordering::Listed => {}
        // stable: equal solids keep their listed order
        Ordering::Simple => order.sort_by(|&a, &b| kinds[a].complexity().total_cmp(&kinds[b].complexity())),
        Ordering::Complex => order.sort_by(|&a, &b| kinds[b].complexity().total_cmp(&kinds[a].complexity())),
        Ordering::Random => fastrand::Rng::with_seed(seed).shuffle(&mut order),
    }
    order
}

/// Compacted ring index per slot (explicit rings honoured, automatic slots
/// fill from the inside), and the number of rings.
fn assign_rings(rings: &[Option<u32>]) -> (Vec<usize>, usize) {
    let mut assigned: Vec<u32> = vec![0; rings.len()];
    let mut count = std::collections::BTreeMap::<u32, usize>::new();
    for r in rings.iter().flatten() {
        *count.entry(*r).or_default() += 1;
    }
    for (i, r) in rings.iter().enumerate() {
        assigned[i] = match r {
            Some(r) => *r,
            None => {
                let mut k = 0;
                while count.get(&k).copied().unwrap_or(0) >= RING_CAPACITY {
                    k += 1;
                }
                *count.entry(k).or_default() += 1;
                k
            }
        };
    }
    let order: Vec<u32> = count.keys().copied().collect();
    let compact: Vec<usize> = assigned.iter().map(|r| order.binary_search(r).unwrap_or(0)).collect();
    (compact, order.len())
}

/// Where the solids go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// Concentric rings on the ground.
    #[default]
    Rings,
    /// Concentric spherical shells floating above the ground.
    Sphere,
    /// One row, evenly spaced, floating above the ground.
    Line,
    /// One above another in the centre, the lowest clear of the ground.
    Stack,
    /// One solid in the centre, the others on orbits of different sizes
    /// around it ([`orbit`]).
    Orbit,
}

impl Layout {
    pub const ALL: [Layout; 5] = [Layout::Rings, Layout::Sphere, Layout::Line, Layout::Stack, Layout::Orbit];

    pub fn id(self) -> &'static str {
        match self {
            Layout::Rings => "rings",
            Layout::Sphere => "sphere",
            Layout::Line => "line",
            Layout::Stack => "stack",
            Layout::Orbit => "orbit",
        }
    }
}

impl std::str::FromStr for Layout {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rings" | "ring" => Ok(Self::Rings),
            "sphere" | "shells" => Ok(Self::Sphere),
            "line" | "row" => Ok(Self::Line),
            "stack" | "column" | "tower" => Ok(Self::Stack),
            "orbit" | "orbits" => Ok(Self::Orbit),
            _ => Err(format!("unknown layout '{s}' (rings|sphere|line|stack|orbit)")),
        }
    }
}

/// Which solid takes which place in the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Ordering {
    /// In the order the metrics are listed.
    #[default]
    Listed,
    /// Simplest solid first (fewest edges), most complex last.
    Simple,
    /// Most complex first.
    Complex,
    /// Shuffled (the same for a given seed).
    Random,
}

impl Ordering {
    pub const ALL: [Ordering; 4] = [Ordering::Listed, Ordering::Simple, Ordering::Complex, Ordering::Random];

    pub fn id(self) -> &'static str {
        match self {
            Ordering::Listed => "listed",
            Ordering::Simple => "simple",
            Ordering::Complex => "complex",
            Ordering::Random => "random",
        }
    }
}

impl std::str::FromStr for Ordering {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|o| o.id() == s).ok_or_else(|| format!("unknown ordering '{s}' (listed|simple|complex|random)"))
    }
}

/// How the solids are placed: the layout, which solid goes where, and how
/// far apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arrangement {
    pub layout: Layout,
    pub ordering: Ordering,
    /// Neighbour spacing as a multiple of the standard (1 = as tuned).
    pub spacing: f32,
    /// Distance scaling, -2..=2: each solid's size times
    /// [`distance_factor`] of its nearest neighbour's distance (0 = off).
    pub distance_scale: f32,
    /// Seed for [`Ordering::Random`] (and the orbits' variation).
    pub seed: u64,
    /// The orbit layout's settings.
    pub orbit: OrbitParams,
    /// The orbit layout's reference slot for a metric centre (CPU, RAM...);
    /// `None` = the first place.
    pub center_slot: Option<usize>,
}

impl Default for Arrangement {
    fn default() -> Self {
        Self {
            layout: Layout::Rings,
            ordering: Ordering::Listed,
            spacing: 1.0,
            distance_scale: 0.0,
            seed: 0,
            orbit: OrbitParams::default(),
            center_slot: None,
        }
    }
}

/// Spacing range for [`Arrangement::spacing`] (a quarter to four times the
/// standard: the dialog's slider is its power of two, -2..=2).
pub const SPACING_RANGE: (f32, f32) = (0.25, 4.0);
/// Orbits keep at least this share of the standard spacing between a
/// periapsis and the centre solid, however tight the spacing.
const ORBIT_MIN_CLEARANCE: f32 = 0.8;
/// Range of [`Arrangement::distance_scale`] (0 = off).
pub const DISTANCE_SCALE_RANGE: (f32, f32) = (-2.0, 2.0);
/// Smallest and largest size multiplier distance scaling gives.
pub const DISTANCE_FACTOR: (f32, f32) = (0.2, 2.5);

fn clamp_distance_scale(k: f32) -> f32 {
    if k.is_finite() { k.clamp(DISTANCE_SCALE_RANGE.0, DISTANCE_SCALE_RANGE.1) } else { 0.0 }
}

/// Distance scaling: the size multiplier for a solid whose nearest
/// neighbour is `d` away, `1 + k × (d / SPACING − 1)`: at the standard
/// spacing it keeps its size; further out a positive `k` grows it and a
/// negative one shrinks it (closer in, the other way round). Clamped to
/// [`DISTANCE_FACTOR`]; 1 with no neighbour.
pub fn distance_factor(k: f32, d: f32) -> f32 {
    if k == 0.0 || !d.is_finite() {
        return 1.0;
    }
    (1.0 + k * (d / SPACING - 1.0)).clamp(DISTANCE_FACTOR.0, DISTANCE_FACTOR.1)
}

/// [`distance_factor`] for every point of `pos` from its nearest neighbour.
fn nearest_factors(k: f32, pos: &[Vec3A]) -> Vec<f32> {
    (0..pos.len())
        .map(|i| {
            if k == 0.0 {
                return 1.0;
            }
            let d = (0..pos.len()).filter(|&j| j != i).map(|j| (pos[i] - pos[j]).length()).fold(f32::INFINITY, f32::min);
            distance_factor(k, d)
        })
        .collect()
}

/// Camera distance multiplier for the sphere layout.
const SPHERE_FRAMING: f32 = 1.45;
/// Camera distance multiplier for the stack: the camera frames layouts by
/// their width, and a stack is all height.
const STACK_FRAMING: f32 = 1.6;
/// Orbits: flat ones frame like rings; steeply tilted ones need the
/// sphere's distance (blended by the steepest orbit's tilt).
const ORBIT_FRAMING_TILTED: f32 = SPHERE_FRAMING;

/// Focus mode: a solid must lead the current focus by this much activity...
const FOCUS_MARGIN: f32 = 0.05;
/// ...for this long (seconds) before the camera switches to it.
const FOCUS_HOLD: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct Scene {
    pub solids: Vec<Instance>,
    /// (pitch, yaw) per solid, radians.
    pub angles: Vec<(f32, f32)>,
    /// (pitch, yaw) rates per solid this frame, radians/s.
    pub spin_rates: Vec<(f32, f32)>,
    /// Fastest possible angular speed, radians/s (normalises spin colours).
    pub spin_max: f32,
    pub ring_pos: Vec<Vec3A>,
    pub ring_index: Vec<usize>,
    pub centroid: Vec3A,
    /// Distance from the centroid to the outermost solid's far edge.
    pub extent: f32,
    pub rig: CameraRig,
    pub camera: Camera,
    pub light: Light,
    /// Radiance of escaping secondary rays.
    pub sky: f32,
    /// Radiance of escaping camera rays.
    pub background: f32,
    /// Background plane at z = 0 (pattern, motion; disabled at luminosity 0).
    pub ground: Ground,
    /// Top of the z-slab (height band) that contains every solid.
    pub slab_top: f32,
    /// Circumradius of a solid at full size.
    pub max_radius: f32,
    /// Rings: `ring_pos` is a ground point and solids rest on it. Sphere:
    /// `ring_pos` is the solid's centre.
    grounded: bool,
    /// Slot the focus camera is on, and a challenger waiting out `FOCUS_HOLD`.
    focus: Option<usize>,
    challenger: Option<(usize, f32)>,
    /// Orbit layout: the orbits (`ring_pos` follows them every frame).
    pub orbits: Option<Orbits>,
    /// Distance scaling factor, -2..=2 (0 = off).
    distance_scale: f32,
    /// Per solid: its size multiplier from distance scaling.
    pub size_factor: Vec<f32>,
}

impl Scene {
    pub fn new(slots: &[SlotLayout], anim: &AnimParams, cam: &CameraParams) -> Self {
        Self::with_layout(slots, anim, cam, Layout::Rings)
    }

    pub fn with_layout(slots: &[SlotLayout], anim: &AnimParams, cam: &CameraParams, layout: Layout) -> Self {
        Self::arranged(slots, anim, cam, &Arrangement { layout, ..Arrangement::default() })
    }

    /// The scene with its solids placed by `arr`.
    pub fn arranged(slots: &[SlotLayout], anim: &AnimParams, cam: &CameraParams, arr: &Arrangement) -> Self {
        let layout = arr.layout;
        let spacing = if arr.spacing.is_finite() { arr.spacing.clamp(SPACING_RANGE.0, SPACING_RANGE.1) } else { 1.0 };
        let n = slots.len().max(1) as f32;
        let mean = |p: &[Vec3A]| p.iter().copied().fold(Vec3A::ZERO, |a, b| a + b) / n;
        // Lay out the slots in placement order, then hand each its place.
        let kinds: Vec<SolidKind> = slots.iter().map(|s| s.solid).collect();
        let order = placement_order(&kinds, arr.ordering, arr.seed);
        let rings: Vec<Option<u32>> = order.iter().map(|&i| slots[i].ring).collect();
        let max_r = BASE_RADIUS * anim.max_scale.max(anim.min_scale);
        let float_z = max_r * HALO_REACH_MAX + FLOAT_GAP;
        let k_dist = clamp_distance_scale(arr.distance_scale);
        // Orbits: the centre is raised so the lowest point of any orbit
        // keeps the largest halo clear of the ground.
        // (each orbit 10% of its radius beyond the last at the standard
        // spacing, stretched by it; periapses stay clear of the centre solid
        // however tight the spacing)
        let mut orbits = (layout == Layout::Orbit).then(|| {
            let clear = SPACING * spacing.max(ORBIT_MIN_CLEARANCE);
            let step = orbit::ORBIT_STEP * spacing;
            Orbits::new(&order, &arr.orbit, arr.center_slot, clear, MIN_RING_RADIUS * spacing, step, float_z, arr.seed)
        });
        // Orbiting solids' neighbours come and go: allow for the largest
        // size distance scaling can give them (a negative factor grows
        // solids as they pass close; a positive one as they spread out).
        let orbit_grow = orbits.as_ref().map(|o| match k_dist {
            k if k < 0.0 => DISTANCE_FACTOR.1,
            k => distance_factor(k, 2.0 * o.reach),
        });
        if let (Some(o), Some(g)) = (&mut orbits, orbit_grow) {
            o.centre.z += max_r * (g - 1.0).max(0.0) * HALO_REACH_MAX;
        }
        let (placed, placed_index, centroid) = match layout {
            Layout::Orbit => {
                let o = orbits.as_ref().expect("orbit layout has orbits");
                (order.iter().map(|&i| o.position(i)).collect(), order.iter().map(|&i| o.index[i]).collect(), o.centre)
            }
            Layout::Rings => {
                let (p, idx) = ring_layout_spaced(&rings, spacing);
                let c = mean(&p);
                (p, idx, c)
            }
            Layout::Sphere => shell_layout_spaced(&rings, spacing),
            Layout::Line | Layout::Stack => {
                let p = if layout == Layout::Line { line_layout(slots.len(), spacing) } else { stack_layout(slots.len(), spacing) };
                let c = mean(&p);
                (p, vec![0; slots.len()], c)
            }
        };
        let (mut ring_pos, mut ring_index) = (vec![Vec3A::ZERO; slots.len()], vec![0; slots.len()]);
        for (k, &i) in order.iter().enumerate() {
            ring_pos[i] = placed[k];
            ring_index[i] = placed_index[k];
        }
        // Distance scaling: each solid's size factor from its nearest
        // neighbour (fixed, but for orbits, where it follows them), and the
        // largest it can get, which sets how high everything floats.
        let size_factor = nearest_factors(k_dist, &ring_pos);
        let grow = orbit_grow.unwrap_or_else(|| size_factor.iter().copied().fold(0.0, f32::max)).max(DISTANCE_FACTOR.0);
        let max_r = max_r * grow;
        let float_z = max_r * HALO_REACH_MAX + FLOAT_GAP;
        // (orbiting solids reach their apoapsis, wherever they start)
        let reach = orbits.as_ref().map_or(0.0, |o| o.reach);
        let extent =
            ring_pos.iter().map(|p| (*p - centroid).length()).fold(reach, f32::max) + BASE_RADIUS * anim.max_scale * grow;
        // Float the solids so their largest halo clears the ground: rings
        // and lines hover at a fixed height (the camera's aim rises with
        // them); a sphere of shells or a stack is lifted if its lowest halo
        // would dip below; orbits were raised when they were made.
        let grounded = matches!(layout, Layout::Rings | Layout::Line);
        let (mut ring_pos, mut centroid) = (ring_pos, centroid);
        match layout {
            Layout::Orbit => {}
            Layout::Rings | Layout::Line => centroid.z += float_z - (max_r + HOVER),
            Layout::Sphere | Layout::Stack => {
                let low = ring_pos.iter().map(|p| p.z).fold(f32::INFINITY, f32::min) - max_r * HALO_REACH_MAX;
                let lift = (FLOAT_GAP - low).max(0.0);
                for p in &mut ring_pos {
                    p.z += lift;
                }
                centroid.z += lift;
            }
        }
        let k = extent / REF_EXTENT;

        let solids = slots
            .iter()
            .enumerate()
            .map(|(i, s)| Instance {
                poly: Polyhedron::new(s.solid),
                center: Vec3A::ZERO,
                scale: 1.0,
                inv_scale: 1.0,
                inv_rot: Mat3A::IDENTITY,
                albedo: ALBEDO[i % ALBEDO.len()],
            })
            .collect();
        // Stagger initial orientations so the idle scene isn't uniform.
        let angles = (0..slots.len()).map(|i| (0.4 + 0.3 * i as f32, 0.7 * i as f32)).collect();
        // A sphere is as tall as it is wide; flat rings appear foreshortened,
        // so the camera framing tuned for rings needs to back off for shells.
        let frame_extent = match layout {
            Layout::Sphere => extent * SPHERE_FRAMING,
            Layout::Stack => extent * STACK_FRAMING,
            Layout::Orbit => {
                let tilt = orbits.as_ref().map_or(0.0, |o| o.max_tilt.sin());
                extent * (1.0 + (ORBIT_FRAMING_TILTED - 1.0) * tilt)
            }
            Layout::Rings | Layout::Line => extent,
        };
        let rig = CameraRig::new(cam, centroid, frame_extent);
        let light = light_beyond(centroid, k, &rig);
        let span = (anim.max_spin - anim.idle_spin).max(0.0);
        let spin_max = (anim.idle_spin + span).hypot(0.5 * anim.idle_spin + span).max(1e-3);
        let mut s = Self {
            spin_rates: vec![(0.0, 0.0); slots.len()],
            spin_max,
            solids,
            angles,
            ring_pos,
            ring_index,
            centroid,
            extent,
            camera: rig.camera(),
            rig,
            light,
            sky: 0.12,
            background: 0.0,
            // Rotation centre: the focal point the camera looks at, on the ground.
            ground: Ground::new(&GroundParams::default(), Vec2::new(centroid.x, centroid.y)),
            slab_top: 2.0 * BASE_RADIUS + HOVER,
            max_radius: max_r,
            grounded,
            focus: None,
            challenger: None,
            orbits,
            distance_scale: k_dist,
            size_factor,
        };
        for i in 0..s.solids.len() {
            s.set_scale(i, anim.min_scale * s.size_factor[i]);
        }
        s.refresh_transforms();
        s
    }

    /// Replace the background settings (keeps the focal-point pivot).
    pub fn with_ground(mut self, p: &GroundParams) -> Self {
        self.ground = Ground::new(p, Vec2::new(self.centroid.x, self.centroid.y));
        self
    }

    /// The classic five-solid scene with a static camera.
    pub fn demo(anim: &AnimParams) -> Self {
        let slots: Vec<SlotLayout> = DEFAULT_SOLIDS.iter().map(|&solid| SlotLayout { solid, ring: None }).collect();
        Self::new(&slots, anim, &CameraParams::default())
    }

    fn set_scale(&mut self, i: usize, scale: f32) {
        let r = (scale * BASE_RADIUS).max(1e-3);
        let float_z = self.float_z();
        let inst = &mut self.solids[i];
        inst.scale = r;
        inst.inv_scale = 1.0 / r;
        inst.center = if self.grounded {
            // Float at a fixed height whatever the size, high enough that the
            // largest halo never dips into the ground.
            self.ring_pos[i] + Vec3A::new(0.0, 0.0, float_z)
        } else {
            self.ring_pos[i]
        };
    }

    /// Give solid `i` another shape (a solid driven by complexity); its
    /// place, size and spin carry on.
    pub fn set_kind(&mut self, i: usize, kind: SolidKind) {
        if let Some(s) = self.solids.get_mut(i).filter(|s| s.poly.kind() != kind) {
            s.poly = Polyhedron::new(kind);
        }
    }

    /// Rings: the height every solid's centre floats at.
    pub fn float_z(&self) -> f32 {
        self.max_radius * HALO_REACH_MAX + FLOAT_GAP
    }

    /// Where solid `i`'s centre is when it is at full size (rings float at
    /// a fixed height, so this is where it always is).
    pub fn full_size_center(&self, i: usize) -> Vec3A {
        let p = self.ring_pos.get(i).copied().unwrap_or(self.centroid);
        if self.grounded { p + Vec3A::new(0.0, 0.0, self.float_z()) } else { p }
    }

    /// Slot the focus camera is on (or heading to).
    pub fn focused(&self) -> Option<usize> {
        self.focus
    }

    /// Track the busiest available solid, with hysteresis so the camera
    /// doesn't flick between near-equal solids.
    fn update_focus(&mut self, channels: &[Channel], dt: f32) {
        let level = |i: usize| channels.get(i).filter(|c| c.available).map_or(0.0, |c| c.level.clamp(0.0, 1.0));
        let n = self.solids.len();
        let Some(best) = (0..n).max_by(|&a, &b| level(a).total_cmp(&level(b))) else { return };
        match self.focus {
            None => {
                if level(best) > 0.01 {
                    self.focus = Some(best);
                }
            }
            Some(cur) if best != cur && level(best) > level(cur) + FOCUS_MARGIN => {
                let held = match self.challenger {
                    Some((c, t)) if c == best => t + dt,
                    _ => dt,
                };
                if held >= FOCUS_HOLD {
                    self.focus = Some(best);
                    self.challenger = None;
                } else {
                    self.challenger = Some((best, held));
                }
            }
            Some(_) => self.challenger = None,
        }
    }

    fn refresh_transforms(&mut self) {
        let mut top: f32 = 0.0;
        for (inst, &(pitch, yaw)) in self.solids.iter_mut().zip(&self.angles) {
            inst.inv_rot = inverse_rotation(pitch, yaw);
            top = top.max(inst.center.z + inst.scale);
        }
        self.slab_top = top + 1e-2;
    }

    /// Each solid's spin as a colour value 0..=1, written to `out`: 0.5 =
    /// still or spinning edge-on to the camera, toward 1 = faster
    /// counter-clockwise as seen from the camera, toward 0 = faster clockwise.
    pub fn spin_values(&self, out: &mut Vec<f32>) {
        out.clear();
        let eye = self.camera.pos;
        for (i, s) in self.solids.iter().enumerate() {
            let (pr, yr) = self.spin_rates.get(i).copied().unwrap_or_default();
            let yaw = self.angles.get(i).map_or(0.0, |a| a.1);
            // R = Rz(yaw) * Rx(pitch): world angular velocity
            let w = Vec3A::Z * yr + Mat3A::from_rotation_z(yaw) * Vec3A::X * pr;
            let view = (eye - s.center).normalize_or_zero();
            out.push((0.5 + 0.5 * w.dot(view) / self.spin_max).clamp(0.0, 1.0));
        }
    }

    /// Per-solid colour values for `by`, written to `out`: spin from
    /// [`Scene::spin_values`], otherwise each channel's load (unavailable =
    /// 0). Sound values come from the caller's meter.
    pub fn color_values(&self, by: crate::color::ColorBy, channels: &[Channel], out: &mut Vec<f32>) {
        if by == crate::color::ColorBy::Spin {
            self.spin_values(out);
        } else {
            out.clear();
            out.extend(channels.iter().map(|c| if c.available { c.level } else { 0.0 }));
        }
    }

    /// Drive scale and spin from smoothed channels and move the camera;
    /// everything uses wall-clock `dt`.
    pub fn update(&mut self, channels: &[Channel], anim: &AnimParams, dt: f32) {
        let span = (anim.max_spin - anim.idle_spin).max(0.0);
        if let Some(o) = &mut self.orbits {
            o.advance(channels, dt);
            for (i, p) in self.ring_pos.iter_mut().enumerate() {
                *p = o.position(i);
            }
            if self.distance_scale != 0.0 {
                self.size_factor = nearest_factors(self.distance_scale, &self.ring_pos);
            }
        }
        for i in 0..self.solids.len() {
            let c = channels.get(i).copied().unwrap_or_default();
            let (level, sy, sx) = if c.available { (c.level, c.spin_y, c.spin_x) } else { (0.0, 0.0, 0.0) };
            let scale = (anim.min_scale + (anim.max_scale - anim.min_scale) * level.clamp(0.0, 1.0))
                * self.size_factor.get(i).copied().unwrap_or(1.0);
            self.set_scale(i, scale);
            let yaw_rate = anim.idle_spin + span * sy.clamp(0.0, 1.0);
            let pitch_rate = 0.5 * anim.idle_spin + span * sx.clamp(0.0, 1.0);
            if let Some(r) = self.spin_rates.get_mut(i) {
                *r = (pitch_rate, yaw_rate);
            }
            let (p, y) = &mut self.angles[i];
            *p = (*p + pitch_rate * dt).rem_euclid(TAU);
            *y = (*y + yaw_rate * dt).rem_euclid(TAU);
        }
        self.refresh_transforms();
        self.update_focus(channels, dt);
        let target = self.focus.and_then(|i| self.solids.get(i)).map(|s| s.center);
        self.rig.advance(dt, target);
        if self.rig.auto_distance() {
            let spheres: Vec<(Vec3A, f32)> = self.solids.iter().map(|s| (s.center, s.scale)).collect();
            self.rig.fit(&spheres, dt);
        }
        self.camera = self.rig.camera();
        self.ground.advance(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(n: usize) -> Vec<Channel> {
        vec![Channel { available: true, level: 1.0, spin_y: 1.0, spin_x: 1.0, activity: 1.0 }; n]
    }

    #[test]
    fn solids_rest_above_ground_and_inside_slab() {
        let anim = AnimParams::default();
        let mut s = Scene::demo(&anim);
        s.update(&full(5), &anim, 0.1);
        for inst in &s.solids {
            assert!(inst.center.z - inst.scale >= HOVER - 1e-5);
            assert!(inst.center.z + inst.scale <= s.slab_top);
            assert!((inst.scale - anim.max_scale * BASE_RADIUS).abs() < 1e-6);
        }
        assert!((s.ring_pos[0] - Vec3A::new(0.0, -MIN_RING_RADIUS, 0.0)).length() < 1e-5, "first slot front-centre");
    }

    fn min_gap(pos: &[Vec3A]) -> f32 {
        let mut m = f32::MAX;
        for i in 0..pos.len() {
            for j in i + 1..pos.len() {
                m = m.min((pos[i] - pos[j]).length());
            }
        }
        m
    }

    #[test]
    fn rings_never_overlap() {
        for n in 1..=16 {
            let (pos, ring) = ring_layout(&vec![None; n]);
            assert!(min_gap(&pos) >= SPACING - 1e-3, "n={n}");
            assert_eq!(ring.iter().filter(|&&r| r == 0).count(), n.min(RING_CAPACITY));
        }
        // single solid sits at the centre
        assert_eq!(ring_layout(&[None]).0[0], Vec3A::ZERO);
    }

    #[test]
    fn explicit_rings_are_honoured_and_compacted() {
        let req = [Some(0), Some(5), Some(5), None, Some(0)];
        let (pos, ring) = ring_layout(&req);
        // ring 5 compacts to 1; the automatic slot joins ring 0 (has room)
        assert_eq!(ring, vec![0, 1, 1, 0, 0]);
        let r0 = pos[0].length();
        let r1 = pos[1].length();
        assert!(r1 >= r0 + SPACING - 1e-4, "outer ring outside inner: {r0} {r1}");
        assert!((pos[1].length() - pos[2].length()).abs() < 1e-5);
        assert!(min_gap(&pos) >= SPACING - 1e-3);
    }

    #[test]
    fn sphere_shells_spaced_and_above_ground() {
        for n in 1..=16 {
            let (pos, shell, c) = shell_layout(&vec![None; n]);
            if n > 1 {
                assert!(min_gap(&pos) >= SPACING - 1e-3, "n={n}: {}", min_gap(&pos));
            }
            for p in &pos {
                assert!(p.z - BASE_RADIUS >= HOVER - 1e-4, "n={n}: full-size solid clears the ground");
            }
            assert_eq!(shell.iter().filter(|&&s| s == 0).count(), n.min(RING_CAPACITY));
            // every member of a shell is the same distance from the centre
            for k in 0..=*shell.iter().max().unwrap() {
                let r: Vec<f32> = (0..n).filter(|&i| shell[i] == k).map(|i| (pos[i] - c).length()).collect();
                assert!(r.iter().all(|x| (x - r[0]).abs() < 1e-3), "n={n} shell {k}: {r:?}");
            }
        }
        // explicit shells: 1 in the centre, 3 around it
        let (pos, shell, c) = shell_layout(&[Some(0), Some(1), Some(1), Some(1)]);
        assert_eq!(shell, vec![0, 1, 1, 1]);
        assert!((pos[0] - c).length() < 1e-5);
        assert!((pos[1] - c).length() >= SPACING - 1e-4);
    }

    #[test]
    fn sphere_scene_keeps_centres_fixed() {
        let anim = AnimParams::default();
        let slots: Vec<SlotLayout> = (0..12).map(|i| SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let mut s = Scene::with_layout(&slots, &anim, &CameraParams::default(), Layout::Sphere);
        let before: Vec<Vec3A> = s.solids.iter().map(|i| i.center).collect();
        s.update(&full(12), &anim, 0.1);
        for (a, b) in before.iter().zip(&s.solids) {
            assert!((*a - b.center).length() < 1e-5, "scaling about a fixed centre");
            assert!(b.center.z - b.scale >= HOVER - 1e-4);
            assert!(b.center.z + b.scale <= s.slab_top);
        }
        assert!(s.centroid.z > 1.0, "shell centre is raised");
        assert!("Sphere".parse::<Layout>().unwrap() == Layout::Sphere && "x".parse::<Layout>().is_err());
    }

    #[test]
    fn focus_follows_busiest_with_hysteresis() {
        let anim = AnimParams::default();
        let cam = CameraParams { mode: camera::CameraMode::Focus, ..CameraParams::default() };
        let slots: Vec<SlotLayout> = DEFAULT_SOLIDS.iter().map(|&solid| SlotLayout { solid, ring: None }).collect();
        let mut s = Scene::new(&slots, &anim, &cam);
        let ch = |levels: [f32; 5]| -> Vec<Channel> {
            levels.iter().map(|&l| Channel { available: true, level: l, spin_y: 0.0, spin_x: 0.0, activity: l }).collect()
        };
        s.update(&ch([0.2, 0.9, 0.1, 0.0, 0.3]), &anim, 0.1);
        assert_eq!(s.focused(), Some(1));
        // slightly busier: within the margin, never switches
        for _ in 0..30 {
            s.update(&ch([0.2, 0.9, 0.1, 0.0, 0.93]), &anim, 0.1);
        }
        assert_eq!(s.focused(), Some(1));
        // clearly busier, but must hold for FOCUS_HOLD
        s.update(&ch([0.2, 0.5, 0.1, 0.0, 0.95]), &anim, 0.5);
        assert_eq!(s.focused(), Some(1));
        s.update(&ch([0.2, 0.5, 0.1, 0.0, 0.95]), &anim, 0.6);
        assert_eq!(s.focused(), Some(4));
        // the camera ends up looking at it
        for _ in 0..100 {
            s.update(&ch([0.2, 0.5, 0.1, 0.0, 0.95]), &anim, 0.1);
        }
        let to = (s.solids[4].center - s.camera.pos).normalize();
        assert!(s.camera.forward.dot(to) > 0.999);
    }

    #[test]
    fn centre_ring_with_one_solid() {
        let (pos, ring) = ring_layout(&[Some(0), Some(1), Some(1), Some(1)]);
        assert_eq!(pos[0], Vec3A::ZERO);
        assert_eq!(ring, vec![0, 1, 1, 1]);
        assert!(pos[1].length() >= SPACING - 1e-4);
    }

    #[test]
    fn spin_values_follow_speed_and_direction_seen_from_the_camera() {
        let anim = AnimParams::default();
        let mut s = Scene::demo(&anim);
        let mut v = Vec::new();
        s.spin_values(&mut v);
        assert!(v.iter().all(|&x| x == 0.5), "before any update: still");
        let still = vec![Channel { available: true, level: 0.5, spin_y: 0.0, spin_x: 0.0, activity: 0.0 }; s.solids.len()];
        s.update(&still, &anim, 0.016);
        s.spin_values(&mut v);
        let idle: Vec<f32> = v.iter().map(|x| (x - 0.5).abs()).collect();
        s.update(&full(s.solids.len()), &anim, 0.016);
        s.spin_values(&mut v);
        for (i, (&x, &calm)) in v.iter().zip(&idle).enumerate() {
            assert!((0.0..=1.0).contains(&x), "{i}: {x}");
            assert!((x - 0.5).abs() > calm, "{i}: full spin {x} further from still than idle ({calm})");
        }
        // the sign is the direction seen from the camera: from the other side it flips
        let eye = s.camera.pos;
        let c = s.solids[0].center;
        s.camera.pos = c - (eye - c);
        let mut back = Vec::new();
        s.spin_values(&mut back);
        assert!(((back[0] - 0.5) + (v[0] - 0.5)).abs() < 1e-4, "{} vs {}", back[0], v[0]);
    }

    #[test]
    fn unavailable_uses_min_scale_and_idle_spin() {        let anim = AnimParams::default();
        let mut s = Scene::demo(&anim);
        let before = s.angles[4];
        let off = vec![Channel { available: false, level: 1.0, spin_y: 1.0, spin_x: 1.0, activity: 1.0 }; 5];
        s.update(&off, &anim, 1.0);
        assert!((s.solids[4].scale - anim.min_scale).abs() < 1e-6);
        let dyaw = (s.angles[4].1 - before.1).rem_euclid(TAU);
        assert!((dyaw - anim.idle_spin).abs() < 1e-5);
    }

    #[test]
    fn line_and_stack_layouts_with_ordering_and_spacing() {
        let anim = AnimParams::default();
        let kinds = [SolidKind::Dodecahedron, SolidKind::Tetrahedron, SolidKind::Cube, SolidKind::Icosahedron, SolidKind::Octahedron];
        let slots: Vec<SlotLayout> = kinds.iter().map(|&solid| SlotLayout { solid, ring: None }).collect();
        let build = |layout, ordering, spacing| {
            let arr = Arrangement { layout, ordering, spacing, seed: 7, ..Arrangement::default() };
            let mut s = Scene::arranged(&slots, &anim, &CameraParams::default(), &arr);
            s.update(&[Channel { available: true, level: 1.0, spin_y: 0.0, spin_x: 0.0, activity: 1.0 }; 5], &anim, 0.1);
            s
        };
        let cx = |s: &Scene| -> Vec<f32> { s.solids.iter().map(|i| i.center.x).collect() };
        // line: one row along x, evenly spaced, centred, all at one height
        let line = build(Layout::Line, Ordering::Listed, 1.0);
        let mut xs = cx(&line);
        xs.sort_by(f32::total_cmp);
        for w in xs.windows(2) {
            assert!((w[1] - w[0] - SPACING).abs() < 1e-4, "even gaps: {xs:?}");
        }
        assert!(xs.iter().sum::<f32>().abs() < 1e-3, "centred");
        assert!(line.solids.iter().all(|i| (i.center.z - line.solids[0].center.z).abs() < 1e-5 && i.center.y.abs() < 1e-5));
        assert!(line.solids.iter().all(|i| i.center.z - i.scale * HALO_REACH_MAX >= FLOAT_GAP - 1e-3), "halos clear the floor");
        // listed order runs left to right; simple-first puts the tetrahedron left, the most complex right
        let x = cx(&line);
        assert!(x.windows(2).all(|w| w[0] < w[1]), "listed order: {x:?}");
        let simple = build(Layout::Line, Ordering::Simple, 1.0);
        let mut by_x: Vec<(f32, SolidKind)> = simple.solids.iter().map(|i| (i.center.x, i.poly.kind())).collect();
        by_x.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(by_x.windows(2).all(|w| w[0].1.complexity() <= w[1].1.complexity()), "{by_x:?}");
        assert_eq!(by_x[0].1, SolidKind::Tetrahedron);
        // spacing doubles the gaps
        let wide = build(Layout::Line, Ordering::Listed, 2.0);
        let gap = |s: &Scene| (s.solids[1].center.x - s.solids[0].center.x).abs();
        assert!((gap(&wide) - 2.0 * gap(&line)).abs() < 1e-4);
        // stack: one column, evenly spaced, the lowest halo clear of the floor
        let stack = build(Layout::Stack, Ordering::Complex, 1.0);
        let mut by_y: Vec<(f32, SolidKind)> = stack.solids.iter().map(|i| (i.center.z, i.poly.kind())).collect();
        by_y.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(stack.solids.iter().all(|i| i.center.x.abs() < 1e-4 && i.center.y.abs() < 1e-4), "a column");
        for w in by_y.windows(2) {
            assert!((w[1].0 - w[0].0 - SPACING).abs() < 1e-4, "even steps");
            assert!(w[0].1.complexity() >= w[1].1.complexity(), "complex at the bottom: {by_y:?}");
        }
        let lowest = stack.solids.iter().map(|i| i.center.z - stack.max_radius * HALO_REACH_MAX).fold(f32::MAX, f32::min);
        assert!(lowest >= FLOAT_GAP - 1e-3, "lowest halo clear: {lowest}");
        // random: a permutation, the same for the same seed
        let k = [SolidKind::Cube; 6];
        let a = placement_order(&k, Ordering::Random, 3);
        assert_eq!(a, placement_order(&k, Ordering::Random, 3));
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..6).collect::<Vec<_>>());
        assert!((0..20).any(|s| placement_order(&k, Ordering::Random, s) != a), "seeds differ");
        // rings and spheres take ordering too: simple-first puts the simplest solid in slot order's first place
        let rings = build(Layout::Rings, Ordering::Simple, 1.0);
        let front = rings.solids.iter().min_by(|a, b| a.center.y.total_cmp(&b.center.y)).unwrap();
        assert_eq!(front.poly.kind(), SolidKind::Tetrahedron, "first place (the front) holds the simplest");
        assert!("Line".parse::<Layout>().unwrap() == Layout::Line && "stack".parse::<Layout>().unwrap() == Layout::Stack);
        assert_eq!("complex".parse::<Ordering>().unwrap(), Ordering::Complex);
        assert!("sideways".parse::<Ordering>().is_err());
    }

    #[test]
    fn orbit_scene_moves_solids_around_the_reference_above_the_floor() {
        let anim = AnimParams::default();
        let slots: Vec<SlotLayout> = (0..6).map(|i| SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let orbit = OrbitParams { inclination: 60.0, eccentricity: 0.5, variation: 1.0, ..OrbitParams::default() };
        let arr = Arrangement { layout: Layout::Orbit, orbit, center_slot: Some(2), seed: 4, ..Arrangement::default() };
        let mut s = Scene::arranged(&slots, &anim, &CameraParams::default(), &arr);
        let start: Vec<Vec3A> = s.solids.iter().map(|i| i.center).collect();
        for _ in 0..600 {
            s.update(&full(6), &anim, 1.0 / 30.0);
            for inst in &s.solids {
                assert!(inst.center.z - inst.scale * HALO_REACH_MAX >= FLOAT_GAP - 1e-3, "halo clear of the floor");
                assert!(inst.center.z + inst.scale <= s.slab_top);
                assert!((inst.center - s.centroid).length() <= s.extent, "inside the framed extent");
            }
        }
        assert!((s.solids[2].center - s.centroid).length() < 1e-4, "CPU's solid holds the centre");
        assert!(s.solids.iter().zip(&start).enumerate().all(|(i, (a, b))| i == 2 || (a.center - *b).length() > 0.1), "the rest moved");
        assert_eq!("orbit".parse::<Layout>().unwrap(), Layout::Orbit);
    }

    #[test]
    fn distance_scaling_sizes_solids_by_their_nearest_neighbour() {
        let anim = AnimParams::default();
        // a line at double spacing: every neighbour is 2 standard spacings away
        let slots: Vec<SlotLayout> = DEFAULT_SOLIDS.iter().map(|&solid| SlotLayout { solid, ring: None }).collect();
        let build = |layout, distance_scale| {
            let arr = Arrangement { layout, spacing: 2.0, distance_scale, ..Arrangement::default() };
            let mut s = Scene::arranged(&slots, &anim, &CameraParams::default(), &arr);
            s.update(&full(5), &anim, 0.1);
            s
        };
        let plain = build(Layout::Line, 0.0);
        let big = build(Layout::Line, 0.5);
        let small = build(Layout::Line, -0.5);
        for i in 0..5 {
            let r = plain.solids[i].scale;
            assert!((big.solids[i].scale - 1.5 * r).abs() < 1e-4, "1 + 0.5 x (2 - 1)");
            assert!((small.solids[i].scale - 0.5 * r).abs() < 1e-4);
            assert!(big.solids[i].center.z - big.solids[i].scale * HALO_REACH_MAX >= FLOAT_GAP - 1e-3, "grown halos clear the floor");
        }
        assert!(big.extent > plain.extent);
        // clamped at both ends
        assert_eq!(distance_factor(2.0, 100.0), DISTANCE_FACTOR.1);
        assert_eq!(distance_factor(-2.0, 100.0), DISTANCE_FACTOR.0);
        assert_eq!(distance_factor(1.5, SPACING), 1.0);
        assert_eq!(distance_factor(1.5, f32::INFINITY), 1.0, "no neighbour");
        // orbits: sizes follow the moving neighbours, and stay clear of the floor
        for k in [-0.5, 0.5] {
            let arr = Arrangement { layout: Layout::Orbit, distance_scale: k, ..Arrangement::default() };
            let mut s = Scene::arranged(&slots, &anim, &CameraParams::default(), &arr);
            let mut seen = Vec::new();
            for _ in 0..300 {
                s.update(&full(5), &anim, 1.0 / 30.0);
                seen.push(s.size_factor[1]);
                for inst in &s.solids {
                    assert!(inst.center.z - inst.scale * HALO_REACH_MAX >= FLOAT_GAP - 1e-3, "k={k}");
                }
            }
            assert!(seen.iter().any(|f| (f - seen[0]).abs() > 1e-3), "k={k}: the factor changes as solids move");
        }
    }

    #[test]
    fn camera_and_light_scale_with_layout() {
        let anim = AnimParams::default();
        let small = Scene::demo(&anim);
        let slots: Vec<SlotLayout> = (0..14).map(|i| SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let big = Scene::new(&slots, &anim, &CameraParams::default());
        assert!(big.extent > small.extent);
        assert!((big.camera.pos - big.centroid).length() > (small.camera.pos - small.centroid).length());
        assert!(big.light.radius > small.light.radius);
    }

    /// The camera, wherever any mode takes it, never reaches the light or
    /// looks through it at a solid, and the light still covers the angle
    /// it was tuned to from the solids (same shading).
    #[test]
    fn camera_never_reaches_or_looks_through_the_light() {
        use crate::scene::camera::CameraMode;
        let anim = AnimParams::default();
        let tuned = (LIGHT_RADIUS / LIGHT_OFFSET.length()).asin();
        for (n, layout) in
            [(5usize, Layout::Rings), (16, Layout::Rings), (16, Layout::Sphere), (5, Layout::Line), (16, Layout::Line), (8, Layout::Stack), (7, Layout::Orbit)]
        {
            let slots: Vec<SlotLayout> =
                (0..n).map(|i| SlotLayout { solid: SolidKind::ALL[i % SolidKind::ALL.len()], ring: None }).collect();
            for mode in CameraMode::ALL {
                for auto_distance in [false, true] {
                    let cp = CameraParams { mode, auto_distance, period: 8.0, seed: Some(3), ..CameraParams::default() };
                    let mut s = Scene::with_layout(&slots, &anim, &cp, layout);
                    let l = s.light;
                    let seen = ((l.center - s.centroid).length().recip() * l.radius).asin();
                    assert!((seen - tuned).abs() < 1e-4, "same apparent size from the solids");
                    for step in 0..400 {
                        let t = step as f32 * 0.25;
                        let ch: Vec<Channel> = (0..n)
                            .map(|i| {
                                let v = 0.5 + 0.5 * (0.7 * t + i as f32).sin();
                                Channel { available: true, level: v, spin_y: v, spin_x: 0.2, activity: v }
                            })
                            .collect();
                        s.update(&ch, &anim, 0.25);
                        let eye = s.camera.pos;
                        let what = format!("{n} {layout:?} {mode:?} auto={auto_distance} t={t}");
                        assert!((eye - l.center).length() > l.radius * 1.1, "{what}: camera at the light");
                        for inst in &s.solids {
                            // segment eye -> solid centre must miss the light sphere
                            let seg = inst.center - eye;
                            let u = ((l.center - eye).dot(seg) / seg.length_squared()).clamp(0.0, 1.0);
                            let gap = (eye + seg * u - l.center).length();
                            assert!(gap > l.radius, "{what}: light in front of a solid");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn instance_sdf_respects_transform() {
        let s = Scene::demo(&AnimParams::default());
        let inst = &s.solids[1]; // cube
        assert!((inst.sdf(inst.center) + inst.poly.inradius() * inst.scale).abs() < 1e-4);
        assert!(inst.sdf(inst.center + Vec3A::new(0.0, 0.0, inst.scale * 1.01)) > 0.0);
    }
}
