//! Monochrome path tracer: Lambertian surfaces, cosine-weighted bounces,
//! next-event estimation toward the spherical area light.

use glam::Vec3A;
use std::f32::consts::TAU;

use super::lut::ShadowGrid;
use super::march::{ALL_SOLIDS, GROUND, Hit, floor_radius, march_in, march_solids_in, normal};
use crate::scene::Scene;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceSettings {
    /// Indirect bounces after the primary hit.
    pub bounces: u32,
    pub max_steps: u32,
    /// The floor's radius round the scene's centre, measured across it.
    pub max_dist: f32,
    /// Floor points further out than this (across the floor) get the cheap
    /// far shading ([`far_ground_radiance`]): no light sampling, shadows or
    /// bounces.
    pub detail_dist: f32,
    /// The angle one pixel (character cell) spans, radians: how large a
    /// patch of floor it covers, so patterns fade to their mean tone where
    /// they would alias. The renderer sets it from the camera each frame.
    pub pixel_angle: f32,
}

impl Default for TraceSettings {
    fn default() -> Self {
        Self { bounces: 2, max_steps: 64, max_dist: 50.0, detail_dist: f32::INFINITY, pixel_angle: DEFAULT_PIXEL_ANGLE }
    }
}

/// [`TraceSettings::pixel_angle`] before the renderer knows better: about
/// 50 degrees over 120 columns.
pub const DEFAULT_PIXEL_ANGLE: f32 = 0.0075;
/// The outer share of the floor's radius over which it fades into the
/// background, so its edge is soft instead of a ragged circle.
const RIM: f32 = 0.15;
/// Grazing rays see a floor patch stretched by 1 / |cos|; capped here.
const GRAZING_MIN: f32 = 0.02;

/// How large a patch of floor a pixel covers where ray `rd` hits it `t`
/// away: the pixel's width there, stretched along the floor at grazing angles.
#[inline(always)]
pub fn footprint(t: f32, rd: Vec3A, pixel_angle: f32) -> f32 {
    t * pixel_angle / (-rd.z).max(GRAZING_MIN)
}

/// Extended floor: how much further out the ground is drawn.
pub const FAR_GROUND_FACTOR: f32 = 8.0;

impl TraceSettings {
    /// With the extended floor: the ground drawn `FAR_GROUND_FACTOR` times
    /// further out, everything past the normal distance shaded cheaply.
    pub fn with_far_ground(self, on: bool) -> Self {
        if on { Self { max_dist: self.max_dist * FAR_GROUND_FACTOR, detail_dist: self.max_dist, ..self } } else { self }
    }
}

/// Radiance of a far ground point `p` with albedo `albedo`, without any
/// sampling: the light's direct contribution in closed form (an unoccluded
/// sphere light over a horizontal surface; solids' shadows never reach this
/// far) plus the averaged sky term the path tracer uses for truncated
/// paths. The pattern has already faded to its mean tone out here.
#[inline]
pub fn far_ground_radiance(scene: &Scene, p: Vec3A, albedo: f32) -> f32 {
    let l = &scene.light;
    let to = l.center - p;
    let d = to.length().max(l.radius * 1.001);
    let cos_max = (1.0 - (l.radius / d).powi(2)).max(0.0).sqrt();
    let ndl = (to.z / d).max(0.0);
    albedo * (l.emission * ndl * 2.0 * (1.0 - cos_max) + scene.sky * AMBIENT_OCCLUSION)
}

/// Offset along the normal for secondary-ray origins.
const RAY_OFFSET: f32 = 2e-3;
/// Rough sky-visibility factor for paths truncated at the last bounce.
const AMBIENT_OCCLUSION: f32 = 0.5;

/// Orthonormal basis around unit `n` (Duff et al. 2017), pivoting on the
/// −y axis (the frame the sampling tables were made in).
#[inline(always)]
pub fn onb(n: Vec3A) -> (Vec3A, Vec3A) {
    let s = 1.0f32.copysign(-n.y);
    let a = -1.0 / (s - n.y);
    let b = n.x * n.z * a;
    (Vec3A::new(1.0 + s * n.x * n.x * a, s * n.x, s * b), Vec3A::new(b, n.z, s + n.z * n.z * a))
}

#[inline(always)]
pub fn cosine_hemisphere(n: Vec3A, rng: &mut fastrand::Rng) -> Vec3A {
    let u1 = rng.f32();
    let u2 = rng.f32();
    let r = u1.sqrt();
    let (s, c) = (TAU * u2).sin_cos();
    let (t, b) = onb(n);
    (t * (r * c) + b * (r * s) + n * (1.0 - u1).max(0.0).sqrt()).normalize()
}

/// Lookup-table size (entries) for the sampling tables below.
const LUT_BITS: u32 = 12;
const LUT_N: usize = 1 << LUT_BITS;

/// Sine and cosine of `TAU · k / LUT_N`.
fn sin_cos_table() -> &'static [(f32, f32)] {
    static T: std::sync::OnceLock<Box<[(f32, f32)]>> = std::sync::OnceLock::new();
    T.get_or_init(|| (0..LUT_N).map(|k| (TAU * (k as f32 + 0.5) / LUT_N as f32).sin_cos()).collect())
}

/// Cosine-weighted directions around +Z (the ground's normal), stratified
/// on a 64 × 64 grid of the two sample coordinates.
fn ground_bounce_table() -> &'static [Vec3A] {
    static T: std::sync::OnceLock<Box<[Vec3A]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let side = 1usize << (LUT_BITS / 2);
        (0..LUT_N)
            .map(|k| {
                let u1 = ((k / side) as f32 + 0.5) / side as f32;
                let u2 = ((k % side) as f32 + 0.5) / side as f32;
                let r = u1.sqrt();
                let (s, c) = (TAU * u2).sin_cos();
                Vec3A::new(r * c, -(r * s), (1.0 - u1).max(0.0).sqrt()).normalize()
            })
            .collect()
    })
}

/// A random table index.
#[inline(always)]
fn lut_index(rng: &mut fastrand::Rng) -> usize {
    (rng.u32(..) >> (32 - LUT_BITS)) as usize
}

/// Cosine-weighted bounce off the ground (normal +Z), from the table.
#[inline(always)]
fn ground_bounce(rng: &mut fastrand::Rng) -> Vec3A {
    ground_bounce_table()[lut_index(rng)]
}

/// Distance along the ray to the light sphere, if hit.
#[inline(always)]
fn light_hit(scene: &Scene, ro: Vec3A, rd: Vec3A) -> Option<f32> {
    let l = &scene.light;
    let oc = ro - l.center;
    let b = oc.dot(rd);
    let c = oc.length_squared() - l.radius * l.radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    (t > 0.0).then_some(t)
}

/// Direct light at `p` (normal `n`) by solid-angle sampling of the light
/// sphere. Returns radiance / albedo.
#[inline]
fn direct_light(scene: &Scene, p: Vec3A, n: Vec3A, rng: &mut fastrand::Rng, s: &TraceSettings, occluders: u64) -> f32 {
    let l = &scene.light;
    let to_c = l.center - p;
    let d2 = to_c.length_squared();
    let d = d2.sqrt();
    let w = to_c / d;
    let sin2_max = (l.radius * l.radius / d2).min(1.0);
    let cos_max = (1.0 - sin2_max).sqrt();

    let u1 = rng.f32();
    let cos_t = 1.0 - u1 * (1.0 - cos_max);
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let (sp, cp) = sin_cos_table()[lut_index(rng)];
    let (t, b) = onb(w);
    let dir = t * (cp * sin_t) + b * (sp * sin_t) + w * cos_t;

    let ndl = n.dot(dir);
    if ndl <= 0.0 {
        return 0.0;
    }
    // Light is above the ground, so only solids can occlude it.
    if march_solids_in(scene, p, dir, 0.0, d - l.radius, s.max_steps, occluders).is_some() {
        return 0.0;
    }
    // (albedo/π)·L·cosθ / pdf, pdf = 1 / (2π(1 − cosθmax)); albedo applied by caller
    l.emission * ndl * 2.0 * (1.0 - cos_max)
}

/// Radiance along one camera ray.
pub fn radiance(scene: &Scene, ro: Vec3A, rd: Vec3A, rng: &mut fastrand::Rng, s: &TraceSettings) -> f32 {
    radiance_primary(scene, ro, rd, rng, s, ALL_SOLIDS, None).0
}

/// Radiance along one camera ray, plus the primary hit (`None` when the ray
/// escapes or sees the light first). `primary_solids`: the solids the camera
/// ray can possibly hit (from the screen-tile table; [`ALL_SOLIDS`] if
/// unknown).
pub fn radiance_primary(
    scene: &Scene,
    mut ro: Vec3A,
    mut rd: Vec3A,
    rng: &mut fastrand::Rng,
    s: &TraceSettings,
    primary_solids: u64,
    shadows: Option<&ShadowGrid>,
) -> (f32, Option<Hit>) {
    let (ro0, rd0) = (ro, rd);
    let mut acc = 0.0f32;
    let mut throughput = 1.0f32;
    let mut primary = None;

    for depth in 0..=s.bounces {
        let allowed = if depth == 0 { primary_solids } else { ALL_SOLIDS };
        let hit = march_in(scene, ro, rd, s.max_dist, s.max_steps, allowed);

        if depth == 0 {
            if let Some(tl) = light_hit(scene, ro, rd)
                && hit.is_none_or(|h| tl < h.t)
            {
                return (scene.light.emission, None);
            }
            primary = hit;
        }

        let Some(h) = hit else {
            acc += throughput * if depth == 0 { scene.background } else { scene.sky };
            break;
        };

        let p = ro + rd * h.t;
        let ground_albedo = || scene.ground.albedo(p, footprint(h.t, rd, s.pixel_angle));
        if h.id == GROUND && floor_radius(scene, p) > s.detail_dist {
            // Extended floor beyond the normal distance: averaged, closed form.
            acc += throughput * far_ground_radiance(scene, p, ground_albedo());
            break;
        }
        let n = normal(scene, h.id, p);
        let albedo = if h.id == GROUND { ground_albedo() } else { scene.solids[h.id as usize].albedo };
        let origin = p + n * RAY_OFFSET;

        // On the ground, only solids whose shadow can reach this point occlude.
        let occluders = match shadows {
            Some(g) if h.id == GROUND => g.at(p.x, p.y),
            _ => ALL_SOLIDS,
        };
        acc += throughput * albedo * direct_light(scene, origin, n, rng, s, occluders);

        if depth == s.bounces {
            // Truncated path: approximate the remaining sky light.
            acc += throughput * albedo * scene.sky * AMBIENT_OCCLUSION;
            break;
        }
        throughput *= albedo;
        ro = origin;
        rd = if h.id == GROUND { ground_bounce(rng) } else { cosine_hemisphere(n, rng) };
    }
    // The floor's outer rim fades into the background.
    if let Some(h) = primary.filter(|h| h.id == GROUND) {
        let r = floor_radius(scene, ro0 + rd0 * h.t) / s.max_dist.max(1e-3);
        let k = ((r - (1.0 - RIM)) / RIM).clamp(0.0, 1.0);
        if k > 0.0 {
            let k = k * k * (3.0 - 2.0 * k);
            acc += (scene.background - acc) * k;
        }
    }
    (acc, primary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn far_ground_matches_the_sampled_average() {
        use crate::scene::{AnimParams, Scene};
        let s = Scene::demo(&AnimParams::default());
        let ts = TraceSettings { bounces: 0, ..TraceSettings::default() };
        let mut rng = fastrand::Rng::with_seed(11);
        // a ground point well away from the solids: sampled direct light + sky term
        for p in [Vec3A::new(30.0, 25.0, 0.0), Vec3A::new(-60.0, -40.0, 0.0)] {
            let origin = p + Vec3A::Z * RAY_OFFSET;
            let n = 20_000;
            let sampled: f32 = (0..n).map(|_| direct_light(&s, origin, Vec3A::Z, &mut rng, &ts, ALL_SOLIDS)).sum::<f32>() / n as f32;
            let exact = far_ground_radiance(&s, p, 1.0) - s.sky * AMBIENT_OCCLUSION;
            assert!((sampled - exact).abs() < 0.03 * exact.max(1e-3) + 1e-4, "{p:?}: sampled {sampled} vs closed form {exact}");
        }
        let far = TraceSettings::default().with_far_ground(true);
        assert_eq!((far.max_dist, far.detail_dist), (50.0 * FAR_GROUND_FACTOR, 50.0));
        assert_eq!(TraceSettings::default().with_far_ground(false), TraceSettings::default());
    }

    #[test]
    fn floor_rim_fades_and_footprint_grows_at_grazing_angles() {
        use crate::scene::{AnimParams, Scene};
        let mut s = Scene::demo(&AnimParams::default());
        s.background = 0.0;
        let ts = TraceSettings { bounces: 0, ..TraceSettings::default() };
        let c = s.centroid;
        let eye = Vec3A::new(c.x, c.y, 200.0);
        let mut rng = fastrand::Rng::with_seed(3);
        let at = |r: f32| (Vec3A::new(c.x + r, c.y, 0.0) - eye).normalize();
        let mean = |rd: Vec3A, rng: &mut fastrand::Rng| (0..200).map(|_| radiance(&s, eye, rd, rng, &ts)).sum::<f32>() / 200.0;
        let inner = mean(at(30.0), &mut rng);
        let rim = mean(at(0.97 * ts.max_dist), &mut rng);
        assert!(inner > 0.01, "the floor is lit from far above: {inner}");
        assert!(rim < 0.3 * inner, "the rim fades to the background: {rim} vs {inner}");
        // the same distance seen edge-on covers far more floor
        let down = footprint(10.0, -Vec3A::Z, 0.01);
        let grazing = footprint(10.0, Vec3A::new(1.0, 0.0, -0.05).normalize(), 0.01);
        assert!((down - 0.1).abs() < 1e-5 && grazing > 15.0 * down, "{down} {grazing}");
    }

    #[test]
    fn onb_is_orthonormal() {
        for n in [Vec3A::Z, Vec3A::Y, Vec3A::new(0.3, -0.8, -0.5).normalize(), -Vec3A::Y, -Vec3A::Z] {
            let (t, b) = onb(n);
            assert!(t.dot(n).abs() < 1e-5 && b.dot(n).abs() < 1e-5 && t.dot(b).abs() < 1e-5);
            assert!((t.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn hemisphere_samples_face_normal() {
        let mut rng = fastrand::Rng::with_seed(7);
        let n = Vec3A::new(0.2, 0.1, 0.9).normalize();
        let mut mean_cos = 0.0;
        for _ in 0..20_000 {
            let d = cosine_hemisphere(n, &mut rng);
            let c = d.dot(n);
            assert!(c >= -1e-5);
            mean_cos += c;
        }
        // E[cosθ] for cosine-weighted sampling = 2/3
        assert!((mean_cos / 20_000.0 - 2.0 / 3.0).abs() < 0.01);
    }
}
