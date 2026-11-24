//! Sphere tracing and normals.
//!
//! The ground plane (z = 0) is intersected analytically; the solids are only
//! marched inside the z-slab (height band) that contains them, which keeps
//! grazing ground rays and sky rays cheap.

use glam::Vec3A;

use crate::scene::Scene;

pub const GROUND: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub t: f32,
    /// Solid index, or [`GROUND`] for the ground plane.
    pub id: u8,
}

/// Use the bounding-sphere distance instead of the exact SDF this far out
/// (as a fraction of circumradius). The sphere bound is conservative.
const SPHERE_SWITCH: f32 = 0.25;

/// Distance to the nearest solid and its index.
#[inline(always)]
pub fn solids_sdf(scene: &Scene, p: Vec3A) -> (f32, u8) {
    let mut best = f32::INFINITY;
    let mut id = 0u8;
    for (i, s) in scene.solids.iter().enumerate() {
        let dc = p - s.center;
        let ds = dc.length() - s.scale;
        if ds >= best {
            continue;
        }
        let d = if ds > SPHERE_SWITCH * s.scale { ds } else { s.sdf_local(dc) };
        if d < best {
            best = d;
            id = i as u8;
        }
    }
    (best, id)
}

/// Parametric interval where `o + t·d` has height (z) in [lo, hi].
#[inline(always)]
fn slab(oz: f32, dz: f32, lo: f32, hi: f32) -> (f32, f32) {
    if dz.abs() < 1e-8 {
        if oz >= lo && oz <= hi { (f32::NEG_INFINITY, f32::INFINITY) } else { (f32::INFINITY, f32::NEG_INFINITY) }
    } else {
        let inv = 1.0 / dz;
        let a = (lo - oz) * inv;
        let b = (hi - oz) * inv;
        (a.min(b), a.max(b))
    }
}

/// Distance to the nearest solid among those set in `mask` (bit `i` = solid
/// `i`; solids past 63 are always included).
#[inline(always)]
fn solids_sdf_masked(scene: &Scene, p: Vec3A, mask: u64) -> (f32, u8) {
    let mut best = f32::INFINITY;
    let mut id = 0u8;
    for (i, s) in scene.solids.iter().enumerate() {
        if i < 64 && mask & (1 << i) == 0 {
            continue;
        }
        let dc = p - s.center;
        let ds = dc.length() - s.scale;
        if ds >= best {
            continue;
        }
        let d = if ds > SPHERE_SWITCH * s.scale { ds } else { s.sdf_local(dc) };
        if d < best {
            best = d;
            id = i as u8;
        }
    }
    (best, id)
}

/// The solids whose bounding spheres the ray meets within `t..end`, as a
/// mask, and the stretch of the ray inside any of them (`None` = no solid
/// can be hit). Spheres are padded by the hit tolerance so culling never
/// drops a hit the march would find.
/// Only solids in `allowed` are considered (see [`ALL_SOLIDS`]).
#[inline(always)]
fn candidates(scene: &Scene, ro: Vec3A, rd: Vec3A, t: f32, end: f32, allowed: u64) -> Option<(u64, f32, f32)> {
    let pad = 1e-3 * end.min(1e4) + 1e-3;
    let (mut mask, mut lo, mut hi) = (0u64, f32::INFINITY, f32::NEG_INFINITY);
    for (i, s) in scene.solids.iter().enumerate() {
        if i < 64 && allowed & (1 << i) == 0 {
            continue;
        }
        let r = s.scale + pad;
        let oc = ro - s.center;
        let b = oc.dot(rd);
        let disc = b * b - (oc.length_squared() - r * r);
        if disc < 0.0 {
            continue;
        }
        let q = disc.sqrt();
        let (t0, t1) = (-b - q, -b + q);
        if t1 < t || t0 > end {
            continue;
        }
        if i < 64 {
            mask |= 1 << i;
        } else {
            mask = u64::MAX; // too many to track: keep them all
        }
        lo = lo.min(t0);
        hi = hi.max(t1);
    }
    (mask != 0).then_some((mask, lo.max(t), hi.min(end)))
}

/// Solid mask meaning "every solid" (solids past 63 are always included).
pub const ALL_SOLIDS: u64 = u64::MAX;

/// March solids only, over `tmin..tmax`.
#[inline]
pub fn march_solids(scene: &Scene, ro: Vec3A, rd: Vec3A, tmin: f32, tmax: f32, max_steps: u32) -> Option<Hit> {
    march_solids_in(scene, ro, rd, tmin, tmax, max_steps, ALL_SOLIDS)
}

/// [`march_solids`] restricted to the solids in `allowed` (bit `i` = solid
/// `i`): the caller knows from a lookup table that the others cannot be hit.
#[inline]
pub fn march_solids_in(
    scene: &Scene,
    ro: Vec3A,
    rd: Vec3A,
    tmin: f32,
    tmax: f32,
    max_steps: u32,
    allowed: u64,
) -> Option<Hit> {
    if allowed == 0 && scene.solids.len() <= 64 {
        return None;
    }
    let (s0, s1) = slab(ro.z, rd.z, 0.0, scene.slab_top);
    let t = tmin.max(s0);
    let end = tmax.min(s1);
    if t > end {
        return None;
    }
    // Skip empty space: only the solids this ray can reach, only where it can
    // reach them.
    let (mask, mut t, end) = candidates(scene, ro, rd, t, end, allowed)?;
    if mask == u64::MAX {
        return march_range(scene, ro, rd, t, end, max_steps);
    }
    for _ in 0..max_steps {
        let (d, id) = solids_sdf_masked(scene, ro + rd * t, mask);
        if d < (1e-3 * t).max(1e-4) {
            return Some(Hit { t, id });
        }
        t += d;
        if t > end {
            return None;
        }
    }
    None
}

/// Plain sphere tracing over `t..end` against every solid.
#[inline]
fn march_range(scene: &Scene, ro: Vec3A, rd: Vec3A, mut t: f32, end: f32, max_steps: u32) -> Option<Hit> {
    for _ in 0..max_steps {
        let (d, id) = solids_sdf(scene, ro + rd * t);
        if d < (1e-3 * t).max(1e-4) {
            return Some(Hit { t, id });
        }
        t += d;
        if t > end {
            return None;
        }
    }
    None
}

/// Nearest hit with anything (solids or ground) within `max_dist`.
#[inline]
pub fn march(scene: &Scene, ro: Vec3A, rd: Vec3A, max_dist: f32, max_steps: u32) -> Option<Hit> {
    march_in(scene, ro, rd, max_dist, max_steps, ALL_SOLIDS)
}

/// [`march`] restricted to the solids in `allowed`.
///
/// `max_dist` is the floor's reach: its radius round the scene's centre,
/// measured across the floor, so a camera high above it sees the same floor
/// as one near it. Solids are found however far the camera is (past the far
/// side of the scene's bounds nothing can be hit).
#[inline]
pub fn march_in(scene: &Scene, ro: Vec3A, rd: Vec3A, max_dist: f32, max_steps: u32, allowed: u64) -> Option<Hit> {
    let t_plane =
        if scene.ground.enabled && rd.z < -1e-7 && ro.z > 0.0 { -ro.z / rd.z } else { f32::INFINITY };
    let beyond = (ro - scene.centroid).length() + scene.extent + SCENE_MARGIN;
    let limit = beyond.min(t_plane);
    if let Some(h) = march_solids_in(scene, ro, rd, 0.0, limit, max_steps, allowed) {
        return Some(h);
    }
    (t_plane.is_finite() && floor_radius(scene, ro + rd * t_plane) <= max_dist).then_some(Hit { t: t_plane, id: GROUND })
}

/// Room past the scene's bounds (solids' halos, rounding).
const SCENE_MARGIN: f32 = 2.0;

/// How far floor point `p` is from the scene's centre, across the floor.
#[inline(always)]
pub fn floor_radius(scene: &Scene, p: Vec3A) -> f32 {
    (p.x - scene.centroid.x).hypot(p.y - scene.centroid.y)
}

/// Tetrahedral 4-tap gradient of one solid's SDF.
#[inline]
pub fn normal(scene: &Scene, id: u8, p: Vec3A) -> Vec3A {
    if id == GROUND {
        return Vec3A::Z;
    }
    let s = &scene.solids[id as usize];
    let h = 5e-4 * s.scale.max(0.1);
    let k0 = Vec3A::new(1.0, 1.0, -1.0);
    let k1 = Vec3A::new(-1.0, -1.0, -1.0);
    let k2 = Vec3A::new(-1.0, 1.0, 1.0);
    let k3 = Vec3A::new(1.0, -1.0, 1.0);
    let n = k0 * s.sdf(p + k0 * h) + k1 * s.sdf(p + k1 * h) + k2 * s.sdf(p + k2 * h) + k3 * s.sdf(p + k3 * h);
    n.try_normalize().unwrap_or(Vec3A::Z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{AnimParams, Scene};

    #[test]
    fn ray_down_hits_ground_ray_up_misses() {
        let s = Scene::demo(&AnimParams::default());
        let far = Vec3A::new(20.0, -20.0, 5.0);
        let h = march(&s, far, -Vec3A::Z, 50.0, 64).unwrap();
        assert_eq!(h.id, GROUND);
        assert!((h.t - 5.0).abs() < 1e-4);
        assert!(march(&s, far, Vec3A::Z, 50.0, 64).is_none());
    }

    #[test]
    fn a_camera_far_above_still_sees_the_floor_and_the_solids() {
        let s = Scene::demo(&AnimParams::default());
        let c = s.centroid;
        // 300 up, far past the 50-unit floor reach along the ray: the floor
        // is still there out to 50 across it, and not beyond
        let eye = Vec3A::new(c.x, c.y, 300.0);
        let at = |dx: f32| (Vec3A::new(c.x + dx, c.y, 0.0) - eye).normalize();
        assert_eq!(march(&s, eye, at(40.0), 50.0, 64).map(|h| h.id), Some(GROUND));
        assert!(march(&s, eye, at(60.0), 50.0, 64).is_none(), "past the floor's edge");
        // and every solid, seen from far away along a slant
        for (i, inst) in s.solids.iter().enumerate() {
            let ro = inst.center + Vec3A::new(120.0, 90.0, 160.0);
            let h = march(&s, ro, (inst.center - ro).normalize(), 50.0, 128).expect("solid seen from far");
            assert_eq!(h.id as usize, i);
        }
    }

    #[test]
    fn luminosity_zero_removes_ground() {
        use crate::scene::GroundParams;
        let s = Scene::demo(&AnimParams::default()).with_ground(&GroundParams { lum: 0.0, ..Default::default() });
        assert!(march(&s, Vec3A::new(20.0, -20.0, 5.0), -Vec3A::Z, 50.0, 64).is_none(), "falls through");
        // solids are still there
        let c = s.solids[0].center;
        assert_eq!(march(&s, c + Vec3A::new(0.0, 0.0, 6.0), -Vec3A::Z, 50.0, 64).unwrap().id, 0);
    }

    #[test]
    fn ray_at_solid_hits_it() {
        let s = Scene::demo(&AnimParams::default());
        for (i, inst) in s.solids.iter().enumerate() {
            let ro = inst.center + Vec3A::new(0.0, 0.0, 6.0);
            let h = march(&s, ro, -Vec3A::Z, 50.0, 64).unwrap();
            assert_eq!(h.id as usize, i);
            let p = ro - Vec3A::Z * h.t;
            assert!(inst.sdf(p).abs() < 1e-2);
            let n = normal(&s, h.id, p);
            assert!((n.length() - 1.0).abs() < 1e-4);
            assert!(n.z > 0.0, "top-facing normal for solid {i}: {n:?}");
        }
    }
}
