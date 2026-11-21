//! Per-frame lookup tables that let rays skip solids they cannot meet.
//!
//! * [`tile_masks`]: for each screen tile, the solids whose bounding sphere
//!   can show in it. Camera rays only test those; most background pixels
//!   test none.
//! * [`ShadowGrid`]: for each cell of a grid on the ground, the solids whose
//!   shadow (from any point of the area light) can fall on it. Shadow rays
//!   from the ground only test those; most of the ground tests none.
//!
//! Both are conservative: a solid is listed wherever it could possibly
//! matter, so the picture is the same as testing every solid.

use glam::Vec3A;

use super::Projection;
use super::march::ALL_SOLIDS;
use crate::scene::Scene;

/// Screen-tile size for [`tile_masks`], pixels.
pub const TILE: usize = 8;

/// Padding added to every solid's bounding sphere: the march's hit
/// tolerance (1e-3 · t, t ≤ 50) plus slack.
const PAD: f32 = 0.06;

/// Fill `out` with, per screen tile, the solids (bit `i` = solid `i`) whose
/// padded bounding sphere projects into it; returns tiles per row.
pub fn tile_masks(scene: &Scene, proj: &Projection, w: usize, h: usize, out: &mut Vec<u64>) -> usize {
    let (tx, ty) = (w.div_ceil(TILE), h.div_ceil(TILE));
    out.clear();
    out.resize(tx * ty, 0);
    if scene.solids.len() > 64 {
        out.fill(ALL_SOLIDS);
        return tx;
    }
    for (i, s) in scene.solids.iter().enumerate() {
        let r = s.scale + PAD;
        let mut corners = [(0.0f32, 0.0f32); 8];
        let mut visible = true;
        // The sphere projects inside the projection of its bounding cube.
        for (k, c) in corners.iter_mut().enumerate() {
            let o = Vec3A::new(
                if k & 1 == 0 { -r } else { r },
                if k & 2 == 0 { -r } else { r },
                if k & 4 == 0 { -r } else { r },
            );
            match proj.project(s.center + o) {
                Some(p) => *c = p,
                None => visible = false,
            }
        }
        let bit = 1u64 << i;
        if !visible {
            // Straddles the camera plane: could be anywhere on screen.
            out.iter_mut().for_each(|m| *m |= bit);
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in &corners {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        // One pixel of margin for rays anywhere inside a pixel.
        let (x0, y0, x1, y1) = (x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0);
        if x1 < 0.0 || y1 < 0.0 || x0 >= w as f32 || y0 >= h as f32 {
            continue;
        }
        let cx = |v: f32| ((v.max(0.0) as usize) / TILE).min(tx - 1);
        let cy = |v: f32| ((v.max(0.0) as usize) / TILE).min(ty - 1);
        for row in cy(y0)..=cy(y1) {
            for m in &mut out[row * tx + cx(x0)..=row * tx + cx(x1)] {
                *m |= bit;
            }
        }
    }
    tx
}

/// Which solids can shadow each part of the ground.
#[derive(Debug, Clone, Default)]
pub struct ShadowGrid {
    /// Every solid everywhere (no table: too many solids, or the light is
    /// not above some solid).
    all: bool,
    x0: f32,
    z0: f32,
    inv_cell: (f32, f32),
    cells: Vec<u64>,
}

impl ShadowGrid {
    /// Cells per axis.
    pub const N: usize = 128;

    /// The solids that can shadow ground point (x, z) (bit `i` = solid `i`).
    #[inline(always)]
    pub fn at(&self, x: f32, z: f32) -> u64 {
        if self.all {
            return ALL_SOLIDS;
        }
        let u = (x - self.x0) * self.inv_cell.0;
        let v = (z - self.z0) * self.inv_cell.1;
        // Outside every shadow footprint (or NaN): nothing can shadow it.
        if !(u >= 0.0 && v >= 0.0 && u < Self::N as f32 && v < Self::N as f32) {
            return 0;
        }
        self.cells[v as usize * Self::N + u as usize]
    }

    /// Rebuild for the scene's current solids and light.
    pub fn build(&mut self, scene: &Scene) {
        self.all = false;
        self.cells.clear();
        self.cells.resize(Self::N * Self::N, 0);
        if scene.solids.len() > 64 {
            self.all = true;
            return;
        }
        let (l, rl) = (scene.light.center, scene.light.radius);
        // Each solid's shadow footprint on y = 0 as an (x0, z0, x1, z1) box.
        let mut boxes = Vec::with_capacity(scene.solids.len());
        for s in &scene.solids {
            let r = s.scale + PAD;
            let top = s.center.y + r;
            // Shadow rays leave just above the ground and only the part of a
            // solid inside the slab y ≥ 0 can block them.
            let bottom = (s.center.y - r).max(0.0);
            let lo_light = l.y - rl;
            if lo_light <= top + 1e-3 {
                // The light is not above this solid: no bounded footprint.
                self.all = true;
                return;
            }
            // A ray from light point x through solid point y meets the ground
            // at x + k (y − x), k = x.y / (x.y − y.y) ≥ 1; horizontally that is
            // within (k − 1)·R + k·r of L + k (c − L).
            let k_min = (l.y + rl) / (l.y + rl - bottom);
            let k_max = lo_light / (lo_light - top);
            let rad = (k_max - 1.0) * rl + k_max * r + 0.05;
            let (a, b) = (l + (s.center - l) * k_min, l + (s.center - l) * k_max);
            boxes.push((a.x.min(b.x) - rad, a.z.min(b.z) - rad, a.x.max(b.x) + rad, a.z.max(b.z) + rad));
        }
        let Some(&first) = boxes.first() else { return };
        let (x0, z0, x1, z1) = boxes
            .iter()
            .fold(first, |u, b| (u.0.min(b.0), u.1.min(b.1), u.2.max(b.2), u.3.max(b.3)));
        if !(x0.is_finite() && z0.is_finite() && x1.is_finite() && z1.is_finite()) {
            self.all = true;
            return;
        }
        let n = Self::N as f32;
        self.x0 = x0;
        self.z0 = z0;
        self.inv_cell = (n / (x1 - x0).max(1e-3), n / (z1 - z0).max(1e-3));
        let cell = |v: f32, o: f32, inv: f32| (((v - o) * inv).max(0.0) as usize).min(Self::N - 1);
        for (i, b) in boxes.iter().enumerate() {
            let bit = 1u64 << i;
            let (u0, u1) = (cell(b.0, x0, self.inv_cell.0), cell(b.2, x0, self.inv_cell.0));
            let (v0, v1) = (cell(b.1, z0, self.inv_cell.1), cell(b.3, z0, self.inv_cell.1));
            for v in v0..=v1 {
                for m in &mut self.cells[v * Self::N + u0..=v * Self::N + u1] {
                    *m |= bit;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::march::{march_solids, march_solids_in};
    use crate::render::{RenderSettings, Renderer};
    use crate::scene::{AnimParams, Scene};

    /// Every ray the full test finds a solid on, the table allows that solid.
    #[test]
    fn tables_never_drop_a_hit() {
        let scene = Scene::demo(&AnimParams::default());
        let (w, h) = (120, 48);
        let rs = RenderSettings::default();
        let proj = Projection::new(scene.camera, w, h, &rs);
        let mut tiles = Vec::new();
        let tx = tile_masks(&scene, &proj, w, h, &mut tiles);
        let mut rng = fastrand::Rng::with_seed(3);
        for _ in 0..20_000 {
            let (x, y) = (rng.f32() * w as f32, rng.f32() * h as f32);
            let rd = proj.ray(x, y);
            let m = tiles[(y as usize / TILE) * tx + x as usize / TILE];
            let full = march_solids(&scene, scene.camera.pos, rd, 0.0, 50.0, 64);
            if let Some(hit) = full {
                assert!(m & (1 << hit.id) != 0, "pixel ({x}, {y}) hits solid {} not in its tile", hit.id);
            }
            assert_eq!(march_solids_in(&scene, scene.camera.pos, rd, 0.0, 50.0, 64, m), full);
        }
        let mut g = ShadowGrid::default();
        g.build(&scene);
        assert!(!g.all, "the demo light is above every solid");
        let l = scene.light;
        let mut shadowed = 0;
        for _ in 0..50_000 {
            let p = Vec3A::new(rng.f32() * 30.0 - 15.0, 2e-3, rng.f32() * 30.0 - 15.0) + scene.centroid * Vec3A::new(1.0, 0.0, 1.0);
            // a random point on the light
            let q = l.center + Vec3A::new(rng.f32() - 0.5, rng.f32() - 0.5, rng.f32() - 0.5).normalize_or_zero() * (l.radius * rng.f32());
            let to = q - p;
            let (d, dir) = (to.length(), to.normalize());
            let full = march_solids(&scene, p, dir, 0.0, d, 64);
            let m = g.at(p.x, p.z);
            if let Some(hit) = full {
                shadowed += 1;
                assert!(m & (1 << hit.id) != 0, "ground ({}, {}) shadowed by solid {} not in its cell", p.x, p.z, hit.id);
            }
            assert_eq!(march_solids_in(&scene, p, dir, 0.0, d, 64, m), full);
        }
        assert!(shadowed > 100, "the test met some shadows: {shadowed}");
        // most of the ground needs no shadow test at all
        let free = (0..10_000)
            .filter(|_| {
                let p = scene.centroid + Vec3A::new(rng.f32() * 40.0 - 20.0, 0.0, rng.f32() * 40.0 - 20.0);
                g.at(p.x, p.z) == 0
            })
            .count();
        assert!(free > 5_000, "{free} of 10000 ground points skip the shadow test");
        // and rendering through the tables still works
        let mut r = Renderer::new();
        assert!(r.render(&scene, w, h, &rs).iter().all(|v| v.is_finite()));
    }
}
