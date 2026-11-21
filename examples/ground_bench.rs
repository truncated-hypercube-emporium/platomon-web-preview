//! Background-plane benchmark: frame render time with the ground off, solid
//! and in each pattern (static and rotating), on a large grid.
//!
//! `cargo run --release --example ground_bench [frames]`

use platomon::metrics::Channel;
use platomon::render::{NO_HIT, RenderSettings, Renderer, TraceSettings};
use platomon::scene::ground::{GroundParams, Pattern, Spin};
use platomon::scene::sdf::SolidKind;
use platomon::scene::{AnimParams, CameraParams, Scene, SlotLayout};
use std::time::Instant;

fn scene(n: usize, bg: &GroundParams) -> Scene {
    let anim = AnimParams::default();
    let slots: Vec<SlotLayout> =
        (0..n).map(|i| SlotLayout { solid: SolidKind::ALL[i % SolidKind::ALL.len()], ring: None }).collect();
    let mut s = Scene::new(&slots, &anim, &CameraParams::default()).with_ground(bg);
    let ch = vec![Channel { available: true, level: 0.6, spin_y: 0.6, spin_x: 0.2, activity: 0.6 }; n];
    s.update(&ch, &anim, 3.0);
    s
}

/// Median and mean ms per frame over `frames` renders (after warm-up).
fn time(r: &mut Renderer, s: &Scene, w: usize, h: usize, rs: &RenderSettings, frames: usize) -> (f64, f64) {
    for _ in 0..5 {
        r.render(s, w, h, rs);
    }
    let mut t: Vec<f64> = (0..frames)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(r.render(s, w, h, rs).len());
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    let mean = t.iter().sum::<f64>() / t.len() as f64;
    t.sort_by(f64::total_cmp);
    (t[t.len() / 2], mean)
}

fn main() {
    let frames: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(60);
    // 1 spp, 1 bounce on 640×240 cells (a WQHD screen in a small font).
    let rs = RenderSettings { spp: 1, trace: TraceSettings { bounces: 1, ..TraceSettings::default() }, ..RenderSettings::default() };
    let cases: Vec<(&str, GroundParams)> = {
        let p = |pattern, rotate| GroundParams { lum: 5.0, pattern, contrast: 0.6, scale: 1.5, rotate, rotate_period: 30.0, ..GroundParams::default() };
        let mut v = vec![("off (lum 0)", GroundParams { lum: 0.0, ..GroundParams::default() }), ("solid", p(Pattern::Solid, Spin::Off))];
        for pat in [Pattern::Checker, Pattern::Diamond, Pattern::Brick, Pattern::Triangles, Pattern::Hex, Pattern::Radial, Pattern::Quasicrystal] {
            v.push((pat.id(), p(pat, Spin::Off)));
        }
        v.push(("checker, rotating + sway", GroundParams { sway: 0.5, ..p(Pattern::Checker, Spin::Cw) }));
        v
    };
    // Rounds alternate over all cases and the best median counts, so load
    // from other programs hits every case alike.
    let rounds = 3;
    for (n, w, h) in [(5usize, 640usize, 240usize), (16, 640, 240)] {
        println!("--- {n} solids, {w}x{h}, 1 spp, 1 bounce, best of {rounds} x {frames} frames");
        let mut r = Renderer::new();
        let scenes: Vec<Scene> = cases.iter().map(|(_, bg)| scene(n, bg)).collect();
        let mut best = vec![(f64::MAX, f64::MAX); cases.len()];
        for _ in 0..rounds {
            for (b, s) in best.iter_mut().zip(&scenes) {
                let (med, mean) = time(&mut r, s, w, h, &rs, frames);
                if med < b.0 {
                    *b = (med, mean);
                }
            }
        }
        for ((name, _), (s, &(med, mean))) in cases.iter().zip(scenes.iter().zip(&best)) {
            r.render(s, w, h, &rs);
            let ground = r.view().map_or(0, |v| v.ids.iter().filter(|&&i| i == platomon::render::march::GROUND).count());
            let sky = r.view().map_or(0, |v| v.ids.iter().filter(|&&i| i == NO_HIT).count());
            println!(
                "{name:<26} median {med:6.2} ms  mean {mean:6.2} ms  ground {:4.1}%  sky {:4.1}%",
                100.0 * ground as f64 / (w * h) as f64,
                100.0 * sky as f64 / (w * h) as f64
            );
        }
    }
}
