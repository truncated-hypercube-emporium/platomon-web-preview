//! Browser build of platomon.
//!
//! A plain C ABI (no wasm-bindgen): the page writes a TOML config into the
//! input buffer, calls [`configure`], then calls [`frame`] on every animation
//! frame it decides to draw and reads the text back from the output buffer.
//! Sensors come from [`platomon::sim`], sampled at `poll_hz` like the real
//! collector, so smoothing, peaks, activity and fps behave as on a machine.

use std::cell::RefCell;

use glam::Vec2;
use platomon::ascii::{self, Ramp};
use platomon::aspect::{self, AspectMode};
use platomon::clock::{activity_to_fps, slew_limit};
use platomon::config::{Config, SlotSpec};
use platomon::metrics::{MetricsState, activity_score};
use platomon::overlay::{OverlayOptions, Overlays};
use platomon::render::{QualityGovernor, RenderSettings, Renderer, TraceSettings};
use platomon::scene::camera::CameraParams;
use platomon::scene::{AnimParams, Ground, Scene, SlotLayout};
use platomon::sim::{Profile, Sim};
use platomon::ui;

/// Everything that lives between frames.
pub struct App {
    cfg: Config,
    slots: Vec<SlotSpec>,
    ratio: Option<f32>,
    anim: AnimParams,
    scene: Scene,
    metrics: MetricsState,
    weights: Vec<f32>,
    renderer: Renderer,
    governor: QualityGovernor,
    overlays: Overlays,
    sim: Sim,
    since_sample: f64,
    target: f32,
    activity: f32,
    labels: Vec<String>,
    text: String,
}

fn anim_params(c: &Config) -> AnimParams {
    AnimParams { min_scale: c.min_scale, max_scale: c.max_scale, idle_spin: c.idle_spin, max_spin: c.max_spin }
}

fn camera_params(c: &Config) -> CameraParams {
    CameraParams {
        mode: c.camera,
        period: c.orbit_period,
        dir: c.orbit_dir,
        radius: c.orbit_radius,
        height: c.orbit_height,
        bob: c.orbit_bob,
        seed: c.seed,
    }
}

/// Parse, sanitise and validate a config.
pub fn load(toml: &str) -> Result<(Config, Vec<SlotSpec>, Option<f32>), String> {
    let mut cfg = Config::from_toml(toml)?;
    cfg.sanitize();
    let slots = cfg.resolve_slots()?;
    let ratio = aspect::parse_aspect(&cfg.aspect)?;
    Ok((cfg, slots, ratio))
}

impl App {
    /// Build everything from scratch (scene, camera, metrics history).
    pub fn new(cfg: Config, slots: Vec<SlotSpec>, ratio: Option<f32>, sim: Sim) -> Self {
        let anim = anim_params(&cfg);
        let layout: Vec<SlotLayout> = slots.iter().map(|s| SlotLayout { solid: s.solid, ring: s.ring }).collect();
        let scene = Scene::new(&layout, &anim, &camera_params(&cfg)).with_ground(&cfg.ground_params());
        let specs: Vec<_> = slots.iter().map(|s| s.metric.clone()).collect();
        let metrics = MetricsState::new(&cfg, &specs);
        let weights = metrics.weights(&cfg.weights);
        let overlays = Overlays::new(&scene);
        let target = if cfg.fps_invert { cfg.fps_max } else { cfg.fps_min };
        Self {
            anim,
            scene,
            metrics,
            weights,
            renderer: Renderer::new(),
            governor: QualityGovernor::default(),
            overlays,
            sim,
            since_sample: f64::INFINITY, // sample on the first frame
            target,
            activity: 0.0,
            labels: Vec::new(),
            text: String::new(),
            cfg,
            slots,
            ratio,
        }
    }

    /// Apply the settings that can change without a restart; the rest of
    /// `new` is ignored until the next restart.
    pub fn apply_live(&mut self, new: &Config, ratio: Option<f32>) {
        let c = &mut self.cfg;
        c.fps_min = new.fps_min;
        c.fps_max = new.fps_max;
        c.fps_curve = new.fps_curve;
        c.fps_slew = new.fps_slew;
        c.fps_invert = new.fps_invert;
        c.spp = new.spp;
        c.bounces = new.bounces;
        c.max_steps = new.max_steps;
        c.max_dist = new.max_dist;
        c.adaptive_quality = new.adaptive_quality;
        c.ramp = new.ramp.clone();
        c.char_aspect = new.char_aspect;
        c.exposure = new.exposure;
        c.show_stats = new.show_stats;
        c.aspect = new.aspect.clone();
        c.aspect_mode = new.aspect_mode;
        c.legend = new.legend;
        c.legend_size = new.legend_size;
        c.legend_opacity = new.legend_opacity;
        c.legend_pos = new.legend_pos;
        c.wireframe = new.wireframe;
        c.min_scale = new.min_scale;
        c.max_scale = new.max_scale;
        c.idle_spin = new.idle_spin;
        c.max_spin = new.max_spin;
        c.weights = new.weights;
        self.weights = self.metrics.weights(&c.weights);
        self.ratio = ratio;
        self.anim = anim_params(c);
        let ground = new.ground_params();
        if ground != c.ground_params() {
            let pivot = Vec2::new(self.scene.centroid.x, self.scene.centroid.z);
            self.scene.ground = Ground::new(&ground, pivot);
            c.bg_lum = new.bg_lum;
            c.bg_pattern = new.bg_pattern;
            c.bg_contrast = new.bg_contrast;
            c.bg_scale = new.bg_scale;
            c.bg_scroll = new.bg_scroll;
            c.bg_sway = new.bg_sway;
            c.bg_sway_period = new.bg_sway_period;
            c.bg_rotate = new.bg_rotate;
            c.bg_rotate_period = new.bg_rotate_period;
        }
        self.target = self.target.clamp(c.fps_min, c.fps_max);
    }

    pub fn target_fps(&self) -> f32 {
        self.target
    }

    /// Advance by `dt` seconds and render a `cols × rows` frame into `text`.
    /// `achieved` is the page's measured fps, `last_render` the previous
    /// frame's render time in seconds (drives the quality governor).
    pub fn frame(&mut self, dt: f32, cols: usize, rows: usize, achieved: f32, last_render: f32) -> &str {
        let dt = if dt.is_finite() { dt.clamp(0.0, 0.25) } else { 0.0 };
        // Sensors at poll_hz, like the collector thread.
        self.since_sample += dt as f64;
        let interval = 1.0 / self.cfg.poll_hz as f64;
        if self.since_sample >= interval {
            let elapsed = if self.since_sample.is_finite() { self.since_sample } else { interval };
            let raw = self.sim.sample(elapsed);
            self.metrics.ingest(&raw);
            self.since_sample = 0.0;
        }
        self.metrics.tick(dt);
        let channels = self.metrics.channels();
        self.activity = activity_score(&channels, &self.weights);
        let a = if self.cfg.fps_invert { 1.0 - self.activity } else { self.activity };
        let want = activity_to_fps(a, self.cfg.fps_min, self.cfg.fps_max, self.cfg.fps_curve);
        self.target = slew_limit(self.target, want, self.cfg.fps_slew, dt).clamp(self.cfg.fps_min, self.cfg.fps_max);
        self.scene.update(&channels, &self.anim, dt);

        let area_h = rows.saturating_sub(self.cfg.show_stats as usize);
        let vp = aspect::viewport(cols, area_h, self.ratio, self.cfg.aspect_mode, self.cfg.char_aspect);
        let base = RenderSettings {
            spp: self.cfg.spp,
            trace: TraceSettings { bounces: self.cfg.bounces, max_steps: self.cfg.max_steps, max_dist: self.cfg.max_dist },
            char_aspect: self.cfg.char_aspect,
            fit_aspect: self.ratio,
        };
        if dt > 0.0 {
            self.governor.update(last_render, self.target, dt);
        }
        let rs = if self.cfg.adaptive_quality { self.governor.apply(base) } else { base };
        self.renderer.render(&self.scene, vp.w, vp.h, &rs);

        let readings = self.metrics.readings();
        if self.cfg.legend {
            self.labels.clear();
            let block = self.cfg.legend_size >= 3;
            self.labels.extend(readings.iter().map(|r| if block { ui::short_label(r) } else { ui::segment(r) }));
        }
        let ramp = Ramp::new(&self.cfg.ramp);
        let legend = self.cfg.legend_params();
        let opt = OverlayOptions {
            wireframe: self.cfg.wireframe,
            legend: self.cfg.legend.then_some(&legend),
            labels: &self.labels,
            char_aspect: self.cfg.char_aspect,
            ramp: &ramp,
            exposure: self.cfg.exposure,
        };
        let mut pic = String::new();
        if let Some(view) = self.renderer.view() {
            let cells = self.overlays.compose(&view, vp.w, vp.h, &self.scene, &opt);
            ascii::compose_plain(cells, vp.w, vp.h, &mut pic);
        }

        // Centre the picture in the cols × area_h area (letterbox margins).
        self.text.clear();
        let pad_l = " ".repeat(vp.x);
        let blank = " ".repeat(cols);
        for _ in 0..vp.y {
            self.text.push_str(&blank);
            self.text.push('\n');
        }
        for line in pic.lines() {
            self.text.push_str(&pad_l);
            self.text.push_str(line);
            self.text.push('\n');
        }
        for _ in (vp.y + vp.h)..area_h {
            self.text.push_str(&blank);
            self.text.push('\n');
        }
        if self.cfg.show_stats {
            let q = self.governor.level();
            let extra = format!(
                "{} | sim {} {:.0}%",
                if q > 0 && self.cfg.adaptive_quality { format!(" | q-{q}") } else { String::new() },
                self.sim.profile.id(),
                100.0 * self.sim.envelope(self.sim.time())
            );
            self.text.push_str(&ui::stats_line_fit(&readings, achieved, self.target, &extra, cols));
        }
        &self.text
    }

    pub fn sim_mut(&mut self) -> &mut Sim {
        &mut self.sim
    }

    pub fn quality(&self) -> u8 {
        self.governor.level()
    }

    pub fn activity(&self) -> f32 {
        self.activity
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn aspect_mode(&self) -> AspectMode {
        self.cfg.aspect_mode
    }
}

// ------------------------------------------------------------------ C ABI

#[derive(Default)]
struct Host {
    app: Option<App>,
    input: Vec<u8>,
    output: Vec<u8>,
    message: Vec<u8>,
    sim: Sim,
}

thread_local! {
    static HOST: RefCell<Host> = RefCell::new(Host::default());
}

/// Reserve `len` bytes for the page to write a string into; returns its address.
#[unsafe(no_mangle)]
pub extern "C" fn input_buffer(len: usize) -> *mut u8 {
    HOST.with_borrow_mut(|h| {
        h.input.clear();
        h.input.resize(len, 0);
        h.input.as_mut_ptr()
    })
}

/// Apply the TOML config in the input buffer (`len` bytes). `restart` = 1
/// rebuilds everything; 0 applies only the live settings. Returns 1 on
/// success, 0 on error (the message buffer holds the reason).
#[unsafe(no_mangle)]
pub extern "C" fn configure(len: usize, restart: u32) -> u32 {
    HOST.with_borrow_mut(|h| {
        let n = len.min(h.input.len());
        let text = String::from_utf8_lossy(&h.input[..n]).into_owned();
        match load(&text) {
            Ok((cfg, slots, ratio)) => {
                match h.app.as_mut() {
                    Some(app) if restart == 0 => app.apply_live(&cfg, ratio),
                    _ => {
                        let sim = h.sim.clone();
                        h.app = Some(App::new(cfg, slots, ratio, sim));
                    }
                }
                h.message.clear();
                1
            }
            Err(e) => {
                h.message = e.into_bytes();
                0
            }
        }
    })
}

/// Simulated load: profile index into [`Profile::ALL`], speed (sim s per s),
/// manual level 0..1, seed. Changing profile or seed restarts the sequence.
#[unsafe(no_mangle)]
pub extern "C" fn set_sim(profile: u32, speed: f64, manual: f64, seed: u32) {
    HOST.with_borrow_mut(|h| {
        let p = Profile::ALL.get(profile as usize).copied().unwrap_or_default();
        let manual = if manual.is_finite() { manual.clamp(0.0, 1.0) } else { 0.5 };
        let speed = if speed.is_finite() { speed.clamp(0.0, 100.0) } else { 1.0 };
        let mut next = h.app.as_ref().map_or(&h.sim, |a| &a.sim).clone();
        if next.profile != p || next.seed() != seed as u64 {
            next = Sim::new(p, speed, seed as u64);
        }
        // Same profile and seed: keep the timeline, change speed / level only.
        next.speed = speed;
        next.manual = manual;
        if let Some(app) = h.app.as_mut() {
            *app.sim_mut() = next.clone();
        }
        h.sim = next;
    })
}

/// Advance and render one frame; returns the text length in the output buffer.
#[unsafe(no_mangle)]
pub extern "C" fn frame(dt: f32, cols: u32, rows: u32, achieved: f32, last_render: f32) -> usize {
    HOST.with_borrow_mut(|h| {
        let Host { app, output, .. } = h;
        output.clear();
        if let Some(app) = app.as_mut() {
            let s = app.frame(dt, cols.min(1000) as usize, rows.min(500) as usize, achieved, last_render);
            output.extend_from_slice(s.as_bytes());
        }
        output.len()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn output_ptr() -> *const u8 {
    HOST.with_borrow(|h| h.output.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn message_ptr() -> *const u8 {
    HOST.with_borrow(|h| h.message.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn message_len() -> usize {
    HOST.with_borrow(|h| h.message.len())
}

#[unsafe(no_mangle)]
pub extern "C" fn target_fps() -> f32 {
    HOST.with_borrow(|h| h.app.as_ref().map_or(30.0, App::target_fps))
}

#[unsafe(no_mangle)]
pub extern "C" fn quality() -> u32 {
    HOST.with_borrow(|h| h.app.as_ref().map_or(0, |a| a.quality() as u32))
}

#[unsafe(no_mangle)]
pub extern "C" fn activity() -> f32 {
    HOST.with_borrow(|h| h.app.as_ref().map_or(0.0, App::activity))
}

/// Current simulated overall load, 0..1.
#[unsafe(no_mangle)]
pub extern "C" fn sim_load() -> f64 {
    HOST.with_borrow(|h| {
        let s = h.app.as_ref().map_or(&h.sim, |a| &a.sim);
        s.envelope(s.time())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(toml: &str) -> App {
        let (c, s, r) = load(toml).unwrap();
        App::new(c, s, r, Sim::new(Profile::Cycle, 4.0, 1))
    }

    #[test]
    fn renders_frames_of_the_requested_size() {
        let mut a = app("show_stats = true\nlegend = true\nwireframe = true\n");
        for _ in 0..20 {
            let t = a.frame(1.0 / 30.0, 80, 24, 30.0, 0.01).to_string();
            let lines: Vec<&str> = t.split('\n').collect();
            assert_eq!(lines.len(), 24, "{t}");
            assert!(lines[..23].iter().all(|l| l.chars().count() == 80));
            assert!(lines[23].contains("sim cycle"), "{}", lines[23]);
        }
    }

    #[test]
    fn live_changes_keep_the_scene_and_restart_rebuilds() {
        let mut a = app("");
        a.frame(0.1, 60, 20, 10.0, 0.01);
        let angles = a.scene.angles.clone();
        let (c, _, r) = load("fps_invert = true\nbg_lum = 0.0\nshow = [\"cpu\"]\n").unwrap();
        a.apply_live(&c, r);
        assert!(a.cfg.fps_invert && !a.scene.ground.enabled);
        assert_eq!(a.slot_count(), 5, "show needs a restart");
        assert_eq!(a.scene.angles, angles);
    }

    #[test]
    fn inverted_fps_rises_when_idle() {
        let (c, s, r) = load("fps_invert = true\nfps_slew = 1000.0\n").unwrap();
        let mut a = App::new(c, s, r, Sim::new(Profile::Idle, 1.0, 1));
        for _ in 0..60 {
            a.frame(0.1, 40, 12, 10.0, 0.001);
        }
        assert!(a.target_fps() > 100.0, "target {}", a.target_fps());
    }

    #[test]
    fn bad_config_reports_error() {
        assert!(load("show = [\"cpu-temp\"]").unwrap_err().contains("unknown metric"));
        assert!(load("nope = 1").is_err());
    }
}
