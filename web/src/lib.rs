//! Browser build of platomon.
//!
//! A plain C ABI (no wasm-bindgen): the page writes a TOML config into the
//! input buffer, calls [`configure`], then calls [`frame`] on every animation
//! frame it decides to draw and reads the text back from the output buffer.
//! Sensors come from [`platomon::sim`], sampled at `poll_hz` like the real
//! collector, so smoothing, peaks, activity and fps behave as on a machine.
//! With a colour gradient on, [`colors_ptr`] / [`colors_len`] give one
//! `0xRRGGBB` (or [`color::NONE`]) per character of the output text.
//!
//! Sound halo ([`sound`]): as in the `platomon-scr` screensaver, a second
//! wireframe of each solid follows the sound level and spectrum. Between full
//! frames the page calls [`halo_frame`] at the display rate, which redraws
//! only the halo over the last rendered picture.

pub mod sound;

use std::cell::RefCell;

use glam::Vec2;
use platomon::ascii::Ramp;
use platomon::aspect::{self, AspectMode, Viewport};
use platomon::clock::{activity_to_fps, slew_limit};
use platomon::color::{self, ColorDepth, ColorMap};
use platomon::config::{Config, SlotSpec};
use platomon::metrics::{MetricsState, activity_score};
use platomon::overlay::{Halo, OverlayOptions, Overlays};
use platomon::render::{QualityGovernor, RenderSettings, Renderer, TraceSettings};
use platomon::scene::camera::CameraParams;
use platomon::scene::{AnimParams, Ground, Scene, SlotLayout};
use platomon::sim::{Profile, Sim};
use platomon::ui;
use sound::{SoundHalo, Source};

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
    cmap: Option<ColorMap>,
    levels: Vec<f32>,
    text: String,
    /// One colour per char of `text` (newlines included); empty when monochrome.
    colors: Vec<u32>,
    /// The picture's glyphs (`vp.w × vp.h`) and display colours.
    pic_chars: Vec<char>,
    pic_colors: Vec<u32>,
    sound: SoundHalo,
    /// Layout of the last full frame, for halo-only frames.
    last: Option<LastFrame>,
    /// Sound time already advanced by halo-only frames since the last full
    /// frame (so a full frame's `dt` is not counted twice).
    halo_time: f32,
}

/// What a halo-only frame reuses from the last full frame.
struct LastFrame {
    vp: Viewport,
    cols: usize,
    area_h: usize,
    stats: Option<String>,
}

/// Lay the picture out in the `cols × area_h` area (letterbox margins) plus
/// the stats line. `colors` tracks `text` char for char when `pic_colors` is
/// non-empty, and is left empty otherwise.
fn assemble(
    text: &mut String,
    colors: &mut Vec<u32>,
    (chars, pic_colors): (&[char], &[u32]),
    vp: &Viewport,
    (cols, area_h): (usize, usize),
    stats: Option<&str>,
) {
    let colored = !pic_colors.is_empty();
    text.clear();
    colors.clear();
    let none = |colors: &mut Vec<u32>, n: usize| {
        if colored {
            colors.extend(std::iter::repeat_n(color::NONE, n));
        }
    };
    let blank = |text: &mut String, n: usize| text.extend(std::iter::repeat_n(' ', n));
    for _ in 0..vp.y {
        blank(text, cols);
        text.push('\n');
        none(colors, cols + 1);
    }
    for y in 0..vp.h {
        blank(text, vp.x);
        none(colors, vp.x);
        let (lo, hi) = ((y * vp.w).min(chars.len()), ((y + 1) * vp.w).min(chars.len()));
        text.extend(&chars[lo..hi]);
        blank(text, vp.w - (hi - lo));
        text.push('\n');
        if colored {
            let row = pic_colors.get(lo..hi.min(pic_colors.len())).unwrap_or(&[]);
            colors.extend_from_slice(row);
            none(colors, vp.w - row.len() + 1);
        }
    }
    for _ in (vp.y + vp.h)..area_h {
        blank(text, cols);
        text.push('\n');
        none(colors, cols + 1);
    }
    if let Some(line) = stats {
        none(colors, line.chars().count());
        text.push_str(line);
    }
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
        focus_zoom: c.focus_zoom,
        focus_tau: c.focus_tau,
    }
}

/// What the page should paint for `c`: the colour itself, or for 256-colour
/// output the xterm palette entry the terminal would show.
fn display_color(c: u32, depth: ColorDepth) -> u32 {
    if c == color::NONE || depth == ColorDepth::True {
        return c;
    }
    xterm256_rgb(color::ansi256(c))
}

/// RGB of an xterm-256 palette index (16..=255; lower ones are not produced).
fn xterm256_rgb(i: u8) -> u32 {
    let i = i as u32;
    if i >= 232 {
        let v = 8 + (i - 232) * 10;
        return (v << 16) | (v << 8) | v;
    }
    const LEVELS: [u32; 6] = [0, 95, 135, 175, 215, 255];
    let k = i.saturating_sub(16);
    (LEVELS[(k / 36) as usize % 6] << 16) | (LEVELS[(k / 6) as usize % 6] << 8) | LEVELS[(k % 6) as usize]
}

/// Parse, sanitise and validate a config.
pub fn load(toml: &str) -> Result<(Config, Vec<SlotSpec>, Option<f32>), String> {
    let mut cfg = Config::from_toml(toml)?;
    cfg.sanitize();
    let slots = cfg.resolve_slots()?;
    let ratio = aspect::parse_aspect(&cfg.aspect)?;
    cfg.color_map().map_err(|e| format!("color: {e}"))?;
    Ok((cfg, slots, ratio))
}

impl App {
    /// Build everything from scratch (scene, camera, metrics history).
    pub fn new(cfg: Config, slots: Vec<SlotSpec>, ratio: Option<f32>, sim: Sim) -> Self {
        let anim = anim_params(&cfg);
        let layout: Vec<SlotLayout> = slots.iter().map(|s| SlotLayout { solid: s.solid, ring: s.ring }).collect();
        let scene =
            Scene::with_layout(&layout, &anim, &camera_params(&cfg), cfg.layout).with_ground(&cfg.ground_params());
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
            cmap: cfg.color_map().ok().flatten(),
            levels: Vec::new(),
            text: String::new(),
            colors: Vec::new(),
            pic_chars: Vec::new(),
            pic_colors: Vec::new(),
            sound: SoundHalo::default(),
            last: None,
            halo_time: 0.0,
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
        c.color = new.color.clone();
        c.color_by = new.color_by;
        c.color_depth = new.color_depth;
        c.color_reverse = new.color_reverse;
        self.cmap = c.color_map().ok().flatten();
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
        // Halo-only frames wait for a full frame with the new settings.
        self.last = None;
    }

    /// Sound halo settings (see [`set_sound`]).
    pub fn set_sound(&mut self, source: Source, wide: bool, bands: bool) {
        self.sound.configure(source, wide, bands, &self.scene);
    }

    pub fn sound(&self) -> &SoundHalo {
        &self.sound
    }

    pub fn sound_mut(&mut self) -> &mut SoundHalo {
        &mut self.sound
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
        self.levels.clear();
        self.levels.extend(channels.iter().map(|c| if c.available { c.level } else { 0.0 }));

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
        let halo = self.sound.next((dt - std::mem::take(&mut self.halo_time)).max(0.0));
        let ramp = Ramp::new(&self.cfg.ramp);
        let legend = self.cfg.legend_params();
        let opt = OverlayOptions {
            wireframe: self.cfg.wireframe,
            legend: self.cfg.legend.then_some(&legend),
            labels: &self.labels,
            char_aspect: self.cfg.char_aspect,
            ramp: &ramp,
            exposure: self.cfg.exposure,
            color: self.cmap.as_ref(),
            levels: &self.levels,
            halo,
            halo_factors: self.sound.factors(),
        };
        self.pic_chars.clear();
        self.pic_colors.clear();
        if let Some(view) = self.renderer.view() {
            let cells = self.overlays.compose(&view, vp.w, vp.h, &self.scene, &opt);
            self.pic_chars.extend_from_slice(cells.chars);
            if let (Some(c), Some(m)) = (cells.colors, self.cmap.as_ref()) {
                self.pic_colors.extend(c.iter().map(|&c| display_color(c, m.depth)));
            }
        }

        let stats = self.cfg.show_stats.then(|| {
            let q = self.governor.level();
            let extra = format!(
                "{} | sim {} {:.0}%",
                if q > 0 && self.cfg.adaptive_quality { format!(" | q-{q}") } else { String::new() },
                self.sim.profile.id(),
                100.0 * self.sim.envelope(self.sim.time())
            );
            ui::stats_line_fit(&readings, achieved, self.target, &extra, cols)
        });
        assemble(
            &mut self.text,
            &mut self.colors,
            (&self.pic_chars, &self.pic_colors),
            &vp,
            (cols, area_h),
            stats.as_deref(),
        );
        self.last = self.renderer.view().is_some().then_some(LastFrame { vp, cols, area_h, stats });
        &self.text
    }

    /// Between full frames: advance the sound by `dt` and redraw only the
    /// halo over the last full frame. `None` when there is nothing to redraw
    /// (halo off, no full frame yet, or no cell changed).
    pub fn halo_frame(&mut self, dt: f32) -> Option<&str> {
        if self.sound.source == Source::Off {
            return None;
        }
        let last = self.last.as_ref()?;
        let dt = if dt.is_finite() { dt.clamp(0.0, 0.25) } else { 0.0 };
        self.halo_time += dt;
        let halo: Option<Halo> = self.sound.next(dt);
        let ramp = Ramp::new(&self.cfg.ramp);
        let legend = self.cfg.legend_params();
        let opt = OverlayOptions {
            wireframe: self.cfg.wireframe,
            legend: self.cfg.legend.then_some(&legend),
            labels: &self.labels,
            char_aspect: self.cfg.char_aspect,
            ramp: &ramp,
            exposure: self.cfg.exposure,
            color: self.cmap.as_ref(),
            levels: &self.levels,
            halo,
            halo_factors: self.sound.factors(),
        };
        let view = self.renderer.view()?;
        let vp = last.vp;
        let u = self.overlays.update_halo(&view, vp.w, vp.h, &self.scene, &opt);
        if u.dirty.is_empty() {
            return None;
        }
        self.pic_chars.clear();
        self.pic_chars.extend_from_slice(u.chars);
        self.pic_colors.clear();
        if let (Some(c), Some(m)) = (u.colors, self.cmap.as_ref()) {
            self.pic_colors.extend(c.iter().map(|&c| display_color(c, m.depth)));
        }
        assemble(
            &mut self.text,
            &mut self.colors,
            (&self.pic_chars, &self.pic_colors),
            &vp,
            (last.cols, last.area_h),
            last.stats.as_deref(),
        );
        Some(&self.text)
    }

    /// Per-character colours of the last frame's text; empty when monochrome.
    pub fn colors(&self) -> &[u32] {
        &self.colors
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
    colors: Vec<u32>,
    message: Vec<u8>,
    sim: Sim,
    /// Sound settings, kept across restarts: source, wide swing, bands.
    sound: (Source, bool, bool),
    /// Samples written by the page (see [`audio_buffer`]).
    audio: Vec<f32>,
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
                        let mut app = App::new(cfg, slots, ratio, sim);
                        let (src, wide, bands) = h.sound;
                        app.set_sound(src, wide, bands);
                        h.app = Some(app);
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

/// Sound halo: `source` 0 = off, 1 = simulated music, 2 = samples from the
/// page ([`audio_push`]); `wide` 1 = size swing ±30% instead of ±5%;
/// `bands` 1 = each solid follows its own frequency band, 0 = overall level.
#[unsafe(no_mangle)]
pub extern "C" fn set_sound(source: u32, wide: u32, bands: u32) {
    HOST.with_borrow_mut(|h| {
        h.sound = (Source::from_index(source), wide != 0, bands != 0);
        if let Some(app) = h.app.as_mut() {
            app.set_sound(h.sound.0, h.sound.1, h.sound.2);
        }
    })
}

/// Reserve room for `len` f32 samples for [`audio_push`]; returns the address.
#[unsafe(no_mangle)]
pub extern "C" fn audio_buffer(len: usize) -> *mut f32 {
    HOST.with_borrow_mut(|h| {
        h.audio.clear();
        h.audio.resize(len.min(1 << 16), 0.0);
        h.audio.as_mut_ptr()
    })
}

/// Hand the page's latest `len` mono samples at `rate` Hz to the halo
/// (source 2 only). Call before each frame / halo frame.
#[unsafe(no_mangle)]
pub extern "C" fn audio_push(len: usize, rate: f32) {
    HOST.with_borrow_mut(|h| {
        let Host { app, audio, .. } = h;
        if let Some(app) = app.as_mut() {
            let n = len.min(audio.len());
            app.sound_mut().set_window(&audio[..n], rate);
        }
    })
}

/// Halo-only frame between full frames; returns the text length in the
/// output buffer, or 0 when the screen does not need repainting.
#[unsafe(no_mangle)]
pub extern "C" fn halo_frame(dt: f32) -> usize {
    HOST.with_borrow_mut(|h| {
        let Host { app, output, colors, .. } = h;
        let Some(app) = app.as_mut() else { return 0 };
        let Some(s) = app.halo_frame(dt) else { return 0 };
        output.clear();
        output.extend_from_slice(s.as_bytes());
        colors.clear();
        colors.extend_from_slice(app.colors());
        output.len()
    })
}

/// Sound meter: 0 = raw peak, 1 = bass-weighted spectrum level, 2 = halo
/// size (x radius), 3 = halo luminosity; 0 when off.
#[unsafe(no_mangle)]
pub extern "C" fn sound_info(what: u32) -> f32 {
    HOST.with_borrow(|h| {
        let Some(s) = h.app.as_ref().map(App::sound) else { return 0.0 };
        let halo = s.last();
        match what {
            0 => s.meter.0,
            1 => s.meter.1,
            2 => halo.map_or(0.0, |x| x.factor),
            3 => halo.map_or(0.0, |x| x.lum),
            _ => 0.0,
        }
    })
}

/// Advance and render one frame; returns the text length in the output buffer.
#[unsafe(no_mangle)]
pub extern "C" fn frame(dt: f32, cols: u32, rows: u32, achieved: f32, last_render: f32) -> usize {
    HOST.with_borrow_mut(|h| {
        let Host { app, output, colors, .. } = h;
        output.clear();
        colors.clear();
        if let Some(app) = app.as_mut() {
            let s = app.frame(dt, cols.min(1000) as usize, rows.min(500) as usize, achieved, last_render);
            output.extend_from_slice(s.as_bytes());
            colors.extend_from_slice(app.colors());
        }
        output.len()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn output_ptr() -> *const u8 {
    HOST.with_borrow(|h| h.output.as_ptr())
}

/// Per-character colours of the last frame (`0xRRGGBB`, `0xFFFFFFFF` = default).
#[unsafe(no_mangle)]
pub extern "C" fn colors_ptr() -> *const u32 {
    HOST.with_borrow(|h| h.colors.as_ptr())
}

/// Number of entries at [`colors_ptr`]: the text's char count, or 0 when monochrome.
#[unsafe(no_mangle)]
pub extern "C" fn colors_len() -> usize {
    HOST.with_borrow(|h| h.colors.len())
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
        assert!(load("color = \"plaid\"").unwrap_err().contains("unknown gradient"));
        assert!(load("layout = \"cube\"").is_err());
    }

    #[test]
    fn colours_track_the_text_char_for_char() {
        let mut a = app("show_stats = true\nlegend = true\naspect = \"square\"\n");
        a.frame(0.1, 70, 20, 10.0, 0.01);
        assert!(a.colors().is_empty(), "monochrome by default");
        for toml in [
            "color = \"inferno\"\nshow_stats = true\naspect = \"square\"\nlegend = true\n",
            "color = \"traffic\"\ncolor_by = \"load\"\ncolor_depth = \"256\"\nshow_stats = true\n",
        ] {
            let (c, _, r) = load(toml).unwrap();
            a.apply_live(&c, r);
            for _ in 0..3 {
                let t = a.frame(0.1, 70, 20, 10.0, 0.01).to_string();
                assert_eq!(a.colors().len(), t.chars().count(), "{toml}");
                assert!(a.colors().iter().any(|&c| c != color::NONE), "{toml}");
                for (ch, &c) in t.chars().zip(a.colors()) {
                    if ch == '\n' {
                        assert_eq!(c, color::NONE);
                    }
                }
            }
        }
    }

    #[test]
    fn sphere_layout_and_focus_camera_render() {
        let mut a = app("layout = \"sphere\"\ncamera = \"focus\"\nfocus_zoom = 0.9\nshow = [\"cpu\", \"ram\", \"gpu\"]\n");
        for _ in 0..30 {
            let t = a.frame(1.0 / 30.0, 80, 24, 30.0, 0.01);
            assert_eq!(t.lines().count(), 24);
        }
        assert_eq!(a.cfg.layout, platomon::scene::Layout::Sphere);
    }

    #[test]
    fn sound_halo_draws_and_halo_frames_redraw_it() {
        let mut a = app("show_stats = true\nlegend = true\n");
        a.set_sound(Source::Off, false, true);
        let quiet = a.frame(0.1, 100, 30, 10.0, 0.01).to_string();
        assert!(a.halo_frame(0.016).is_none(), "halo off: nothing to redraw");
        a.set_sound(Source::Sim, true, true);
        let mut changed = 0;
        let mut sizes = Vec::new();
        for i in 0..40 {
            let t = if i % 8 == 0 {
                a.frame(0.1, 100, 30, 10.0, 0.01).to_string()
            } else {
                match a.halo_frame(1.0 / 60.0) {
                    Some(t) => {
                        changed += 1;
                        t.to_string()
                    }
                    None => continue,
                }
            };
            assert_eq!(t.split('\n').count(), 30, "same layout");
            let bad: Vec<_> = t.lines().enumerate().filter(|(_, l)| l.chars().count() != 100).map(|(i, l)| (i, l.chars().count())).collect();
            assert!(bad.iter().all(|&(i, _)| i == 29), "{bad:?}");
            sizes.push(a.sound().last().unwrap().factor);
        }
        assert!(changed > 0, "halo-only frames repaint");
        assert_ne!(a.frame(0.1, 100, 30, 10.0, 0.01), quiet, "halo visible");
        let (lo, hi) = sizes.iter().fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(lo >= 0.9 && hi <= 1.5 && hi > lo, "wide swing range {lo}..{hi}");
        assert_eq!(a.sound().factors().len(), 5, "one size per solid by frequency");
    }

    #[test]
    fn host_samples_drive_the_halo() {
        let mut a = app("");
        a.set_sound(Source::Host, false, false);
        let rate = 48_000.0;
        let tone: Vec<f32> = (0..4096).map(|i| 0.8 * (std::f32::consts::TAU * 60.0 * i as f32 / rate).sin()).collect();
        for _ in 0..20 {
            a.sound_mut().set_window(&tone, rate);
            a.frame(0.05, 60, 20, 20.0, 0.01);
        }
        let s = a.sound();
        assert!(s.meter.0 > 0.7, "peak {:?}", s.meter);
        assert!(s.meter.1 > 0.3, "60 Hz counts as bass {:?}", s.meter);
        assert!(s.last().unwrap().factor > 1.2 && s.factors().is_empty());
    }

    #[test]
    fn palette_256_rgb() {
        assert_eq!(xterm256_rgb(16), 0x000000);
        assert_eq!(xterm256_rgb(231), 0xffffff);
        assert_eq!(xterm256_rgb(196), 0xff0000);
        assert_eq!(xterm256_rgb(244), 0x808080);
        assert_eq!(display_color(0xff0101, ColorDepth::Ansi256), 0xff0000);
        assert_eq!(display_color(0xff0101, ColorDepth::True), 0xff0101);
    }
}
