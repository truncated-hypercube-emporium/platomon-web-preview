//! Sound halo for the browser build: the screensaver's halo logic
//! (`platomon-scr`'s `SoundHalo`) fed from a simulated music track or from
//! samples the page captures (microphone, or a shared tab / system audio via
//! Web Audio). Everything after the samples is the shared code in
//! [`platomon::audio`] and [`platomon::overlay`]: peak envelope with auto-gain
//! for the halo size, a 2048-point FFT into 12 log bands, bass-weighted
//! brightness, and per-solid bands along the frequency gradient.

use platomon::audio::{Analyzer, Envelope, FFT_N, SolidLevels, bass_weighted, frequency_gradient};
use platomon::overlay::{HALO_SWINGS, Halo, halo_factor, halo_lum, halo_reach};
use platomon::scene::Scene;

/// Where the samples come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Source {
    /// No halo.
    Off,
    /// [`music`]: a deterministic synthetic track, no permissions needed.
    #[default]
    Sim,
    /// Samples pushed by the page ([`SoundHalo::set_window`]).
    Host,
}

impl Source {
    pub fn from_index(i: u32) -> Self {
        match i {
            0 => Self::Off,
            2 => Self::Host,
            _ => Self::Sim,
        }
    }
}

/// Sample rate of the simulated track.
pub const SIM_RATE: f32 = 48_000.0;

/// Halo state between frames.
pub struct SoundHalo {
    pub source: Source,
    /// Halo size swing, a share of the radius either way (see
    /// [`HALO_SWINGS`]).
    pub swing: f32,
    /// Each solid's halo follows its own band (simple solids bass, complex
    /// ones treble); off = one size from the overall level.
    pub bands_on: bool,
    size: Envelope,
    bass: Envelope,
    solid: SolidLevels,
    /// Per-solid band levels for `color_by = "sound"` (always by frequency).
    colour: SolidLevels,
    colour_levels: Vec<f32>,
    analyzer: Analyzer,
    /// The latest `FFT_N` samples (mono) and their rate.
    window: Vec<f32>,
    rate: f32,
    /// Highest |sample| since the last update.
    peak: f32,
    /// Simulated track position, seconds.
    sim_t: f64,
    factors: Vec<f32>,
    last: Option<Halo>,
    /// Latest raw readings, for the page's meter: peak, bass-weighted.
    pub meter: (f32, f32),
}

impl Default for SoundHalo {
    fn default() -> Self {
        Self {
            source: Source::Sim,
            swing: HALO_SWINGS[0],
            bands_on: true,
            size: Envelope::new(),
            bass: Envelope::new(),
            solid: SolidLevels::new(Vec::new()),
            colour: SolidLevels::new(Vec::new()),
            colour_levels: Vec::new(),
            analyzer: Analyzer::new(),
            window: vec![0.0; FFT_N],
            rate: SIM_RATE,
            peak: 0.0,
            sim_t: 0.0,
            factors: Vec::new(),
            last: None,
            meter: (0.0, 0.0),
        }
    }
}

impl SoundHalo {
    /// Settings from the page; places come from the scene's solids.
    pub fn configure(&mut self, source: Source, swing: f32, bands_on: bool, scene: &Scene) {
        if source != self.source {
            // Fresh auto-gain for a new source.
            self.size = Envelope::new();
            self.bass = Envelope::new();
            self.window.fill(0.0);
            self.peak = 0.0;
        }
        self.source = source;
        self.swing = if swing.is_finite() { swing.clamp(0.0, 1.0) } else { HALO_SWINGS[0] };
        if self.colour.places().len() != scene.solids.len() {
            self.colour = SolidLevels::new(self.places_for(scene, true));
        }
        if bands_on != self.bands_on || self.solid.places().len() != self.places_for(scene, bands_on).len() {
            self.bands_on = bands_on;
            self.reset_places(scene);
        }
        if source == Source::Off {
            self.last = None;
        }
    }

    fn places_for(&self, scene: &Scene, bands_on: bool) -> Vec<f32> {
        if bands_on {
            let kinds: Vec<_> = scene.solids.iter().map(|i| i.poly.kind()).collect();
            frequency_gradient(&kinds)
        } else {
            Vec::new()
        }
    }

    /// Recompute each solid's place on the frequency gradient (after a
    /// scene rebuild).
    pub fn reset_places(&mut self, scene: &Scene) {
        self.solid = SolidLevels::new(self.places_for(scene, self.bands_on));
        self.colour = SolidLevels::new(self.places_for(scene, true));
    }

    /// The page's latest capture: the newest samples (any length; the last
    /// `FFT_N` are kept) at `rate` Hz.
    pub fn set_window(&mut self, samples: &[f32], rate: f32) {
        if self.source != Source::Host || !(rate.is_finite() && rate > 0.0) {
            return;
        }
        self.rate = rate;
        let n = samples.len().min(FFT_N);
        let fresh = &samples[samples.len() - n..];
        self.window.copy_within(n.., 0);
        self.window[FFT_N - n..].copy_from_slice(fresh);
        let p = fresh.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
        self.peak = self.peak.max(p.min(1.0));
    }

    /// Advance by `dt` seconds: the halo for this frame (`None` when off) and
    /// per-solid sizes in [`SoundHalo::factors`].
    pub fn next(&mut self, dt: f32) -> Option<Halo> {
        if self.source == Source::Off {
            self.last = None;
            self.factors.clear();
            self.colour_levels.clear();
            return None;
        }
        let dt = if dt.is_finite() { dt.clamp(0.0, 0.25) } else { 0.0 };
        if self.source == Source::Sim {
            self.sim_t += dt as f64;
            self.rate = SIM_RATE;
            let start = self.sim_t - FFT_N as f64 / SIM_RATE as f64;
            let fresh = ((dt * SIM_RATE) as usize).clamp(1, FFT_N);
            let mut peak = 0.0f32;
            for (i, s) in self.window.iter_mut().enumerate() {
                *s = music(start + i as f64 / SIM_RATE as f64);
                if i >= FFT_N - fresh {
                    peak = peak.max(s.abs());
                }
            }
            self.peak = peak;
        }
        let bands = self.analyzer.bands(&self.window, self.rate);
        let raw_peak = std::mem::take(&mut self.peak);
        let bw = bass_weighted(&bands);
        self.meter = (raw_peak, bw);
        let size = self.size.update(raw_peak, dt);
        let bass = self.bass.update(bw, dt);
        self.factors.clear();
        let swing = self.swing;
        self.factors.extend(self.solid.update(&bands, dt).iter().map(|&l| halo_factor(l, swing)));
        self.colour_levels.clear();
        self.colour_levels.extend_from_slice(self.colour.update(&bands, dt));
        let halo = Halo { factor: halo_factor(size, swing), lum: halo_lum(bass), reach: halo_reach(swing) };
        self.last = Some(halo);
        Some(halo)
    }

    /// Per-solid halo sizes of the last [`SoundHalo::next`] (empty = all use
    /// the halo's own factor).
    pub fn factors(&self) -> &[f32] {
        &self.factors
    }

    /// Each solid's level of its own frequency band, 0..=1 (empty when off):
    /// the values for `color_by = "sound"`.
    pub fn colour_levels(&self) -> &[f32] {
        &self.colour_levels
    }

    pub fn last(&self) -> Option<Halo> {
        self.last
    }
}

// ------------------------------------------------------------------ track

/// Deterministic white noise in -1..1 for sample `n`.
fn noise(n: u64) -> f32 {
    let mut x = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

fn sine(hz: f32, t: f32) -> f32 {
    (std::f32::consts::TAU * hz * t).sin()
}

/// The simulated track at `t` seconds, mono -1..1: 120 bpm, 16-second
/// sections cycling through full mix, breakdown (pads only), bass & kick and
/// a hi-hat build, so bass, mid and treble solids each get their turn.
pub fn music(t: f64) -> f32 {
    if t < 0.0 {
        return 0.0;
    }
    const BEAT: f64 = 0.5;
    let beat = (t / BEAT).floor();
    let tb = (t - beat * BEAT) as f32; // time since the beat
    let section = ((t / 16.0) as u64) % 4;
    let (kick_on, bass_on, pad_on, hats_on) = match section {
        0 => (1.0, 1.0, 1.0, 1.0),
        1 => (0.0, 0.0, 1.0, 0.3),
        2 => (1.0, 1.0, 0.2, 0.0),
        _ => (0.4, 0.0, 0.5, 1.0),
    };
    // Kick: pitch drop 110 -> 45 Hz (integrated phase), short decay.
    let phase = std::f32::consts::TAU * (45.0 * tb + 65.0 * 0.03 * (1.0 - (-tb / 0.03).exp()));
    let kick = phase.sin() * (-tb / 0.18).exp();
    // Bass line, one note per beat, 3 harmonics.
    const ROOTS: [f32; 8] = [55.0, 55.0, 73.42, 65.41, 49.0, 49.0, 65.41, 61.74];
    let root = ROOTS[(beat as usize) % ROOTS.len()];
    let tt = t as f32;
    let bass = (sine(root, tt) + 0.5 * sine(2.0 * root, tt) + 0.25 * sine(3.0 * root, tt)) * (-tb / 0.35).exp() * 0.55;
    // Pads: a chord per bar (4 beats), slow swell.
    const CHORDS: [[f32; 3]; 4] = [[220.0, 261.63, 329.63], [293.66, 349.23, 440.0], [261.63, 329.63, 392.0], [246.94, 293.66, 392.0]];
    let chord = CHORDS[((t / 2.0) as usize) % CHORDS.len()];
    let swell = 0.5 + 0.5 * (std::f32::consts::TAU * (t as f32) / 8.0).sin();
    let pad: f32 = chord.iter().map(|&hz| sine(hz, tt) + 0.3 * sine(2.0 * hz, tt)).sum::<f32>() * (0.06 + 0.1 * swell);
    // Hi-hats on every eighth: high-passed noise bursts.
    let te = (t - (t / 0.25).floor() * 0.25) as f32;
    let n = (t * SIM_RATE as f64) as u64;
    let hiss = noise(n) - noise(n.wrapping_sub(1));
    let hats = hiss * (-te / 0.025).exp() * 0.25;
    (kick_on * kick * 0.7 + bass_on * bass + pad_on * pad + hats_on * hats).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use platomon::audio::{Analyzer, BANDS};

    #[test]
    fn track_has_bass_mid_and_treble_in_turn() {
        let mut a = Analyzer::new();
        let block = |t0: f64| -> Vec<f32> { (0..FFT_N).map(|i| music(t0 + i as f64 / SIM_RATE as f64)).collect() };
        // Full mix right on a beat: strong bass.
        let full = a.bands(&block(4.0), SIM_RATE);
        let lo: f32 = full[..3].iter().sum();
        assert!(lo > 0.1, "bass {full:?}");
        // Hi-hat build (section 3) has treble energy.
        let build = a.bands(&block(48.0 + 0.25), SIM_RATE);
        assert!(build[BANDS - 3..].iter().any(|&b| b > 0.005), "treble {build:?}");
        assert!((0..48_000 * 4).all(|i| music(i as f64 / 12_000.0).abs() <= 1.0));
    }

    #[test]
    fn noise_is_bounded() {
        assert!((0..10_000).map(noise).all(|v| (-1.0..=1.0).contains(&v)));
    }
}
