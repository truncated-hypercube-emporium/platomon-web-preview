//! A looping chiptune for the sound halo: 16 bars in A minor at 150 bpm,
//! four voices in the style of an 8-bit sound chip — a pulse-wave lead with
//! vibrato, a quiet pulse arpeggio, a 4-bit triangle bass and noise drums.
//!
//! Chords Am – F – C – G, one per bar. Bars 1-4 are an intro (bass, arpeggio,
//! hats), 5-8 bring in the drums and the first melody, 9-12 the second,
//! higher melody, and 13-16 a breakdown: the first melody with an echo over
//! bass and drums, ending in a snare fill that leads back to the top.
//!
//! [`render`] synthesises the whole loop once; the page plays it on repeat
//! and feeds what it plays to the halo, like any other sound source.

/// Tempo, beats per minute.
pub const BPM: f32 = 150.0;
/// One sixteenth note, seconds.
pub const STEP: f32 = 60.0 / BPM / 4.0;
pub const STEPS_PER_BAR: usize = 16;
pub const BARS: usize = 16;

/// Length of the loop, seconds.
pub fn loop_seconds() -> f32 {
    STEP * (STEPS_PER_BAR * BARS) as f32
}

/// Chord roots (MIDI) and qualities per bar of the 4-bar progression.
const CHORDS: [(u8, [u8; 3]); 4] = [
    (57, [0, 3, 7]), // Am
    (53, [0, 4, 7]), // F
    (48, [0, 4, 7]), // C
    (55, [0, 4, 7]), // G
];

/// A melody note: start step in its bar, MIDI note, length in steps.
type Note = (u8, u8, u8);

/// Melody 1, one entry per bar of the progression.
const MELODY_1: [&[Note]; 4] = [
    &[(0, 76, 2), (2, 81, 2), (4, 79, 2), (6, 76, 2), (8, 74, 4), (12, 72, 4)],
    &[(0, 72, 2), (2, 74, 2), (4, 76, 4), (8, 77, 2), (10, 76, 2), (12, 74, 4)],
    &[(0, 76, 2), (2, 79, 2), (4, 84, 4), (8, 83, 2), (10, 79, 2), (12, 76, 4)],
    &[(0, 74, 4), (4, 71, 2), (6, 74, 2), (8, 79, 6)],
];

/// Melody 2: the answer, higher and busier.
const MELODY_2: [&[Note]; 4] = [
    &[(0, 81, 3), (3, 79, 1), (4, 76, 2), (6, 81, 2), (8, 83, 2), (10, 84, 2), (12, 83, 2), (14, 81, 2)],
    &[(0, 81, 4), (4, 77, 2), (6, 81, 2), (8, 84, 4), (12, 81, 4)],
    &[(0, 79, 2), (2, 76, 2), (4, 79, 2), (6, 84, 2), (8, 88, 4), (12, 86, 4)],
    &[(0, 83, 2), (2, 81, 2), (4, 79, 2), (6, 74, 2), (8, 79, 4), (12, 83, 2), (14, 86, 2)],
];

fn hz(midi: f32) -> f32 {
    440.0 * 2f32.powf((midi - 69.0) / 12.0)
}

/// Pulse wave with duty cycle `duty` at phase `ph` (cycles).
fn pulse(ph: f32, duty: f32) -> f32 {
    if ph.fract() < duty { 1.0 } else { -1.0 }
}

/// 4-bit stepped triangle, like the sound chips' bass channel.
fn triangle4(ph: f32) -> f32 {
    let x = ph.fract();
    let tri = if x < 0.5 { 4.0 * x - 1.0 } else { 3.0 - 4.0 * x };
    ((tri * 7.5 + 7.5).round() - 7.5) / 7.5
}

/// 15-bit LFSR noise, as the chips' noise channel: sample `n` of a sequence
/// clocked once per sample (`short` = the metallic 93-step mode).
struct Lfsr {
    reg: u16,
    short: bool,
}

impl Lfsr {
    fn new(short: bool) -> Self {
        Self { reg: 1, short }
    }
    fn next(&mut self) -> f32 {
        let tap = if self.short { 6 } else { 1 };
        let bit = (self.reg ^ (self.reg >> tap)) & 1;
        self.reg = (self.reg >> 1) | (bit << 14);
        if self.reg & 1 == 1 { 1.0 } else { -1.0 }
    }
}

/// Attack / decay / sustain envelope with a release at `len` seconds.
fn env(t: f32, len: f32, attack: f32, decay: f32, sustain: f32, release: f32) -> f32 {
    if t < 0.0 || t > len + release {
        return 0.0;
    }
    let a = if t < attack { t / attack } else { sustain + (1.0 - sustain) * (-(t - attack) / decay).exp() };
    if t > len { a * (1.0 - (t - len) / release) } else { a }
}

/// The melody note sounding at `step` of `bar` (0..4 of the phrase), with
/// the time since it started.
fn melody_at(phrase: &[&[Note]; 4], bar: usize, step: f32) -> Option<(f32, f32, f32)> {
    let notes = phrase[bar % 4];
    notes.iter().rev().find(|n| step >= n.0 as f32).and_then(|&(s, m, l)| {
        let since = (step - s as f32) * STEP;
        (since < l as f32 * STEP + 0.03).then_some((m as f32, since, l as f32 * STEP))
    })
}

/// A pulse lead voice: `midi` held for `len` seconds, `since` seconds in.
fn lead(midi: f32, since: f32, len: f32, duty: f32) -> f32 {
    let f = hz(midi);
    // vibrato after a moment: ±0.25 semitone at 6 Hz, as phase modulation
    let depth = if since > 0.15 { (since - 0.15).min(0.1) * 10.0 } else { 0.0 };
    let vib = depth * f * 0.0145 / (std::f32::consts::TAU * 6.0) * (std::f32::consts::TAU * 6.0 * since).sin();
    pulse(f * since + vib, duty) * env(since, len - 0.02, 0.005, 0.12, 0.7, 0.03)
}

/// The whole loop, mono, at `rate` Hz (-1..1).
pub fn render(rate: f32) -> Vec<f32> {
    let rate = if rate.is_finite() { rate.clamp(8_000.0, 192_000.0) } else { 48_000.0 };
    let n = (loop_seconds() * rate).round() as usize;
    let mut out = Vec::with_capacity(n);
    let mut long_noise = Lfsr::new(false);
    let mut short_noise = Lfsr::new(true);
    for i in 0..n {
        let t = i as f32 / rate;
        let gstep = t / STEP;
        let bar = (gstep as usize / STEPS_PER_BAR) % BARS;
        let step = gstep - (bar * STEPS_PER_BAR) as f32;
        let istep = step as usize;
        let ts = (step - istep as f32) * STEP; // time since this step began
        let (root, quality) = CHORDS[bar % 4];
        let section = bar / 4; // 0 intro, 1 melody 1, 2 melody 2, 3 breakdown

        // Bass: root and its octave on alternate eighths.
        let eighth = istep / 2;
        let since8 = (step - (eighth * 2) as f32) * STEP;
        let bass_note = root as f32 - 12.0 + if eighth % 2 == 1 { 12.0 } else { 0.0 };
        let bass = triangle4(hz(bass_note) * since8) * env(since8, STEP * 0.9, 0.002, 0.08, 0.35, 0.03);

        // Arpeggio: chord tones climbing every sixteenth, two octaves up.
        let arp = if section < 3 {
            let tone = [quality[0], quality[1], quality[2], 12][istep % 4];
            let m = root as f32 + 12.0 + tone as f32;
            pulse(hz(m) * ts, 0.25) * env(ts, STEP * 0.6, 0.001, 0.05, 0.3, 0.01)
        } else {
            0.0
        };

        // Lead.
        let lead_v = match section {
            1 => melody_at(&MELODY_1, bar, step).map_or(0.0, |(m, s, l)| lead(m, s, l, 0.25)),
            2 => melody_at(&MELODY_2, bar, step).map_or(0.0, |(m, s, l)| lead(m, s, l, 0.125)),
            3 => {
                let dry = melody_at(&MELODY_1, bar, step).map_or(0.0, |(m, s, l)| lead(m - 12.0, s, l, 0.5));
                // echo three sixteenths later
                let back = step - 3.0;
                let wet = if back >= 0.0 {
                    melody_at(&MELODY_1, bar, back).map_or(0.0, |(m, s, l)| lead(m - 12.0, s, l, 0.5))
                } else {
                    0.0
                };
                dry + 0.35 * wet
            }
            _ => 0.0,
        };

        // Drums.
        let long = long_noise.next();
        let short = short_noise.next();
        let drums_on = section > 0;
        let kick = if drums_on && (istep == 0 || istep == 8 || (istep == 10 && bar % 2 == 1)) {
            // pitch drop 150 -> 50 Hz (integrated phase)
            let ph = 50.0 * ts + 100.0 * 0.02 * (1.0 - (-ts / 0.02).exp());
            triangle4(ph) * (-ts / 0.09).exp()
        } else {
            0.0
        };
        let fill = bar == BARS - 1 && istep >= 8;
        let snare = if drums_on && (istep == 4 || istep == 12 || fill) {
            let tail = if fill { 0.04 } else { 0.09 };
            (0.8 * long + 0.4 * triangle4(185.0 * ts)) * (-ts / tail).exp()
        } else {
            0.0
        };
        let hat = if istep % 2 == 1 { short * (-ts / 0.018).exp() } else { 0.0 };

        let mix = 0.34 * bass + 0.07 * arp + 0.16 * lead_v + 0.42 * kick + 0.20 * snare + 0.06 * hat;
        out.push((mix * 1.3).tanh() * 0.9);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use platomon::audio::{Analyzer, BANDS, FFT_N};

    #[test]
    fn loop_length_and_level() {
        let rate = 24_000.0;
        let s = render(rate);
        assert_eq!(s.len(), (loop_seconds() * rate).round() as usize);
        assert!((loop_seconds() - 25.6).abs() < 1e-3, "16 bars at 150 bpm");
        assert!(s.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        let peak = s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.5, "loud enough: {peak}");
    }

    #[test]
    fn sections_differ_in_the_spectrum() {
        let rate = 48_000.0;
        let s = render(rate);
        let mut a = Analyzer::new();
        let at = |bar: f32, step: f32| ((bar * 16.0 + step) * STEP * rate) as usize;
        // Intro (no lead) vs melody 2 on a long high note (bar 11, E6 at step 8).
        let intro = a.bands(&s[at(1.0, 8.5)..at(1.0, 8.5) + FFT_N], rate);
        let high = a.bands(&s[at(10.0, 9.0)..at(10.0, 9.0) + FFT_N], rate);
        let upper = |b: &[f32; BANDS]| b[7..].iter().sum::<f32>();
        assert!(upper(&high) > upper(&intro), "lead adds upper bands: {intro:?} vs {high:?}");
        // The bass is always there.
        assert!(intro[..4].iter().any(|&v| v > 0.05), "{intro:?}");
    }

    #[test]
    fn the_loop_is_seamless() {
        let rate = 48_000.0;
        let s = render(rate);
        // The last and first samples are close: no click at the wrap.
        assert!((s[0] - s[s.len() - 1]).abs() < 0.5);
    }
}

#[cfg(test)]
mod probe {
    use super::*;
    use crate::sound::{SoundHalo, Source};
    use platomon::overlay::HALO_SWINGS;

    /// Halo sizes over the loop, sampled the way the page does (latest 2048
    /// samples each 1/60 s): printed per section for tuning.
    #[test]
    #[ignore]
    fn halo_dynamics() {
        let rate = 48_000.0;
        let s = render(rate);
        let cfg = platomon::config::Config::default();
        let slots = cfg.resolve_slots().unwrap();
        let layout: Vec<_> = slots.iter().map(|s| platomon::scene::SlotLayout { solid: s.solid, ring: s.ring }).collect();
        let scene = platomon::scene::Scene::new(&layout, &platomon::scene::AnimParams::default(), &Default::default());
        let mut h = SoundHalo::default();
        h.configure(Source::Host, HALO_SWINGS[0], true, &scene);
        let hop = (rate / 60.0) as usize;
        for sec in 0..4 {
            let (mut lo, mut hi) = (9.0f32, 0.0f32);
            let (mut slo, mut shi) = (vec![9.0f32; 5], vec![0.0f32; 5]);
            let start = sec * 4 * 16 * (STEP * rate) as usize;
            let end = start + 4 * 16 * (STEP * rate) as usize;
            let mut i = start + 2048;
            while i < end {
                h.set_window(&s[i - 2048..i], rate);
                let halo = h.next(1.0 / 60.0).unwrap();
                if i > start + 2 * 48_000 {
                    lo = lo.min(halo.factor); hi = hi.max(halo.factor);
                    for (k, &f) in h.factors().iter().enumerate() { slo[k] = slo[k].min(f); shi[k] = shi[k].max(f); }
                }
                i += hop;
            }
            println!("section {sec}: overall {lo:.2}..{hi:.2}  per solid {:?}", slo.iter().zip(&shi).map(|(a, b)| format!("{a:.2}..{b:.2}")).collect::<Vec<_>>());
        }
    }
}

/// Write the loop as 16-bit mono WAV to `$PLATOMON_WAV` (to listen to it
/// outside the page): `PLATOMON_WAV=tune.wav cargo test -p platomon-web
/// chiptune_wav -- --ignored`.
#[cfg(test)]
#[test]
#[ignore]
fn chiptune_wav() {
    let Ok(path) = std::env::var("PLATOMON_WAV") else { return };
    let rate = 44_100u32;
    let s = render(rate as f32);
    let mut b = Vec::with_capacity(44 + s.len() * 2);
    let data = (s.len() * 2) as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for v in s {
        b.extend_from_slice(&((v * 32_000.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}
