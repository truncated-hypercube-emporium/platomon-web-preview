//! Colour mapping: a gradient sampled into a fixed palette of 16, 32, 64 or
//! 128 colours, indexed per cell by brightness or by a per-solid value (load,
//! spin, sound), with the background mapped on its own.

use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::str::FromStr;

/// "No colour": the terminal's default foreground.
pub const NONE: u32 = u32::MAX;

/// What the solids' palette position comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ColorBy {
    /// Every cell by its brightness (dark → first stop, bright → last stop).
    #[default]
    Lum,
    /// Each solid by its metric's load (idle → first stop, full → last
    /// stop), shaded by brightness.
    Load,
    /// Each solid by its spin as seen from the camera: the middle of the
    /// gradient is still (or edge-on), the first stop fast clockwise and the
    /// last stop fast counter-clockwise; shaded by brightness.
    Spin,
    /// Each solid by the loudness of its own frequency band of the output
    /// device (simplest solid = bass, most complex = treble); shaded by
    /// brightness. Screensaver only.
    Sound,
}

impl ColorBy {
    pub const ALL: [ColorBy; 4] = [ColorBy::Lum, ColorBy::Load, ColorBy::Spin, ColorBy::Sound];

    pub fn id(self) -> &'static str {
        match self {
            ColorBy::Lum => "lum",
            ColorBy::Load => "load",
            ColorBy::Spin => "spin",
            ColorBy::Sound => "sound",
        }
    }

    /// Whether solids are coloured by a per-solid value (not brightness).
    pub fn per_solid(self) -> bool {
        self != ColorBy::Lum
    }
}

/// What the sound halo's colour comes from (with a colour gradient on).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum HaloColor {
    /// Whatever it passes over: the solid or background under each cell.
    #[default]
    Under,
    /// Its own solid's colour, wherever it is drawn.
    Solid,
    /// Its size: quiet (small) at the gradient's start, loud (large) at
    /// its end; per solid with the frequency gradient on.
    Size,
    /// The bass level (which also sets its brightness).
    Bass,
}

impl HaloColor {
    pub const ALL: [HaloColor; 4] = [HaloColor::Under, HaloColor::Solid, HaloColor::Size, HaloColor::Bass];

    pub fn id(self) -> &'static str {
        match self {
            HaloColor::Under => "under",
            HaloColor::Solid => "solid",
            HaloColor::Size => "size",
            HaloColor::Bass => "bass",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|h| h.id() == s)
    }
}

impl FromStr for ColorBy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lum" | "luminance" | "brightness" => Ok(Self::Lum),
            "load" | "metric" => Ok(Self::Load),
            "spin" | "rotation" | "velocity" => Ok(Self::Spin),
            "sound" | "music" | "audio" | "frequency" | "freq" => Ok(Self::Sound),
            _ => Err(format!("unknown colour source '{s}' (lum|load|spin|sound)")),
        }
    }
}

/// Palette sizes on offer.
pub const PALETTE_SIZES: [usize; 4] = [16, 32, 64, 128];
/// Default palette size.
pub const DEFAULT_COLORS: usize = 16;

/// The nearest offered palette size to `n`.
pub fn palette_size(n: usize) -> usize {
    PALETTE_SIZES.into_iter().min_by_key(|&s| s.abs_diff(n)).unwrap_or(DEFAULT_COLORS)
}

/// Background colouring: `auto` (by brightness with the solids' gradient
/// when they are coloured by brightness, else uncoloured), `off`, `match`
/// (the solids' gradient by brightness, whatever colours the solids) or a
/// gradient of its own (preset or hex stops), by brightness.
pub const BG_COLOR_DEFAULT: &str = "auto";

/// The background's gradient for `spec` (see [`BG_COLOR_DEFAULT`]), given
/// the solids' gradient and source; `None` = uncoloured.
pub fn bg_gradient(spec: &str, solid: Option<&Gradient>, by: ColorBy) -> Result<Option<Gradient>, String> {
    match spec.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(solid.filter(|_| by == ColorBy::Lum).cloned()),
        "off" | "none" | "mono" => Ok(None),
        "match" | "same" | "solids" => Ok(solid.cloned()),
        _ => Gradient::parse(spec).map(Some).map_err(|e| format!("background colour: {e}")),
    }
}

/// A fixed set of colours sampled from a gradient: `steps` positions along
/// it × `shades` brightness levels (1 shade = full brightness only).
#[derive(Debug, Clone, PartialEq)]
pub struct Palette {
    colors: Vec<u32>,
    steps: usize,
    shades: usize,
}

/// Brightness of shade `s` of `shades` (the top shade is full brightness;
/// never fully black, so the hue stays readable on dark faces).
fn shade_level(s: usize, shades: usize) -> f32 {
    0.35 + 0.65 * (s + 1) as f32 / shades as f32
}

impl Palette {
    /// `n` colours along `g` at full brightness (colour by brightness).
    pub fn ramp(g: &Gradient, n: usize) -> Self {
        let n = n.max(2);
        let colors = (0..n).map(|k| g.sample(k as f32 / (n - 1) as f32, 1.0)).collect();
        Self { colors, steps: n, shades: 1 }
    }

    /// `n` colours: positions along `g` × brightness shades (colour by a
    /// per-solid value, shaded by the glyph's brightness). 16 colours are 8
    /// positions × 2 shades; larger palettes use 4 shades.
    pub fn shaded(g: &Gradient, n: usize) -> Self {
        let shades = if n <= 16 { 2 } else { 4 };
        let steps = (n / shades).max(2);
        let mut colors = Vec::with_capacity(steps * shades);
        for v in 0..steps {
            let t = v as f32 / (steps - 1) as f32;
            colors.extend((0..shades).map(|s| g.sample(t, shade_level(s, shades))));
        }
        Self { colors, steps, shades }
    }

    pub fn len(&self) -> usize {
        self.colors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    pub fn colors(&self) -> &[u32] {
        &self.colors
    }

    pub fn shades(&self) -> usize {
        self.shades
    }

    /// Position index of `t` in 0..=1: step `k` covers `[k/n, (k+1)/n)`.
    #[inline(always)]
    fn step(&self, t: f32) -> usize {
        let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
        ((t * self.steps as f32) as usize).min(self.steps - 1)
    }

    /// The brightest colour at position `t`.
    #[inline(always)]
    pub fn at(&self, t: f32) -> u32 {
        self.colors[self.step(t) * self.shades + self.shades - 1]
    }

    /// The shades (dark to bright) at position `t`.
    #[inline(always)]
    pub fn shade_row(&self, t: f32) -> &[u32] {
        let i = self.step(t) * self.shades;
        &self.colors[i..i + self.shades]
    }
}

/// Built-in gradients, first stop = dark / idle.
pub const PRESETS: [(&str, &str, &[u32]); 12] = [
    ("inferno", "black, purple, red, orange, pale yellow", &[0x000004, 0x420a68, 0x932667, 0xdd513a, 0xfca50a, 0xfcffa4]),
    ("viridis", "purple, blue, teal, green, yellow", &[0x440154, 0x3b528b, 0x21918c, 0x5ec962, 0xfde725]),
    ("plasma", "indigo, magenta, orange, yellow", &[0x0d0887, 0x7e03a8, 0xcc4778, 0xf89540, 0xf0f921]),
    ("fire", "black, deep red, orange, gold, white", &[0x000000, 0x5a0000, 0xc81e00, 0xff8c00, 0xffd700, 0xffffe0]),
    ("ice", "navy, blue, cyan, white", &[0x000010, 0x002050, 0x0070c0, 0x40c0ff, 0xe0ffff]),
    ("ocean", "deep blue to aqua", &[0x000814, 0x001d3d, 0x003566, 0x0096c7, 0x90e0ef]),
    ("sunset", "violet, magenta, coral, gold", &[0x2d0b59, 0x8a1c7c, 0xda4167, 0xf78154, 0xfcd581]),
    ("matrix", "phosphor green", &[0x001000, 0x003b00, 0x008f11, 0x00ff41, 0xc8ffc8]),
    ("amber", "amber monochrome monitor", &[0x100800, 0x6b3a00, 0xffb000, 0xffe0a0]),
    ("grey", "neutral grey ramp", &[0x000000, 0xffffff]),
    ("traffic", "green, yellow, red (good for --color-by load)", &[0x00c040, 0xa0e000, 0xffd000, 0xff7000, 0xff2020]),
    ("rainbow", "red through violet", &[0xff0000, 0xff8000, 0xffff00, 0x00ff00, 0x00ffff, 0x0040ff, 0x8000ff]),
];

/// Default gradient when a colour source is chosen without one.
pub fn default_for(by: ColorBy) -> &'static str {
    match by {
        ColorBy::Lum => "inferno",
        ColorBy::Load => "traffic",
        ColorBy::Spin => "plasma",
        ColorBy::Sound => "viridis",
    }
}

/// Evenly spaced RGB stops.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    stops: Vec<[f32; 3]>,
}

fn rgb_f(c: u32) -> [f32; 3] {
    [((c >> 16) & 0xff) as f32, ((c >> 8) & 0xff) as f32, (c & 0xff) as f32]
}

fn parse_hex(s: &str) -> Option<u32> {
    let h = s.trim().trim_start_matches('#');
    match h.len() {
        6 => u32::from_str_radix(h, 16).ok(),
        3 => {
            let v = u32::from_str_radix(h, 16).ok()?;
            let (r, g, b) = ((v >> 8) & 0xf, (v >> 4) & 0xf, v & 0xf);
            Some(((r * 17) << 16) | ((g * 17) << 8) | (b * 17))
        }
        _ => None,
    }
}

impl Gradient {
    pub fn from_stops(stops: &[u32]) -> Self {
        Self { stops: stops.iter().map(|&c| rgb_f(c)).collect() }
    }

    /// A preset name, or two or more hex colours separated by `,` `:` `;`
    /// `/` or spaces, e.g. `#000:#f80:#fff` or `001020,ff8800`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let name = s.trim().to_ascii_lowercase();
        if let Some((_, _, stops)) = PRESETS.iter().find(|(id, _, _)| *id == name) {
            return Ok(Self::from_stops(stops));
        }
        let parts: Vec<&str> = s.split([',', ':', ';', '/', ' ']).filter(|p| !p.trim().is_empty()).collect();
        let stops: Option<Vec<u32>> = parts.iter().map(|p| parse_hex(p)).collect();
        match stops {
            Some(v) if v.len() >= 2 => Ok(Self::from_stops(&v)),
            _ => Err(format!(
                "unknown gradient '{s}': use a preset (see --list gradients) or 2+ hex colours like \"#000:#f80:#fff\""
            )),
        }
    }

    pub fn reversed(mut self) -> Self {
        self.stops.reverse();
        self
    }

    /// Colour at `t` in 0..=1 as `0xRRGGBB`, scaled by `bright` (0..=1).
    pub fn sample(&self, t: f32, bright: f32) -> u32 {
        let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
        let b = if bright.is_finite() { bright.clamp(0.0, 1.0) } else { 0.0 };
        let n = self.stops.len();
        let f = t * (n - 1) as f32;
        let i = (f as usize).min(n - 2);
        let u = f - i as f32;
        let (a, c) = (self.stops[i], self.stops[i + 1]);
        let ch = |k: usize| ((a[k] + (c[k] - a[k]) * u) * b).round().clamp(0.0, 255.0) as u32;
        (ch(0) << 16) | (ch(1) << 8) | ch(2)
    }
}

// ------------------------------------------------------------------ tag colours

/// WCAG relative luminance of `0xRRGGBB`, 0 (black) ..= 1 (white).
pub fn rel_luminance(c: u32) -> f32 {
    let lin = |v: u32| {
        let s = v as f32 / 255.0;
        if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    0.2126 * lin((c >> 16) & 0xff) + 0.7152 * lin((c >> 8) & 0xff) + 0.0722 * lin(c & 0xff)
}

/// WCAG contrast ratio of `c` against black (1 ..= 21).
pub fn contrast_on_black(c: u32) -> f32 {
    (rel_luminance(c) + 0.05) / 0.05
}

/// Smallest luminance a tag colour is lifted to: 11:1 against the black
/// cell background (WCAG AAA asks for 7:1), so labels stay crisp.
pub const TAG_MIN_LUMINANCE: f32 = 0.5;

/// `c` mixed toward white just enough to reach `min_l` relative luminance
/// (keeps the hue; colours already that bright are unchanged).
pub fn lift_to(c: u32, min_l: f32) -> u32 {
    if rel_luminance(c) >= min_l {
        return c;
    }
    let mix = |t: f32| {
        let ch = |s: u32| {
            let v = ((c >> s) & 0xff) as f32;
            (v + (255.0 - v) * t).round().clamp(0.0, 255.0) as u32
        };
        (ch(16) << 16) | (ch(8) << 8) | ch(0)
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..20 {
        let m = 0.5 * (lo + hi);
        if rel_luminance(mix(m)) >= min_l { hi = m } else { lo = m }
    }
    mix(hi)
}

/// (hue degrees, saturation, value) of `0xRRGGBB`.
fn to_hsv(c: u32) -> (f32, f32, f32) {
    let [r, g, b] = rgb_f(c).map(|v| v / 255.0);
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    let d = mx - mn;
    let h = if d <= 1e-6 {
        0.0
    } else if mx == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if mx == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (h, if mx <= 1e-6 { 0.0 } else { d / mx }, mx)
}

fn from_hsv(h: f32, s: f32, v: f32) -> u32 {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let ch = |u: f32| ((u + m) * 255.0).round().clamp(0.0, 255.0) as u32;
    (ch(r) << 16) | (ch(g) << 8) | ch(b)
}

/// How tag text and leader lines are coloured when a gradient is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TagColor {
    /// The gradient's most vivid bright colour, lifted to a light tint.
    #[default]
    Theme,
    /// The hue opposite the gradient's dominant hue, as a light tint: stands
    /// out from the picture while going with it.
    Complement,
    /// Plain white.
    White,
    /// A fixed colour (still lifted to the readable minimum).
    Custom(u32),
}

impl TagColor {
    pub const CHOICES: [TagColor; 3] = [TagColor::Theme, TagColor::Complement, TagColor::White];

    pub fn id(self) -> String {
        match self {
            TagColor::Theme => "theme".into(),
            TagColor::Complement => "complement".into(),
            TagColor::White => "white".into(),
            TagColor::Custom(c) => format!("#{c:06x}"),
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            TagColor::Theme => "Match the colour theme (light tint)",
            TagColor::Complement => "Complementary hue (light tint)",
            TagColor::White => "White",
            TagColor::Custom(_) => "Custom colour",
        }
    }

    /// Tag colour for `gradient`; [`NONE`] (the default text colour) without
    /// one. Always at least [`TAG_MIN_LUMINANCE`].
    pub fn resolve(self, gradient: Option<&Gradient>) -> u32 {
        let Some(g) = gradient else { return NONE };
        let samples: Vec<u32> = (0..=20).map(|k| g.sample(k as f32 / 20.0, 1.0)).collect();
        let c = match self {
            TagColor::White => 0xFF_FF_FF,
            TagColor::Custom(c) => c,
            TagColor::Theme => {
                // most vivid among the brighter half (chroma, a little value)
                samples[8..]
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        let score = |c: u32| {
                            let (_, s, v) = to_hsv(c);
                            s * v + 0.15 * v
                        };
                        score(a).total_cmp(&score(b))
                    })
                    .unwrap_or(0xFF_FF_FF)
            }
            TagColor::Complement => {
                // chroma-weighted circular mean hue, turned half way round
                let (mut x, mut y) = (0.0f32, 0.0f32);
                for &c in &samples[4..] {
                    let (h, s, v) = to_hsv(c);
                    let w = s * v;
                    x += w * h.to_radians().cos();
                    y += w * h.to_radians().sin();
                }
                if x.hypot(y) < 0.5 {
                    0x7F_D8_FF // no dominant hue (grey, rainbow): a cool cyan
                } else {
                    from_hsv(y.atan2(x).to_degrees() + 180.0, 0.7, 1.0)
                }
            }
        };
        lift_to(c, TAG_MIN_LUMINANCE)
    }
}

impl FromStr for TagColor {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "theme" | "auto" | "match" => Ok(TagColor::Theme),
            "complement" | "complementary" => Ok(TagColor::Complement),
            "white" => Ok(TagColor::White),
            v => parse_hex(v)
                .map(TagColor::Custom)
                .ok_or_else(|| format!("unknown tag colour '{s}' (theme | complement | white | #RRGGBB)")),
        }
    }
}

/// Append the SGR code that sets the foreground to palette colour `c`
/// ([`NONE`] = default).
pub fn push_sgr(out: &mut String, c: u32) {
    if c == NONE {
        out.push_str("\x1b[39m");
        return;
    }
    let _ = write!(out, "\x1b[38;2;{};{};{}m", (c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff);
}

/// Colouring for one run: the solids' palette and source, and the
/// background's own palette. Every colour drawn is a palette entry.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorMap {
    /// The solids' gradient (the background's when the solids are
    /// uncoloured): tag colours are chosen from it.
    pub gradient: Gradient,
    pub by: ColorBy,
    /// Colours per palette (16, 32, 64 or 128).
    pub size: usize,
    /// The solids' palette; `None` = solids uncoloured.
    pub solid: Option<Palette>,
    /// The background's palette, by brightness; `None` = uncoloured.
    pub bg: Option<Palette>,
}

impl ColorMap {
    /// `None` when neither the solids nor the background are coloured.
    pub fn new(solid: Option<Gradient>, by: ColorBy, size: usize, bg: Option<Gradient>) -> Option<Self> {
        let size = palette_size(size);
        let gradient = solid.clone().or_else(|| bg.clone())?;
        let solid = solid.map(|g| if by.per_solid() { Palette::shaded(&g, size) } else { Palette::ramp(&g, size) });
        let bg = bg.map(|g| Palette::ramp(&g, size));
        Some(Self { gradient, by, size, solid, bg })
    }

    /// Colour of a solid cell: brightness `l` (0..=1) in glyph bucket
    /// `bucket` of `buckets`, and the solid's value (per-solid sources).
    #[inline]
    pub fn solid_color(&self, value: Option<f32>, l: f32, bucket: usize, buckets: usize) -> u32 {
        match (&self.solid, self.by.per_solid()) {
            (None, _) => NONE,
            (Some(p), false) => p.at(l),
            (Some(p), true) => value.map_or(NONE, |v| p.shade_row(v)[bucket * p.shades() / buckets.max(1)]),
        }
    }

    /// Colour of a background cell with brightness `l`.
    #[inline]
    pub fn bg_color(&self, l: f32) -> u32 {
        self.bg.as_ref().map_or(NONE, |p| p.at(l))
    }

    /// Colour for position `v` (0..=1) along the gradient, shaded for glyph
    /// bucket `bucket` of `buckets` where the palette has shades: the
    /// solids' palette, else the background's.
    #[inline]
    pub fn value_color(&self, v: f32, bucket: usize, buckets: usize) -> u32 {
        match self.solid.as_ref().or(self.bg.as_ref()) {
            None => NONE,
            Some(p) if p.shades() > 1 => p.shade_row(v)[bucket * p.shades() / buckets.max(1)],
            Some(p) => p.at(v),
        }
    }

    /// A solid's legend/icon colour: its value's brightest shade, or the
    /// palette's brightest colour when coloured by brightness.
    pub fn icon_color(&self, value: Option<f32>) -> u32 {
        match &self.solid {
            None => NONE,
            Some(p) if self.by.per_solid() => p.at(value.unwrap_or(0.0)),
            Some(p) => p.at(1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hue_gap(a: u32, b: u32) -> f32 {
        let d = (to_hsv(a).0 - to_hsv(b).0).rem_euclid(360.0);
        d.min(360.0 - d)
    }

    #[test]
    fn tag_colours_are_readable_and_go_with_the_theme() {
        for (id, _, _) in PRESETS {
            let g = Gradient::parse(id).unwrap();
            let bright = g.sample(0.8, 1.0);
            for mode in TagColor::CHOICES {
                let c = mode.resolve(Some(&g));
                assert!(contrast_on_black(c) >= 11.0 - 1e-3, "{id} {mode:?}: #{c:06x} contrast {:.1}", contrast_on_black(c));
            }
            let theme = TagColor::Theme.resolve(Some(&g));
            let comp = TagColor::Complement.resolve(Some(&g));
            let (_, s_theme, _) = to_hsv(theme);
            if !matches!(id, "grey" | "rainbow") {
                // the theme tint keeps a colour of the theme, the complement sits well away from it
                assert!(s_theme > 0.08, "{id}: theme tag #{theme:06x} keeps some colour");
                assert!(hue_gap(comp, bright) > 60.0, "{id}: complement #{comp:06x} vs #{bright:06x}");
                assert!(hue_gap(comp, theme) > 60.0, "{id}: complement #{comp:06x} vs theme #{theme:06x}");
            }
        }
        // monochrome: the default text colour
        assert_eq!(TagColor::Theme.resolve(None), NONE);
        assert_eq!(TagColor::White.resolve(Some(&Gradient::parse("fire").unwrap())), 0xFFFFFF);
        // a dark custom colour is lifted, keeping its hue
        let navy = TagColor::Custom(0x000080).resolve(Some(&Gradient::parse("ice").unwrap()));
        assert!(rel_luminance(navy) >= TAG_MIN_LUMINANCE && to_hsv(navy).0.round() == 240.0, "#{navy:06x}");
        // parsing
        assert_eq!("Complement".parse::<TagColor>().unwrap(), TagColor::Complement);
        assert_eq!("#ff8800".parse::<TagColor>().unwrap(), TagColor::Custom(0xff8800));
        assert!("plaid".parse::<TagColor>().is_err());
        for m in TagColor::CHOICES {
            assert_eq!(m.id().parse::<TagColor>().unwrap(), m);
        }
    }

    #[test]
    fn presets_and_custom_parse() {
        for (id, _, stops) in PRESETS {
            let g = Gradient::parse(id).unwrap();
            assert_eq!(g.sample(0.0, 1.0), stops[0], "{id} start");
            assert_eq!(g.sample(1.0, 1.0), *stops.last().unwrap(), "{id} end");
        }
        let g = Gradient::parse("#000:#f80:#fff").unwrap();
        assert_eq!(g.sample(0.5, 1.0), 0xff8800);
        assert_eq!(g.sample(0.25, 1.0), 0x804400);
        assert_eq!(Gradient::parse("000000, ffffff").unwrap().sample(1.0, 0.5), 0x808080);
        assert_eq!(Gradient::parse("grey").unwrap().reversed().sample(0.0, 1.0), 0xffffff);
        assert!(Gradient::parse("#fff").is_err(), "one stop");
        assert!(Gradient::parse("plaid").is_err());
        assert!(Gradient::parse("#12345g:#000").is_err());
        assert_eq!(Gradient::parse("fire").unwrap().sample(f32::NAN, 1.0), 0x000000);
    }

    #[test]
    fn palettes_hold_exactly_their_size_in_colours() {
        let g = Gradient::parse("inferno").unwrap();
        for n in PALETTE_SIZES {
            let ramp = Palette::ramp(&g, n);
            let shaded = Palette::shaded(&g, n);
            assert_eq!((ramp.len(), shaded.len()), (n, n), "{n}");
            assert_eq!(ramp.at(0.0), g.sample(0.0, 1.0));
            assert_eq!(ramp.at(1.0), g.sample(1.0, 1.0));
            // every lookup lands on a palette entry
            let mut seen = std::collections::HashSet::new();
            for k in 0..=1000 {
                let t = k as f32 / 1000.0;
                seen.insert(ramp.at(t));
                for &c in shaded.shade_row(t) {
                    assert!(shaded.colors().contains(&c));
                }
            }
            assert_eq!(seen.len(), n, "{n}: every ramp colour is reachable");
            // shades get brighter; the top shade is full brightness
            let row = shaded.shade_row(0.9);
            assert!(row.windows(2).all(|w| rel_luminance(w[0]) < rel_luminance(w[1])), "{n}: {row:x?}");
            assert_eq!(*row.last().unwrap(), shaded.at(0.9));
        }
        assert_eq!(palette_size(24), 16);
        assert_eq!(palette_size(100), 128);
        assert_eq!(palette_size(16_000_000), 128);
        assert_eq!(Palette::ramp(&g, 16).at(f32::NAN), g.sample(0.0, 1.0));
    }

    #[test]
    fn background_maps_independently() {
        let inferno = Gradient::parse("inferno").unwrap();
        let ice = Gradient::parse("ice").unwrap();
        assert_eq!(bg_gradient("auto", Some(&inferno), ColorBy::Lum).unwrap(), Some(inferno.clone()));
        assert_eq!(bg_gradient("auto", Some(&inferno), ColorBy::Load).unwrap(), None);
        assert_eq!(bg_gradient("match", Some(&inferno), ColorBy::Spin).unwrap(), Some(inferno.clone()));
        assert_eq!(bg_gradient("off", Some(&inferno), ColorBy::Lum).unwrap(), None);
        assert_eq!(bg_gradient("ice", None, ColorBy::Lum).unwrap(), Some(ice.clone()));
        assert!(bg_gradient("plaid", None, ColorBy::Lum).is_err());
        // background only: solids uncoloured, tags pick from the background's gradient
        let m = ColorMap::new(None, ColorBy::Lum, 32, Some(ice.clone())).unwrap();
        assert_eq!((m.solid.is_none(), m.gradient == ice), (true, true));
        assert_eq!(m.solid_color(None, 0.5, 3, 16), NONE);
        assert_eq!(m.bg_color(1.0), ice.sample(1.0, 1.0));
        assert!(ColorMap::new(None, ColorBy::Lum, 16, None).is_none());
        // per-solid sources shade by glyph bucket and need a value
        let m = ColorMap::new(Some(inferno.clone()), ColorBy::Sound, 16, None).unwrap();
        assert_eq!(m.solid_color(None, 0.5, 3, 16), NONE);
        assert_eq!(m.solid_color(Some(1.0), 1.0, 15, 16), inferno.sample(1.0, 1.0));
        assert_eq!(m.bg_color(0.5), NONE);
    }

    #[test]
    fn sgr_codes() {
        let mut s = String::new();
        push_sgr(&mut s, 0x0a141e);
        push_sgr(&mut s, NONE);
        assert_eq!(s, "\x1b[38;2;10;20;30m\x1b[39m");
        for by in ColorBy::ALL {
            assert_eq!(by.id().parse::<ColorBy>().unwrap(), by);
        }
        assert!("cmyk".parse::<ColorBy>().is_err());
    }
}
