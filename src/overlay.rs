//! Overlays drawn over the shaded picture:
//!
//! * **wireframe** — each solid's visible edges, drawn into the picture with
//!   inverted luminosity so neighbouring solids separate clearly;
//! * **legend** — a panel in one corner with a small wireframe of every solid
//!   (spinning with it) next to the metric it shows.

use glam::Vec3A;
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};
use std::str::FromStr;

use crate::ascii::{self, Ramp};
use crate::color::{self, ColorMap, HaloColor};
use crate::render::FrameView;
use crate::scene::Scene;
use crate::scene::sdf::SolidKind;

// ------------------------------------------------------------------ params

/// Legend corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum LegendPos {
    #[default]
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl FromStr for LegendPos {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "tl" | "top-left" => Ok(Self::TopLeft),
            "tr" | "top-right" => Ok(Self::TopRight),
            "bl" | "bottom-left" => Ok(Self::BottomLeft),
            "br" | "bottom-right" => Ok(Self::BottomRight),
            _ => Err(format!("unknown legend position '{s}' (tl|tr|bl|br)")),
        }
    }
}

/// Smallest and largest legend size.
pub const LEGEND_SIZE_MIN: u8 = 1;
pub const LEGEND_SIZE_MAX: u8 = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegendParams {
    /// 1..=5. 1–2: terminal text; 3–5: block letters, 1–3 cells per font pixel.
    /// Icons grow with the size.
    pub size: u8,
    /// Panel background: 0 = see-through, 1 = solid blank.
    pub opacity: f32,
    pub pos: LegendPos,
}

impl Default for LegendParams {
    fn default() -> Self {
        Self { size: 2, opacity: 0.85, pos: LegendPos::TopLeft }
    }
}

/// Where the solid tags go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TagPlace {
    /// On a circle round the picture's centre, each at the point nearest
    /// its solid ([`TAG_CIRCLE`]).
    #[default]
    Circle,
    /// Beside each solid, on the diagonal toward the picture's nearest
    /// corner.
    Beside,
    /// Stacked in the picture's corner on its solid's side.
    Corner,
    /// Centred under each solid.
    Below,
    /// Behind each solid as it moves across the picture, like a trail.
    Trailing,
}

impl TagPlace {
    pub const ALL: [TagPlace; 5] = [TagPlace::Circle, TagPlace::Beside, TagPlace::Corner, TagPlace::Below, TagPlace::Trailing];

    pub fn id(self) -> &'static str {
        match self {
            TagPlace::Circle => "circle",
            TagPlace::Beside => "beside",
            TagPlace::Corner => "corner",
            TagPlace::Below => "below",
            TagPlace::Trailing => "trailing",
        }
    }
}

impl FromStr for TagPlace {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|p| p.id() == s)
            .ok_or_else(|| format!("unknown tag position '{s}' (circle|beside|corner|below|trailing)"))
    }
}

/// The block-letter font of tags on small text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TagFont {
    /// Three rows of strokes (`_ | / \ ( )`).
    #[default]
    Strokes,
    /// Five-row pixel letters drawn with `#`.
    Hash,
    /// ...with `o`.
    Dots,
    /// ...with `*`.
    Stars,
    /// ...with `@`.
    At,
    /// ...with `[]` bricks (two columns per pixel).
    Bricks,
}

impl TagFont {
    pub const ALL: [TagFont; 6] = [TagFont::Strokes, TagFont::Hash, TagFont::Dots, TagFont::Stars, TagFont::At, TagFont::Bricks];

    /// A font chosen at random from `seed` (the same for the same seed).
    pub fn random(seed: u64) -> Self {
        Self::ALL[fastrand::Rng::with_seed(seed ^ 0x666f_6e74).usize(0..Self::ALL.len())]
    }

    /// The glyph(s) a lit pixel of the pixel fonts is drawn with.
    fn ink(self) -> Option<&'static str> {
        match self {
            TagFont::Strokes => None,
            TagFont::Hash => Some("#"),
            TagFont::Dots => Some("o"),
            TagFont::Stars => Some("*"),
            TagFont::At => Some("@"),
            TagFont::Bricks => Some("[]"),
        }
    }
}

// ------------------------------------------------------------------ wire models

/// Unit-circumradius edge model of one solid.
#[derive(Debug, Clone)]
struct Wire {
    kind: SolidKind,
    verts: Vec<Vec3A>,
    /// Face normals and offsets.
    faces: Vec<(Vec3A, f32)>,
    edges: Vec<(usize, usize, usize, usize)>,
}

impl Wire {
    fn new(kind: SolidKind) -> Self {
        let p = crate::scene::sdf::Polyhedron::new(kind);
        Self { kind, verts: p.vertices(), faces: p.face_planes(), edges: p.edges() }
    }
}

/// Inverted luminosity, pushed at least `MIN_SEP` away from the original so
/// edges stay visible on mid-grey too.
#[inline]
pub fn invert(l: f32) -> f32 {
    const MIN_SEP: f32 = 0.4;
    let l = l.clamp(0.0, 1.0);
    let i = 1.0 - l;
    if (i - l).abs() >= MIN_SEP {
        i
    } else if l < 0.5 {
        l + MIN_SEP
    } else {
        l - MIN_SEP
    }
}

// ------------------------------------------------------------------ block font

/// 3×5 pixel font, one byte per row, bit 2 = left column.
fn font_glyph(c: char) -> [u8; 5] {
    match c.to_ascii_uppercase() {
        'A' => [2, 5, 7, 5, 5],
        'B' => [6, 5, 6, 5, 6],
        'C' => [3, 4, 4, 4, 3],
        'D' => [6, 5, 5, 5, 6],
        'E' => [7, 4, 6, 4, 7],
        'F' => [7, 4, 6, 4, 4],
        'G' => [3, 4, 5, 5, 3],
        'H' => [5, 5, 7, 5, 5],
        'I' => [7, 2, 2, 2, 7],
        'J' => [1, 1, 1, 5, 2],
        'K' => [5, 5, 6, 5, 5],
        'L' => [4, 4, 4, 4, 7],
        'M' => [5, 7, 7, 5, 5],
        'N' => [6, 5, 5, 5, 5],
        'O' => [2, 5, 5, 5, 2],
        'P' => [6, 5, 6, 4, 4],
        'Q' => [2, 5, 5, 6, 3],
        'R' => [6, 5, 6, 5, 5],
        'S' => [3, 4, 2, 1, 6],
        'T' => [7, 2, 2, 2, 2],
        'U' => [5, 5, 5, 5, 7],
        'V' => [5, 5, 5, 5, 2],
        'W' => [5, 5, 7, 7, 5],
        'X' => [5, 5, 2, 5, 5],
        'Y' => [5, 5, 2, 2, 2],
        'Z' => [7, 1, 2, 4, 7],
        '0' => [7, 5, 5, 5, 7],
        '1' => [2, 6, 2, 2, 7],
        '2' => [6, 1, 2, 4, 7],
        '3' => [6, 1, 2, 1, 6],
        '4' => [5, 5, 7, 1, 1],
        '5' => [7, 4, 6, 1, 6],
        '6' => [3, 4, 7, 5, 7],
        '7' => [7, 1, 2, 2, 2],
        '8' => [7, 5, 7, 5, 7],
        '9' => [7, 5, 7, 1, 6],
        '%' => [5, 1, 2, 4, 5],
        '.' => [0, 0, 0, 0, 2],
        ':' => [0, 2, 0, 2, 0],
        '-' => [0, 0, 7, 0, 0],
        '+' => [0, 2, 7, 2, 0],
        '/' => [1, 1, 2, 4, 4],
        '\\' => [4, 4, 2, 1, 1],
        '_' => [0, 0, 0, 0, 7],
        '(' => [2, 4, 4, 4, 2],
        ')' => [2, 1, 1, 1, 2],
        ' ' => [0; 5],
        _ => [7, 1, 2, 0, 2], // '?'
    }
}

/// Label typography for a legend size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Font {
    /// 0 = terminal text, else cells per font pixel (vertically).
    scale: usize,
    /// Columns per font pixel.
    px_w: usize,
}

impl Font {
    fn for_size(size: u8, char_aspect: f32) -> Self {
        if size < 3 {
            return Self { scale: 0, px_w: 1 };
        }
        let scale = (size - 2) as usize;
        let px_w = ((scale as f32 * 0.5 / char_aspect.max(0.1)).round() as usize).max(1);
        Self { scale, px_w }
    }

    fn height(&self) -> usize {
        if self.scale == 0 { 1 } else { 5 * self.scale }
    }

    fn width(&self, text: &str) -> usize {
        let n = text.chars().count();
        if self.scale == 0 { n } else { (n * 4).saturating_sub(1) * self.px_w }
    }
}

// ------------------------------------------------------------------ layout

#[derive(Debug, Clone, PartialEq)]
struct Panel {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    icon_w: usize,
    icon_h: usize,
    label_w: usize,
    font: Font,
    /// (slot, x, y) of each entry's icon.
    entries: Vec<(usize, usize, usize)>,
}

/// Rows of an icon for a legend size (at least 2 more than the label).
fn icon_rows(size: u8, font: &Font) -> usize {
    (3 + 2 * size as usize).max(font.height() + 2)
}

const PAD: usize = 1;
const GAP_COLS: usize = 2;
const MAX_LABEL: usize = 48;

fn layout(p: &LegendParams, w: usize, h: usize, labels: &[String], char_aspect: f32) -> Option<Panel> {
    let n = labels.len();
    if n == 0 || w < 8 || h < 4 {
        return None;
    }
    let size = p.size.clamp(LEGEND_SIZE_MIN, LEGEND_SIZE_MAX);
    let font = Font::for_size(size, char_aspect);
    let mut icon_h = icon_rows(size, &font).min(h.saturating_sub(2 * PAD));
    if icon_h < 3 {
        return None;
    }
    if icon_h < font.height() {
        icon_h = font.height().min(h);
    }
    let icon_w = ((icon_h as f32 / char_aspect.max(0.1)).round() as usize).max(3);
    let mut label_w = labels.iter().map(|l| font.width(l)).max().unwrap_or(0).min(MAX_LABEL * 4 * font.px_w);
    let entry_h = icon_h + 1;
    let per_col = ((h - 2 * PAD + 1) / entry_h).max(1);
    let mut ncols = n.div_ceil(per_col);
    let panel_w = |ncols: usize, label_w: usize| 2 * PAD + ncols * (icon_w + 1 + label_w) + (ncols - 1) * GAP_COLS;
    while ncols > 1 && panel_w(ncols, label_w) > w {
        ncols -= 1;
    }
    if panel_w(ncols, label_w) > w {
        // One column still too wide: clip the labels.
        label_w = w.checked_sub(2 * PAD + icon_w + 1)?;
        if label_w < 3 {
            return None;
        }
    }
    let shown = n.min(ncols * per_col);
    let rows_used = shown.min(per_col);
    let pw = panel_w(ncols, label_w);
    let ph = (2 * PAD + rows_used * entry_h - 1).min(h);
    let (x, y) = match p.pos {
        LegendPos::TopLeft => (0, 0),
        LegendPos::TopRight => (w - pw, 0),
        LegendPos::BottomLeft => (0, h - ph),
        LegendPos::BottomRight => (w - pw, h - ph),
    };
    let entries = (0..shown)
        .map(|i| {
            let (c, r) = (i / per_col, i % per_col);
            (i, x + PAD + c * (icon_w + 1 + label_w + GAP_COLS), y + PAD + r * entry_h)
        })
        .collect();
    Some(Panel { x, y, w: pw, h: ph, icon_w, icon_h, label_w, font, entries })
}

// ------------------------------------------------------------------ drawing helpers

struct Grid<'a> {
    cells: &'a mut [char],
    /// Per-cell colour, when colour is on; `put` paints `ink`.
    colors: Option<&'a mut [u32]>,
    ink: u32,
    w: usize,
    h: usize,
}

impl<'a> Grid<'a> {
    fn new(cells: &'a mut [char], w: usize, h: usize) -> Self {
        Self { cells, colors: None, ink: color::NONE, w, h }
    }

    #[inline]
    fn put(&mut self, x: isize, y: isize, c: char) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = y as usize * self.w + x as usize;
            self.cells[i] = c;
            if let Some(cols) = self.colors.as_deref_mut() {
                cols[i] = self.ink;
            }
        }
    }

    #[inline]
    fn get(&self, x: isize, y: isize) -> Option<char> {
        (x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h)
            .then(|| self.cells[y as usize * self.w + x as usize])
    }
}

/// Glyph for a line segment's direction (cell units, y down).
fn slope_char(dx: f32, dy: f32, char_aspect: f32) -> char {
    let (px, py) = (dx * char_aspect, dy);
    let a = py.atan2(px).to_degrees().rem_euclid(180.0);
    if !(22.5..157.5).contains(&a) {
        '-'
    } else if (67.5..112.5).contains(&a) {
        '|'
    } else if a < 90.0 {
        '\\' // going right and down
    } else {
        '/'
    }
}

/// Draw a line between continuous cell positions, clipped to `clip` (x0, y0, x1, y1).
fn draw_line(
    g: &mut Grid<'_>,
    (x0, y0): (f32, f32),
    (x1, y1): (f32, f32),
    clip: (isize, isize, isize, isize),
    ch: impl Fn(char) -> Option<char>,
) {
    let n = (x1 - x0).abs().max((y1 - y0).abs()).ceil().max(1.0) as usize;
    for s in 0..=n {
        let t = s as f32 / n as f32;
        let x = (x0 + (x1 - x0) * t).floor() as isize;
        let y = (y0 + (y1 - y0) * t).floor() as isize;
        if x < clip.0 || y < clip.1 || x >= clip.2 || y >= clip.3 {
            continue;
        }
        if let Some(old) = g.get(x, y)
            && let Some(c) = ch(old)
        {
            g.put(x, y, c);
        }
    }
}

fn draw_text(g: &mut Grid<'_>, x: usize, y: usize, max_w: usize, text: &str, font: &Font, ink: char) {
    if font.scale == 0 {
        for (i, c) in text.chars().filter(|c| !c.is_control()).take(max_w).enumerate() {
            g.put((x + i) as isize, y as isize, c);
        }
        return;
    }
    let (s, pw) = (font.scale, font.px_w);
    for (i, c) in text.chars().enumerate() {
        let cx = x + i * 4 * pw;
        if cx + 3 * pw > x + max_w {
            break;
        }
        for (row, bits) in font_glyph(c).iter().enumerate() {
            for col in 0..3 {
                if bits & (4 >> col) == 0 {
                    continue;
                }
                for dy in 0..s {
                    for dx in 0..pw {
                        g.put((cx + col * pw + dx) as isize, (y + row * s + dy) as isize, ink);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ the layers

/// Wire models per slot plus scratch buffers, reused across frames.
pub struct Overlays {
    wires: Vec<Wire>,
    mask: Vec<bool>,
    lum: Vec<f32>,
    cells: Vec<char>,
    colors: Vec<u32>,
    /// Luminance under the halo (after the other overlays), for halo-only
    /// updates.
    base: Vec<f32>,
    /// Cells the halo currently covers, as a mask and as a list.
    halo_mask: Vec<bool>,
    halo_cells: Vec<usize>,
    /// Per cell, the solid whose halo was drawn there (valid where
    /// `halo_mask` is set).
    halo_owner: Vec<u8>,
    scratch: Vec<usize>,
    dirty: Vec<usize>,
    tags: TagState,
    journal: Journal,
    /// Per-frame colour table: solid × glyph level (per-solid sources).
    lut: Vec<u32>,
}

/// What to draw this frame.
pub struct OverlayOptions<'a> {
    pub wireframe: bool,
    pub legend: Option<&'a LegendParams>,
    /// One label per slot.
    pub labels: &'a [String],
    pub char_aspect: f32,
    pub ramp: &'a Ramp,
    pub exposure: f32,
    /// Colour gradient mapping; `None` = monochrome.
    pub color: Option<&'a ColorMap>,
    /// Each solid's colour value 0..=1 for the per-solid colour sources
    /// (load, spin, sound).
    pub levels: &'a [f32],
    /// Halo: a second, larger wireframe of every solid drawn on top, its size
    /// and brightness driven by the caller (e.g. a sound level).
    pub halo: Option<Halo>,
    /// Per-slot halo sizes overriding `halo.factor` (empty = all solids use
    /// `halo.factor`).
    pub halo_factors: &'a [f32],
    /// What the halo's colour comes from (with colour on).
    pub halo_color: HaloColor,
    /// Solid tags: each solid's label floats at a fixed place on screen with
    /// a leader line to the solid ([`TagState`]), on top of everything.
    pub tags: bool,
    /// One tag text per slot (falls back to the solid's name).
    pub tag_labels: &'a [String],
    /// Tag text size: 0 = one line, k = block letters drawn k times as
    /// large (for small text).
    pub tag_scale: u8,
    /// The block letters' font.
    pub tag_font: TagFont,
    /// Where the tags go.
    pub tag_place: TagPlace,
    /// Fixed tag length (beside, below and trailing tags): placed for each
    /// solid's current size, so the leader line keeps one length; off: for
    /// the solid at full size, so tags hold still as solids grow.
    pub tag_fixed: bool,
    /// Tag colour when colour is on ([`color::NONE`] = default text colour;
    /// see [`color::TagColor`]).
    pub tag_ink: u32,
}

// ------------------------------------------------------------------ tags

/// Where one solid's tag sits this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TagAnchor {
    /// Text row.
    y: usize,
    /// Column of the text's attaching end (the end nearest the solid).
    x: usize,
    /// Text runs right from `x` (the solid is to its lower/upper left),
    /// else left.
    rightward: bool,
    /// The tag is above its solid.
    above: bool,
    /// Where its leader line starts, (column, row), when not at the box's
    /// inner corner (a tag centred under its solid starts at its top middle).
    lead: Option<(f32, f32)>,
}

/// Underscores either side of a tag's label (`__RAM 42%__`).
const TAG_RULE: &str = "__";
/// Columns reserved per tag when laying them out (labels change length).
const TAG_MIN_W: usize = 14;
/// Room kept around a solid for its halo before the tag starts, as a
/// multiple of the solid's radius: the halo's reach (1.5x when not given),
/// or just clear of the solid without a halo. The tag does not bounce with the
/// halo; its leader line ends at the halo wherever it is.
const TAG_CLEAR_HALO: f32 = 1.5;
const TAG_CLEAR_PLAIN: f32 = 1.05;
/// Rows of leader line beyond that room.
const TAG_GAP_ROWS: f32 = 3.0;
/// How much further out (rows along the diagonal) a tag first looks for a
/// spot clear of the other solids before trying its other sides.
const TAG_SLACK_ROWS: f32 = 6.0;
/// A tag only moves to the solid's other side once the solid is this far
/// (share of the width or height) past the picture's centre line, so it
/// doesn't flicker.
const TAG_FLIP: f32 = 0.06;

/// Tags on a circle: its diameter as a share of the picture's height.
pub const TAG_CIRCLE: f32 = 0.8;
/// Trailing tags: how far their trail direction turns toward the solid's
/// latest motion each frame.
const TAG_TRAIL_EASE: f32 = 0.12;
/// Columns kept clear between a corner's stacked tags and the picture's
/// edge, and rows between them.
const TAG_CORNER_MARGIN: usize = 1;
/// How far a circle tag moves toward its target angle each frame (eased,
/// so tags glide round the circle).
const TAG_CIRCLE_EASE: f32 = 0.2;
/// A circle tag turns its text to the circle's other side only once it is
/// this far (cosine or sine of its angle) past the dividing axis.
const TAG_CIRCLE_FLIP: f32 = 0.15;

/// A trailing tag's memory: its solid's last screen position (column,
/// row) and the trail direction (unit; columns scaled by the cell aspect).
type Trail = ((f32, f32), (f32, f32));

/// Each solid's tag side from last frame, and the cells the tags covered
/// last frame, with what lay under them.
#[derive(Debug, Clone, Default)]
struct TagState {
    /// (horizontal, vertical) side per solid: +1 = right / below, −1 =
    /// left / above, 0 = not placed yet.
    sides: Vec<(i8, i8)>,
    /// Circle tags: each solid's tag angle round the circle (radians, NaN =
    /// not placed yet).
    angles: Vec<f32>,
    /// Trailing tags: each solid's last screen position and its trail
    /// direction (unit, rows; columns scaled by the cell aspect).
    trails: Vec<Option<Trail>>,
    /// Cells holding tag text this frame (cleared after drawing).
    text_mask: Vec<bool>,
    /// The legend panel (rows r0..=r1, columns c0..=c1) from the last full
    /// compose: tags and their leader lines keep out of it.
    keep_out: Option<(usize, usize, usize, usize)>,
    /// (cell, glyph, colour) under each tag cell drawn last.
    under: Vec<(usize, char, u32)>,
    /// Per-cell stamp: cells already in `under` this frame.
    stamp: Vec<u32>,
    epoch: u32,
}

impl TagState {
    fn begin(&mut self, n: usize) {
        self.under.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 || self.stamp.len() != n {
            self.stamp.clear();
            self.stamp.resize(n, 0);
            self.epoch = 1;
        }
    }
}

/// Records each changed cell's previous glyph and colour once, so
/// [`Overlays::update_halo`] can report exactly what changed on screen.
#[derive(Debug, Clone, Default)]
struct Journal {
    on: bool,
    epoch: u32,
    stamp: Vec<u32>,
    saved: Vec<(usize, char, u32)>,
}

impl Journal {
    fn begin(&mut self, n: usize) {
        self.on = true;
        self.saved.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 || self.stamp.len() != n {
            self.stamp.clear();
            self.stamp.resize(n, 0);
            self.epoch = 1;
        }
    }

    #[inline]
    fn touch(&mut self, c: usize, cells: &[char], colors: &[u32]) {
        if self.on && self.stamp[c] != self.epoch {
            self.stamp[c] = self.epoch;
            self.saved.push((c, cells[c], colors.get(c).copied().unwrap_or(color::NONE)));
        }
    }
}

/// Screen radius (rows) of a sphere of `radius` around `center`, and its
/// projected centre.
fn screen_circle(frame: &FrameView<'_>, center: Vec3A, radius: f32) -> Option<((f32, f32), f32)> {
    let proj = frame.proj;
    let c = proj.project(center)?;
    let up = proj.project(center + proj.cam.up * radius)?;
    Some((c, (up.1 - c.1).abs().max(0.5)))
}

/// Three-row block letters for tags on small text, built from ordinary
/// glyphs (`_ | / \ ( ) < > . o`). Unknown characters show as `?`.
fn block_glyph(c: char) -> [&'static str; 3] {
    match c.to_ascii_uppercase() {
        'A' => [" _ ", "|_|", "| |"],
        'B' => [" _ ", "|_)", "|_)"],
        'C' => [" _ ", "|  ", "|_ "],
        'D' => [" _ ", "| \\", "|_/"],
        'E' => [" _ ", "|_ ", "|_ "],
        'F' => [" _ ", "|_ ", "|  "],
        'G' => [" _ ", "|  ", "|_]"],
        'H' => ["   ", "|_|", "| |"],
        'I' => ["   ", " | ", " | "],
        'J' => ["   ", "  |", "|_|"],
        'K' => ["   ", "|_/", "| \\"],
        'L' => ["   ", "|  ", "|_ "],
        'M' => ["    ", "|\\/|", "|  |"],
        'N' => ["    ", "|\\ |", "| \\|"],
        'O' => [" _ ", "| |", "|_|"],
        'P' => [" _ ", "|_)", "|  "],
        'Q' => [" _ ", "| |", "|_\\"],
        'R' => [" _ ", "|_)", "| \\"],
        'S' => [" _ ", "(_ ", " _)"],
        'T' => ["___", " | ", " | "],
        'U' => ["   ", "| |", "|_|"],
        'V' => ["   ", "\\ /", " v "],
        'W' => ["    ", "|  |", "|/\\|"],
        'X' => ["   ", "\\_/", "/ \\"],
        'Y' => ["   ", "\\_/", " | "],
        'Z' => ["__ ", " / ", "/_ "],
        '0' => [" _ ", "|/|", "|_|"],
        '1' => ["  ", "/|", " |"],
        '2' => [" _ ", " _)", "(_ "],
        '3' => [" _ ", " _)", " _)"],
        '4' => ["   ", "|_|", "  |"],
        '5' => [" _ ", "|_ ", " _)"],
        '6' => [" _ ", "|_ ", "|_)"],
        '7' => [" _ ", "  /", " / "],
        '8' => [" _ ", "(_)", "(_)"],
        '9' => [" _ ", "(_|", "  |"],
        '%' => ["   ", "o/ ", "/o "],
        '.' => [" ", " ", "."],
        ',' => [" ", " ", ","],
        ':' => [" ", ".", "."],
        '-' => ["  ", "__", "  "],
        '+' => ["   ", "_|_", " | "],
        '/' => ["  ", " /", "/ "],
        '(' => [" ", "(", "("],
        ')' => [" ", ")", ")"],
        '<' => ["  ", "/_", "\\ "],
        '>' => ["  ", "_\\", " /"],
        ' ' => ["  ", "  ", "  "],
        _ => [" _ ", " _)", " . "],
    }
}

/// The rows of one tag's text: one row (`__RAM 42%__`) at `scale` 0, or
/// three rows of [`block_glyph`] letters (one blank column between letters
/// and either side, so the box hides what is behind it), drawn `scale`
/// times as large ([`scale_blocks`]).
#[cfg(test)]
fn tag_lines(label: &str, scale: u8) -> Vec<Vec<char>> {
    tag_lines_in(label, scale, TagFont::Strokes)
}

/// [`tag_lines`] in block-letter font `font`: the strokes font's three
/// rows, or five rows of [`pixel_glyph`] letters drawn with the font's ink.
fn tag_lines_in(label: &str, scale: u8, font: TagFont) -> Vec<Vec<char>> {
    let label: String = label.chars().filter(|c| !c.is_control()).collect();
    if scale == 0 {
        return vec![format!("{TAG_RULE}{label}{TAG_RULE}").chars().collect()];
    }
    let Some(ink) = font.ink() else {
        let mut rows = vec![vec![' '], vec![' '], vec![' ']];
        for c in label.chars() {
            for (row, part) in rows.iter_mut().zip(block_glyph(c)) {
                row.extend(part.chars());
                row.push(' ');
            }
        }
        return scale_blocks(&rows, scale, false);
    };
    let ink: Vec<char> = ink.chars().collect();
    let blank = vec![' '; ink.len()];
    let mut rows = vec![blank.clone(); 5];
    for c in label.chars() {
        for (row, part) in rows.iter_mut().zip(pixel_glyph(c)) {
            for p in part.chars() {
                row.extend(if p == '#' { &ink } else { &blank });
            }
            row.extend(&blank);
        }
    }
    scale_blocks(&rows, scale, true)
}

/// Five-row pixel letters (`#` = lit) for the pixel fonts. Unknown
/// characters show as `?`.
fn pixel_glyph(c: char) -> [&'static str; 5] {
    match c.to_ascii_uppercase() {
        'A' => [".#.", "#.#", "###", "#.#", "#.#"],
        'B' => ["##.", "#.#", "##.", "#.#", "##."],
        'C' => [".##", "#..", "#..", "#..", ".##"],
        'D' => ["##.", "#.#", "#.#", "#.#", "##."],
        'E' => ["###", "#..", "##.", "#..", "###"],
        'F' => ["###", "#..", "##.", "#..", "#.."],
        'G' => [".##", "#..", "#.#", "#.#", ".##"],
        'H' => ["#.#", "#.#", "###", "#.#", "#.#"],
        'I' => ["###", ".#.", ".#.", ".#.", "###"],
        'J' => ["..#", "..#", "..#", "#.#", ".#."],
        'K' => ["#.#", "#.#", "##.", "#.#", "#.#"],
        'L' => ["#..", "#..", "#..", "#..", "###"],
        'M' => ["#...#", "##.##", "#.#.#", "#...#", "#...#"],
        'N' => ["#..#", "##.#", "#.##", "#..#", "#..#"],
        'O' => [".#.", "#.#", "#.#", "#.#", ".#."],
        'P' => ["##.", "#.#", "##.", "#..", "#.."],
        'Q' => [".#.", "#.#", "#.#", "##.", ".##"],
        'R' => ["##.", "#.#", "##.", "#.#", "#.#"],
        'S' => [".##", "#..", ".#.", "..#", "##."],
        'T' => ["###", ".#.", ".#.", ".#.", ".#."],
        'U' => ["#.#", "#.#", "#.#", "#.#", "###"],
        'V' => ["#.#", "#.#", "#.#", "#.#", ".#."],
        'W' => ["#...#", "#...#", "#.#.#", "##.##", "#...#"],
        'X' => ["#.#", "#.#", ".#.", "#.#", "#.#"],
        'Y' => ["#.#", "#.#", ".#.", ".#.", ".#."],
        'Z' => ["###", "..#", ".#.", "#..", "###"],
        '0' => ["###", "#.#", "#.#", "#.#", "###"],
        '1' => [".#.", "##.", ".#.", ".#.", "###"],
        '2' => ["##.", "..#", ".#.", "#..", "###"],
        '3' => ["##.", "..#", ".#.", "..#", "##."],
        '4' => ["#.#", "#.#", "###", "..#", "..#"],
        '5' => ["###", "#..", "##.", "..#", "##."],
        '6' => [".##", "#..", "###", "#.#", "###"],
        '7' => ["###", "..#", ".#.", ".#.", ".#."],
        '8' => ["###", "#.#", "###", "#.#", "###"],
        '9' => ["###", "#.#", "###", "..#", "##."],
        '%' => ["#.#", "..#", ".#.", "#..", "#.#"],
        '.' => [".", ".", ".", ".", "#"],
        ',' => [".", ".", ".", "#", "#"],
        ':' => [".", "#", ".", "#", "."],
        '-' => ["...", "...", "###", "...", "..."],
        '+' => ["...", ".#.", "###", ".#.", "..."],
        '/' => ["..#", "..#", ".#.", "#..", "#.."],
        '(' => [".#", "#.", "#.", "#.", ".#"],
        ')' => ["#.", ".#", ".#", ".#", "#."],
        '<' => ["..#", ".#.", "#..", ".#.", "..#"],
        '>' => ["#..", ".#.", "..#", ".#.", "#.."],
        ' ' => ["..", "..", "..", "..", ".."],
        _ => ["##.", "..#", ".#.", "...", ".#."],
    }
}

/// Block letters `k` times as large: every cell becomes a k × k square.
/// Pixel fonts (`pixel`) fill it with the cell's glyph; the strokes font
/// redraws it so strokes stay strokes: `_` along its bottom, `|`, `(` and
/// `)` down its middle, `/` and `\` across its diagonal, anything else once
/// in its middle.
fn scale_blocks(rows: &[Vec<char>], k: u8, pixel: bool) -> Vec<Vec<char>> {
    let k = k.max(1) as usize;
    if k == 1 {
        return rows.to_vec();
    }
    if pixel {
        return rows
            .iter()
            .flat_map(|row| {
                let wide: Vec<char> = row.iter().flat_map(|&c| std::iter::repeat_n(c, k)).collect();
                std::iter::repeat_n(wide, k)
            })
            .collect();
    }
    let wide = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![vec![' '; wide * k]; rows.len() * k];
    let mid = (k - 1) / 2;
    for (r, row) in rows.iter().enumerate() {
        for (c, &ch) in row.iter().enumerate() {
            let mut set = |i: usize, j: usize, g: char| out[r * k + i][c * k + j] = g;
            match ch {
                ' ' => {}
                '_' => (0..k).for_each(|j| set(k - 1, j, '_')),
                '|' | '(' | ')' => (0..k).for_each(|i| set(i, mid, ch)),
                '/' => (0..k).for_each(|i| set(i, k - 1 - i, '/')),
                '\\' => (0..k).for_each(|i| set(i, i, '\\')),
                '.' | ',' => set(k - 1, mid, ch),
                _ => set(k / 2, mid, ch),
            }
        }
    }
    out
}

/// A tag's box, (columns, rows); one-line tags reserve a little width (labels
/// change length).
fn tag_box(rows: &[Vec<char>]) -> (usize, usize) {
    let wide = rows.iter().map(Vec::len).max().unwrap_or(0);
    (if rows.len() == 1 { wide.max(TAG_MIN_W) } else { wide }, rows.len())
}

/// Place every solid's tag box for this frame, next to the solid, always
/// pointing from the inside out: on the 45° diagonal from the solid toward
/// the picture's corner nearest it (up and right for a solid in the upper
/// right quarter, and so on), with the text running outward so the leader
/// line only spans the gap between box and solid. Along that diagonal (the
/// leader line's path) the box sits as close to its solid as it can while
/// staying clear of it (at full size, with the room its halo needs, plus a
/// short gap for the line), of every other solid and of the other tags,
/// going further out when it must. `sides` keeps each tag's corner from
/// frame to frame, changing only once the solid is clearly past the
/// picture's centre line, so tags don't flip back and forth. `boxes` is
/// each tag's (columns, rows). Leader lines never cross one another when
/// any other spot avoids it. With `fixed` (fixed tag length) tags are
/// placed for each solid's current size and position, so the leader keeps
/// one length and only grows to keep the box off other solids; otherwise
/// for the solid at full size, so tags hold still as solids grow.
#[allow(clippy::too_many_arguments)]
fn place_tags(
    sides: &mut Vec<(i8, i8)>,
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    boxes: &[(usize, usize)],
    clear: f32,
    aspect: f32,
    keep_out: Option<(usize, usize, usize, usize)>,
    fixed: bool,
) -> Vec<Option<TagAnchor>> {
    let n = scene.solids.len();
    sides.resize(n, (0, 0));
    let (hw, hh) = (w as f32 * 0.5, h as f32 * 0.5);
    // Placed for each solid at full size (rings: a bigger solid also sits
    // higher), so a solid growing and shrinking with its load doesn't move
    // its tag; only the camera does. The radius includes the halo's room.
    let circles: Vec<Option<((f32, f32), f32)>> = scene
        .solids
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if fixed {
                screen_circle(frame, s.center, s.scale)
            } else {
                screen_circle(frame, scene.full_size_center(i), s.scale.max(scene.max_radius))
            }
        })
        .map(|c| c.filter(|((x, y), _)| x.is_finite() && y.is_finite()).map(|(c, r)| (c, r * clear)))
        .collect();
    // Does the box (rows r0..=r1, columns c0..=c1) come within `margin` rows
    // of the circle?
    let touches = |(r0, r1, c0, c1): (f32, f32, f32, f32), ((cx, cy), r): ((f32, f32), f32), margin: f32| {
        let px = cx.clamp(c0, c1 + 1.0);
        let py = cy.clamp(r0, r1 + 1.0);
        ((px - cx) * aspect).hypot(py - cy) < r + margin
    };
    // Nearer solids (larger on screen) choose first.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let r = |i: usize| circles[i].map_or(0.0, |c| c.1);
        r(b).total_cmp(&r(a))
    });
    // (row0, row1, col0, col1); the legend panel counts as taken.
    let mut placed: Vec<(usize, usize, usize, usize)> = keep_out.into_iter().collect();
    // Leader lines placed so far, (x, y) from the tag's corner to the
    // solid's edge.
    let mut leaders: Vec<Segment> = Vec::new();
    let mut out = vec![None; n];
    for i in order {
        let Some(((cx, cy), rr)) = circles[i] else { continue };
        let (tw, th) = boxes.get(i).copied().unwrap_or((TAG_MIN_W, 1));
        let (tw, th) = (tw.clamp(1, w.max(1)), th.clamp(1, h.max(1)));
        // The corner nearest the solid: its quarter of the picture.
        let (mut sx, mut sy) = sides[i];
        let (out_x, out_y) = (cx - hw, cy - hh);
        if sx == 0 {
            sx = if out_x >= 0.0 { 1 } else { -1 };
        } else if out_x * (sx as f32) < -TAG_FLIP * w as f32 {
            sx = -sx; // clearly on the other half now: follow it outward
        }
        if sy == 0 {
            sy = if out_y >= 0.0 { 1 } else { -1 };
        } else if out_y * (sy as f32) < -TAG_FLIP * h as f32 {
            sy = -sy;
        }
        // The box with its inner corner `d` rows out along the diagonal:
        // (near row, attach column, row0, row1, col0, col1).
        let spot = |sx: i8, sy: i8, d: f32| {
            let y = (cy + sy as f32 * d).round();
            let x = (cx + sx as f32 * d / aspect).round();
            let (r0, r1) = if sy < 0 { (y - th as f32 + 1.0, y) } else { (y, y + th as f32 - 1.0) };
            let (c0, c1) = if sx > 0 { (x, x + tw as f32 - 1.0) } else { (x - tw as f32 + 1.0, x) };
            (r0 >= 0.0 && r1 < h as f32 && c0 >= 0.0 && c1 < w as f32).then_some((y, x, r0, r1, c0, c1))
        };
        let clash = |(r0, r1, c0, c1): (f32, f32, f32, f32), placed: &[(usize, usize, usize, usize)]| {
            placed.iter().any(|&(p0, p1, q0, q1)| {
                r0 <= p1 as f32 + 1.0 && p0 as f32 <= r1 + 1.0 && c0 <= q1 as f32 + 1.0 && q0 as f32 <= c1 + 1.0
            })
        };
        // Closest spot along the diagonal: clear of its own solid by the
        // leader gap, of the others outright (when `others`), and of tags.
        // `slack`: how many rows past the closest spot clear of its own solid
        // the search may go.
        let own_d = |sx: i8, sy: i8| {
            let mut d = 0.0;
            while d <= rr + 4.0 * th as f32 + 12.0 {
                if spot(sx, sy, d).is_some_and(|s| !touches((s.2, s.3, s.4, s.5), ((cx, cy), rr), TAG_GAP_ROWS)) {
                    return d;
                }
                d += 1.0;
            }
            d
        };
        // The leader from a tag corner at (x, y) to the solid's edge.
        let leader = |x: f32, y: f32| {
            let (vx, vy) = ((x - cx) * aspect, y - cy);
            let len = vx.hypot(vy).max(1e-3);
            let k = (rr / len).min(1.0);
            ((x, y), (cx + (x - cx) * k, cy + (y - cy) * k))
        };
        let nearest = |sx: i8, sy: i8, others: bool, slack: f32, uncrossed: bool, placed: &[(usize, usize, usize, usize)], leaders: &[Segment]| {
            let d0 = own_d(sx, sy);
            let far = (d0 + slack).min(rr * 2.0 + 4.0 * th as f32 + 24.0);
            let mut d = d0;
            while d <= far {
                if let Some(s) = spot(sx, sy, d) {
                    let b = (s.2, s.3, s.4, s.5);
                    let free = !touches(b, ((cx, cy), rr), TAG_GAP_ROWS)
                        && !(others && circles.iter().enumerate().any(|(j, c)| j != i && c.is_some_and(|c| touches(b, c, 0.0))))
                        && !clash(b, placed)
                        && !(uncrossed && leaders.iter().any(|&l| segments_cross(leader(s.1, s.0), l)));
                    if free {
                        return Some(s);
                    }
                }
                d += 1.0;
            }
            None
        };
        // The outward diagonal first: clear of every solid and close; then
        // however far; then clear of its own solid only. Only when that whole
        // diagonal is blocked (the picture's edge, the legend, other tags)
        // the other corner on the same side (up for down or down for up):
        // a tag never points in toward the picture's centre.
        // Leaders that cross another tag's leader are a last resort.
        let tries = [(true, TAG_SLACK_ROWS), (true, f32::INFINITY), (false, TAG_SLACK_ROWS), (false, f32::INFINITY)];
        let choice = [true, false].into_iter().find_map(|uncrossed| {
            [(sx, sy), (sx, -sy)].into_iter().find_map(|(hx, vy)| {
                tries
                    .iter()
                    .find_map(|&(others, slack)| nearest(hx, vy, others, slack, uncrossed, &placed, &leaders).map(|s| (s, hx, vy)))
            })
        });
        let found = choice.is_some();
        // Nowhere free: the outward side, pulled onto the picture (dropped
        // for this frame if that would cover the legend).
        let ((y, x, r0, r1, c0, c1), hx, vy) = choice.unwrap_or_else(|| {
            let d = rr + TAG_GAP_ROWS;
            let vy = sy;
            let y = (cy + vy as f32 * d).round();
            let y = if vy < 0 { y.clamp(th as f32 - 1.0, h as f32 - 1.0) } else { y.clamp(0.0, (h - th) as f32) };
            let x = (cx + sx as f32 * d / aspect).round();
            let x = if sx > 0 { x.clamp(0.0, (w - tw) as f32) } else { x.clamp(tw as f32 - 1.0, w as f32 - 1.0) };
            let (r0, r1) = if vy < 0 { (y - th as f32 + 1.0, y) } else { (y, y + th as f32 - 1.0) };
            let (c0, c1) = if sx > 0 { (x, x + tw as f32 - 1.0) } else { (x - tw as f32 + 1.0, x) };
            ((y, x, r0, r1, c0, c1), sx, vy)
        });
        if !found && keep_out.is_some_and(|k| clash((r0, r1, c0, c1), &[k])) {
            continue;
        }
        placed.push((r0 as usize, r1 as usize, c0 as usize, c1 as usize));
        leaders.push(leader(x, y));
        // Remember the outward corner (a neighbouring one is only a detour).
        sides[i] = (sx, sy);
        out[i] = Some(TagAnchor { y: y as usize, x: x as usize, rightward: hx > 0, above: vy < 0, lead: None });
    }
    out
}
/// Place the tags on a circle round the picture's centre, [`TAG_CIRCLE`] of
/// its height across (fixed tag length off). Each tag heads for the point
/// of the circle nearest its solid (straight out from the centre through
/// the solid), so its leader line is as short as the circle allows, and
/// eases toward it frame by frame ([`TAG_CIRCLE_EASE`]), gliding round as
/// the solids and camera move. Neighbouring tags push each other apart
/// along the circle until their boxes no longer overlap. A tag's text runs
/// outward from the circle, turning to the other side only once the tag
/// is clearly past the dividing axis. `angles` and `sides` carry each
/// tag's angle and text side from frame to frame.
#[allow(clippy::too_many_arguments)]
fn place_tags_circle(
    angles: &mut Vec<f32>,
    sides: &mut Vec<(i8, i8)>,
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    boxes: &[(usize, usize)],
    aspect: f32,
    keep_out: Option<(usize, usize, usize, usize)>,
) -> Vec<Option<TagAnchor>> {
    let n = scene.solids.len();
    angles.resize(n, f32::NAN);
    sides.resize(n, (0, 0));
    let (hw, hh) = (w as f32 * 0.5, h as f32 * 0.5);
    // radius in rows; a column is `aspect` rows wide
    let radius = (TAG_CIRCLE * 0.5 * h as f32).max(1.0);
    let centres: Vec<Option<(f32, f32)>> = scene
        .solids
        .iter()
        .map(|s| frame.proj.project(s.center).filter(|(x, y)| x.is_finite() && y.is_finite()))
        .collect();
    let wrap = |a: f32| (a + PI).rem_euclid(TAU) - PI;
    // Ease each tag toward the angle of its solid seen from the centre (a
    // solid right at the centre keeps its tag where it is).
    let placed: Vec<usize> = (0..n).filter(|&i| centres[i].is_some()).collect();
    for (k, &i) in placed.iter().enumerate() {
        let (cx, cy) = centres[i].expect("placed");
        let (dx, dy) = ((cx - hw) * aspect, cy - hh);
        let target = if dx.hypot(dy) > 0.5 {
            dy.atan2(dx)
        } else if angles[i].is_nan() {
            k as f32 * TAU / placed.len() as f32 - PI
        } else {
            angles[i]
        };
        angles[i] = if angles[i].is_nan() { target } else { wrap(angles[i] + wrap(target - angles[i]) * TAG_CIRCLE_EASE) };
    }
    // Half each box's length along the circle at its angle (in rows), and
    // push neighbours apart until they clear one another.
    let half = |i: usize, a: f32| {
        let (tw, th) = boxes.get(i).copied().unwrap_or((TAG_MIN_W, 1));
        0.5 * (tw as f32 * aspect * a.sin().abs() + th as f32 * a.cos().abs()) + 1.0
    };
    for _ in 0..24 {
        let mut order = placed.clone();
        order.sort_by(|&a, &b| angles[a].total_cmp(&angles[b]));
        let m = order.len();
        if m < 2 {
            break;
        }
        let mut moved = false;
        for k in 0..m {
            let (a, b) = (order[k], order[(k + 1) % m]);
            let gap = wrap(angles[b] - angles[a]).rem_euclid(TAU);
            let need = ((half(a, angles[a]) + half(b, angles[b])) / radius).min(TAU / m as f32);
            if gap < need {
                let push = 0.5 * (need - gap);
                angles[a] = wrap(angles[a] - push);
                angles[b] = wrap(angles[b] + push);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    let mut out = vec![None; n];
    for &i in &placed {
        let a = angles[i];
        let (px, py) = (hw + radius * a.cos() / aspect, hh + radius * a.sin());
        // The text runs outward from the circle (right on its right half, up
        // on its top half), so tags ring the picture instead of crossing
        // it; near the vertical (horizontal) axis it keeps its side.
        let (mut sx, mut sy) = sides[i];
        if sx == 0 || a.cos() * (sx as f32) < -TAG_CIRCLE_FLIP {
            sx = if a.cos() >= 0.0 { 1 } else { -1 };
        }
        if sy == 0 || a.sin() * (sy as f32) < -TAG_CIRCLE_FLIP {
            sy = if a.sin() < 0.0 { -1 } else { 1 };
        }
        sides[i] = (sx, sy);
        let (tw, th) = boxes.get(i).copied().unwrap_or((TAG_MIN_W, 1));
        let (tw, th) = (tw.clamp(1, w.max(1)) as f32, th.clamp(1, h.max(1)) as f32);
        // keep the whole box on the picture
        let x = if sx > 0 { px.round().clamp(0.0, w as f32 - tw) } else { px.round().clamp(tw - 1.0, w as f32 - 1.0) };
        let y = if sy < 0 { py.round().clamp(th - 1.0, h as f32 - 1.0) } else { py.round().clamp(0.0, h as f32 - th) };
        // a tag that would cover the legend is left out this frame
        let (r0, r1) = if sy < 0 { (y - th + 1.0, y) } else { (y, y + th - 1.0) };
        let (c0, c1) = if sx > 0 { (x, x + tw - 1.0) } else { (x - tw + 1.0, x) };
        if keep_out.is_some_and(|(p0, p1, q0, q1)| r0 <= p1 as f32 && p0 as f32 <= r1 && c0 <= q1 as f32 && q0 as f32 <= c1) {
            continue;
        }
        out[i] = Some(TagAnchor { y: y as usize, x: x as usize, rightward: sx > 0, above: sy < 0, lead: None });
    }
    out
}

/// Each solid's circle on screen for the tags placed beside it: at its
/// current size (`fixed`) or at full size, widened by `clear` for its halo.
fn tag_circles(frame: &FrameView<'_>, scene: &Scene, fixed: bool, clear: f32) -> Vec<Option<((f32, f32), f32)>> {
    scene
        .solids
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if fixed {
                screen_circle(frame, s.center, s.scale)
            } else {
                screen_circle(frame, scene.full_size_center(i), s.scale.max(scene.max_radius))
            }
        })
        .map(|c| c.filter(|((x, y), _)| x.is_finite() && y.is_finite()).map(|(c, r)| (c, r * clear)))
        .collect()
}

/// Does a box (rows r0..=r1, columns c0..=c1) meet one in `placed`, with a
/// cell between them?
fn box_clash((r0, r1, c0, c1): (f32, f32, f32, f32), placed: &[(f32, f32, f32, f32)]) -> bool {
    placed.iter().any(|&(p0, p1, q0, q1)| r0 <= p1 + 1.0 && p0 <= r1 + 1.0 && c0 <= q1 + 1.0 && q0 <= c1 + 1.0)
}

fn legend_box(keep_out: Option<(usize, usize, usize, usize)>) -> Vec<(f32, f32, f32, f32)> {
    keep_out.into_iter().map(|(r0, r1, c0, c1)| (r0 as f32, r1 as f32, c0 as f32, c1 as f32)).collect()
}

/// Tags centred under their solids (clear of the halo, with a short gap for
/// the leader, which runs straight up from the box's top middle). A tag
/// that would sit on another steps down below it.
#[allow(clippy::too_many_arguments)]
fn place_tags_below(
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    boxes: &[(usize, usize)],
    clear: f32,
    fixed: bool,
    keep_out: Option<(usize, usize, usize, usize)>,
) -> Vec<Option<TagAnchor>> {
    let circles = tag_circles(frame, scene, fixed, clear);
    let mut placed = legend_box(keep_out);
    // the highest solids first, so lower tags step down past them
    let mut order: Vec<usize> = (0..circles.len()).filter(|&i| circles[i].is_some()).collect();
    order.sort_by(|&a, &b| circles[a].map_or(0.0, |c| c.0.1).total_cmp(&circles[b].map_or(0.0, |c| c.0.1)));
    let mut out = vec![None; circles.len()];
    for i in order {
        let ((cx, cy), rr) = circles[i].expect("placed");
        let (tw, th) = boxes.get(i).copied().unwrap_or((TAG_MIN_W, 1));
        let (tw, th) = (tw.clamp(1, w.max(1)) as f32, th.clamp(1, h.max(1)) as f32);
        let x0 = (cx - tw * 0.5).round().clamp(0.0, w as f32 - tw);
        let mut top = (cy + rr + TAG_GAP_ROWS).round();
        for _ in 0..8 {
            if !box_clash((top, top + th - 1.0, x0, x0 + tw - 1.0), &placed) {
                break;
            }
            top += th + 1.0;
        }
        let top = top.clamp(0.0, h as f32 - th);
        let b = (top, top + th - 1.0, x0, x0 + tw - 1.0);
        if keep_out.is_some() && box_clash(b, &legend_box(keep_out)) {
            continue;
        }
        placed.push(b);
        let lead = Some(((x0 + (tw * 0.5).floor()).min(w as f32 - 1.0) + 0.5, top - 1.0 + 0.5));
        out[i] = Some(TagAnchor { y: top as usize, x: x0 as usize, rightward: true, above: false, lead });
    }
    out
}

/// Tags stacked in the picture's corners: each in the corner on its
/// solid's side (its quarter of the picture, kept until the solid is
/// clearly past the centre line), the stack running in from the corner in
/// the order of the solids' heights so leaders rarely cross.
#[allow(clippy::too_many_arguments)]
fn place_tags_corner(
    sides: &mut Vec<(i8, i8)>,
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    boxes: &[(usize, usize)],
    keep_out: Option<(usize, usize, usize, usize)>,
) -> Vec<Option<TagAnchor>> {
    let n = scene.solids.len();
    sides.resize(n, (0, 0));
    let (hw, hh) = (w as f32 * 0.5, h as f32 * 0.5);
    let centres: Vec<Option<(f32, f32)>> =
        scene.solids.iter().map(|s| frame.proj.project(s.center).filter(|(x, y)| x.is_finite() && y.is_finite())).collect();
    for (i, c) in centres.iter().enumerate() {
        let Some((cx, cy)) = *c else { continue };
        let (mut sx, mut sy) = sides[i];
        if sx == 0 || (cx - hw) * (sx as f32) < -TAG_FLIP * w as f32 {
            sx = if cx >= hw { 1 } else { -1 };
        }
        if sy == 0 || (cy - hh) * (sy as f32) < -TAG_FLIP * h as f32 {
            sy = if cy >= hh { 1 } else { -1 };
        }
        sides[i] = (sx, sy);
    }
    let legend = legend_box(keep_out);
    let m = TAG_CORNER_MARGIN as f32;
    let mut out = vec![None; n];
    for (sx, sy) in [(-1i8, -1i8), (1, -1), (-1, 1), (1, 1)] {
        let mut group: Vec<usize> = (0..n).filter(|&i| centres[i].is_some() && sides[i] == (sx, sy)).collect();
        // top corners stack downward, highest solid first; bottom ones upward
        group.sort_by(|&a, &b| {
            let y = |i: usize| centres[i].map_or(0.0, |c| c.1);
            if sy < 0 { y(a).total_cmp(&y(b)) } else { y(b).total_cmp(&y(a)) }
        });
        let mut run = m;
        for i in group {
            let (tw, th) = boxes.get(i).copied().unwrap_or((TAG_MIN_W, 1));
            let (tw, th) = (tw.clamp(1, w.max(1)) as f32, th.clamp(1, h.max(1)) as f32);
            let (r0, r1) = if sy < 0 { (run, run + th - 1.0) } else { (h as f32 - 1.0 - run - th + 1.0, h as f32 - 1.0 - run) };
            run += th + 1.0;
            if r0 < 0.0 || r1 > h as f32 - 1.0 {
                break; // the corner is full
            }
            let (c0, c1) = if sx < 0 { (m, m + tw - 1.0) } else { (w as f32 - m - tw, w as f32 - 1.0 - m) };
            if c0 < 0.0 || box_clash((r0, r1, c0, c1), &legend) {
                continue;
            }
            // attach at the inner end, toward the picture's middle
            let (x, rightward) = if sx < 0 { (c1, false) } else { (c0, true) };
            let (y, above) = if sy < 0 { (r1, true) } else { (r0, false) };
            out[i] = Some(TagAnchor { y: y as usize, x: x as usize, rightward, above, lead: None });
        }
    }
    out
}

/// Tags trailing behind their solids: each sits just past its solid's halo
/// on the side away from the way the solid is moving across the picture,
/// the direction easing round as the motion changes ([`TAG_TRAIL_EASE`]);
/// a solid that holds still keeps its tag where it is (below and to the
/// right at first). `trails` carries each solid's last position and trail
/// direction from frame to frame.
#[allow(clippy::too_many_arguments)]
fn place_tags_trailing(
    trails: &mut Vec<Option<Trail>>,
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    boxes: &[(usize, usize)],
    clear: f32,
    fixed: bool,
    aspect: f32,
) -> Vec<Option<TagAnchor>> {
    let circles = tag_circles(frame, scene, fixed, clear);
    trails.resize(circles.len(), None);
    let unit = |(x, y): (f32, f32)| {
        let l = x.hypot(y);
        if l > 1e-6 { (x / l, y / l) } else { (std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2) }
    };
    let mut out = vec![None; circles.len()];
    for (i, c) in circles.iter().enumerate() {
        let Some(((cx, cy), rr)) = *c else {
            trails[i] = None;
            continue;
        };
        let (last, dir) = trails[i].unwrap_or(((cx, cy), unit((1.0, 1.0))));
        let v = ((cx - last.0) * aspect, cy - last.1);
        let speed = v.0.hypot(v.1);
        let dir = if speed > 0.05 {
            let back = (-v.0 / speed, -v.1 / speed);
            unit((dir.0 + (back.0 - dir.0) * TAG_TRAIL_EASE, dir.1 + (back.1 - dir.1) * TAG_TRAIL_EASE))
        } else {
            dir
        };
        trails[i] = Some(((cx, cy), dir));
        let d = rr + TAG_GAP_ROWS;
        let (px, py) = (cx + dir.0 * d / aspect, cy + dir.1 * d);
        let (sx, sy) = (if dir.0 >= 0.0 { 1 } else { -1 }, if dir.1 < 0.0 { -1 } else { 1 });
        let (tw, th) = boxes.get(i).copied().unwrap_or((TAG_MIN_W, 1));
        let (tw, th) = (tw.clamp(1, w.max(1)) as f32, th.clamp(1, h.max(1)) as f32);
        let x = if sx > 0 { px.round().clamp(0.0, w as f32 - tw) } else { px.round().clamp(tw - 1.0, w as f32 - 1.0) };
        let y = if sy < 0 { py.round().clamp(th - 1.0, h as f32 - 1.0) } else { py.round().clamp(0.0, h as f32 - th) };
        out[i] = Some(TagAnchor { y: y as usize, x: x as usize, rightward: sx > 0, above: sy < 0, lead: None });
    }
    out
}

/// A line on the grid: (x, y) to (x, y).
type Segment = ((f32, f32), (f32, f32));

/// Whether segments `a` and `b` cross (touching end points do not count).
fn segments_cross(a: Segment, b: Segment) -> bool {
    let cross = |o: (f32, f32), p: (f32, f32), q: (f32, f32)| (p.0 - o.0) * (q.1 - o.1) - (p.1 - o.1) * (q.0 - o.0);
    let (d1, d2) = (cross(b.0, b.1, a.0), cross(b.0, b.1, a.1));
    let (d3, d4) = (cross(a.0, a.1, b.0), cross(a.0, a.1, b.1));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Width / height of a cell in the frame's projection.
fn frame_aspect(frame: &FrameView<'_>) -> f32 {
    let p = frame.proj;
    // a unit step right vs up at the same depth, in cells
    let o = p.cam.pos + p.cam.forward * 10.0;
    match (p.project(o), p.project(o + p.cam.right), p.project(o + p.cam.up)) {
        (Some(a), Some(b), Some(c)) => ((c.1 - a.1).abs() / (b.0 - a.0).abs().max(1e-6)).clamp(0.05, 20.0),
        _ => 0.5,
    }
}

/// Draw every solid's tag and leader line on top of `cells`, remembering
/// what lay under them in `t.under`.
#[allow(clippy::too_many_arguments)]
fn draw_tags(
    t: &mut TagState,
    j: &mut Journal,
    cells: &mut [char],
    colors: &mut [u32],
    halo_mask: &[bool],
    wire_mask: Option<&[bool]>,
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    opt: &OverlayOptions<'_>,
) {
    t.under.clear();
    let n = scene.solids.len();
    if !opt.tags || n == 0 || w < 4 || h < 2 || cells.len() < w * h || frame.ids.len() < w * h {
        return;
    }
    t.begin(w * h);
    let color_on = colors.len() == cells.len();
    let ink = opt.tag_ink;
    let aspect = frame_aspect(frame);
    let sizes = opt.halo_sizes();
    let lines: Vec<Vec<Vec<char>>> = scene
        .solids
        .iter()
        .enumerate()
        .map(|(i, inst)| {
            let label = opt.tag_labels.get(i).cloned().unwrap_or_else(|| inst.poly.kind().id().to_string());
            tag_lines_in(&label, opt.tag_scale, opt.tag_font)
        })
        .collect();
    let boxes: Vec<(usize, usize)> = lines.iter().map(|l| tag_box(l)).collect();
    let clear = if sizes.is_some() { opt.halo.map_or(TAG_CLEAR_HALO, |hl| hl.reach.max(TAG_CLEAR_PLAIN)) } else { TAG_CLEAR_PLAIN };
    let fixed = opt.tag_fixed;
    let anchors = match opt.tag_place {
        TagPlace::Circle => place_tags_circle(&mut t.angles, &mut t.sides, frame, w, h, scene, &boxes, aspect, t.keep_out),
        TagPlace::Beside => place_tags(&mut t.sides, frame, w, h, scene, &boxes, clear, aspect, t.keep_out, fixed),
        TagPlace::Corner => place_tags_corner(&mut t.sides, frame, w, h, scene, &boxes, t.keep_out),
        TagPlace::Below => place_tags_below(frame, w, h, scene, &boxes, clear, fixed, t.keep_out),
        TagPlace::Trailing => place_tags_trailing(&mut t.trails, frame, w, h, scene, &boxes, clear, fixed, aspect),
    };
    let mut text_cells: Vec<usize> = Vec::new();
    let put = |c: usize, ch: char, t: &mut TagState, j: &mut Journal, cells: &mut [char], colors: &mut [u32]| {
        j.touch(c, cells, colors);
        if t.stamp[c] != t.epoch {
            t.stamp[c] = t.epoch;
            t.under.push((c, cells[c], if color_on { colors[c] } else { color::NONE }));
        }
        cells[c] = ch;
        if color_on {
            colors[c] = ink;
        }
    };
    // All tag texts first, then the leader lines, which never cross any
    // tag's text.
    t.text_mask.resize(w * h, false);
    for (i, rows) in lines.iter().enumerate() {
        let Some(a) = anchors[i] else { continue };
        // The box's rows run away from the solid from the near row `a.y`; its
        // columns run outward from `a.x`. Every cell of the box is drawn
        // (spaces included), so it hides what is behind it.
        let (th, tw) = (rows.len(), rows.iter().map(Vec::len).max().unwrap_or(0));
        let top = if a.above { a.y as isize + 1 - th as isize } else { a.y as isize };
        let x0 = if a.rightward { a.x as isize } else { a.x as isize + 1 - tw as isize };
        for (k, row) in rows.iter().enumerate() {
            let y = top + k as isize;
            if y < 0 || y >= h as isize {
                continue;
            }
            for col in 0..tw {
                let x = x0 + col as isize;
                if x >= 0 && (x as usize) < w {
                    let c = y as usize * w + x as usize;
                    text_cells.push(c);
                    t.text_mask[c] = true;
                    put(c, row.get(col).copied().unwrap_or(' '), t, j, cells, colors);
                }
            }
        }
    }
    for (i, inst) in scene.solids.iter().enumerate() {
        let Some(a) = anchors[i] else { continue };
        // Leader line from beside the attaching end toward the solid's
        // centre, stopping at its halo / wireframe / surface.
        let factor = sizes.as_ref().map_or(1.0, |s| s.get(i).or(s.last()).copied().unwrap_or(1.0));
        let Some(((cx, cy), r)) = screen_circle(frame, inst.center, inst.scale * factor.max(1.0)) else { continue };
        // Start at the tag's inner corner, on the solid's side: the line runs
        // only through the gap between the two, never under the text.
        let (sx, sy) = a.lead.unwrap_or((
            if a.rightward { a.x as f32 - 1.0 } else { a.x as f32 + 1.0 } + 0.5,
            a.y as f32 + if a.above { 1.0 } else { -1.0 } + 0.5,
        ));
        let (dx, dy) = (cx - sx, cy - sy);
        if dy * if a.above { 1.0 } else { -1.0 } <= 0.0 {
            continue; // the solid isn't past the tag's row: no line needed
        }
        // Diagonal or vertical strokes only, one glyph per row (a 45° line
        // crosses two columns per row: `/` steps two columns at a time).
        let ch = match slope_char(dx, dy, aspect) {
            '-' => {
                if (dx > 0.0) == (dy > 0.0) { '\\' } else { '/' }
            }
            c => c,
        };
        let steps = dy.abs().ceil().max(1.0) as usize;
        let mut last = usize::MAX;
        for s in 0..=steps {
            let f = s as f32 / steps as f32;
            let (x, y) = (sx + dx * f, sy + dy * f);
            if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                break;
            }
            let c = y as usize * w + x as usize;
            if c == last {
                continue;
            }
            last = c;
            // distance to the centre in rows (columns scaled by the aspect)
            let d = ((x - cx) * aspect).hypot(y - cy);
            if d <= r * 1.05 {
                let edge = frame.ids[c] as usize == i
                    || (sizes.is_some() && halo_mask.get(c).copied().unwrap_or(false))
                    || wire_mask.is_some_and(|m| m.get(c).copied().unwrap_or(false));
                if edge || d <= r * 0.35 {
                    break;
                }
            }
            let in_legend = t.keep_out.is_some_and(|(r0, r1, c0, c1)| {
                let (row, col) = (c / w, c % w);
                (r0..=r1).contains(&row) && (c0..=c1).contains(&col)
            });
            if t.text_mask[c] || in_legend {
                continue;
            }
            put(c, ch, t, j, cells, colors);
        }
    }
    for &c in &text_cells {
        t.text_mask[c] = false;
    }
}

impl OverlayOptions<'_> {
    /// Solid `i`'s halo size as a level 0..=1 (smallest to largest of its
    /// swing).
    fn halo_level(&self, i: usize) -> f32 {
        let Some(hl) = self.halo else { return 0.0 };
        let f = self.halo_factors.get(i).copied().unwrap_or(hl.factor);
        let swing = hl.reach - HALO_BASE;
        if swing > 1e-4 { (0.5 + 0.5 * (f - HALO_BASE) / swing).clamp(0.0, 1.0) } else { 0.5 }
    }

    /// Halo size per solid, or `None` without a halo.
    fn halo_sizes(&self) -> Option<Vec<f32>> {
        self.halo.map(|hl| {
            let f = if self.halo_factors.is_empty() { &[hl.factor][..] } else { self.halo_factors };
            f.iter().map(|v| if v.is_finite() { v.max(1.0) } else { 1.0 }).collect()
        })
    }
}

/// The halo for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Halo {
    /// Size as a multiple of each solid's radius ([`halo_factor`]).
    pub factor: f32,
    /// Luminosity of its edges 0..=1 ([`halo_lum`]).
    pub lum: f32,
    /// The largest size it can reach this run ([`halo_reach`]): tag boxes
    /// keep this far from their solids so the halo never runs into them.
    pub reach: f32,
}

/// Halo size at the middle level, as a multiple of the solid's radius.
pub const HALO_BASE: f32 = 1.20;
/// Halo size swings on offer, as a share of the solid's radius either way:
/// ±10% (1.1x..1.3x), ±30% (0.9x..1.5x) and ±60% (0.6x..1.8x).
pub const HALO_SWINGS: [f32; 3] = [0.10, 0.30, 0.60];

/// Halo size: [`HALO_BASE`] times the solid's radius, ±`swing` of the radius
/// with a level 0..=1 (e.g. a sound level: 0 = silent, 1 = loudest recently).
pub fn halo_factor(level: f32, swing: f32) -> f32 {
    let l = if level.is_finite() { level.clamp(0.0, 1.0) } else { 0.0 };
    HALO_BASE + swing.max(0.0) * (2.0 * l - 1.0)
}

/// The largest halo with `swing`: how far tags keep clear of a solid.
pub fn halo_reach(swing: f32) -> f32 {
    HALO_BASE + swing.max(0.0)
}

/// Dimmest halo (silence, or no low frequencies).
pub const HALO_LUM_MIN: f32 = 0.2;

/// Halo luminosity from a level 0..=1 (e.g. bass-weighted spectrum): the
/// more low-frequency energy, the brighter.
pub fn halo_lum(level: f32) -> f32 {
    let l = if level.is_finite() { level.clamp(0.0, 1.0) } else { 0.0 };
    HALO_LUM_MIN + (1.0 - HALO_LUM_MIN) * l
}

/// Cells changed by [`Overlays::update_halo`], with the whole updated grid.
pub struct HaloUpdate<'a> {
    /// Changed cell indices, ascending.
    pub dirty: &'a [usize],
    pub chars: &'a [char],
    pub colors: Option<&'a [u32]>,
}

/// Colour of one cell with luminance `l`, as [`Overlays::colorize`] gives it.
fn cell_color(cm: &ColorMap, frame: &FrameView<'_>, ramp: &Ramp, levels: &[f32], n_solids: usize, i: usize, l: f32) -> u32 {
    let id = frame.ids.get(i).copied().unwrap_or(crate::render::NO_HIT) as usize;
    if id < n_solids {
        cm.solid_color(levels.get(id).copied(), l, ramp.level(l), ramp.len())
    } else {
        cm.bg_color(l)
    }
}

/// Colour of halo cell `c` (brightness `l`) drawn for solid `owner`, by
/// `opt.halo_color`.
fn halo_cell_color(cm: &ColorMap, opt: &OverlayOptions<'_>, frame: &FrameView<'_>, n_solids: usize, owner: usize, c: usize, l: f32) -> u32 {
    let (lvl, nr) = (opt.ramp.level(l), opt.ramp.len());
    match opt.halo_color {
        HaloColor::Under => cell_color(cm, frame, opt.ramp, opt.levels, n_solids, c, l),
        HaloColor::Solid if cm.solid.is_some() => cm.solid_color(opt.levels.get(owner).copied(), l, lvl, nr),
        HaloColor::Solid => cm.bg_color(l),
        HaloColor::Size => cm.value_color(opt.halo_level(owner), lvl, nr),
        HaloColor::Bass => {
            let bass = opt.halo.map_or(0.0, |hl| (hl.lum - HALO_LUM_MIN) / (1.0 - HALO_LUM_MIN));
            cm.value_color(bass.clamp(0.0, 1.0), lvl, nr)
        }
    }
}

/// Mark every visible edge cell of every solid's wire model scaled by
/// `factors[i]` for solid `i` (1 = the solid itself; past the end of
/// `factors`, its last value) in `mask`; newly marked cells are also pushed
/// to `list`, and their solid's index written to `owner`.
#[allow(clippy::too_many_arguments)]
fn mark_edges(
    wires: &[Wire],
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    factors: &[f32],
    mask: &mut [bool],
    mut list: Option<&mut Vec<usize>>,
    mut owner: Option<&mut [u8]>,
) {
    if frame.depth.len() < w * h || frame.ids.len() < w * h || mask.len() < w * h {
        return;
    }
    let proj = frame.proj;
    let eye = proj.cam.pos;
    for (i, (inst, wire)) in scene.solids.iter().zip(wires).enumerate() {
        let rot = inst.inv_rot.transpose();
        let factor = factors.get(i).or(factors.last()).copied().unwrap_or(1.0);
        let scale = inst.scale * factor;
        let front: Vec<bool> = wire
            .faces
            .iter()
            .map(|&(n, o)| {
                let nw = rot * n;
                nw.dot(eye - (inst.center + nw * (o * scale))) > 0.0
            })
            .collect();
        let world: Vec<Vec3A> = wire.verts.iter().map(|&v| inst.center + rot * (v * scale)).collect();
        // Depth slack: an edge is hidden only by something clearly in front of it.
        let tol = 0.2 * inst.scale + 0.02;
        for &(a, b, f1, f2) in &wire.edges {
            if !(front[f1] || front[f2]) {
                continue; // back edge: the solid itself hides it
            }
            let (pa, pb) = (world[a], world[b]);
            let (Some(sa), Some(sb)) = (proj.project(pa), proj.project(pb)) else { continue };
            let n = (2.0 * (sb.0 - sa.0).abs().max((sb.1 - sa.1).abs())).ceil().max(1.0) as usize;
            for s in 0..=n {
                let p = pa.lerp(pb, s as f32 / n as f32);
                let Some((x, y)) = proj.project(p) else { continue };
                if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                    continue;
                }
                let c = y as usize * w + x as usize;
                if mask[c] {
                    continue;
                }
                if frame.ids[c] as usize == i || frame.depth[c] >= (p - eye).length() - tol {
                    mask[c] = true;
                    if let Some(l) = list.as_deref_mut() {
                        l.push(c);
                    }
                    if let Some(slot) = owner.as_deref_mut().and_then(|o| o.get_mut(c)) {
                        *slot = i as u8;
                    }
                }
            }
        }
    }
}

/// The composed picture: glyphs and, when colour is on, per-cell colours.
pub struct Cells<'a> {
    pub chars: &'a [char],
    pub colors: Option<&'a [u32]>,
}

impl Overlays {
    pub fn new(scene: &Scene) -> Self {
        let wires = scene.solids.iter().map(|s| Wire::new(s.poly.kind())).collect();
        Self {
            wires,
            mask: Vec::new(),
            lum: Vec::new(),
            cells: Vec::new(),
            colors: Vec::new(),
            base: Vec::new(),
            halo_mask: Vec::new(),
            halo_cells: Vec::new(),
            halo_owner: Vec::new(),
            scratch: Vec::new(),
            dirty: Vec::new(),
            tags: TagState::default(),
            journal: Journal::default(),
            lut: Vec::new(),
        }
    }

    /// Redraw only the halo (and tags) over the last composed frame (same `frame`,
    /// size and scene as that [`Overlays::compose`]): the halo moves to
    /// `opt.halo` and only cells whose glyph or colour changed are reported.
    /// `opt.halo = None` removes it. Legend panels are not re-applied (call
    /// `compose` for those).
    pub fn update_halo(
        &mut self,
        frame: &FrameView<'_>,
        w: usize,
        h: usize,
        scene: &Scene,
        opt: &OverlayOptions<'_>,
    ) -> HaloUpdate<'_> {
        self.dirty.clear();
        let color_on = opt.color.is_some();
        if self.base.len() == w * h && self.cells.len() == w * h && self.halo_mask.len() == w * h {
            // Every write goes through the journal, which keeps each cell's
            // on-screen value from before this update.
            self.journal.begin(w * h);
            let colors: &mut [u32] = if color_on && self.colors.len() == w * h { &mut self.colors } else { &mut [] };
            // Lift last frame's tags off the picture.
            for &(c, g, k) in &self.tags.under {
                self.journal.touch(c, &self.cells, colors);
                self.cells[c] = g;
                if let Some(slot) = colors.get_mut(c) {
                    *slot = k;
                }
            }
            for &c in &self.halo_cells {
                self.halo_mask[c] = false;
            }
            let mut fresh = std::mem::take(&mut self.scratch);
            fresh.clear();
            self.halo_owner.resize(w * h, 0);
            if let Some(sizes) = opt.halo_sizes() {
                mark_edges(&self.wires, frame, w, h, scene, &sizes, &mut self.halo_mask, Some(&mut fresh), Some(&mut self.halo_owner));
            }
            let lum_on = opt.halo.map_or(0.0, |hl| hl.lum);
            let n = scene.solids.len();
            for &c in self.halo_cells.iter().chain(&fresh) {
                let on = self.halo_mask[c];
                let l = if on { lum_on } else { self.base[c] };
                self.journal.touch(c, &self.cells, colors);
                self.cells[c] = opt.ramp.glyph(l);
                if let Some(slot) = colors.get_mut(c) {
                    *slot = opt.color.map_or(color::NONE, |cm| {
                        if on {
                            halo_cell_color(cm, opt, frame, n, self.halo_owner[c] as usize, c, l)
                        } else {
                            cell_color(cm, frame, opt.ramp, opt.levels, n, c, l)
                        }
                    });
                }
            }
            self.scratch = std::mem::replace(&mut self.halo_cells, fresh);
            // Tags back on top; their leader lines follow the halo.
            let wire = opt.wireframe.then_some(&self.mask[..]).filter(|m| m.len() == w * h);
            draw_tags(
                &mut self.tags,
                &mut self.journal,
                &mut self.cells,
                colors,
                &self.halo_mask,
                wire,
                frame,
                w,
                h,
                scene,
                opt,
            );
            let (cells, colors) = (&self.cells, &*colors);
            self.dirty.extend(
                self.journal
                    .saved
                    .iter()
                    .filter(|&&(c, g, k)| cells[c] != g || colors.get(c).is_some_and(|&now| now != k))
                    .map(|s| s.0),
            );
            self.journal.on = false;
            self.dirty.sort_unstable();
        }
        HaloUpdate {
            dirty: &self.dirty,
            chars: &self.cells,
            colors: color_on.then_some(&self.colors[..]),
        }
    }

    /// Fill `self.colors` from the final luminances.
    /// Glyphs and palette colours from the final luminances in one pass
    /// (same result as [`ascii::shade`] plus [`cell_color`] per cell): the
    /// glyph level is found once, per-solid colours come from a solid ×
    /// level table built per frame, brightness colours index the palette
    /// directly.
    fn shade_colorize(&mut self, cm: &ColorMap, frame: &FrameView<'_>, ramp: &Ramp, levels: &[f32], n_solids: usize) {
        let nr = ramp.len();
        let per_solid = cm.by.per_solid() && cm.solid.is_some();
        self.lut.clear();
        if per_solid {
            for k in 0..n_solids {
                let v = levels.get(k).copied();
                self.lut.extend((0..nr).map(|b| cm.solid_color(v, 0.0, b, nr)));
            }
        }
        // Brightness palettes: `l` in 0..=1 → entry `⌊l·n⌋` (saturating for
        // negatives and NaN, clamped at the top), the same as `Palette::at`.
        let solid_lum: &[u32] = match &cm.solid {
            Some(p) if !per_solid => p.colors(),
            _ => &[],
        };
        let bg: &[u32] = cm.bg.as_ref().map_or(&[], |p| p.colors());
        let pick = |pal: &[u32], l: f32| -> u32 {
            match pal.len() {
                0 => color::NONE,
                n => pal[((l * n as f32) as usize).min(n - 1)],
            }
        };
        let n_cells = self.lum.len();
        self.cells.clear();
        self.colors.clear();
        self.cells.reserve(n_cells);
        self.colors.reserve(n_cells);
        let ids = frame.ids;
        for (i, &l) in self.lum.iter().enumerate() {
            let lvl = ramp.level(l);
            self.cells.push(ramp.glyph_at(lvl));
            let id = ids.get(i).copied().unwrap_or(crate::render::NO_HIT) as usize;
            let c = if id >= n_solids {
                pick(bg, l)
            } else if per_solid {
                self.lut[id * nr + lvl]
            } else {
                pick(solid_lum, l)
            };
            self.colors.push(c);
        }
    }

    /// Tone-map the frame, apply the overlays and return the `w × h` grid.
    pub fn compose(
        &mut self,
        frame: &FrameView<'_>,
        w: usize,
        h: usize,
        scene: &Scene,
        opt: &OverlayOptions<'_>,
    ) -> Cells<'_> {
        // (a solid driven by complexity changes its shape)
        self.wires.resize_with(scene.solids.len(), || Wire::new(SolidKind::Tetrahedron));
        for (wire, s) in self.wires.iter_mut().zip(&scene.solids) {
            if wire.kind != s.poly.kind() {
                *wire = Wire::new(s.poly.kind());
            }
        }
        let ramp = opt.ramp;
        ascii::tone_buffer(frame.radiance, opt.exposure, &mut self.lum);
        self.lum.resize(w * h, 0.0);
        if opt.wireframe {
            self.mask.clear();
            self.mask.resize(w * h, false);
            mark_edges(&self.wires, frame, w, h, scene, &[1.0], &mut self.mask, None, None);
            for (l, &m) in self.lum.iter_mut().zip(&self.mask) {
                if m {
                    *l = invert(*l);
                }
            }
        }
        let panel = opt.legend.and_then(|p| layout(p, w, h, opt.labels, opt.char_aspect).map(|panel| (p, panel)));
        if let Some((p, panel)) = &panel {
            let keep = 1.0 - p.opacity.clamp(0.0, 1.0);
            for y in panel.y..panel.y + panel.h {
                for l in &mut self.lum[y * w + panel.x..y * w + panel.x + panel.w] {
                    *l *= keep;
                }
            }
        }
        // Halo on top, at its own luminosity; what lies under it is kept
        // for halo-only updates.
        self.halo_mask.clear();
        self.halo_mask.resize(w * h, false);
        self.halo_cells.clear();
        self.halo_owner.clear();
        self.halo_owner.resize(w * h, 0);
        if let (Some(hl), Some(sizes)) = (opt.halo, opt.halo_sizes()) {
            self.base.clone_from(&self.lum);
            mark_edges(&self.wires, frame, w, h, scene, &sizes, &mut self.halo_mask, Some(&mut self.halo_cells), Some(&mut self.halo_owner));
            for &c in &self.halo_cells {
                self.lum[c] = hl.lum;
            }
        } else {
            self.base.clear();
        }
        if let Some(cm) = opt.color {
            self.shade_colorize(cm, frame, ramp, opt.levels, scene.solids.len());
            if opt.halo_color != HaloColor::Under {
                let n = scene.solids.len();
                for &c in &self.halo_cells {
                    self.colors[c] = halo_cell_color(cm, opt, frame, n, self.halo_owner[c] as usize, c, self.lum[c]);
                }
            }
            if let Some((_, panel)) = &panel {
                // Panel background stays uncoloured so labels read clearly.
                for y in panel.y..panel.y + panel.h {
                    self.colors[y * w + panel.x..y * w + panel.x + panel.w].fill(color::NONE);
                }
            }
        }
        if opt.color.is_none() {
            ascii::shade(&self.lum, ramp, &mut self.cells);
        }
        if let Some((_, panel)) = &panel {
            let mut g = Grid::new(&mut self.cells, w, h);
            if opt.color.is_some() {
                g.colors = Some(&mut self.colors);
            }
            let icon_ink = |slot: usize| opt.color.map_or(color::NONE, |cm| cm.icon_color(opt.levels.get(slot).copied()));
            draw_legend(&mut g, panel, &self.wires, scene, opt, &icon_ink);
        }
        // Tags float above everything, the legend included.
        self.journal.on = false;
        let colors: &mut [u32] = if opt.color.is_some() && self.colors.len() == w * h { &mut self.colors } else { &mut [] };
        let wire = opt.wireframe.then_some(&self.mask[..]).filter(|m| m.len() == w * h);
        self.tags.keep_out = panel.as_ref().map(|(_, p)| (p.y, p.y + p.h.max(1) - 1, p.x, p.x + p.w.max(1) - 1));
        draw_tags(&mut self.tags, &mut self.journal, &mut self.cells, colors, &self.halo_mask, wire, frame, w, h, scene, opt);
        Cells { chars: &self.cells, colors: opt.color.is_some().then_some(&self.colors[..]) }
    }
}

fn draw_legend(
    g: &mut Grid<'_>,
    panel: &Panel,
    wires: &[Wire],
    scene: &Scene,
    opt: &OverlayOptions<'_>,
    icon_ink: &dyn Fn(usize) -> u32,
) {
    let (labels, ramp, char_aspect) = (opt.labels, opt.ramp, opt.char_aspect);
    let cam = scene.camera;
    let toward = -cam.forward;
    let ink = ramp.glyph(1.0);
    let hidden = '.';
    let (iw, ih) = (panel.icon_w, panel.icon_h);
    // Unit circumradius → icon cells, matching physical proportions.
    let r_rows = (ih as f32 - 1.0) * 0.5;
    let r_cols = (r_rows / char_aspect.max(0.1)).min((iw as f32 - 1.0) * 0.5);
    let r_rows = r_cols * char_aspect.max(0.1);

    for &(slot, ex, ey) in &panel.entries {
        let (Some(inst), Some(wire)) = (scene.solids.get(slot), wires.get(slot)) else { continue };
        let rot = inst.inv_rot.transpose();
        let (cx, cy) = (ex as f32 + iw as f32 * 0.5, ey as f32 + ih as f32 * 0.5);
        let screen: Vec<(f32, f32)> = wire
            .verts
            .iter()
            .map(|&v| {
                let q = rot * v;
                (cx + q.dot(cam.right) * r_cols, cy - q.dot(cam.up) * r_rows)
            })
            .collect();
        let front: Vec<bool> = wire.faces.iter().map(|&(n, _)| (rot * n).dot(toward) > 0.0).collect();
        let clip = (ex as isize, ey as isize, (ex + iw) as isize, (ey + ih) as isize);
        g.ink = icon_ink(slot);
        // Hidden edges first (dotted, only on blank cells), then visible ones on top.
        for pass in 0..2 {
            for &(a, b, f1, f2) in &wire.edges {
                let visible = front[f1] || front[f2];
                if visible != (pass == 1) {
                    continue;
                }
                let (pa, pb) = (screen[a], screen[b]);
                if visible {
                    let c = slope_char(pb.0 - pa.0, pb.1 - pa.1, char_aspect);
                    draw_line(g, pa, pb, clip, |_| Some(c));
                } else {
                    draw_line(g, pa, pb, clip, |old| (old == ' ').then_some(hidden));
                }
            }
        }
        // Label, vertically centred on the icon.
        let label = labels.get(slot).map(String::as_str).unwrap_or("");
        g.ink = color::NONE;
        let lx = ex + iw + 1;
        let ly = ey + (ih.saturating_sub(panel.font.height())) / 2;
        draw_text(g, lx, ly, panel.label_w, label, &panel.font, ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorBy;
    use crate::render::{RenderSettings, Renderer};
    use crate::scene::AnimParams;

    #[test]
    fn invert_keeps_separation() {
        for i in 0..=100 {
            let l = i as f32 / 100.0;
            let v = invert(l);
            assert!((0.0..=1.0).contains(&v));
            assert!((v - l).abs() >= 0.4 - 1e-6, "{l} -> {v}");
        }
        assert_eq!(invert(0.0), 1.0);
        assert_eq!(invert(1.0), 0.0);
    }

    #[test]
    fn positions_parse() {
        assert_eq!("tl".parse::<LegendPos>().unwrap(), LegendPos::TopLeft);
        assert_eq!("Bottom_Right".parse::<LegendPos>().unwrap(), LegendPos::BottomRight);
        assert!("middle".parse::<LegendPos>().is_err());
    }

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("CPU {i}%")).collect()
    }

    #[test]
    fn legend_fits_and_sits_in_its_corner() {
        let (w, h) = (120, 40);
        for pos in [LegendPos::TopLeft, LegendPos::TopRight, LegendPos::BottomLeft, LegendPos::BottomRight] {
            for size in LEGEND_SIZE_MIN..=LEGEND_SIZE_MAX {
                for n in [1, 5, 16] {
                    let p = LegendParams { size, opacity: 1.0, pos };
                    let Some(pl) = layout(&p, w, h, &labels(n), 0.5) else { continue };
                    assert!(pl.x + pl.w <= w && pl.y + pl.h <= h, "{pos:?} {size} {n}: {pl:?}");
                    let right = matches!(pos, LegendPos::TopRight | LegendPos::BottomRight);
                    let bottom = matches!(pos, LegendPos::BottomLeft | LegendPos::BottomRight);
                    assert_eq!(pl.x == w - pl.w, right || pl.w == w);
                    assert_eq!(pl.y == h - pl.h, bottom || pl.h == h);
                    for &(_, ex, ey) in &pl.entries {
                        assert!(ex + pl.icon_w <= pl.x + pl.w && ey + pl.icon_h <= pl.y + pl.h);
                    }
                }
            }
        }
        // 16 entries at size 2 need more than one column at 40 rows
        let pl = layout(&LegendParams::default(), w, h, &labels(16), 0.5).unwrap();
        assert_eq!(pl.entries.len(), 16);
        assert!(pl.entries.iter().any(|e| e.1 != pl.entries[0].1));
    }

    #[test]
    fn block_font_metrics() {
        let f = Font::for_size(3, 0.5);
        assert_eq!((f.scale, f.px_w, f.height(), f.width("AB")), (1, 1, 5, 7));
        let f = Font::for_size(5, 0.5);
        assert_eq!((f.scale, f.px_w, f.height()), (3, 3, 15));
        assert_eq!(Font::for_size(2, 0.5).width("CPU 5%"), 6);
        let mut cells = vec![' '; 8 * 5];
        draw_text(&mut Grid::new(&mut cells, 8, 5), 0, 0, 8, "T1", &Font::for_size(3, 0.5), '#');
        let rows: Vec<String> = cells.chunks(8).map(|r| r.iter().collect()).collect();
        assert_eq!(rows, ["###  #  ", " #  ##  ", " #   #  ", " #   #  ", " #  ### "]);
    }

    const W: usize = 100;
    const H: usize = 34;

    /// (plain glyphs, overlaid glyphs, colours, ids)
    fn frame_full(
        wireframe: bool,
        legend: Option<&LegendParams>,
        color: Option<&ColorMap>,
    ) -> (Vec<char>, Vec<char>, Option<Vec<u32>>, Vec<u8>) {
        frame_halo(wireframe, legend, color, None)
    }

    fn frame_halo(
        wireframe: bool,
        legend: Option<&LegendParams>,
        color: Option<&ColorMap>,
        halo: Option<Halo>,
    ) -> (Vec<char>, Vec<char>, Option<Vec<u32>>, Vec<u8>) {
        let scene = Scene::demo(&AnimParams::default());
        let mut r = Renderer::new();
        let rs = RenderSettings { spp: 1, ..RenderSettings::default() };
        r.render(&scene, W, H, &rs);
        let view = r.view().unwrap();
        let ramp = Ramp::default();
        let plain = ascii::glyphs(view.radiance, &ramp, 1.0);
        let mut o = Overlays::new(&scene);
        let labels = labels(scene.solids.len());
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let opt = OverlayOptions {
            wireframe,
            legend,
            labels: &labels,
            char_aspect: 0.5,
            ramp: &ramp,
            exposure: 1.0,
            color,
            levels: &levels,
            halo,
            halo_factors: &[],
            halo_color: Default::default(),
            tags: false,
            tag_labels: &[],
            tag_ink: color::NONE,
            tag_scale: 0,
            tag_place: TagPlace::Circle,
            tag_font: TagFont::Strokes,
            tag_fixed: false,
        };
        let out = o.compose(&view, W, H, &scene, &opt);
        (plain, out.chars.to_vec(), out.colors.map(<[u32]>::to_vec), view.ids.to_vec())
    }

    fn frame_with(wireframe: bool, legend: Option<&LegendParams>) -> (Vec<char>, Vec<char>) {
        let (p, o, _, _) = frame_full(wireframe, legend, None);
        (p, o)
    }

    #[test]
    fn lum_colours_follow_glyph_brightness() {
        let grey = crate::color::Gradient::parse("grey").unwrap();
        let cm = ColorMap::new(Some(grey.clone()), ColorBy::Lum, 16, Some(grey)).unwrap();
        let (_, chars, cols, _) = frame_full(false, None, Some(&cm));
        let cols = cols.expect("colours on");
        let ramp: Vec<char> = crate::ascii::DEFAULT_RAMP.chars().collect();
        // 16 colours over the 16-glyph ramp: one palette entry per glyph
        for (c, col) in chars.iter().zip(&cols) {
            let b = ramp.iter().position(|r| r == c).unwrap();
            assert_eq!(*col & 0xff, (b as f32 / 15.0 * 255.0).round() as u32, "glyph {c:?}");
        }
        // larger palettes add steps between glyphs but stay within their size
        for n in crate::color::PALETTE_SIZES {
            let grey = crate::color::Gradient::parse("grey").unwrap();
            let cm = ColorMap::new(Some(grey.clone()), ColorBy::Lum, n, Some(grey)).unwrap();
            let (_, _, cols, _) = frame_full(false, None, Some(&cm));
            let distinct: std::collections::HashSet<u32> = cols.unwrap().into_iter().collect();
            assert!(distinct.len() <= n && distinct.len() > n.min(16) / 2, "{n}: {} colours", distinct.len());
        }
        let (_, _, none, _) = frame_full(false, None, None);
        assert!(none.is_none(), "monochrome has no colour buffer");
    }

    #[test]
    fn load_colours_solids_only_and_legend_is_plain() {
        let red = crate::color::Gradient::parse("#000000:#ff0000").unwrap();
        let cm = ColorMap::new(Some(red), ColorBy::Load, 16, None).unwrap();
        let (_, _, cols, ids) = frame_full(false, None, Some(&cm));
        let cols = cols.unwrap();
        for (i, (&col, &id)) in cols.iter().zip(&ids).enumerate() {
            if (id as usize) < 5 {
                assert_eq!(col & 0x00ffff, 0, "pure red hue at {i}");
                if id == 0 {
                    assert_eq!(col, 0, "slot 0 is idle → first stop (black)");
                }
            } else {
                assert_eq!(col, color::NONE, "background uncoloured at {i}");
            }
        }
        assert!(cols.iter().any(|&c| c != color::NONE && c >> 16 > 0x80), "busy solids are bright red");

        // opaque legend: panel is uncoloured except icon strokes in the slot colour
        let p = LegendParams { size: 2, opacity: 1.0, pos: LegendPos::TopLeft };
        let (_, chars, cols, _) = frame_full(false, Some(&p), Some(&cm));
        let cols = cols.unwrap();
        let text: String = chars.chunks(W).map(|r| r.iter().collect::<String>() + "\n").collect();
        let row = text.lines().position(|l| l.contains("CPU 4%")).unwrap();
        let col = text.lines().nth(row).unwrap().find("CPU 4%").unwrap();
        assert_eq!(cols[row * W + col], color::NONE, "label text in default colour");
        assert!(cols.contains(&0xff0000), "slot 4 (full load) icon drawn in the last stop");
    }

    #[test]
    fn background_palette_is_independent_of_the_solids() {
        let g = |s: &str| crate::color::Gradient::parse(s).ok();
        let ice = crate::color::Palette::ramp(&g("ice").unwrap(), 32);
        for (solid, by) in [(g("#000000:#ff0000"), ColorBy::Load), (g("fire"), ColorBy::Lum), (None, ColorBy::Lum)] {
            let solid_pal = solid.clone();
            let cm = ColorMap::new(solid, by, 32, g("ice")).unwrap();
            let (_, _, cols, ids) = frame_full(false, None, Some(&cm));
            let cols = cols.unwrap();
            let mut bg_seen = 0;
            for (&col, &id) in cols.iter().zip(&ids) {
                if (id as usize) < 5 {
                    match &solid_pal {
                        None => assert_eq!(col, color::NONE, "solids uncoloured"),
                        Some(_) => assert!(cm.solid.as_ref().unwrap().colors().contains(&col), "solid #{col:06x}"),
                    }
                } else {
                    assert!(ice.colors().contains(&col), "background #{col:06x} from the ice palette");
                    bg_seen += 1;
                }
            }
            assert!(bg_seen > 0);
        }
    }

    fn tag_opts<'a>(ramp: &'a Ramp, labels: &'a [String], halo: Option<Halo>, wireframe: bool) -> OverlayOptions<'a> {
        OverlayOptions {
            wireframe,
            legend: None,
            labels,
            char_aspect: 0.5,
            ramp,
            exposure: 1.0,
            color: None,
            levels: &[],
            halo,
            halo_factors: &[],
            halo_color: Default::default(),
            tags: true,
            tag_labels: labels,
            tag_ink: color::NONE,
            tag_scale: 0,
            // beside each solid (the circle has its own test)
            tag_place: TagPlace::Beside,
            tag_font: TagFont::Strokes,
            tag_fixed: true,
        }
    }

    /// Row and column of `needle` in the grid.
    fn find(chars: &[char], w: usize, needle: &str) -> Option<(usize, usize)> {
        chars.chunks(w).enumerate().find_map(|(y, r)| r.iter().collect::<String>().find(needle).map(|x| (y, x)))
    }

    #[test]
    fn tags_follow_their_solids_with_short_leaders() {
        let (w, h) = (200, 70);
        let anim = AnimParams::default();
        // orbiting camera so the solids move across the picture
        let slots: Vec<crate::scene::SlotLayout> =
            (0..5).map(|i| crate::scene::SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let cam = crate::scene::CameraParams { mode: crate::scene::camera::CameraMode::Orbit, ..Default::default() };
        let mut scene = Scene::new(&slots, &anim, &cam);
        let labels: Vec<String> = ["CPU 12%", "RAM 40%", "GPU 3%", "DSK 1.0M", "NET 0.2M"].map(String::from).to_vec();
        let ramp = Ramp::default();
        let mut r = Renderer::new();
        let mut ov = Overlays::new(&scene);
        let mut sides = Vec::new();
        let mut prev: Option<Vec<(usize, usize)>> = None;
        let mut moved = false;
        for step in 0..6 {
            r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
            let view = r.view().unwrap();
            let aspect = frame_aspect(&view);
            let opt = tag_opts(&ramp, &labels, None, false);
            let mut plain_opt = tag_opts(&ramp, &labels, None, false);
            plain_opt.tags = false;
            let plain = Overlays::new(&scene).compose(&view, w, h, &scene, &plain_opt).chars.to_vec();
            let chars = ov.compose(&view, w, h, &scene, &opt).chars.to_vec();
            let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, 0))).collect();
            let anchors = place_tags(&mut sides, &view, w, h, &scene, &boxes, TAG_CLEAR_PLAIN, aspect, None, true);
            let mut at = Vec::new();
            for (i, s) in scene.solids.iter().enumerate() {
                let a = anchors[i].unwrap_or_else(|| panic!("step {step}: solid {i} tagged"));
                let ((cx, cy), rr) = screen_circle(&view, s.center, s.scale).unwrap();
                // the tag text is where the layout says, running away from its solid
                let text = format!("{TAG_RULE}{}{TAG_RULE}", labels[i]);
                let x0 = if a.rightward { a.x } else { a.x + 1 - text.len() };
                let row: String = chars[a.y * w + x0..a.y * w + x0 + text.len()].iter().collect();
                assert_eq!(row, text, "step {step}: tag {i} drawn at its anchor");
                assert_eq!(a.rightward, a.x as f32 > cx, "step {step}: tag {i} text runs outward");
                assert_eq!(a.above, (a.y as f32) < cy, "step {step}: tag {i} side");
                // next to the solid: past it (at full size), but only a short way
                let dy = (a.y as f32 - cy).abs();
                assert!(dy > rr * 0.7, "step {step}: tag {i} clear of its solid ({dy:.1} rows, r {rr:.1})");
                // The box keeps clear of its own solid (full size) and, unless
                // it sits close by, of every other solid too: it only goes
                // further out along the leader to get clear of them.
                // (fixed tag length: placed for each solid's current size)
                let full = |j: usize| {
                    let ((x, y), r) = screen_circle(&view, scene.solids[j].center, scene.solids[j].scale).unwrap();
                    ((x, y), r * TAG_CLEAR_PLAIN)
                };
                let n = labels[i].len() + 2 * TAG_RULE.len();
                let (c0, c1) = if a.rightward { (a.x, a.x + n.max(TAG_MIN_W) - 1) } else { (a.x + 1 - n.max(TAG_MIN_W), a.x) };
                let hits = |j: usize, margin: f32| {
                    let ((qx, qy), qr) = full(j);
                    let px = qx.clamp(c0 as f32, c1 as f32 + 1.0);
                    let py = qy.clamp(a.y as f32, a.y as f32 + 1.0);
                    ((px - qx) * aspect).hypot(py - qy) < qr + margin - 0.75
                };
                assert!(!hits(i, TAG_GAP_ROWS), "step {step}: tag {i}'s box clear of its solid");
                let ((_, fy), fr) = full(i);
                let fdy = (a.y as f32 - fy).abs();
                let close = fdy <= fr * 0.71 + TAG_GAP_ROWS + TAG_SLACK_ROWS + 2.0;
                let clear_of_all = (0..5).all(|j| j == i || !hits(j, 0.0));
                assert!(close || clear_of_all, "step {step}: tag {i} far out ({fdy:.1} rows) only to clear the other solids");
                at.push((a.y, a.x));
            }
            // no two tags overlap
            for (i, a) in anchors.iter().flatten().enumerate() {
                for (j, b) in anchors.iter().flatten().enumerate() {
                    let span = |t: &TagAnchor, k: usize| {
                        let n = labels[k].len() + 4;
                        if t.rightward { (t.x, t.x + n - 1) } else { (t.x + 1 - n, t.x) }
                    };
                    let (sa, sb) = (span(a, i), span(b, j));
                    assert!(i == j || a.y != b.y || sa.1 < sb.0 || sb.1 < sa.0, "step {step}: tags {i} and {j} overlap");
                }
            }
            // leader lines: only diagonal / vertical strokes, never horizontal,
            // and every stroke lies between a tag and its solid
            let underscores = chars.iter().filter(|&&c| c == '_').count();
            assert_eq!(underscores, 5 * 2 * TAG_RULE.len(), "step {step}: no horizontal leaders");
            for c in 0..w * h {
                if chars[c] == plain[c] || !matches!(chars[c], '/' | '\\' | '|') {
                    continue;
                }
                let (x, y) = ((c % w) as f32 + 0.5, (c / w) as f32 + 0.5);
                let between = scene.solids.iter().enumerate().any(|(i, s)| {
                    let a = anchors[i].unwrap();
                    let ((cx, cy), _) = screen_circle(&view, s.center, s.scale).unwrap();
                    let (ax, ay) = (a.x as f32 + 0.5, a.y as f32 + 0.5);
                    let in_rows = (y - ay) * (cy - ay) > 0.0 && (y - cy) * (ay - cy) >= -1.0;
                    let in_cols = x >= ax.min(cx) - 1.5 && x <= ax.max(cx) + 1.5;
                    in_rows && in_cols
                });
                assert!(between, "step {step}: stroke at ({x}, {y}) lies between a tag and its solid");
            }
            // each solid has a leader reaching it (unless its tag sits right beside it)
            for (i, s) in scene.solids.iter().enumerate() {
                let ((cx, cy), rr) = screen_circle(&view, s.center, s.scale).unwrap();
                let near = (0..w * h).any(|c| {
                    let (x, y) = ((c % w) as f32 + 0.5, (c / w) as f32 + 0.5);
                    let d = ((x - cx) * aspect).hypot(y - cy);
                    chars[c] != plain[c] && matches!(chars[c], '/' | '\\' | '|') && d <= rr * 1.1 + 2.0
                });
                assert!(near, "step {step}: solid {i}'s leader reaches it");
            }
            if let Some(p) = &prev {
                moved |= p != &at;
            }
            prev = Some(at);
            let ch = vec![crate::metrics::Channel { available: true, level: 0.7, spin_y: 0.7, spin_x: 0.3, activity: 0.7 }; 5];
            scene.update(&ch, &anim, 1.5);
        }
        assert!(moved, "tags move with their solids");
    }

    #[test]
    fn block_letter_tags_are_three_even_rows_and_clear_their_solids() {
        // every glyph's three rows line up
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%.,:-+/()<> ?".chars() {
            let g = block_glyph(c);
            assert!(g.iter().all(|r| r.chars().count() == g[0].chars().count()), "{c:?}: {g:?}");
        }
        let rows = tag_lines("RAM 42%", 1);
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.len() == rows[0].len()));
        let text: Vec<String> = rows.iter().map(|r| r.iter().collect()).collect();
        assert_eq!(text[1], " |_) |_| |\\/|    |_|  _) o/  ", "{text:?}");
        assert_eq!(text[2], " | \\ | | |  |      | (_  /o  ", "{text:?}");
        // drawn: three rows per tag, in place, halo-only updates agree
        let scene = Scene::demo(&AnimParams::default());
        let (w, h) = (220, 80);
        let mut r = Renderer::new();
        r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
        let view = r.view().unwrap();
        let ramp = Ramp::default();
        let labels: Vec<String> = ["CPU 12%", "RAM 40%", "GPU 3%", "DSK 1.0M", "NET 0.2M"].map(String::from).to_vec();
        let opt = |f: f32| OverlayOptions { tag_scale: 1, ..tag_opts(&ramp, &labels, Some(Halo { factor: f, lum: 0.9, reach: TAG_CLEAR_HALO }), false) };
        let mut inc = Overlays::new(&scene);
        let mut chars = inc.compose(&view, w, h, &scene, &opt(1.2)).chars.to_vec();
        let aspect = frame_aspect(&view);
        let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, 1))).collect();
        let anchors = place_tags(&mut Vec::new(), &view, w, h, &scene, &boxes, TAG_CLEAR_HALO, aspect, None, true);
        for (i, (a, l)) in anchors.iter().zip(&labels).enumerate() {
            let a = a.unwrap();
            let rows = tag_lines(l, 1);
            let tw = rows[0].len();
            let top = if a.above { a.y - 2 } else { a.y };
            let x0 = if a.rightward { a.x } else { a.x + 1 - tw };
            for (k, row) in rows.iter().enumerate() {
                let got: String = chars[(top + k) * w + x0..(top + k) * w + x0 + tw].iter().collect();
                assert_eq!(got, row.iter().collect::<String>(), "tag {i} row {k}");
            }
            // the whole box is clear of its own solid's halo room (at its
            // current size: fixed tag length)
            let s = &scene.solids[i];
            let ((cx, cy), rr) = screen_circle(&view, s.center, s.scale).unwrap();
            let px = cx.clamp(x0 as f32, (x0 + tw) as f32);
            let py = cy.clamp(top as f32, (top + 3) as f32);
            assert!(((px - cx) * aspect).hypot(py - cy) >= rr * TAG_CLEAR_HALO, "tag {i} box clear of its solid");
        }
        for f in [1.4, 1.25, 1.5] {
            let u = inc.update_halo(&view, w, h, &scene, &opt(f));
            for &c in u.dirty {
                chars[c] = u.chars[c];
            }
            let full = Overlays::new(&scene).compose(&view, w, h, &scene, &opt(f)).chars.to_vec();
            assert_eq!(chars, full, "halo {f}: incremental matches full");
        }
    }

    #[test]
    fn pixel_fonts_draw_every_glyph_in_five_even_rows() {
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%.,:-+/()<> ?".chars() {
            let g = pixel_glyph(c);
            assert!(g.iter().all(|r| r.len() == g[0].len() && r.chars().all(|p| p == '#' || p == '.')), "{c:?}: {g:?}");
        }
        for font in TagFont::ALL {
            for k in [1, 2] {
                let rows = tag_lines_in("NET 7%", k, font);
                let want = if font == TagFont::Strokes { 3 } else { 5 } * k as usize;
                assert_eq!(rows.len(), want, "{font:?} {k}x");
                assert!(rows.iter().all(|r| r.len() == rows[0].len()), "{font:?} {k}x: even rows");
                if let Some(ink) = font.ink() {
                    let lit: String = rows.iter().flatten().filter(|c| **c != ' ').collect();
                    assert!(!lit.is_empty() && lit.chars().all(|c| ink.contains(c)), "{font:?}: only its ink");
                }
            }
            // one-line tags are the same in every font
            assert_eq!(tag_lines_in("CPU 1%", 0, font), tag_lines("CPU 1%", 0));
        }
        assert_eq!(TagFont::random(9), TagFont::random(9), "the same for a seed");
        let seen: std::collections::HashSet<_> = (0..60).map(|s| format!("{:?}", TagFont::random(s))).collect();
        assert!(seen.len() >= 4, "seeds pick different fonts: {seen:?}");
    }

    #[test]
    fn every_tag_position_keeps_tags_on_the_picture_and_beside_the_right_solid() {
        let (w, h) = (200, 70);
        let anim = AnimParams::default();
        let slots: Vec<crate::scene::SlotLayout> =
            (0..5).map(|i| crate::scene::SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let cam = crate::scene::CameraParams { mode: crate::scene::camera::CameraMode::Orbit, ..Default::default() };
        let labels: Vec<String> = ["CPU 12%", "RAM 40%", "GPU 3%", "DSK 1.0M", "NET 0.2M"].map(String::from).to_vec();
        let ramp = Ramp::default();
        let ch = vec![crate::metrics::Channel { available: true, level: 0.6, spin_y: 0.5, spin_x: 0.2, activity: 0.6 }; 5];
        for place in TagPlace::ALL {
            let mut scene = Scene::new(&slots, &anim, &cam);
            let mut r = Renderer::new();
            let mut ov = Overlays::new(&scene);
            let opt = OverlayOptions { tag_place: place, ..tag_opts(&ramp, &labels, None, false) };
            for step in 0..12 {
                r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
                let view = r.view().unwrap();
                let chars = ov.compose(&view, w, h, &scene, &opt).chars.to_vec();
                // every tag drawn once, whole
                for l in &labels {
                    let text = format!("{TAG_RULE}{l}{TAG_RULE}");
                    let found = chars.chunks(w).filter(|row| row.iter().collect::<String>().contains(&text)).count();
                    assert_eq!(found, 1, "{place:?} step {step}: {text} drawn once");
                }
                let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, 0))).collect();
                match place {
                    TagPlace::Below => {
                        let a = place_tags_below(&view, w, h, &scene, &boxes, TAG_CLEAR_PLAIN, true, None);
                        for (i, s) in scene.solids.iter().enumerate() {
                            let ((cx, cy), _) = screen_circle(&view, s.center, s.scale).unwrap();
                            let a = a[i].unwrap();
                            let mid = a.x as f32 + boxes[i].0 as f32 * 0.5;
                            assert!(a.y as f32 > cy, "{step}: tag {i} under its solid");
                            assert!((mid - cx).abs() < 1.5 || a.x == 0 || a.x + boxes[i].0 == w, "{step}: tag {i} centred ({mid} vs {cx})");
                        }
                    }
                    TagPlace::Corner => {
                        let a = place_tags_corner(&mut Vec::new(), &view, w, h, &scene, &boxes, None);
                        for (i, s) in scene.solids.iter().enumerate() {
                            let (cx, cy) = view.proj.project(s.center).unwrap();
                            let a = a[i].unwrap();
                            let at_edge = if cx < w as f32 / 2.0 { a.x + 1 - boxes[i].0 == TAG_CORNER_MARGIN } else { a.x + boxes[i].0 == w - TAG_CORNER_MARGIN };
                            assert!(at_edge, "{step}: tag {i} at its side's edge");
                            assert_eq!(a.above, cy < h as f32 / 2.0, "{step}: tag {i} in its half's corner");
                        }
                    }
                    _ => {}
                }
                scene.update(&ch, &anim, 0.25);
            }
        }
    }

    #[test]
    fn trailing_tags_settle_behind_the_motion() {
        let (w, h) = (200, 70);
        let anim = AnimParams::default();
        let cam = crate::scene::CameraParams { mode: crate::scene::camera::CameraMode::Orbit, ..Default::default() };
        let slots: Vec<crate::scene::SlotLayout> =
            (0..5).map(|i| crate::scene::SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let mut scene = Scene::new(&slots, &anim, &cam);
        let boxes = vec![(TAG_MIN_W, 1); 5];
        let mut trails = Vec::new();
        let mut r = Renderer::new();
        let ch = vec![crate::metrics::Channel { available: true, level: 0.5, spin_y: 0.3, spin_x: 0.1, activity: 0.5 }; 5];
        let mut last: Vec<(f32, f32)> = Vec::new();
        for step in 0..60 {
            r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
            let view = r.view().unwrap();
            let aspect = frame_aspect(&view);
            let a = place_tags_trailing(&mut trails, &view, w, h, &scene, &boxes, TAG_CLEAR_PLAIN, true, aspect);
            let now: Vec<(f32, f32)> = scene.solids.iter().map(|s| view.proj.project(s.center).unwrap()).collect();
            if step >= 40 {
                for i in 0..5 {
                    let v = ((now[i].0 - last[i].0) * aspect, now[i].1 - last[i].1);
                    let (_, dir) = trails[i].unwrap();
                    if v.0.hypot(v.1) > 0.3 {
                        // the trail points (mostly) the other way from the motion
                        assert!(dir.0 * v.0 + dir.1 * v.1 < 0.0, "step {step}: tag {i} trails its solid");
                    }
                    assert!(a[i].is_some());
                }
            }
            last = now;
            scene.update(&ch, &anim, 0.25);
        }
    }

    #[test]
    fn larger_block_letters_keep_their_strokes() {
        let one = tag_lines("H1", 1);
        for k in 2..=4u8 {
            let big = tag_lines("H1", k);
            let k = k as usize;
            assert_eq!(big.len(), 3 * k);
            assert!(big.iter().all(|r| r.len() == one[0].len() * k));
            // the same strokes, k times over: every glyph of the 1x letters
            // shows up, and nothing else
            let glyphs = |rows: &[Vec<char>]| rows.iter().flatten().filter(|c| **c != ' ').copied().collect::<std::collections::BTreeSet<char>>();
            assert_eq!(glyphs(&big), glyphs(&one), "{k}x");
            let count = |rows: &[Vec<char>], c: char| rows.iter().flatten().filter(|x| **x == c).count();
            assert_eq!(count(&big, '|'), count(&one, '|') * k, "{k}x: bars run k rows");
            assert_eq!(count(&big, '_'), count(&one, '_') * k, "{k}x: rules run k columns");
        }
    }

    #[test]
    fn circle_tags_sit_on_the_circle_nearest_their_solids_and_glide() {
        let (w, h) = (200, 70);
        let anim = AnimParams::default();
        let slots: Vec<crate::scene::SlotLayout> =
            (0..5).map(|i| crate::scene::SlotLayout { solid: SolidKind::ALL[i], ring: None }).collect();
        let cam = crate::scene::CameraParams { mode: crate::scene::camera::CameraMode::Orbit, ..Default::default() };
        let mut scene = Scene::new(&slots, &anim, &cam);
        let labels: Vec<String> = ["CPU 12%", "RAM 40%", "GPU 3%", "DSK 1.0M", "NET 0.2M"].map(String::from).to_vec();
        let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, 0))).collect();
        let (mut angles, mut sides) = (Vec::new(), Vec::new());
        let mut r = Renderer::new();
        let mut last: Option<Vec<f32>> = None;
        let ch = vec![crate::metrics::Channel { available: true, level: 0.6, spin_y: 0.5, spin_x: 0.2, activity: 0.6 }; 5];
        for step in 0..40 {
            r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
            let view = r.view().unwrap();
            let aspect = frame_aspect(&view);
            let anchors = place_tags_circle(&mut angles, &mut sides, &view, w, h, &scene, &boxes, aspect, None);
            let radius = TAG_CIRCLE * 0.5 * h as f32;
            for (i, a) in anchors.iter().enumerate() {
                let a = a.unwrap_or_else(|| panic!("step {step}: tag {i} placed"));
                // on the circle (the box may be nudged onto the picture)
                let d = ((a.x as f32 - w as f32 / 2.0) * aspect).hypot(a.y as f32 - h as f32 / 2.0);
                assert!((d - radius).abs() < 3.0, "step {step}: tag {i} {d:.1} rows from the centre, circle {radius:.1}");
            }
            // once settled, each tag is near the circle point closest to its solid
            if step >= 20 {
                for (i, s) in scene.solids.iter().enumerate() {
                    let (cx, cy) = view.proj.project(s.center).unwrap();
                    let want = (cy - h as f32 / 2.0).atan2((cx - w as f32 / 2.0) * aspect);
                    let off = (angles[i] - want + PI).rem_euclid(TAU) - PI;
                    assert!(off.abs() < 0.9, "step {step}: tag {i} {off:.2} rad from its solid's direction");
                }
            }
            // no two tags overlap
            let rect = |i: usize| {
                let a = anchors[i].unwrap();
                let (tw, th) = boxes[i];
                let (r0, r1) = if a.above { (a.y + 1 - th, a.y) } else { (a.y, a.y + th - 1) };
                let (c0, c1) = if a.rightward { (a.x, a.x + tw - 1) } else { (a.x + 1 - tw, a.x) };
                (r0, r1, c0, c1)
            };
            for i in 0..5 {
                for j in i + 1..5 {
                    let (a, b) = (rect(i), rect(j));
                    assert!(a.1 < b.0 || b.1 < a.0 || a.3 < b.2 || b.3 < a.2, "step {step}: tags {i} and {j} overlap");
                }
            }
            // gliding: no tag jumps far round the circle in one frame
            if let Some(prev) = &last {
                for (a, b) in prev.iter().zip(&angles) {
                    assert!(((b - a + PI).rem_euclid(TAU) - PI).abs() < 0.6, "step {step}: smooth");
                }
            }
            last = Some(angles.clone());
            scene.update(&ch, &anim, 0.25);
        }
    }

    /// A scene of `n` solids at `level`, rendered at `w × h`.
    fn tag_scene(n: usize, level: f32, layout: crate::scene::Layout, secs: f32) -> (Scene, Renderer) {
        let anim = AnimParams::default();
        let slots: Vec<crate::scene::SlotLayout> =
            (0..n).map(|i| crate::scene::SlotLayout { solid: crate::scene::sdf::SolidKind::ALL[i % 15], ring: None }).collect();
        let cam = crate::scene::CameraParams { mode: crate::scene::camera::CameraMode::Orbit, ..Default::default() };
        let mut scene = Scene::with_layout(&slots, &anim, &cam, layout);
        let ch = vec![crate::metrics::Channel { available: true, level, spin_y: 0.2, spin_x: 0.1, activity: level }; n];
        // scale follows the level at once; the camera turns by `secs`
        scene.update(&ch, &anim, secs);
        (scene, Renderer::new())
    }

    fn leader_of(view: &FrameView<'_>, scene: &Scene, i: usize, a: &TagAnchor, fixed: bool, aspect: f32) -> ((f32, f32), (f32, f32)) {
        let s = &scene.solids[i];
        let ((cx, cy), rr) = if fixed {
            screen_circle(view, s.center, s.scale).unwrap()
        } else {
            screen_circle(view, scene.full_size_center(i), s.scale.max(scene.max_radius)).unwrap()
        };
        let rr = rr * TAG_CLEAR_PLAIN;
        let (x, y) = (a.x as f32, a.y as f32);
        let len = ((x - cx) * aspect).hypot(y - cy).max(1e-3);
        let k = (rr / len).min(1.0);
        ((x, y), (cx + (x - cx) * k, cy + (y - cy) * k))
    }

    #[test]
    fn leader_lines_never_cross() {
        let (w, h) = (200, 64);
        let mut checked = 0;
        for (n, layout) in [(6, crate::scene::Layout::Rings), (9, crate::scene::Layout::Rings), (8, crate::scene::Layout::Sphere)] {
            for secs in [0.5, 7.0, 19.0, 33.0] {
                for fixed in [false, true] {
                    let (scene, mut r) = tag_scene(n, 0.6, layout, secs);
                    r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
                    let view = r.view().unwrap();
                    let aspect = frame_aspect(&view);
                    let boxes = vec![(TAG_MIN_W, 1); n];
                    let anchors = place_tags(&mut Vec::new(), &view, w, h, &scene, &boxes, TAG_CLEAR_PLAIN, aspect, None, fixed);
                    let lines: Vec<_> = anchors
                        .iter()
                        .enumerate()
                        .filter_map(|(i, a)| a.as_ref().map(|a| leader_of(&view, &scene, i, a, fixed, aspect)))
                        .collect();
                    for (p, a) in lines.iter().enumerate() {
                        for b in &lines[p + 1..] {
                            assert!(!segments_cross(*a, *b), "{n} {layout:?} t={secs} fixed={fixed}: {a:?} crosses {b:?}");
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 100);
        assert!(segments_cross(((0.0, 0.0), (2.0, 2.0)), ((0.0, 2.0), (2.0, 0.0))));
        assert!(!segments_cross(((0.0, 0.0), (1.0, 1.0)), ((2.0, 0.0), (3.0, 1.0))));
    }

    #[test]
    fn fixed_tag_length_keeps_leaders_the_same_length() {
        let (w, h) = (200, 64);
        let aspect_of = |r: &Renderer| frame_aspect(&r.view().unwrap());
        let gap = |level: f32| {
            let (scene, mut r) = tag_scene(5, level, crate::scene::Layout::Rings, 3.0);
            r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
            let aspect = aspect_of(&r);
            let view = r.view().unwrap();
            let boxes = vec![(TAG_MIN_W, 1); 5];
            let anchors = place_tags(&mut Vec::new(), &view, w, h, &scene, &boxes, TAG_CLEAR_PLAIN, aspect, None, true);
            anchors
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let a = a.unwrap();
                    let ((x0, y0), (x1, y1)) = leader_of(&view, &scene, i, &a, true, aspect);
                    ((x1 - x0) * aspect).hypot(y1 - y0)
                })
                .collect::<Vec<f32>>()
        };
        let (small, big) = (gap(0.2), gap(0.9));
        let same = small.iter().zip(&big).filter(|(a, b)| (*a - *b).abs() <= 1.5).count();
        assert!(same >= 4, "leader lengths small {small:?} vs big {big:?}");
    }

    #[test]
    fn tags_point_toward_their_nearest_corner() {
        let anim = AnimParams::default();
        let (w, h) = (200, 64);
        for (layout, yaw) in [(crate::scene::Layout::Rings, 0.0f32), (crate::scene::Layout::Sphere, 1.3)] {
            let slots: Vec<crate::scene::SlotLayout> = (0..8)
                .map(|i| crate::scene::SlotLayout { solid: crate::scene::sdf::SolidKind::ALL[i], ring: None })
                .collect();
            let cam = crate::scene::CameraParams { mode: crate::scene::camera::CameraMode::Orbit, ..Default::default() };
            let mut scene = Scene::with_layout(&slots, &anim, &cam, layout);
            scene.update(&[crate::metrics::Channel { available: true, level: 0.5, spin_y: 0.2, spin_x: 0.1, activity: 0.5 }; 8], &anim, yaw);
            let mut r = Renderer::new();
            r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
            let view = r.view().unwrap();
            let boxes = vec![(TAG_MIN_W, 1); 8];
            let anchors = place_tags(&mut Vec::new(), &view, w, h, &scene, &boxes, TAG_CLEAR_PLAIN, frame_aspect(&view), None, false);
            let mut checked = 0;
            for (i, a) in anchors.iter().enumerate() {
                let Some(a) = a else { continue };
                let Some(((cx, cy), _)) = screen_circle(&view, scene.full_size_center(i), scene.max_radius) else { continue };
                let (dx, dy) = (cx - w as f32 / 2.0, cy - h as f32 / 2.0);
                if dx.abs() < 0.1 * w as f32 || dy.abs() < 0.1 * h as f32 {
                    continue; // near a centre line: either corner is fine
                }
                assert_eq!(a.rightward, dx > 0.0, "{layout:?} solid {i} at ({cx:.0}, {cy:.0}): tag points left/right away from the centre");
                assert_eq!(a.above, dy < 0.0, "{layout:?} solid {i} at ({cx:.0}, {cy:.0}): tag points up/down away from the centre");
                checked += 1;
            }
            assert!(checked >= 2, "{layout:?}: {checked} solids clearly off-centre");
        }
    }

    #[test]
    fn tags_keep_out_of_the_legend() {
        let scene = Scene::demo(&AnimParams::default());
        let ramp = Ramp::default();
        let labels: Vec<String> = ["CPU 12%", "RAM 40%", "GPU 3%", "DSK 1.0M", "NET 0.2M"].map(String::from).to_vec();
        for (w, h, pos, size) in [(120, 40, LegendPos::TopLeft, 2), (160, 50, LegendPos::BottomRight, 3), (100, 34, LegendPos::TopRight, 1)] {
            let mut r = Renderer::new();
            r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
            let view = r.view().unwrap();
            let legend = LegendParams { size, opacity: 1.0, pos };
            let with = |tags: bool| {
                let opt = OverlayOptions { legend: Some(&legend), tags, ..tag_opts(&ramp, &labels, None, true) };
                Overlays::new(&scene).compose(&view, w, h, &scene, &opt).chars.to_vec()
            };
            let (plain, tagged) = (with(false), with(true));
            let p = layout(&legend, w, h, &labels, 0.5).unwrap();
            for y in p.y..p.y + p.h {
                for x in p.x..p.x + p.w {
                    assert_eq!(tagged[y * w + x], plain[y * w + x], "{pos:?} size {size}: legend cell ({x}, {y}) untouched");
                }
            }
            assert!(tagged != plain, "{pos:?}: tags still drawn outside the legend");
        }
    }

    #[test]
    fn tags_hold_still_while_solids_grow_and_shrink() {
        let (w, h) = (200, 70);
        let anim = AnimParams::default();
        let mut scene = Scene::demo(&anim); // static camera
        let labels: Vec<String> = (0..5).map(|i| format!("S{i} 50%")).collect();
        let mut r = Renderer::new();
        let mut sides = Vec::new();
        let mut first: Option<Vec<Option<TagAnchor>>> = None;
        for level in [0.0, 1.0, 0.3, 0.8] {
            let ch = vec![crate::metrics::Channel { available: true, level, spin_y: 0.0, spin_x: 0.0, activity: level }; 5];
            scene.update(&ch, &anim, 0.05);
            r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
            let view = r.view().unwrap();
            let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, 0))).collect();
            let a = place_tags(&mut sides, &view, w, h, &scene, &boxes, TAG_CLEAR_HALO, frame_aspect(&view), None, false);
            assert!(a.iter().all(Option::is_some));
            match &first {
                None => first = Some(a),
                Some(f) => assert_eq!(&a, f, "load {level}: tags stay where they were"),
            }
        }
    }

    #[test]
    fn tags_cover_the_picture_and_go_with_the_option() {
        let scene = Scene::demo(&AnimParams::default());
        let (w, h) = (W, H);
        let mut r = Renderer::new();
        r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
        let view = r.view().unwrap();
        let ramp = Ramp::default();
        let labels: Vec<String> = (0..5).map(|i| format!("S{i}")).collect();
        let mut off = tag_opts(&ramp, &labels, None, false);
        off.tags = false;
        let plain = Overlays::new(&scene).compose(&view, w, h, &scene, &off).chars.to_vec();
        let tagged = Overlays::new(&scene).compose(&view, w, h, &scene, &tag_opts(&ramp, &labels, None, false)).chars.to_vec();
        assert!(find(&plain, w, "__S0__").is_none());
        let (y, x) = find(&tagged, w, "__S0__").unwrap();
        // the text replaces whatever was rendered there
        let row: String = tagged[y * w + x..y * w + x + 6].iter().collect();
        assert_eq!(row, "__S0__");
        assert!(tagged.iter().zip(&plain).filter(|(a, b)| a != b).count() > 5 * 6);
    }

    #[test]
    fn halo_only_updates_keep_tags_and_match_a_full_compose() {
        let scene = Scene::demo(&AnimParams::default());
        let (w, h) = (W, H);
        let mut r = Renderer::new();
        r.render(&scene, w, h, &RenderSettings { spp: 1, ..RenderSettings::default() });
        let view = r.view().unwrap();
        let ramp = Ramp::default();
        let labels: Vec<String> = (0..5).map(|i| format!("SOLID {i}")).collect();
        for wireframe in [false, true] {
            let mut inc = Overlays::new(&scene);
            let steps = [1.15, 1.45, 1.45, 1.2, 1.0, 1.5];
            let opt = |f: f32| tag_opts(&ramp, &labels, Some(Halo { factor: f, lum: 0.9, reach: TAG_CLEAR_HALO }), wireframe);
            let mut chars = inc.compose(&view, w, h, &scene, &opt(steps[0])).chars.to_vec();
            for (k, &f) in steps.iter().enumerate().skip(1) {
                let u = inc.update_halo(&view, w, h, &scene, &opt(f));
                for &c in u.dirty {
                    chars[c] = u.chars[c];
                }
                assert!(u.dirty.windows(2).all(|p| p[0] < p[1]), "ascending, no repeats");
                if k == 2 {
                    assert!(u.dirty.is_empty(), "same halo again: nothing to redraw");
                }
                let full = Overlays::new(&scene).compose(&view, w, h, &scene, &opt(f)).chars.to_vec();
                assert_eq!(chars, full, "step {k} (wireframe {wireframe})");
                assert!(find(&chars, w, "__SOLID 3__").is_some(), "tags survive halo frames");
            }
        }
    }

    #[test]
    fn no_overlays_is_plain_shading() {
        let (plain, out) = frame_with(false, None);
        assert_eq!(plain, out);
    }

    #[test]
    fn wireframe_changes_only_solid_cells() {
        let (plain, out) = frame_with(true, None);
        let changed = plain.iter().zip(&out).filter(|(a, b)| a != b).count();
        assert!(changed > 50, "edges drawn: {changed}");
        assert!(changed < plain.len() / 3, "not the whole picture: {changed}");
    }

    #[test]
    fn halo_factor_swings_10_30_or_60_percent_around_1_2() {
        for (swing, lo, hi) in [(0.10, 1.10, 1.30), (0.30, 0.90, 1.50), (0.60, 0.60, 1.80)] {
            assert!((halo_factor(0.0, swing) - lo).abs() < 1e-6, "{swing}");
            assert!((halo_factor(0.5, swing) - 1.20).abs() < 1e-6, "{swing}");
            assert!((halo_factor(1.0, swing) - hi).abs() < 1e-6, "{swing}");
            assert!((halo_reach(swing) - hi).abs() < 1e-6, "{swing}");
            assert_eq!(halo_factor(7.0, swing), halo_factor(1.0, swing));
            assert_eq!(halo_factor(f32::NAN, swing), halo_factor(0.0, swing));
        }
        assert_eq!(HALO_SWINGS, [0.10, 0.30, 0.60]);
    }

    #[test]
    fn halo_lum_rises_with_bass_level() {
        assert_eq!(halo_lum(0.0), HALO_LUM_MIN);
        assert_eq!(halo_lum(1.0), 1.0);
        assert!(halo_lum(0.3) < halo_lum(0.6));
        assert_eq!(halo_lum(f32::NAN), HALO_LUM_MIN);
    }

    fn bright(factor: f32) -> Option<Halo> {
        Some(Halo { factor, lum: 1.0, reach: TAG_CLEAR_HALO })
    }

    #[test]
    fn sound_halo_draws_its_luminosity() {
        for lum in [0.0, 0.5, 1.0] {
            let (_, out, _, _) = frame_halo(false, None, None, Some(Halo { factor: 1.2, lum, reach: TAG_CLEAR_HALO }));
            let (_, plain, _, _) = frame_halo(false, None, None, None);
            let g = Ramp::default().glyph(lum);
            let changed: Vec<char> = plain.iter().zip(&out).filter(|(a, b)| a != b).map(|(_, b)| *b).collect();
            assert!(!changed.is_empty(), "lum {lum}");
            assert!(changed.iter().all(|&c| c == g), "lum {lum}: halo cells are all {g:?}");
        }
    }

    #[test]
    fn halo_only_updates_match_a_full_compose() {
        let scene = Scene::demo(&AnimParams::default());
        let mut r = Renderer::new();
        r.render(&scene, W, H, &RenderSettings { spp: 1, ..RenderSettings::default() });
        let view = r.view().unwrap();
        let ramp = Ramp::default();
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        let g = |s: &str| crate::color::Gradient::parse(s).ok();
        let lum_cm = ColorMap::new(g("inferno"), ColorBy::Lum, 64, g("inferno")).unwrap();
        let load_cm = ColorMap::new(g("inferno"), ColorBy::Load, 16, None).unwrap();
        let spin_cm = ColorMap::new(g("plasma"), ColorBy::Spin, 128, g("ice")).unwrap();
        for (wireframe, cm) in [(false, None), (true, Some(&lum_cm)), (true, Some(&load_cm)), (false, Some(&spin_cm))] {
            let opt = |halo| OverlayOptions {
                wireframe,
                legend: None,
                labels: &[],
                char_aspect: 0.5,
                ramp: &ramp,
                exposure: 1.0,
                color: cm,
                levels: &levels,
                halo,
                halo_factors: &[],
                halo_color: Default::default(),
                tags: false,
                tag_labels: &[],
                tag_ink: color::NONE,
                tag_scale: 0,
                tag_place: TagPlace::Circle,
                tag_font: TagFont::Strokes,
                tag_fixed: false,
            };
            let steps = [
                Some(Halo { factor: 1.15, lum: 0.2, reach: TAG_CLEAR_HALO }),
                Some(Halo { factor: 1.25, lum: 0.9, reach: TAG_CLEAR_HALO }),
                Some(Halo { factor: 1.25, lum: 0.9, reach: TAG_CLEAR_HALO }),
                Some(Halo { factor: 1.18, lum: 0.5, reach: TAG_CLEAR_HALO }),
                None,
                Some(Halo { factor: 1.2, lum: 1.0, reach: TAG_CLEAR_HALO }),
            ];
            let mut inc = Overlays::new(&scene);
            let first = inc.compose(&view, W, H, &scene, &opt(steps[0]));
            let (mut chars, mut colors) = (first.chars.to_vec(), first.colors.map(<[u32]>::to_vec));
            for (k, &h) in steps.iter().enumerate().skip(1) {
                let u = inc.update_halo(&view, W, H, &scene, &opt(h));
                // apply only the reported cells to the previous picture
                for &c in u.dirty {
                    chars[c] = u.chars[c];
                    if let (Some(cs), Some(uc)) = (colors.as_mut(), u.colors) {
                        cs[c] = uc[c];
                    }
                }
                assert!(u.dirty.windows(2).all(|p| p[0] < p[1]), "ascending, no repeats");
                if k == 2 {
                    assert!(u.dirty.is_empty(), "same halo again: nothing to redraw");
                }
                let mut full = Overlays::new(&scene);
                let f = full.compose(&view, W, H, &scene, &opt(h));
                assert_eq!(chars, f.chars, "step {k} ({wireframe}, {:?}) glyphs", cm.map(|c| c.by));
                assert_eq!(colors.as_deref(), f.colors, "step {k} colours");
            }
        }
    }

    #[test]
    fn halo_colour_sources() {
        use crate::color::Gradient;
        let scene = Scene::demo(&AnimParams::default());
        let mut r = Renderer::new();
        r.render(&scene, W, H, &RenderSettings { spp: 1, ..RenderSettings::default() });
        let view = r.view().unwrap();
        let ramp = Ramp::default();
        let levels = [0.1f32, 0.3, 0.5, 0.7, 0.9];
        let g = Gradient::parse("rainbow").unwrap();
        for by in [ColorBy::Load, ColorBy::Lum] {
            let cm = ColorMap::new(Some(g.clone()), by, 32, Some(g.clone())).unwrap();
            let opt = |src: HaloColor, halo: Halo, factors: &'static [f32]| OverlayOptions {
                wireframe: false,
                legend: None,
                labels: &[],
                char_aspect: 0.5,
                ramp: &ramp,
                exposure: 1.0,
                color: Some(&cm),
                levels: &levels,
                halo: Some(halo),
                halo_factors: factors,
                halo_color: src,
                tags: false,
                tag_labels: &[],
                tag_ink: color::NONE,
                tag_scale: 0,
                tag_place: TagPlace::Circle,
                tag_font: TagFont::Strokes,
                tag_fixed: false,
            };
            let loud = Halo { factor: 1.2, lum: 1.0, reach: 1.5 };
            let quiet = Halo { lum: HALO_LUM_MIN, ..loud };
            let draw = |o: &OverlayOptions<'_>| {
                let mut ov = Overlays::new(&scene);
                let colors = ov.compose(&view, W, H, &scene, o).colors.unwrap().to_vec();
                (ov.halo_cells.clone(), ov.halo_owner.clone(), colors)
            };
            // under: the colour of what lies beneath, as before
            let (cells, _, under) = draw(&opt(HaloColor::Under, loud, &[]));
            assert!(cells.len() > 50, "{by:?}: a halo is drawn");
            for &c in &cells {
                assert_eq!(under[c], cell_color(&cm, &view, &ramp, &levels, 5, c, 1.0));
            }
            // solid: each cell its own solid's colour, also over the background
            let (cells, owner, solid) = draw(&opt(HaloColor::Solid, loud, &[]));
            let over_bg = cells.iter().filter(|&&c| view.ids[c] as usize >= 5).count();
            assert!(over_bg > 10, "{by:?}: some halo lies over the background");
            for &c in &cells {
                let o = owner[c] as usize;
                assert_eq!(solid[c], cm.solid_color(Some(levels[o]), 1.0, ramp.level(1.0), ramp.len()), "{by:?} cell {c}");
            }
            if by == ColorBy::Load {
                assert!(cells.iter().any(|&c| solid[c] != under[c]), "solid differs from under somewhere");
            }
            // size: a large halo and a small one sit at opposite ends of the gradient
            let (cells, owner, big) = draw(&opt(HaloColor::Size, loud, &[1.5, 0.9, 1.2, 1.2, 1.2]));
            let of = |k: usize| cells.iter().find(|&&c| owner[c] as usize == k).map(|&c| big[c]).unwrap();
            assert_ne!(of(0), of(1), "{by:?}: size colours differ");
            assert_eq!(of(0), cm.value_color(1.0, ramp.level(1.0), ramp.len()));
            assert_eq!(of(1), cm.value_color(0.0, ramp.level(1.0), ramp.len()));
            // bass: one colour for all, following the bass level
            let (cells, _, b_loud) = draw(&opt(HaloColor::Bass, loud, &[]));
            let (cells_q, _, b_quiet) = draw(&opt(HaloColor::Bass, quiet, &[]));
            assert!(cells.iter().all(|&c| b_loud[c] == b_loud[cells[0]]), "{by:?}: one bass colour");
            assert_ne!(b_loud[cells[0]], b_quiet[cells_q[0]], "{by:?}: bass colour follows the level");
            // halo-only updates agree with a full compose in every mode
            for src in HaloColor::ALL {
                let mut inc = Overlays::new(&scene);
                inc.compose(&view, W, H, &scene, &opt(src, quiet, &[]));
                let u = inc.update_halo(&view, W, H, &scene, &opt(src, loud, &[1.4, 1.0, 1.2, 1.3, 1.1])).colors.unwrap().to_vec();
                let full = Overlays::new(&scene).compose(&view, W, H, &scene, &opt(src, loud, &[1.4, 1.0, 1.2, 1.3, 1.1])).colors.unwrap().to_vec();
                assert_eq!(u, full, "{by:?} {src:?}: incremental matches full");
            }
        }
    }

    #[test]
    fn per_solid_halo_sizes_move_only_their_solid() {
        let scene = Scene::demo(&AnimParams::default());
        let mut r = Renderer::new();
        r.render(&scene, W, H, &RenderSettings { spp: 1, ..RenderSettings::default() });
        let view = r.view().unwrap();
        let ramp = Ramp::default();
        let centres: Vec<(f32, f32)> = scene.solids.iter().filter_map(|s| view.proj.project(s.center)).collect();
        let opt = |f: &'static [f32]| OverlayOptions {
            wireframe: false,
            legend: None,
            labels: &[],
            char_aspect: 0.5,
            ramp: &ramp,
            exposure: 1.0,
            color: None,
            levels: &[],
            halo: Some(Halo { factor: 1.2, lum: 1.0, reach: TAG_CLEAR_HALO }),
            halo_factors: f,
            halo_color: Default::default(),
            tags: false,
            tag_labels: &[],
            tag_ink: color::NONE,
            tag_scale: 0,
            tag_place: TagPlace::Circle,
            tag_font: TagFont::Strokes,
            tag_fixed: false,
        };
        let calm: &'static [f32] = &[1.15; 5];
        let base = Overlays::new(&scene).compose(&view, W, H, &scene, &opt(calm)).chars.to_vec();
        // the factors override `halo.factor`
        let single = Overlays::new(&scene).compose(&view, W, H, &scene, &opt(&[])).chars.to_vec();
        assert_ne!(base, single);
        let bumps: [&'static [f32]; 5] = [
            &[1.4, 1.15, 1.15, 1.15, 1.15],
            &[1.15, 1.4, 1.15, 1.15, 1.15],
            &[1.15, 1.15, 1.4, 1.15, 1.15],
            &[1.15, 1.15, 1.15, 1.4, 1.15],
            &[1.15, 1.15, 1.15, 1.15, 1.4],
        ];
        let mut inc = Overlays::new(&scene);
        inc.compose(&view, W, H, &scene, &opt(calm));
        for (k, f) in bumps.into_iter().enumerate() {
            let out = Overlays::new(&scene).compose(&view, W, H, &scene, &opt(f)).chars.to_vec();
            let diff: Vec<usize> = (0..W * H).filter(|&i| out[i] != base[i]).collect();
            assert!(!diff.is_empty(), "solid {k}'s halo moved");
            let nearest = |i: usize| {
                let (x, y) = ((i % W) as f32 + 0.5, (i / W) as f32 + 0.5);
                (0..centres.len())
                    .min_by(|&a, &b| {
                        let d = |c: (f32, f32)| ((x - c.0) * 0.5).hypot(y - c.1);
                        d(centres[a]).total_cmp(&d(centres[b]))
                    })
                    .unwrap()
            };
            let near_k = diff.iter().filter(|&&i| nearest(i) == k).count();
            assert!(near_k * 10 >= diff.len() * 8, "solid {k}: {near_k}/{} changed cells near it", diff.len());
            // halo-only updates agree with a full compose
            let u = inc.update_halo(&view, W, H, &scene, &opt(f));
            assert_eq!(u.chars, &out[..], "solid {k}: incremental matches full");
        }
    }

    #[test]
    fn sound_halo_draws_outside_the_solids_and_grows_with_level() {
        let (plain, _, _, ids) = frame_halo(false, None, None, None);
        let changed = |f: f32| {
            let (_, out, _, _) = frame_halo(false, None, None, bright(f));
            plain.iter().zip(&out).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect::<Vec<_>>()
        };
        let quiet = changed(halo_factor(0.0, HALO_SWINGS[0]));
        let loud = changed(halo_factor(1.0, HALO_SWINGS[0]));
        assert!(quiet.len() > 50, "halo drawn: {}", quiet.len());
        assert!(quiet.len() < plain.len() / 3);
        // ids 0..5 are the demo's solids; anything else is background
        let outside = quiet.iter().filter(|&&i| ids[i] >= 5).count();
        // a 1.1x halo's front edges fall partly over its own solid; a good
        // share still lies outside, over the background
        assert!(outside * 3 > quiet.len(), "around the solids, not only on them: {outside}/{}", quiet.len());
        assert_ne!(quiet, loud, "the halo moves with the level");
        // the loud halo sits further from the solids' centres on average
        let scene = Scene::demo(&AnimParams::default());
        let mut r = Renderer::new();
        r.render(&scene, W, H, &RenderSettings { spp: 1, ..RenderSettings::default() });
        let proj = r.view().unwrap().proj;
        let centres: Vec<(f32, f32)> = scene.solids.iter().filter_map(|s| proj.project(s.center)).collect();
        assert_eq!(centres.len(), 5);
        let spread = |cells: &[usize]| {
            cells
                .iter()
                .map(|&i| {
                    let (x, y) = ((i % W) as f32 + 0.5, (i / W) as f32 + 0.5);
                    centres.iter().map(|c| ((x - c.0) * 0.5).hypot(y - c.1)).fold(f32::MAX, f32::min)
                })
                .sum::<f32>()
                / cells.len() as f32
        };
        assert!(spread(&loud) > spread(&quiet), "{} vs {}", spread(&loud), spread(&quiet));
        // the wide swing goes further out when loud and inside the solid when quiet
        let wide_loud = changed(halo_factor(1.0, HALO_SWINGS[1]));
        let wide_quiet = changed(halo_factor(0.0, HALO_SWINGS[1]));
        assert!(spread(&wide_loud) > spread(&loud), "{} vs {}", spread(&wide_loud), spread(&loud));
        assert!(spread(&wide_quiet) < spread(&quiet), "{} vs {}", spread(&wide_quiet), spread(&quiet));
        let on_solids = wide_quiet.iter().filter(|&&i| ids[i] < 5).count();
        assert!(on_solids * 2 > wide_quiet.len(), "0.9x lies over the solids: {on_solids}/{}", wide_quiet.len());
        // wireframe and halo together mark both
        let (_, both, _, _) = frame_halo(true, None, None, bright(1.2));
        let (_, wire, _, _) = frame_halo(true, None, None, None);
        assert!(both.iter().zip(&wire).filter(|(a, b)| a != b).count() > 50);
    }

    #[test]
    fn opaque_legend_blanks_its_panel_and_draws_labels() {
        let p = LegendParams { size: 2, opacity: 1.0, pos: LegendPos::BottomRight };
        let (_, out) = frame_with(false, Some(&p));
        let text: String = out.chunks(100).map(|r| r.iter().collect::<String>() + "\n").collect();
        for i in 0..5 {
            assert!(text.contains(&format!("CPU {i}%")), "label {i}:\n{text}");
        }
        assert!(text.contains('/') || text.contains('\\') || text.contains('|'), "icons drawn");
    }
}
