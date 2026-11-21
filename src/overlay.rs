//! Overlays drawn over the shaded picture:
//!
//! * **wireframe** — each solid's visible edges, drawn into the picture with
//!   inverted luminosity so neighbouring solids separate clearly;
//! * **legend** — a panel in one corner with a small wireframe of every solid
//!   (spinning with it) next to the metric it shows.

use glam::Vec3A;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::ascii::{self, Ramp};
use crate::color::{self, ColorBy, ColorMap};
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

// ------------------------------------------------------------------ wire models

/// Unit-circumradius edge model of one solid.
#[derive(Debug, Clone)]
struct Wire {
    verts: Vec<Vec3A>,
    /// Face normals and offsets.
    faces: Vec<(Vec3A, f32)>,
    edges: Vec<(usize, usize, usize, usize)>,
}

impl Wire {
    fn new(kind: SolidKind) -> Self {
        let p = crate::scene::sdf::Polyhedron::new(kind);
        Self { verts: p.vertices(), faces: p.face_planes(), edges: p.edges() }
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
    scratch: Vec<usize>,
    dirty: Vec<usize>,
    tags: TagState,
    journal: Journal,
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
    /// Each slot's load 0..=1 (for `--color-by load`).
    pub levels: &'a [f32],
    /// Halo: a second, larger wireframe of every solid drawn on top, its size
    /// and brightness driven by the caller (e.g. a sound level).
    pub halo: Option<Halo>,
    /// Per-slot halo sizes overriding `halo.factor` (empty = all solids use
    /// `halo.factor`).
    pub halo_factors: &'a [f32],
    /// Solid tags: each solid's label floats at a fixed place on screen with
    /// a leader line to the solid ([`TagState`]), on top of everything.
    pub tags: bool,
    /// One tag text per slot (falls back to the solid's name).
    pub tag_labels: &'a [String],
    /// Draw tag text as three-row block letters (for small text).
    pub tag_blocks: bool,
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
}

/// Underscores either side of a tag's label (`__RAM 42%__`).
const TAG_RULE: &str = "__";
/// Columns reserved per tag when laying them out (labels change length).
const TAG_MIN_W: usize = 14;
/// Room kept around a solid for its halo before the tag starts, as a
/// multiple of the solid's radius: the widest halo swing (1.5x), or just
/// clear of the solid without a halo. The tag does not bounce with the
/// halo; its leader line ends at the halo wherever it is.
const TAG_CLEAR_HALO: f32 = 1.5;
const TAG_CLEAR_PLAIN: f32 = 1.05;
/// Rows of leader line beyond that room.
const TAG_GAP_ROWS: f32 = 3.0;
/// How much further out (rows along the diagonal) a tag first looks for a
/// spot clear of the other solids before trying its other sides.
const TAG_SLACK_ROWS: f32 = 6.0;
/// A tag only moves to the solid's other side once the solid is this far
/// (share of the width) past the picture's centre, so it doesn't flicker.
const TAG_FLIP: f32 = 0.06;

/// Each solid's tag side from last frame, and the cells the tags covered
/// last frame, with what lay under them.
#[derive(Debug, Clone, Default)]
struct TagState {
    /// (horizontal, vertical) side per solid: +1 = right / below, −1 =
    /// left / above, 0 = not placed yet.
    sides: Vec<(i8, i8)>,
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

/// The rows of one tag's text: one row (`__RAM 42%__`) normally, or three
/// rows of [`block_glyph`] letters (one blank column between letters and
/// either side, so the box hides what is behind it).
fn tag_lines(label: &str, blocks: bool) -> Vec<Vec<char>> {
    let label: String = label.chars().filter(|c| !c.is_control()).collect();
    if !blocks {
        return vec![format!("{TAG_RULE}{label}{TAG_RULE}").chars().collect()];
    }
    let mut rows = vec![vec![' '], vec![' '], vec![' ']];
    for c in label.chars() {
        for (row, part) in rows.iter_mut().zip(block_glyph(c)) {
            row.extend(part.chars());
            row.push(' ');
        }
    }
    rows
}

/// A tag's box, (columns, rows); one-line tags reserve a little width (labels
/// change length).
fn tag_box(rows: &[Vec<char>]) -> (usize, usize) {
    let wide = rows.iter().map(Vec::len).max().unwrap_or(0);
    (if rows.len() == 1 { wide.max(TAG_MIN_W) } else { wide }, rows.len())
}

/// Place every solid's tag box for this frame, next to the solid: on the 45°
/// diagonal away from the picture's centre (toward the periphery) and above
/// it, with the text running outward so the leader line only spans the gap
/// between box and solid. Along that diagonal (the leader line's path) the
/// box sits as close to its solid as it can while staying clear of it (at
/// full size, with the room its halo needs, plus a short gap for the line)
/// and of every other solid. A box that would leave the picture or overlap
/// another tag tries the other diagonals. `sides` keeps each tag's side
/// from frame to frame (with hysteresis) so tags don't flip back and forth.
/// `boxes` is each tag's (columns, rows).
#[allow(clippy::too_many_arguments)]
fn place_tags(
    sides: &mut Vec<(i8, i8)>,
    frame: &FrameView<'_>,
    w: usize,
    h: usize,
    scene: &Scene,
    boxes: &[(usize, usize)],
    halo: bool,
    aspect: f32,
    keep_out: Option<(usize, usize, usize, usize)>,
) -> Vec<Option<TagAnchor>> {
    let n = scene.solids.len();
    sides.resize(n, (0, 0));
    let hw = w as f32 * 0.5;
    let clear = if halo { TAG_CLEAR_HALO } else { TAG_CLEAR_PLAIN };
    // Placed for each solid at full size (rings: a bigger solid also sits
    // higher), so a solid growing and shrinking with its load doesn't move
    // its tag; only the camera does. The radius includes the halo's room.
    let circles: Vec<Option<((f32, f32), f32)>> = scene
        .solids
        .iter()
        .enumerate()
        .map(|(i, s)| screen_circle(frame, scene.full_size_center(i), s.scale.max(scene.max_radius)))
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
    let mut out = vec![None; n];
    for i in order {
        let Some(((cx, cy), rr)) = circles[i] else { continue };
        let (tw, th) = boxes.get(i).copied().unwrap_or((TAG_MIN_W, 1));
        let (tw, th) = (tw.clamp(1, w.max(1)), th.clamp(1, h.max(1)));
        let (mut sx, mut sy) = sides[i];
        let outward = cx - hw;
        if sx == 0 {
            sx = if outward >= 0.0 { 1 } else { -1 };
        } else if outward * (sx as f32) < -TAG_FLIP * w as f32 {
            sx = -sx; // clearly on the other half now: follow it outward
        }
        if sy == 0 {
            sy = -1;
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
        let nearest = |sx: i8, sy: i8, others: bool, slack: f32, placed: &[(usize, usize, usize, usize)]| {
            let d0 = own_d(sx, sy);
            let far = (d0 + slack).min(rr * 2.0 + 4.0 * th as f32 + 24.0);
            let mut d = d0;
            while d <= far {
                if let Some(s) = spot(sx, sy, d) {
                    let b = (s.2, s.3, s.4, s.5);
                    let free = !touches(b, ((cx, cy), rr), TAG_GAP_ROWS)
                        && !(others && circles.iter().enumerate().any(|(j, c)| j != i && c.is_some_and(|c| touches(b, c, 0.0))))
                        && !clash(b, placed);
                    if free {
                        return Some(s);
                    }
                }
                d += 1.0;
            }
            None
        };
        // Above is preferred: a tag below its solid goes back up once there
        // is room again.
        let up_fits = nearest(sx, -1, true, TAG_SLACK_ROWS, &placed).is_some();
        let ys = if sy > 0 && up_fits { [-1, 1] } else { [sy, -sy] };
        let sides_to_try = [(sx, ys[0]), (sx, ys[1]), (-sx, ys[0]), (-sx, ys[1])];
        // Clear of every solid and close; then clear of every solid however
        // far; then clear of its own solid only.
        let choice = [(true, TAG_SLACK_ROWS), (true, f32::INFINITY), (false, TAG_SLACK_ROWS)].into_iter().find_map(|(others, slack)| {
            sides_to_try.iter().find_map(|&(hx, vy)| nearest(hx, vy, others, slack, &placed).map(|s| (s, hx, vy)))
        });
        // Nowhere free: the preferred side, pulled onto the picture.
        let ((y, x, r0, r1, c0, c1), hx, vy) = choice.unwrap_or_else(|| {
            let d = rr + TAG_GAP_ROWS;
            let vy = ys[0];
            let y = (cy + vy as f32 * d).round();
            let y = if vy < 0 { y.clamp(th as f32 - 1.0, h as f32 - 1.0) } else { y.clamp(0.0, (h - th) as f32) };
            let x = (cx + sx as f32 * d / aspect).round();
            let x = if sx > 0 { x.clamp(0.0, (w - tw) as f32) } else { x.clamp(tw as f32 - 1.0, w as f32 - 1.0) };
            let (r0, r1) = if vy < 0 { (y - th as f32 + 1.0, y) } else { (y, y + th as f32 - 1.0) };
            let (c0, c1) = if sx > 0 { (x, x + tw as f32 - 1.0) } else { (x - tw as f32 + 1.0, x) };
            ((y, x, r0, r1, c0, c1), sx, vy)
        });
        placed.push((r0 as usize, r1 as usize, c0 as usize, c1 as usize));
        sides[i] = (hx, vy);
        out[i] = Some(TagAnchor { y: y as usize, x: x as usize, rightward: hx > 0, above: vy < 0 });
    }
    out
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
            tag_lines(&label, opt.tag_blocks)
        })
        .collect();
    let boxes: Vec<(usize, usize)> = lines.iter().map(|l| tag_box(l)).collect();
    let anchors = place_tags(&mut t.sides, frame, w, h, scene, &boxes, sizes.is_some(), aspect, t.keep_out);
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
        let sx = if a.rightward { a.x as f32 - 1.0 } else { a.x as f32 + 1.0 } + 0.5;
        let sy = a.y as f32 + if a.above { 1.0 } else { -1.0 } + 0.5;
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
}

/// Halo size swing, as a share of the solid's radius either way.
pub const HALO_SWING: f32 = 0.05;
/// Extra swing with the wide option.
pub const HALO_SWING_WIDE_EXTRA: f32 = 0.25;

/// Halo size: 1.2x the solid's radius, ±5% of the radius with a level
/// 0..=1 (e.g. a sound level: 0 = silent, 1 = loudest recently); `wide` adds another ±25%
/// (0.9x..1.5x).
pub fn halo_factor(level: f32, wide: bool) -> f32 {
    let l = if level.is_finite() { level.clamp(0.0, 1.0) } else { 0.0 };
    let swing = HALO_SWING + if wide { HALO_SWING_WIDE_EXTRA } else { 0.0 };
    1.20 + swing * (2.0 * l - 1.0)
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
fn cell_color(cm: &ColorMap, frame: &FrameView<'_>, ramp: &Ramp, levels: &[f32], i: usize, l: f32) -> u32 {
    let (n, b) = (ramp.len(), ramp.level(l));
    match cm.by {
        ColorBy::Lum => cm.gradient.sample((b as f32 + 0.5) / n as f32, 1.0),
        ColorBy::Load => {
            let id = frame.ids.get(i).copied().unwrap_or(crate::render::NO_HIT) as usize;
            levels.get(id).map_or(color::NONE, |&lv| cm.gradient.sample(lv, load_shade(b, n)))
        }
    }
}

/// Mark every visible edge cell of every solid's wire model scaled by
/// `factors[i]` for solid `i` (1 = the solid itself; past the end of
/// `factors`, its last value) in `mask`; newly marked cells are also pushed
/// to `list`.
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

/// Brightness of `--color-by load` colours for glyph bucket `b` of `n`:
/// never fully black, so the hue stays readable on dark faces.
fn load_shade(b: usize, n: usize) -> f32 {
    0.35 + 0.65 * (b as f32 + 0.5) / n as f32
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
            scratch: Vec::new(),
            dirty: Vec::new(),
            tags: TagState::default(),
            journal: Journal::default(),
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
            if let Some(sizes) = opt.halo_sizes() {
                mark_edges(&self.wires, frame, w, h, scene, &sizes, &mut self.halo_mask, Some(&mut fresh));
            }
            let lum_on = opt.halo.map_or(0.0, |hl| hl.lum);
            for &c in self.halo_cells.iter().chain(&fresh) {
                let l = if self.halo_mask[c] { lum_on } else { self.base[c] };
                self.journal.touch(c, &self.cells, colors);
                self.cells[c] = opt.ramp.glyph(l);
                if let Some(slot) = colors.get_mut(c) {
                    *slot = opt.color.map_or(color::NONE, |cm| cell_color(cm, frame, opt.ramp, opt.levels, c, l));
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
    fn colorize(&mut self, cm: &ColorMap, frame: &FrameView<'_>, ramp: &Ramp, levels: &[f32]) {
        let n = ramp.len();
        self.colors.clear();
        match cm.by {
            ColorBy::Lum => {
                let table: Vec<u32> = (0..n).map(|b| cm.gradient.sample((b as f32 + 0.5) / n as f32, 1.0)).collect();
                self.colors.extend(self.lum.iter().map(|&l| table[ramp.level(l)]));
            }
            ColorBy::Load => {
                // slot × bucket table
                let table: Vec<Vec<u32>> = levels
                    .iter()
                    .map(|&lv| (0..n).map(|b| cm.gradient.sample(lv, load_shade(b, n))).collect())
                    .collect();
                self.colors.extend(self.lum.iter().enumerate().map(|(i, &l)| {
                    let id = frame.ids.get(i).copied().unwrap_or(crate::render::NO_HIT) as usize;
                    table.get(id).map_or(color::NONE, |t| t[ramp.level(l)])
                }));
            }
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
        let ramp = opt.ramp;
        ascii::tone_buffer(frame.radiance, opt.exposure, &mut self.lum);
        self.lum.resize(w * h, 0.0);
        if opt.wireframe {
            self.mask.clear();
            self.mask.resize(w * h, false);
            mark_edges(&self.wires, frame, w, h, scene, &[1.0], &mut self.mask, None);
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
        if let (Some(hl), Some(sizes)) = (opt.halo, opt.halo_sizes()) {
            self.base.clone_from(&self.lum);
            mark_edges(&self.wires, frame, w, h, scene, &sizes, &mut self.halo_mask, Some(&mut self.halo_cells));
            for &c in &self.halo_cells {
                self.lum[c] = hl.lum;
            }
        } else {
            self.base.clear();
        }
        ascii::shade(&self.lum, ramp, &mut self.cells);
        if let Some(cm) = opt.color {
            self.colorize(cm, frame, ramp, opt.levels);
            if let Some((_, panel)) = &panel {
                // Panel background stays uncoloured so labels read clearly.
                for y in panel.y..panel.y + panel.h {
                    self.colors[y * w + panel.x..y * w + panel.x + panel.w].fill(color::NONE);
                }
            }
        }
        if let Some((_, panel)) = &panel {
            let mut g = Grid::new(&mut self.cells, w, h);
            if opt.color.is_some() {
                g.colors = Some(&mut self.colors);
            }
            let icon_ink = |slot: usize| match opt.color {
                Some(cm) if cm.by == ColorBy::Load => {
                    cm.gradient.sample(opt.levels.get(slot).copied().unwrap_or(0.0), 1.0)
                }
                Some(cm) => cm.gradient.sample(1.0, 1.0),
                None => color::NONE,
            };
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
            tags: false,
            tag_labels: &[],
            tag_ink: color::NONE,
            tag_blocks: false,
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
        let cm = ColorMap {
            gradient: crate::color::Gradient::parse("grey").unwrap(),
            by: ColorBy::Lum,
            depth: crate::color::ColorDepth::True,
        };
        let (_, chars, cols, _) = frame_full(false, None, Some(&cm));
        let cols = cols.expect("colours on");
        let ramp: Vec<char> = crate::ascii::DEFAULT_RAMP.chars().collect();
        for (c, col) in chars.iter().zip(&cols) {
            let b = ramp.iter().position(|r| r == c).unwrap();
            assert_eq!(*col & 0xff, ((b as f32 + 0.5) / 16.0 * 255.0).round() as u32, "glyph {c:?}");
        }
        let (_, _, none, _) = frame_full(false, None, None);
        assert!(none.is_none(), "monochrome has no colour buffer");
    }

    #[test]
    fn load_colours_solids_only_and_legend_is_plain() {
        let cm = ColorMap {
            gradient: crate::color::Gradient::parse("#000000:#ff0000").unwrap(),
            by: ColorBy::Load,
            depth: crate::color::ColorDepth::True,
        };
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
            tags: true,
            tag_labels: labels,
            tag_ink: color::NONE,
            tag_blocks: false,
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
            let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, false))).collect();
            let anchors = place_tags(&mut sides, &view, w, h, &scene, &boxes, false, aspect, None);
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
                let full = |j: usize| {
                    let ((x, y), r) = screen_circle(&view, scene.full_size_center(j), scene.max_radius).unwrap();
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
        let rows = tag_lines("RAM 42%", true);
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
        let opt = |f: f32| OverlayOptions { tag_blocks: true, ..tag_opts(&ramp, &labels, Some(Halo { factor: f, lum: 0.9 }), false) };
        let mut inc = Overlays::new(&scene);
        let mut chars = inc.compose(&view, w, h, &scene, &opt(1.2)).chars.to_vec();
        let aspect = frame_aspect(&view);
        let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, true))).collect();
        let anchors = place_tags(&mut Vec::new(), &view, w, h, &scene, &boxes, true, aspect, None);
        for (i, (a, l)) in anchors.iter().zip(&labels).enumerate() {
            let a = a.unwrap();
            let rows = tag_lines(l, true);
            let tw = rows[0].len();
            let top = if a.above { a.y - 2 } else { a.y };
            let x0 = if a.rightward { a.x } else { a.x + 1 - tw };
            for (k, row) in rows.iter().enumerate() {
                let got: String = chars[(top + k) * w + x0..(top + k) * w + x0 + tw].iter().collect();
                assert_eq!(got, row.iter().collect::<String>(), "tag {i} row {k}");
            }
            // the whole box is clear of its own solid's halo room
            let ((cx, cy), rr) = screen_circle(&view, scene.full_size_center(i), scene.max_radius).unwrap();
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
            let boxes: Vec<(usize, usize)> = labels.iter().map(|l| tag_box(&tag_lines(l, false))).collect();
            let a = place_tags(&mut sides, &view, w, h, &scene, &boxes, true, frame_aspect(&view), None);
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
            let opt = |f: f32| tag_opts(&ramp, &labels, Some(Halo { factor: f, lum: 0.9 }), wireframe);
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
    fn halo_factor_spans_plus_minus_five_percent_around_1_2() {
        assert!((halo_factor(0.0, false) - 1.15).abs() < 1e-6);
        assert!((halo_factor(0.5, false) - 1.20).abs() < 1e-6);
        assert!((halo_factor(1.0, false) - 1.25).abs() < 1e-6);
        assert_eq!(halo_factor(7.0, false), halo_factor(1.0, false));
        assert_eq!(halo_factor(f32::NAN, false), halo_factor(0.0, false));
        // wide: an extra ±25% of the radius, ±30% in all
        assert!((halo_factor(0.0, true) - 0.90).abs() < 1e-6);
        assert!((halo_factor(0.5, true) - 1.20).abs() < 1e-6);
        assert!((halo_factor(1.0, true) - 1.50).abs() < 1e-6);
        assert_eq!(halo_factor(f32::NAN, true), halo_factor(0.0, true));
    }

    #[test]
    fn halo_lum_rises_with_bass_level() {
        assert_eq!(halo_lum(0.0), HALO_LUM_MIN);
        assert_eq!(halo_lum(1.0), 1.0);
        assert!(halo_lum(0.3) < halo_lum(0.6));
        assert_eq!(halo_lum(f32::NAN), HALO_LUM_MIN);
    }

    fn bright(factor: f32) -> Option<Halo> {
        Some(Halo { factor, lum: 1.0 })
    }

    #[test]
    fn sound_halo_draws_its_luminosity() {
        for lum in [0.0, 0.5, 1.0] {
            let (_, out, _, _) = frame_halo(false, None, None, Some(Halo { factor: 1.2, lum }));
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
        let lum_cm = ColorMap {
            gradient: crate::color::Gradient::parse("inferno").unwrap(),
            by: ColorBy::Lum,
            depth: crate::color::ColorDepth::True,
        };
        let load_cm = ColorMap { by: ColorBy::Load, ..lum_cm.clone() };
        for (wireframe, cm) in [(false, None), (true, Some(&lum_cm)), (true, Some(&load_cm))] {
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
                tags: false,
                tag_labels: &[],
                tag_ink: color::NONE,
                tag_blocks: false,
            };
            let steps = [
                Some(Halo { factor: 1.15, lum: 0.2 }),
                Some(Halo { factor: 1.25, lum: 0.9 }),
                Some(Halo { factor: 1.25, lum: 0.9 }),
                Some(Halo { factor: 1.18, lum: 0.5 }),
                None,
                Some(Halo { factor: 1.2, lum: 1.0 }),
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
            halo: Some(Halo { factor: 1.2, lum: 1.0 }),
            halo_factors: f,
            tags: false,
            tag_labels: &[],
            tag_ink: color::NONE,
            tag_blocks: false,
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
        let quiet = changed(halo_factor(0.0, false));
        let loud = changed(halo_factor(1.0, false));
        assert!(quiet.len() > 50, "halo drawn: {}", quiet.len());
        assert!(quiet.len() < plain.len() / 3);
        // ids 0..5 are the demo's solids; anything else is background
        let outside = quiet.iter().filter(|&&i| ids[i] >= 5).count();
        assert!(outside * 2 > quiet.len(), "mostly around the solids, not on them: {outside}/{}", quiet.len());
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
        let wide_loud = changed(halo_factor(1.0, true));
        let wide_quiet = changed(halo_factor(0.0, true));
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
